"""Garment factory for creating garment instances.

This module implements a factory pattern for creating garment components,
replacing the dynamic class lookup approach used in MetaGarment.
"""

from typing import Any

from loguru import logger

import weon_garment_code.pygarment.garmentcode as pyg
from weon_garment_code.config import AttachmentConstraint
from weon_garment_code.garment_programs.weon_bands import StraightWB
from weon_garment_code.garment_programs.weon_circle_skirt import (
    AsymmSkirtCircle,
    SkirtCircle,
)
from weon_garment_code.garment_programs.weon_pants import Pants
from weon_garment_code.garment_programs.weon_shirt import FittedShirt, Shirt
from weon_garment_code.pattern_definitions.body_definition import BodyDefinition
from weon_garment_code.pattern_definitions.pants_design import PantsDesign
from weon_garment_code.pattern_definitions.shirt_design import ShirtDesign
from weon_garment_code.pattern_definitions.skirt_design import SkirtDesign
from weon_garment_code.pattern_definitions.waistband_design import WaistbandDesign
from weon_garment_code.perfect_fit_templates.pants_templates import (
    CrotchDisplacementRatio,
    GeneralFit,
    LegLength,
    PerfectFitPantsDesign,
    PerfectFitPantsStyleSpec,
    RaiseWaist,
)
from weon_garment_code.perfect_fit_templates.shirt_templates import (
    PerfectFitShirtDesign,
)


class GarmentFactory:
    """Factory for creating garment instances.

    This factory encapsulates the logic for creating different types of garments
    based on design parameters. It uses the new class-based parameter definitions
    (BodyDefinition, PantsDesign, etc.) instead of raw dictionaries.

    Attributes:
    -----------
    _body : BodyDefinition
        The body measurements object (private, set during creation).
    _design : dict
        The full design parameters dictionary (private, set during creation).
    """

    def __init__(self, body: BodyDefinition, design: dict) -> None:
        """Initialize the garment factory.

        Parameters:
        -----------
        body : BodyDefinition
            Body measurements object containing all body parameters.
        design : dict
            Full design parameters dictionary containing all garment specifications.
        """
        self._body = body
        self._design = design

    @property
    def body(self) -> BodyDefinition:
        """Get the body measurements object."""
        return self._body

    @property
    def design(self) -> dict:
        """Get the full design parameters dictionary."""
        return self._design

    def create_pants(self) -> Pants:
        """Create a pants garment instance from regular design specification.

        Creates a Pants garment using the factory's body and design parameters.
        Uses regular PantsDesign conversion from the design dictionary.

        Returns:
        --------
        Pants
            A fully initialized Pants garment instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain pants parameters.
        """
        # Validate that pants design exists
        if "pants" not in self._design:
            raise ValueError(
                "Design dictionary must contain 'pants' key for pants creation"
            )

        pants_design = PantsDesign(self._design)
        logger.info("Creating pants from regular design specification")

        pants = Pants(design=pants_design)
        pants.apply_body_alignment(self._body)

        return pants

    def create_pants_perfect_fit(self) -> Pants:
        """Create a pants garment instance from perfect fit specification.

        Creates a Pants garment using perfect fit parameters from the design dictionary.
        The design dictionary should contain perfect fit parameters at the top level:
        - garment_type: str (e.g., "pants")
        - raise_to_waist_ratio: str (e.g., "MID")
        - crotch_displacement_ratio: str (e.g., "LOW_CROTCH")
        - general_fit_ratio: str (e.g., "REGULAR")
        - leg_length_ratio: str (e.g., "ANKLE")
        - pants_style_spec: str (e.g., "BOOT_CUT")

        Returns:
        --------
        Pants
            A fully initialized Pants garment instance.

        Raises:
        -------
        ValueError
            If required perfect fit parameters are missing or contain invalid enum values.
        """
        # Extract perfect fit parameters from design dict
        try:
            raise_to_waist = RaiseWaist[self._design.get("raise_to_waist_ratio", "MID")]
            crotch_displacement = CrotchDisplacementRatio[
                self._design.get("crotch_displacement_ratio", "LOW_CROTCH")
            ]
            general_fit = GeneralFit[self._design.get("general_fit_ratio", "REGULAR")]
            leg_length = LegLength[self._design.get("leg_length_ratio", "ANKLE")]
            pants_style = PerfectFitPantsStyleSpec[
                self._design.get("pants_style_spec", "STRAIGHT")
            ]
            cuff_length_ratio = self._design.get("cuff_length_ratio", 0)
        except KeyError as e:
            raise ValueError(
                f"Invalid enum value in perfect fit specification: {e}. "
                f"Available values: RaiseWaist: {[e.name for e in RaiseWaist]}, "
                f"CrotchDisplacementRatio: {[e.name for e in CrotchDisplacementRatio]}, "
                f"GeneralFit: {[e.name for e in GeneralFit]}, "
                f"LegLength: {[e.name for e in LegLength]}, "
                f"PerfectFitPantsStyleSpec: {[e.name for e in PerfectFitPantsStyleSpec]}"
            )

        # Create perfect fit pants design
        pants_design = PerfectFitPantsDesign(
            body_definition=self._body,
            raise_to_waist_ratio=raise_to_waist,
            crotch_displacement_ratio=crotch_displacement,
            general_fit_ratio=general_fit,
            leg_length_ratio=leg_length,
            pants_style_spec=pants_style,
            cuff_length_ratio=cuff_length_ratio,
        ).get_design()

        logger.info(
            f"Creating perfect fit pants: "
            f"style={pants_style.name}, "
            f"fit={general_fit.name}, "
            f"length={leg_length.name}"
        )

        pants = Pants(design=pants_design)
        pants.apply_body_alignment(self._body)

        return pants

    def create_shirt_perfect_fit(self, fitted: bool = False) -> Shirt:
        """Create a shirt garment instance from perfect fit specification.

        Parameters
        ----------
        fitted : bool, optional
             If True, creates a fitted shirt; otherwise, a looser fit.
             Default is False.

        Returns
        -------
        Shirt
            A fully initialized Shirt garment instance.
        """
        # Extract perfect fit parameters
        pf_design = PerfectFitShirtDesign(
            body_definition=self._body,
            smallest_width_above_chest_ratio=self._design.get(
                "smallest_width_above_chest_ratio", 0.9
            ),
            width_chest_ratio=self._design.get("width_chest_ratio", 0.9),
            width_waist_ratio=self._design.get("width_waist_ratio", 0.8),
            width_hip_ratio=self._design.get("width_hip_ratio", 0.95),
            neck_to_shoulder_distance_ratio=self._design.get(
                "neck_to_shoulder_distance_ratio", 0.3
            ),
            neck_width_ratio=self._design.get("neck_width_ratio", 0.3),
            shoulder_slant_ratio=self._design.get("shoulder_slant_ratio", 0.05),
            waist_over_bust_line_height_ratio=self._design.get(
                "waist_over_bust_line_height_ratio", 1.0
            ),
            back_width_ratio=self._design.get("back_width_ratio", 1.05),
            front_length_ratio=self._design.get("front_length_ratio", 1.3),
            back_length_ratio=self._design.get("back_length_ratio", 1.35),
            scye_depth_ratio=self._design.get("scye_depth_ratio", 0.5),
            waistband_width_ratio=self._design.get("waistband_width_ratio", 0.1),
            waistband_length_ratio=self._design.get("waistband_length_ratio", 1.0),
            shirttail_offset_ratio=self._design.get("shirttail_offset_ratio", 0),
            sleeveless=self._design.get("sleeveless", False),
            sleeve_length_ratio=self._design.get("sleeve_length_ratio", 1.0),
            bicep_width_ratio=self._design.get("bicep_width_ratio", 0.25),
            elbow_width_ratio=self._design.get("elbow_width_ratio", 0.2),
            wrist_width_ratio=self._design.get("wrist_width_ratio", 0.155),
            cuff_length=self._design.get("cuff_length", 0.0),
            cuff_width_ratio=self._design.get("cuff_width_ratio", 0.0),
        )

        design_data = pf_design.get_design()

        # Create ShirtDesign and inject calculated components
        # We start with empty dict as we populate fields manually
        shirt_design = ShirtDesign(self._design)
        shirt_design._torso_design = design_data["torso"]
        shirt_design._sleeve_design = design_data["sleeve"]
        shirt_design._waistband_design = design_data["waistband"]

        shirt = Shirt(
            shirt_design=shirt_design,
            arm_pose_angle=self._body.arm_pose_angle,
            fitted=fitted,
        )
        shirt.apply_body_alignment(self._body)

        return shirt

    def create_shirt(self, fitted: bool = False) -> Shirt:
        """Create a shirt garment instance.

        Parameters:
        -----------
        fitted : bool, optional
            If True, creates a fitted shirt; otherwise, a looser fit.
            Default is False.

        Returns:
        --------
        Shirt
            A fully initialized Shirt garment instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain shirt parameters.
        """
        if "shirt" not in self._design:
            # Check if this is a perfect fit design by looking for key parameters
            if "width_chest_ratio" in self._design:
                return self.create_shirt_perfect_fit(fitted=fitted)
            raise ValueError(
                "Design dictionary must contain 'shirt' key for shirt creation"
            )

        shirt = Shirt(
            shirt_design=ShirtDesign(self._design),
            arm_pose_angle=self._body.arm_pose_angle,
            fitted=fitted,
        )
        shirt.apply_body_alignment(self._body)

        return shirt

    def create_fitted_shirt(self) -> FittedShirt:
        """Create a fitted shirt garment instance.

        Returns:
        --------
        FittedShirt
            A fully initialized FittedShirt garment instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain shirt parameters.
        """
        if "shirt" not in self._design:
            raise ValueError(
                "Design dictionary must contain 'shirt' key for fitted shirt creation"
            )

        shirt = FittedShirt(
            shirt_design=ShirtDesign(self._design),
            arm_pose_angle=self._body.arm_pose_angle,
        )
        shirt.apply_body_alignment(self._body)

        return shirt

    def create_skirt_circle(self, **kwargs) -> SkirtCircle:
        """Create a circle skirt garment instance.

        Parameters:
        -----------
        **kwargs
            Additional keyword arguments passed to SkirtCircle constructor
            (e.g., tag, length, slit, asymm, min_len).

        Returns:
        --------
        SkirtCircle
            A fully initialized SkirtCircle garment instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain skirt parameters.
        """
        if "flare-skirt" not in self._design:
            raise ValueError(
                "Design dictionary must contain 'flare-skirt' key for skirt creation"
            )

        skirt_design = SkirtDesign(self._design)
        waist_front = self._body.waist - self._body.waist_back_width
        waist_back = self._body.waist_back_width

        skirt = SkirtCircle(
            skirt_design=skirt_design,
            waist_front=waist_front,
            waist_back=waist_back,
            **kwargs,
        )
        skirt.apply_body_alignment(self._body)

        return skirt

    def create_asymm_skirt_circle(self, **kwargs) -> AsymmSkirtCircle:
        """Create an asymmetric circle skirt garment instance.

        Parameters:
        -----------
        **kwargs
            Additional keyword arguments passed to AsymmSkirtCircle constructor
            (e.g., tag, length, slit).

        Returns:
        --------
        AsymmSkirtCircle
            A fully initialized AsymmSkirtCircle garment instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain skirt parameters.
        """
        if "flare-skirt" not in self._design:
            raise ValueError(
                "Design dictionary must contain 'flare-skirt' key for asymmetric skirt creation"
            )

        skirt_design = SkirtDesign(self._design)
        waist_front = self._body.waist - self._body.waist_back_width
        waist_back = self._body.waist_back_width

        skirt = AsymmSkirtCircle(
            skirt_design=skirt_design,
            waist_front=waist_front,
            waist_back=waist_back,
            **kwargs,
        )
        skirt.apply_body_alignment(self._body)

        return skirt

    def create_straight_wb(
        self,
        waist: float | None = None,
        waist_back: float | None = None,
        rise: float = 1.0,
    ) -> StraightWB:
        """Create a straight waistband instance.

        Parameters:
        -----------
        waist : float, optional
            Total waist measurement. If None, will be calculated from body.
        waist_back : float, optional
            Back waist width. If None, will be calculated from body.
        rise : float, optional
            The rise value of the bottoms that the WB is attached to.
            Default is 1.0.

        Returns:
        --------
        StraightWB
            A fully initialized StraightWB instance.

        Raises:
        -------
        ValueError
            If the design dictionary does not contain waistband parameters.
        """
        if "waistband" not in self._design:
            raise ValueError(
                "Design dictionary must contain 'waistband' key for waistband creation"
            )

        waistband_design = WaistbandDesign(self._design)

        # Use provided values or calculate from body
        if waist is None:
            waist = self._body.waist
        if waist_back is None:
            waist_back = self._body.waist_back_width
        waist_front = waist - waist_back

        wb = StraightWB(
            waistband_design=waistband_design,
            waist=waist,
            waist_back=waist_back,
            waist_front=waist_front,
            rise=rise,
        )
        wb.apply_body_alignment(self._body)

        return wb

    def create_composite_garment(self) -> pyg.Component:
        """Create a composite garment (dress/jumpsuit) with upper, waistband, and lower components.

        This method creates a composite garment similar to MetaGarment, combining:
        - Upper garment (if meta.upper.v is specified)
        - Waistband (if meta.wb.v is specified and lower garment exists)
        - Lower garment (if meta.bottom.v is specified)

        The components are connected in order: upper -> waistband -> lower

        Returns:
        --------
        pyg.Component
            A composite garment component containing the upper, waistband (if any),
            and lower garments connected together.

        Raises:
        -------
        ValueError
            If neither upper nor lower garment is specified, or if required
            design parameters are missing.
        """

        # Get garment types from design
        upper_type = self._design.get("meta", {}).get("upper", {}).get("v")
        lower_type = self._design.get("meta", {}).get("bottom", {}).get("v")
        wb_type = self._design.get("meta", {}).get("wb", {}).get("v")

        if not upper_type and not lower_type:
            raise ValueError(
                "Composite garment must specify either 'meta.upper.v' or 'meta.bottom.v'"
            )

        # Create upper garment if specified
        upper: pyg.Component | None = None
        if upper_type:
            if upper_type == "Shirt":
                upper = self.create_shirt(fitted=False)
            elif upper_type == "FittedShirt":
                upper = self.create_fitted_shirt()
            else:
                raise ValueError(f"Unsupported upper garment type: {upper_type}")

        # Create lower garment if specified
        lower: pyg.Component | None = None
        if lower_type:
            if lower_type == "Pants":
                lower = self.create_pants()
            elif lower_type == "SkirtCircle":
                lower = self.create_skirt_circle()
            elif lower_type == "AsymmSkirtCircle":
                lower = self.create_asymm_skirt_circle()
            else:
                raise ValueError(f"Unsupported lower garment type: {lower_type}")

        # Create waistband if specified and lower garment exists
        wb: pyg.Component | None = None
        if wb_type and lower:
            # Get waist width from lower garment
            if hasattr(lower, "get_waist_width"):
                # For skirts (SkirtCircle, AsymmSkirtCircle)
                waist_width = lower.get_waist_width()  # type: ignore[attr-defined]
            elif hasattr(lower, "right") and hasattr(lower.right, "pants_design"):
                # For pants, access via right.pants_design
                waist_width = lower.right.pants_design.waist  # type: ignore[attr-defined]
            else:
                # Fallback: use body waist measurement
                waist_width = self._body.waist

            if wb_type == "StraightWB":
                wb = self.create_straight_wb(
                    waist=waist_width, waist_back=self._body.waist_back_width, rise=1.0
                )
            else:
                raise ValueError(f"Unsupported waistband type: {wb_type}")

        # Create composite wrapper
        composite = pyg.CompositeGarment(
            "CompositeGarment", upper=upper, lower=lower, waistband=wb
        )

        return composite

    @staticmethod
    def create_pants_from_params(body: BodyDefinition, design: dict) -> Pants:
        """Static factory method to create pants directly.

        Convenience method for creating pants without instantiating the factory.

        Parameters:
        -----------
        body : BodyDefinition
            Body measurements object containing all body parameters.
        design : dict
            Design parameters dictionary containing pants specifications.

        Returns:
        --------
        Pants
            A fully initialized Pants garment instance.
        """
        factory = GarmentFactory(body, design)
        return factory.create_pants()


def create_garment(
    garment_type: str, body: BodyDefinition, design: dict, **kwargs: Any
) -> tuple[pyg.Component, list[AttachmentConstraint]]:
    """Factory function to create garments by type name.

    This function provides a simple interface for creating garments based on
    their type name, similar to the original MetaGarment approach but using
    the factory pattern.

    Parameters:
    -----------
    garment_type : str
        Type of garment to create. Supported types:
        - 'Pants': Creates a pants garment
        - 'Shirt': Creates a shirt garment (use fitted=False in kwargs for loose fit)
        - 'FittedShirt': Creates a fitted shirt garment
        - 'SkirtCircle': Creates a circle skirt
        - 'AsymmSkirtCircle': Creates an asymmetric circle skirt
        - 'StraightWB': Creates a straight waistband
    body : BodyDefinition
        Body measurements object containing all body parameters.
    design : dict
        Design parameters dictionary containing garment specifications.
    **kwargs
        Additional keyword arguments passed to the garment constructor.
        For example:
        - For Shirt: fitted=True/False
        - For SkirtCircle/AsymmSkirtCircle: tag, length, slit, etc.
        - For StraightWB: waist, waist_back, rise

    Returns:
    --------
    tuple[pyg.Component, list[AttachmentConstraint]]
        A tuple containing:
        - The created garment instance (Pants, Shirt, FittedShirt, SkirtCircle,
          AsymmSkirtCircle, StraightWB)
        - A list of AttachmentConstraint objects for the garment.
          For pants, this includes a crotch constraint. For other garment types,
          this is an empty list.

    Raises:
    -------
    ValueError
        If the garment_type is not recognized or if required design parameters
        are missing.
    """
    factory = GarmentFactory(body, design)

    garment_creators = {
        "Pants": factory.create_pants,
        "Shirt": lambda: factory.create_shirt(fitted=kwargs.get("fitted", False)),
        "FittedShirt": factory.create_fitted_shirt,
        "SkirtCircle": lambda: factory.create_skirt_circle(
            **{k: v for k, v in kwargs.items() if k not in ["fitted"]}
        ),
        "AsymmSkirtCircle": lambda: factory.create_asymm_skirt_circle(
            **{k: v for k, v in kwargs.items() if k not in ["fitted"]}
        ),
        "StraightWB": lambda: factory.create_straight_wb(
            waist=kwargs.get("waist"),
            waist_back=kwargs.get("waist_back"),
            rise=kwargs.get("rise", 1.0),
        ),
    }

    if garment_type not in garment_creators:
        raise ValueError(
            f"Unknown garment type: {garment_type}. "
            f"Supported types: {list(garment_creators.keys())}"
        )

    garment = garment_creators[garment_type]()

    # Get attachment constraints from the garment class
    # All garment classes implement get_attachment_constraints() static method
    garment_class = garment.__class__
    constraints = garment_class.get_attachment_constraints()  # type: ignore[attr-defined]

    return garment, constraints
