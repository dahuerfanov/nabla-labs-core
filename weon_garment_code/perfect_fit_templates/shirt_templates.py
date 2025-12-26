from loguru import logger

from weon_garment_code.garment_programs.garment_enums import SleeveArmholeShape
from weon_garment_code.pattern_definitions.body_definition import BodyDefinition
from weon_garment_code.pattern_definitions.sleeve_design import SleeveDesign
from weon_garment_code.pattern_definitions.torso_design import TorsoDesign

# ============================================================================
# VALIDATION RULES SYSTEM
# ============================================================================


class ShirtDesignRule:
    """Base class for shirt design validation rules.

    Each rule validates a specific constraint on shirt design parameters.
    Rules are checked before creating the design to prevent invalid combinations.
    """

    @staticmethod
    def validate(**params) -> tuple[bool, str]:
        """Validate design parameters.

        Parameters
        ----------
        **params : dict
            Design parameters to validate.

        Returns
        -------
        tuple[bool, str]
            (is_valid, error_message). If valid, error_message is empty.
        """
        raise NotImplementedError("Subclasses must implement validate()")


class WidthConsistencyRule(ShirtDesignRule):
    """Validates that narrower body parts don't exceed wider parts.

    Rule: smallest_width_above_chest_ratio and back_width_ratio must be ≤ width_chest_ratio

    Rationale: The chest is typically the widest part of the torso. Having narrower
    parts (above chest, back) exceed the chest width creates invalid garment geometry.
    """

    @staticmethod
    def validate(**params) -> tuple[bool, str]:
        smallest = params.get("smallest_width_above_chest_ratio")
        back = params.get("back_width_ratio")
        chest = params.get("width_chest_ratio")

        if smallest is None or back is None or chest is None:
            return True, ""  # Skip validation if params not provided

        if smallest > chest:
            return (
                False,
                f"smallest_width_above_chest_ratio ({smallest:.3f}) must be ≤ "
                f"width_chest_ratio ({chest:.3f})",
            )

        if back > chest:
            return (
                False,
                f"back_width_ratio ({back:.3f}) must be ≤ width_chest_ratio ({chest:.3f})",
            )

        return True, ""


class LengthProportionRule(ShirtDesignRule):
    """Validates torso length proportions.

    Rule: back_length, front_length > waist_over_bust_line_height > scye_depth

    Rationale: These measurements represent vertical distances on the torso.
    The waist-over-bust line must be below the scye (armhole) depth, and the
    garment length must extend below the waist line.
    """

    @staticmethod
    def validate(**params) -> tuple[bool, str]:
        front_length = params.get("front_length_ratio")
        back_length = params.get("back_length_ratio")
        waist_height = params.get("waist_over_bust_line_height_ratio")
        scye_depth = params.get("scye_depth_ratio")

        # Skip validation if any param not provided
        if (
            front_length is None
            or back_length is None
            or waist_height is None
            or scye_depth is None
        ):
            return True, ""

        # Check: scye_depth < waist_height
        if scye_depth >= waist_height:
            return (
                False,
                f"scye_depth_ratio ({scye_depth:.3f}) must be < "
                f"waist_over_bust_line_height_ratio ({waist_height:.3f})",
            )

        # Check: waist_height < front_length
        if waist_height >= front_length:
            return (
                False,
                f"waist_over_bust_line_height_ratio ({waist_height:.3f}) must be < "
                f"front_length_ratio ({front_length:.3f})",
            )

        # Check: waist_height < back_length
        if waist_height >= back_length:
            return (
                False,
                f"waist_over_bust_line_height_ratio ({waist_height:.3f}) must be < "
                f"back_length_ratio ({back_length:.3f})",
            )

        return True, ""


class SleeveGeometryRule(ShirtDesignRule):
    """Validates that sleeve bicep width fits within the armhole opening.

    Prevents NaN error in sleeve creation where bicep_width > armhole_opening.
    """

    @staticmethod
    def validate(**params) -> tuple[bool, str]:
        # Required params
        bicep_ratio = params.get("bicep_width_ratio")
        scye_ratio = params.get("scye_depth_ratio")
        slant_ratio = params.get("shoulder_slant_ratio")
        sleeveless = params.get("sleeveless")
        body = params.get("body_definition")

        if sleeveless or body is None:
            return True, ""

        if bicep_ratio is None or scye_ratio is None or slant_ratio is None:
            return True, ""

        waist_to_neck = body.waist_over_bust_line
        arm_length = body.arm_length
        bicep_width = bicep_ratio * arm_length
        armhole_height = (scye_ratio - slant_ratio) * waist_to_neck

        if bicep_width >= armhole_height:
            return (
                False,
                f"Sleeve too wide for armhole: bicep_width ({bicep_width:.1f}) >= "
                f"approx_armhole_height ({armhole_height:.1f}). "
                f"bicep_width_ratio={bicep_ratio}, scye_depth_ratio={scye_ratio}. "
                f"Reduce bicep_width_ratio or increase scye_depth_ratio.",
            )

        return True, ""


# Registry of all validation rules
# Add new rules here to automatically include them in validation
SHIRT_DESIGN_RULES: list[type[ShirtDesignRule]] = [
    WidthConsistencyRule,
    LengthProportionRule,
    SleeveGeometryRule,
]


# ============================================================================
# SHIRT DESIGN CLASSES
# ============================================================================


class PerfectFitShirtDesign:
    """
    Collection of PerfectFitTorsoDesign and PerfectFitSleeveDesign and default collar and assimetry config.
    """

    def __init__(
        self,
        body_definition: BodyDefinition,
        smallest_width_above_chest_ratio: float,
        width_chest_ratio: float,
        width_waist_ratio: float,
        width_hip_ratio: float,
        neck_to_shoulder_distance_ratio: float,
        neck_width_ratio: float,
        shoulder_slant_ratio: float,
        waist_over_bust_line_height_ratio: float,
        back_width_ratio: float,
        front_length_ratio: float,
        back_length_ratio: float,
        scye_depth_ratio: float,
        sleeveless: bool,
        armhole_size_ratio: float,
        sleeve_length_ratio: float,
        bicep_width_ratio: float,
        elbow_width_ratio: float,
        wrist_width_ratio: float,
        cuff_length: float,
    ) -> None:
        """Initialize shirt template.

        Parameters:
        -----------
        body_definition: BodyDefinition
            The body definition.
        smallest_width_above_chest_ratio: float
            The smallest width above chest ratio.
        width_chest_ratio: float
            The width chest ratio.
        width_waist_ratio: float
            The width waist ratio.
        width_hip_ratio: float
            The width hip ratio.
        neck_to_shoulder_distance_ratio: float
            The neck to shoulder distance ratio.
        neck_width_ratio: float
            The neck width ratio.
        shoulder_slant_ratio: float
            The shoulder slant ratio.
        waist_over_bust_line_height_ratio: float
            The waist over bust line height ratio.
        back_width_ratio: float
            The back width ratio.
        front_length_ratio: float
            The front length ratio.
        back_length_ratio: float
            The back length ratio.
        scye_depth_ratio: float
            The scye depth ratio.
        sleeveless: bool
            Whether the garment is sleeveless.
        armhole_size_ratio: float
            The armhole size ratio.
        sleeve_length_ratio: float
            The sleeve length ratio.
        bicep_width_ratio: float
            The bicep width ratio.
        elbow_width_ratio: float
            The elbow width ratio.
        wrist_width_ratio: float
            The wrist width ratio.
        cuff_length: float
            The cuff length.

        Raises
        ------
        ValueError
            If any validation rule is violated.
        """

        # Validate design parameters against all rules
        for rule_class in SHIRT_DESIGN_RULES:
            is_valid, error_msg = rule_class.validate(
                smallest_width_above_chest_ratio=smallest_width_above_chest_ratio,
                width_chest_ratio=width_chest_ratio,
                width_waist_ratio=width_waist_ratio,
                width_hip_ratio=width_hip_ratio,
                back_width_ratio=back_width_ratio,
                front_length_ratio=front_length_ratio,
                back_length_ratio=back_length_ratio,
                waist_over_bust_line_height_ratio=waist_over_bust_line_height_ratio,
                scye_depth_ratio=scye_depth_ratio,
                # Sleeve params
                body_definition=body_definition,
                sleeveless=sleeveless,
                bicep_width_ratio=bicep_width_ratio,
                shoulder_slant_ratio=shoulder_slant_ratio,
            )
            if not is_valid:
                raise ValueError(f"Shirt design validation failed: {error_msg}")

        self._torso_design = PerfectFitTorsoDesign(
            body_definition,
            smallest_width_above_chest_ratio,
            width_chest_ratio,
            width_waist_ratio,
            width_hip_ratio,
            neck_to_shoulder_distance_ratio,
            neck_width_ratio,
            shoulder_slant_ratio,
            waist_over_bust_line_height_ratio,
            back_width_ratio,
            front_length_ratio,
            back_length_ratio,
            scye_depth_ratio,
        )
        self._sleeve_design = PerfectFitSleeveDesign(
            body_definition,
            sleeveless,
            armhole_size_ratio,
            sleeve_length_ratio,
            bicep_width_ratio,
            elbow_width_ratio,
            wrist_width_ratio,
            cuff_length,
        )

    def get_design(self) -> dict:
        return {
            "torso": self._torso_design.get_design(),
            "sleeve": self._sleeve_design.get_design(),
        }


class PerfectFitTorsoDesign:
    STRAPLESS_DEFAULT: bool = False
    """
    Perfect fit torso design.

    Horizontal ratios are relative to the body's shoulder width,
    meaning the distance from the furthest left end of collarbone to the furthest right end of collarbone.

    Vertical ratios are relative to the body's waist over bust line height,
    meaning the vertical line passing thorugh bust point, measured from the neck base (on the balance line)
    to the waist line from the front over the bust. Follows the convex hull of the geodesic measurement,
    similar to tailor's tape measurement.
    """

    def __init__(
        self,
        body_definition: BodyDefinition,
        smallest_width_above_chest_ratio: float,
        width_chest_ratio: float,
        width_waist_ratio: float,
        width_hip_ratio: float,
        neck_to_shoulder_distance_ratio: float,
        neck_width_ratio: float,
        shoulder_slant_ratio: float,
        waist_over_bust_line_height_ratio: float,
        back_width_ratio: float,
        front_length_ratio: float,
        back_length_ratio: float,
        scye_depth_ratio: float,
    ) -> None:
        """Initialize shirt template.

        Parameters:
        -----------
        body_definition: BodyDefinition
            The body definition.
        smallest_width_above_chest_ratio: float
            The smallest width above chest ratio.
        width_chest_ratio: float
            The width chest ratio.
        width_waist_ratio: float
            The width waist ratio.
        width_hip_ratio: float
            The width hip ratio.
        neck_to_shoulder_distance_ratio: float
            The neck to shoulder distance ratio.
        neck_width_ratio: float
            The neck width ratio.
        shoulder_slant_ratio: float
            The shoulder slant ratio.
        waist_over_bust_line_height_ratio: float
            The waist over bust line height ratio.
        back_width_ratio: float
            The back width ratio.
        front_length_ratio: float
            The front length ratio.
        back_length_ratio: float
            The back length ratio.
        scye_depth_ratio: float
            The scye depth ratio.
        """

        shoulder_width = body_definition.shoulder_w
        waist_to_neck_distance = body_definition.waist_over_bust_line

        self._smallest_width: float = smallest_width_above_chest_ratio * shoulder_width
        self._width_chest: float = width_chest_ratio * shoulder_width
        self._width_waist: float = width_waist_ratio * shoulder_width
        self._width_hip: float = width_hip_ratio * shoulder_width
        self._neck_to_shoulder_distance: float = (
            neck_to_shoulder_distance_ratio * shoulder_width
        )
        self._neck_width: float = neck_width_ratio * shoulder_width
        self._back_width: float = back_width_ratio * shoulder_width

        self._shoulder_slant: float = shoulder_slant_ratio * waist_to_neck_distance
        self._waist_over_bust_line_height: float = (
            waist_over_bust_line_height_ratio * waist_to_neck_distance
        )
        self._front_length: float = front_length_ratio * waist_to_neck_distance
        self._back_length: float = back_length_ratio * waist_to_neck_distance
        self._scye_depth: float = scye_depth_ratio * waist_to_neck_distance

    def get_design(self) -> TorsoDesign:
        """Get torso design."""
        torso_dict = {
            "smallest_width": {"v": self._smallest_width},
            "width_chest": {"v": self._width_chest},
            "width_waist": {"v": self._width_waist},
            "width_hip": {"v": self._width_hip},
            "neck_to_shoulder_distance": {"v": self._neck_to_shoulder_distance},
            "neck_width": {"v": self._neck_width},
            "shoulder_slant": {"v": self._shoulder_slant},
            "waist_over_bust_line_height": {"v": self._waist_over_bust_line_height},
            "back_width": {"v": self._back_width},
            "front_length": {"v": self._front_length},
            "back_length": {"v": self._back_length},
            "scye_depth": {"v": self._scye_depth},
        }
        logger.debug("PerfectFitShirtDesign torso values:")
        for k, v in torso_dict.items():
            logger.debug(f"  {k}: {v['v']}")
        return TorsoDesign(torso_dict)


class PerfectFitSleeveDesign:
    """
    Perfect fit sleeve design. The ratios are relative to the body's arm length.
    """

    ARMHOLE_DEFAULT_SHAPE: SleeveArmholeShape = SleeveArmholeShape.ARMHOLE_CURVE
    DEFAULT_SLEEVE_ANGLE: float = 10.0  # degrees
    DEFAULT_OPENING_DIR_MIX: float = 0.1
    DEFAULT_STANDING_SHOULDER: bool = False
    DEFAULT_STANDING_SHOULDER_LEN: float = 0.0
    DEFAULT_CONNECT_RUFFLE: float = 1
    DEFAULT_SMOOTHING_COEFF: float = 0.25

    def __init__(
        self,
        body_definition: BodyDefinition,
        sleeveless: bool,
        armhole_size_ratio: float,
        sleeve_length_ratio: float,
        bicep_width_ratio: float,
        elbow_width_ratio: float,
        wrist_width_ratio: float,
        cuff_length: float,
    ):
        """
        Attributes:
        -----------
        sleeveless : bool
            Whether the sleeve is sleeveless. Overwrites the rest of the sleeve parameters.
        armhole_size_ratio: float
            Size (height) ratio of the armhole.
        sleeve_length_ratio : float
            Length ratio of the sleeve.
        bicep_width_ratio : float
            Width ratio at the bicep level.
        elbow_width_ratio : float
            Width ratio at the elbow level.
        wrist_width_ratio : float
            Width ratio at the end of the sleeve at the wrist.
        cuff : CuffDesign
            Cuff design parameters (reusing from waistband design).
        """

        arm_length = body_definition.arm_length
        self._sleeveless: bool = sleeveless

        self._armhole_shape: SleeveArmholeShape = (
            PerfectFitSleeveDesign.ARMHOLE_DEFAULT_SHAPE
        )
        self._connecting_width: float = armhole_size_ratio * arm_length
        self._length: float = sleeve_length_ratio * arm_length
        self._bicep_width: float = bicep_width_ratio * arm_length
        self._elbow_width: float = elbow_width_ratio * arm_length
        self._end_width: float = wrist_width_ratio * arm_length
        self._cuff_length: float = cuff_length * arm_length

        self._sleeve_angle: float = PerfectFitSleeveDesign.DEFAULT_SLEEVE_ANGLE
        self._opening_dir_mix: float = PerfectFitSleeveDesign.DEFAULT_OPENING_DIR_MIX
        self._standing_shoulder: bool = PerfectFitSleeveDesign.DEFAULT_STANDING_SHOULDER
        self._standing_shoulder_len: float = (
            PerfectFitSleeveDesign.DEFAULT_STANDING_SHOULDER_LEN
        )
        self._connect_ruffle: float = PerfectFitSleeveDesign.DEFAULT_CONNECT_RUFFLE
        self._smoothing_coeff: float = PerfectFitSleeveDesign.DEFAULT_SMOOTHING_COEFF

    def get_design(self) -> SleeveDesign:
        """Get sleeve design."""

        sleeve_dict = {
            "sleeveless": {"v": self._sleeveless},
            "armhole_shape": {"v": self._armhole_shape.value},
            "length": {"v": self._length},
            "bicep_width": {"v": self._bicep_width},
            "elbow_width": {"v": self._elbow_width},
            "connecting_width": {"v": self._connecting_width},
            "end_width": {"v": self._end_width},
            "sleeve_angle": {"v": self._sleeve_angle},
            "opening_dir_mix": {"v": self._opening_dir_mix},
            "standing_shoulder": {"v": self._standing_shoulder},
            "standing_shoulder_len": {"v": self._standing_shoulder_len},
            "connect_ruffle": {"v": self._connect_ruffle},
            "smoothing_coeff": {"v": self._smoothing_coeff},
            "cuff": {"type": {"v": "CuffBand"}, "cuff_len": {"v": self._cuff_length}},
        }

        logger.debug("\n--- SleeveDesign parameter values ---")
        for k, v in sleeve_dict.items():
            if k == "cuff":
                logger.debug(f"Cuff type: {v['type']['v']}")  # type: ignore
                logger.debug(f"Cuff length: {v['cuff_len']['v']}")  # type: ignore
            else:
                logger.debug(
                    f"{k}: {v['v'] if isinstance(v, dict) and 'v' in v else v}"
                )
        logger.debug("--- End SleeveDesign parameter values ---\n")

        return SleeveDesign(sleeve_dict)
