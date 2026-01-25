import copy
from collections.abc import Callable

import numpy as np

import weon_garment_code.pygarment.garmentcode as pyg
from weon_garment_code.config import AttachmentConstraint
from weon_garment_code.garment_programs.base_classes import BaseBottoms
from weon_garment_code.garment_programs.garment_enums import (
    EdgeLabel,
    InterfaceName,
    PanelAlignment,
)
from weon_garment_code.garment_programs.garment_program_utils import AttachmentHandler
from weon_garment_code.garment_programs.weon_bands import CuffBand
from weon_garment_code.pattern_definitions.body_definition import BodyDefinition
from weon_garment_code.pattern_definitions.pants_design import PantsDesign
from weon_garment_code.pygarment.meshgen.arap.arap_types import (
    GarmentRing,
    RingConnection,
    RingConnector,
)
from weon_garment_code.pygarment.meshgen.arap.core_types import (
    GarmentCategory,
    GarmentMetadata,
    GarmentRingType,
    PanelPosition,
)
from weon_garment_code.pygarment.meshgen.box_mesh_gen.box_mesh import BoxMesh
from weon_garment_code.pygarment.meshgen.box_mesh_gen.stitch_types import (
    EdgeLabel as BoxEdgeLabel,
)


class PantPanel(pyg.Panel):
    """A pant panel component with optional dart fitting.

    This panel represents one half of a pants leg (front or back) and can be
    fitted with single or double darts at the waistline. The panel includes
    interfaces for connecting to other panels (outside seam, inside seam,
    crotch, top waistline, and bottom hem).

    Attributes:
    -----------
    DART_DEPTH_MULTIPLIER : float
        Multiplier for calculating dart depth based on waist-to-hip length.
    SMALL_DART_DEPTH_RATIO : float
        Ratio for smaller dart depth when using double darts.
    DART_POSITION_MULTIPLIER : float
        Multiplier for positioning double darts relative to single dart position.
    DART_OFFSET_MULTIPLIER_QUARTER : float
        Quarter-width offset multiplier for dart positioning.
    GRAINLINE_OFFSET_RATIO : float
        Ratio for positioning the grainline along the crotch extension.
    """

    # Dart and fit constants
    DART_DEPTH_MULTIPLIER: float = 0.8
    SMALL_DART_DEPTH_RATIO: float = 0.9
    DART_POSITION_MULTIPLIER: float = 0.5
    DART_OFFSET_MULTIPLIER_QUARTER: float = 0.25
    GRAINLINE_OFFSET_RATIO: float = 0.4
    EPSILON: float = 1e-6

    def __init__(
        self,
        name: str,
        design: PantsDesign,
        rise: float,
        mid_rise: float,
        waist: float,
        hips: float,
        crotch_extension: float,
        hips_crotch_diff: float,
        crotch_hip_diff: float,
        dart_position: float,
        translation: np.ndarray,
        match_top_int_to: float | None = None,
        hipline_ext: float = 1,
        double_dart: bool = False,
    ) -> None:
        """Initialize a pant panel with optional dart fitting.

        Creates a pant panel with curved edges for the outside seam, inside seam,
        crotch, and waistline. The panel can be fitted with darts at the waistline
        to accommodate the difference between waist and hip measurements.

        Parameters:
        -----------
        name : str
            Name identifier for this panel.
        design : PantsDesign
            Pants design parameters object containing measurements for widths and
            lengths (e.g., width_thigh, width_knee, width_calf, width_ankle, length_thigh_to_knee, length_knee_to_calf, length_calf_to_ankle,
            length_waist_to_hip).
        rise : float
            Total rise measurement from crotch to waist for this panel.
        mid_rise : float
            Average rise between front and back panels, used for alignment.
        waist : float
            Half-waist measurement (one side of the body).
        hips : float
            Half-hip measurement (one side of the body).
        crotch_extension : float
            Additional extension at the crotch point for fit and ease.
        hips_crotch_diff : float
            Vertical difference between the hips line and crotch point along the rise.
        crotch_hip_diff : float
            Vertical difference between the crotch point and hips line along the rise.
        dart_position : float
            Horizontal position of the dart(s) from the center front/back edge,
            measured along the waistline.
        translation : np.ndarray, optional
            3D translation vector for positioning the panel in space.
        match_top_int_to : float | None, optional
            If provided, the top interface will have a ruffle factor to match
            this target measurement. Used for correct balance line matching when
            connecting panels. Default is None.
        hipline_ext : float, optional
            Multiplier to extend the hipline for better fit adjustment.
            Default is 1.0.
        double_dart : bool, optional
            If True, creates two darts instead of one at the waistline.
            Default is False.

        Notes:
        ------
        The panel creates interfaces for:
        - 'outside': The outer side seam (right side for right leg)
        - 'inside': The inner leg seam
        - 'crotch': The crotch seam connecting front and back panels
        - 'top': The waistline interface (with optional darts)
        - 'bottom': The hem interface
        """

        super().__init__(name)

        dart_depth = (
            design.length_waist_to_hip * hipline_ext * self.DART_DEPTH_MULTIPLIER
        )
        dart_width = (hips - waist) / 2

        # --- Edges definition ---
        # Right
        total_length = (
            design.length_thigh_to_knee
            + design.length_knee_to_calf
            + design.length_calf_to_ankle
        )
        right_5 = pyg.CurveEdgeFactory.curve_from_tangents(
            start=[-hips, hips_crotch_diff + total_length],
            end=[-waist, mid_rise + total_length],
            target_tan0=np.array([0, 1]),
            initial_guess=[0.2, 0],
        )
        ext_point_thigh = right_5.evaluate_at_length(-hips_crotch_diff)

        right_4 = pyg.CurveEdgeFactory.curve_from_tangents(
            start=ext_point_thigh,
            end=right_5.start,
            target_tan0=np.array([0, 1]),
        )

        x_grainline = (
            ext_point_thigh[0]
            + (design.width_thigh + crotch_extension) * self.GRAINLINE_OFFSET_RATIO
        )

        right_3 = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=[
                x_grainline - design.width_knee / 2,
                (design.length_knee_to_calf + design.length_calf_to_ankle),
            ],
            end=right_4.start,
            target_tan1=[
                right_4.as_curve().unit_tangent(0.0).real,
                right_4.as_curve().unit_tangent(0.0).imag,
            ],
            pre_start=[
                x_grainline - design.width_calf / 2,
                design.length_calf_to_ankle,
            ],
            initial_guess=[0.1, 0],
        )

        right_2 = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=[x_grainline - design.width_calf / 2, design.length_calf_to_ankle],
            end=right_3.start,
            pre_start=[x_grainline - design.width_ankle / 2, 0],
            post_end=right_3.end,
            initial_guess=[0.5, 0],
        )

        right_1 = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=[x_grainline - design.width_ankle / 2, 0],
            end=right_2.start,
            post_end=right_2.end,
            initial_guess=[0.5, 0],
        )

        top = pyg.Edge(
            start=right_5.end,
            end=[0, (rise + total_length)],
        )

        crotch_top = pyg.Edge(
            start=top.end,
            end=[0, crotch_hip_diff + total_length],
        )
        crotch_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            start=crotch_top.end,
            end=[
                ext_point_thigh[0] + design.width_thigh + crotch_extension,
                total_length,
            ],
            target_tan0=np.array([0, -1]),
            target_tan1=np.array([1, 0]),
            initial_guess=[0.5, -0.5],
        )
        crotch_bottom.label = EdgeLabel.CROTCH_POINT_SEAM

        # Calculate the angle (in radians) between the start and end point of the segment and the horizontal axis.
        left_top_start = crotch_bottom.end
        left_top_end = [
            right_2.end[0] + design.width_knee,
            design.length_knee_to_calf + design.length_calf_to_ankle,
        ]
        dx = left_top_end[0] - left_top_start[0]
        dy = left_top_end[1] - left_top_start[1]
        if abs(dx) < self.EPSILON:
            left_top_angle = np.pi / 2 if dy > 0 else -np.pi / 2
        else:
            left_top_angle = np.arctan2(dy, -dx)

        left_top = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=left_top_start,
            end=left_top_end,
            post_end=[x_grainline + design.width_calf / 2, design.length_calf_to_ankle],
            initial_guess=[0.3, 0],
            tan0_min_angle=left_top_angle,
            constraint_penalty=1e3,
        )

        left_middle = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=left_top.end,
            end=[x_grainline + design.width_calf / 2, design.length_calf_to_ankle],
            post_end=[x_grainline + design.width_ankle / 2, 0],
            initial_guess=[0.5, 0],
        )

        left_bottom = pyg.CurveEdgeFactory.interpolate_with_tangents(
            start=left_middle.end,
            end=[x_grainline + design.width_ankle / 2, 0],
            pre_start=left_middle.start,
            initial_guess=[0.5, 0],
        )
        self.edges = pyg.EdgeSequence(
            right_1,
            right_2,
            right_3,
            right_4,
            right_5,
            top,
            crotch_top,
            crotch_bottom,
            left_top,
            left_middle,
            left_bottom,
        ).close_loop()
        bottom = self.edges[-1]

        # Default placement
        self.set_pivot(crotch_bottom.end)
        rightmost_thigh_alignment = crotch_bottom.end[0]
        self.translation = translation + [-rightmost_thigh_alignment, 0, 0]

        # Out interfaces (easier to define before adding a dart)
        self.interfaces = {
            InterfaceName.OUTSIDE: pyg.Interface(
                self, pyg.EdgeSequence(right_1, right_2, right_3, right_4, right_5)
            ),
            InterfaceName.CROTCH: pyg.Interface(
                self, pyg.EdgeSequence(crotch_top, crotch_bottom)
            ),
            InterfaceName.INSIDE: pyg.Interface(
                self, pyg.EdgeSequence(left_top, left_middle, left_bottom)
            ),
            InterfaceName.BOTTOM: pyg.Interface(self, bottom),
        }

        # Add top dart
        # NOTE: Ruffle indicator to match to waistline proportion for correct balance line matching
        if dart_width > 0:
            top_edges, int_edges = self.add_darts(
                top,
                dart_width,
                dart_depth,
                dart_position,
                double_dart=double_dart,
            )
            self.interfaces[InterfaceName.TOP] = pyg.Interface(
                self,
                top_edges,
                ruffle=waist / match_top_int_to
                if match_top_int_to is not None
                else 1.0,
            )
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces[InterfaceName.TOP] = pyg.Interface(
                self,
                top,
                ruffle=waist / match_top_int_to
                if match_top_int_to is not None
                else 1.0,
            )

    def add_darts(
        self,
        top: pyg.Edge,
        dart_width: float,
        dart_depth: float,
        dart_position: float,
        double_dart: bool = False,
    ) -> tuple[pyg.EdgeSequence, pyg.EdgeSequence]:
        """Add dart(s) to the top edge of the panel.

        Creates one or two darts along the waistline edge to provide shaping
        for the difference between waist and hip measurements. When using
        double darts, the first dart is smaller (using SMALL_DART_DEPTH_RATIO).

        Parameters:
        -----------
        top : pyg.Edge
            The top edge (waistline) where darts will be inserted.
        dart_width : float
            Total width of the dart(s). For double darts, this is split in half.
        dart_depth : float
            Depth (length) of the dart(s) extending into the panel.
        dart_position : float
            Horizontal position of the dart(s) from the center of the edge,
            measured along the edge length.
        double_dart : bool, optional
            If True, creates two darts instead of one. The first dart is
            positioned slightly offset and uses a reduced depth.
            Default is False.

        Returns:
        --------
        top_edges : pyg.EdgeSequence
            New edge sequence representing the top edge with dart(s) inserted.
        int_edges : pyg.EdgeSequence
            Edge sequence containing the internal dart edges (the folded dart
            lines inside the panel).
        """

        if double_dart:
            dist = (
                dart_position * self.DART_POSITION_MULTIPLIER
            )  # Dist between darts -> dist between centers
            offsets_mid = [
                -dart_position
                + dist / 2
                + dart_width / 2
                + dart_width * self.DART_OFFSET_MULTIPLIER_QUARTER,
                -dart_position
                - dist / 2
                - dart_width * self.DART_OFFSET_MULTIPLIER_QUARTER,
            ]

            darts = [
                pyg.EdgeSeqFactory.dart_shape(
                    dart_width / 2, dart_depth * self.SMALL_DART_DEPTH_RATIO
                ),  # smaller
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth),
            ]
        else:
            offsets_mid = [
                -dart_position + dart_width / 2,
            ]
            darts = [pyg.EdgeSeqFactory.dart_shape(dart_width, dart_depth)]

        top_edges, int_edges = pyg.EdgeSequence(top), pyg.EdgeSequence(top)

        for off, dart in zip(offsets_mid, darts, strict=True):
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=-off,
                edge_seq=top_edges,
                int_edge_seq=int_edges,
            )

        return top_edges, int_edges


class PantsHalf(BaseBottoms):
    """Half of a pants garment (left or right leg).

    A pants half consists of a front panel and a back panel, connected along
    the outside and inside seams. The front and back panels are positioned
    with different Z translations to create proper 3D shape.

    Attributes:
    -----------
    front : PantPanel
        The front panel of this pants half.
    back : PantPanel
        The back panel of this pants half.
    pants_design : PantsDesign
        The pants design parameters object containing all design measurements.
    BACK_HIPLINE_EXTENSION : float
        Multiplier for extending the back panel hipline for better fit.
    BACK_CROTCH_EXTENSION_OFFSET : float
        Additional crotch extension offset for the back panel (in cm).
    DEFAULT_TRANSLATION_Z_FRONT : float
        Default Z-axis translation for front panel positioning.
    DEFAULT_TRANSLATION_Z_BACK : float
        Default Z-axis translation for back panel positioning.
    """

    # Back panel constants
    BACK_HIPLINE_EXTENSION: float = 1.1
    BACK_CROTCH_EXTENSION_OFFSET: float = 3

    # Panel pivot and translation constants
    DEFAULT_TRANSLATION_Z_FRONT: float = 25
    DEFAULT_TRANSLATION_Z_BACK: float = -20

    front: PantPanel
    back: PantPanel
    pants_design: PantsDesign
    cuff: CuffBand | None = None

    def __init__(
        self,
        tag: str,
        design: PantsDesign,
    ) -> None:
        """Initialize a pants half (left or right leg).

        Creates both front and back panels for one leg of the pants, calculates
        the rise measurements, and sets up the stitching rules to connect the
        panels along their seams.

        Parameters:
        -----------
        tag : str
            Side identifier: 'r' for right leg or 'l' for left leg.
        design : PantsDesign
            Pants design parameters object containing measurements for widths
            and lengths needed to construct the panels.

        Notes:
        ------
        The front and back panels are created with different parameters:
        - Back panel has extended hipline and additional crotch extension
        - Both panels use calculated rise measurements based on front and back rises
        - Panels are positioned with different Z translations for 3D shaping
        """
        super().__init__(tag)

        # Create PantsDesign from the design dict and store it
        self.pants_design = design
        self.cuff: CuffBand | None = None

        rise_middle = (self.pants_design.back_rise + self.pants_design.front_rise) / 2
        hips_crotch_diff = rise_middle - self.pants_design.length_waist_to_hip

        self.front = PantPanel(
            name=f"pant_f_{tag}",
            design=self.pants_design,
            rise=self.pants_design.front_rise,
            mid_rise=rise_middle,
            waist=self.pants_design.waist / 2,
            hips=self.pants_design.width_hips / 2,
            hips_crotch_diff=hips_crotch_diff,
            crotch_extension=self.pants_design.width_gusset_crotch,
            crotch_hip_diff=rise_middle - self.pants_design.length_waist_to_hip,
            dart_position=self.pants_design.waist / 4,
            translation=np.array([0, 0, self.DEFAULT_TRANSLATION_Z_FRONT]),
            match_top_int_to=self.pants_design.waist / 2,
        )

        if self.pants_design.width_gusset_crotch_back is None:
            crotch_extension_back = (
                self.pants_design.width_gusset_crotch
                + self.BACK_CROTCH_EXTENSION_OFFSET
            )
        else:
            crotch_extension_back = self.pants_design.width_gusset_crotch_back

        self.back = PantPanel(
            name=f"pant_b_{tag}",
            design=self.pants_design,
            rise=self.pants_design.back_rise,
            mid_rise=rise_middle,
            waist=self.pants_design.waist / 2,
            hips=self.pants_design.width_hips / 2,
            hips_crotch_diff=hips_crotch_diff,
            crotch_extension=crotch_extension_back,
            crotch_hip_diff=rise_middle - self.pants_design.length_waist_to_hip,
            hipline_ext=self.BACK_HIPLINE_EXTENSION,
            dart_position=self.pants_design.waist / 4,
            translation=np.array([0, 0, self.DEFAULT_TRANSLATION_Z_BACK]),
            double_dart=False,
        )

        self.stitching_rules = pyg.Stitches(
            (
                self.front.interfaces[InterfaceName.OUTSIDE],
                self.back.interfaces[InterfaceName.OUTSIDE],
            ),
            (
                self.front.interfaces[InterfaceName.INSIDE],
                self.back.interfaces[InterfaceName.INSIDE],
            ),
        )

        if design.cuff_design and design.cuff_design.cuff_length > 0:
            pant_bottom = pyg.Interface.from_multiple(
                self.front.interfaces[InterfaceName.BOTTOM],
                self.back.interfaces[InterfaceName.BOTTOM],
            )

            self.cuff = CuffBand(f"pant_{tag}", design.cuff_design)
            self.cuff.front.translate_by([0, 0, self.DEFAULT_TRANSLATION_Z_FRONT])
            self.cuff.back.translate_by([0, 0, self.DEFAULT_TRANSLATION_Z_BACK])
            self.cuff.place_by_interface(
                self.cuff.interfaces[InterfaceName.TOP],
                pant_bottom,
                gap=5,
                alignment=PanelAlignment.CENTER,
            )
            self.stitching_rules.append(
                (pant_bottom, self.cuff.interfaces[InterfaceName.TOP])
            )

        self.interfaces = {
            InterfaceName.CROTCH_F: self.front.interfaces[InterfaceName.CROTCH],
            InterfaceName.CROTCH_B: self.back.interfaces[InterfaceName.CROTCH],
            InterfaceName.TOP_F: self.front.interfaces[InterfaceName.TOP],
            InterfaceName.TOP_B: self.back.interfaces[InterfaceName.TOP],
        }

    def length(self) -> float:
        """Get the length of the pants half.

        Returns the length measurement of the front panel, which represents
        the overall length of this pants half from waist to hem.

        Returns:
        --------
        float
            The length of the pants half in the design units.
        """
        if self.pants_design.cuff_design:
            return self.front.length() + self.pants_design.cuff_design.cuff_length

        return self.front.length()


class Pants(BaseBottoms):
    """Complete pants garment consisting of left and right halves.

    A full pants garment combines left and right pants halves, each containing
    front and back panels. The halves are mirrored and positioned based on body
    measurements, then connected at the front and back crotch seams.

    Attributes:
    -----------
    right : PantsHalf
        The right leg half of the pants.
    left : PantsHalf
        The left leg half of the pants (mirrored from right).
    body : BodyDefinition
        The body measurements object used for positioning the pants.
    """

    right: PantsHalf
    left: PantsHalf
    body: BodyDefinition

    def __init__(self, design: PantsDesign) -> None:
        """Initialize a complete pants garment.

        Creates both left and right pants halves, positions them based on body
        measurements, and sets up stitching rules to connect the halves at the
        crotch seams. The left half is created by mirroring the right half.

        Parameters:
        -----------
        design : dict
            Design parameters dictionary containing all measurements needed
            for constructing the pants panels.

        Notes:
        ------
        The pants are positioned vertically based on the body's waist level and
        crotch height. The left half is automatically mirrored from the right
        half to ensure symmetry. Interfaces are created for connecting to
        waistbands or other components at the top.
        """
        super().__init__()

        self.right = PantsHalf("r", design)
        # Mirroring creates the left half.
        self.left = PantsHalf("l", design).mirror()

        self.stitching_rules.append(
            (
                self.right.interfaces[InterfaceName.CROTCH_F],
                self.left.interfaces[InterfaceName.CROTCH_F].clone(),
            )
        )
        self.stitching_rules.append(
            (
                self.right.interfaces[InterfaceName.CROTCH_B],
                self.left.interfaces[InterfaceName.CROTCH_B],
            )
        )

        self.interfaces = {
            InterfaceName.TOP_F: pyg.Interface.from_multiple(
                self.right.interfaces[InterfaceName.TOP_F],
                self.left.interfaces[InterfaceName.TOP_F],
            ),
            InterfaceName.TOP_B: pyg.Interface.from_multiple(
                self.right.interfaces[InterfaceName.TOP_B],
                self.left.interfaces[InterfaceName.TOP_B],
            ),
            # Some are reversed for correct connection
            InterfaceName.TOP: pyg.Interface.from_multiple(  # around the body starting from front right
                self.right.interfaces[InterfaceName.TOP_F],
                self.left.interfaces[InterfaceName.TOP_F]
                .clone()
                .reverse(with_edge_dir_reverse=False),
                self.left.interfaces[InterfaceName.TOP_B],
                self.right.interfaces[InterfaceName.TOP_B]
                .clone()
                .reverse(with_edge_dir_reverse=False),
            ),
        }

        # Add lower_interface label for attachment constraints
        self.interfaces[InterfaceName.TOP].edges.propagate_label(
            EdgeLabel.LOWER_INTERFACE
        )

        # Propagate ring labels for deterministic ARAP detection
        self._propagate_ring_labels()

    def length(self) -> float:
        """Get the length of the pants garment.

        Returns the length measurement of the right pants half, which represents
        the overall length of the pants from waist to hem.

        Returns:
        --------
        float
            The length of the pants in the design units.
        """
        return self.right.length()

    @staticmethod
    def get_attachment_constraints() -> list[AttachmentConstraint]:
        """Get the list of attachment constraints for pants garments.

        Returns the attachment constraints required for simulating pants garments.
        Pants require both a crotch constraint to anchor the garment at the crotch level
        and a lower_interface constraint at the waist level.

        The crotch constraint finds vertices by looking for edges with CROTCH_POINT_SEAM label,
        then labels all vertices below those reference vertices (in Y direction).

        The lower_interface constraint finds vertices by looking for edges with LOWER_INTERFACE label.

        Returns:
        --------
        list[AttachmentConstraint]
            A list containing the crotch and lower_interface attachment constraints for pants.
        """
        crotch_constraint = AttachmentConstraint(
            label=BoxEdgeLabel.CROTCH,
            vertex_labels_to_find=[BoxEdgeLabel.CROTCH_POINT_SEAM],
            stiffness=1000.0,
            damping=10.0,
            position_calculation={
                "x": {"positive": [], "negative": []},
                "y": {"positive": ["_leg_length"], "negative": ["crotch_hip_diff"]},
                "z": {"positive": [], "negative": []},
            },
            direction=None,  # Will use default [0, -1, 0] (downward)
        )

        lower_interface_constraint = AttachmentConstraint(
            label=BoxEdgeLabel.LOWER_INTERFACE,
            vertex_labels_to_find=[BoxEdgeLabel.LOWER_INTERFACE],
            stiffness=1000000,
            damping=10.0,
            position_calculation={
                "x": {"positive": [], "negative": []},
                "y": {"positive": ["fit_waist_level"], "negative": []},
                "z": {"positive": [], "negative": []},
            },
            direction=[0, 1, 0],  # Will use default [0, 1, 0] (upward)
        )

        return [crotch_constraint, lower_interface_constraint]

    def get_vertex_processor_callback(self) -> Callable[[BoxMesh], None]:
        """Get the vertex processor callback for pants attachment constraints.

        Returns a callback that handles all pants-specific constraint processing:
        - Gets constraints internally from garment_program.get_attachment_constraints()
        - CROTCH: finds edges with CROTCH_POINT_SEAM, then labels all vertices below in Y
        - LOWER_INTERFACE: uses vertices already labeled with LOWER_INTERFACE from edges
        - Stores constraints in BoxMesh for later use

        Returns:
        --------
        Callable[[BoxMesh], None]
            The vertex processor callback function.
        """
        return (
            lambda box_mesh: AttachmentHandler.process_attachment_constraints_generic(
                Pants, box_mesh
            )
        )

    def apply_body_alignment(self, body: BodyDefinition) -> None:
        """Apply body alignment to position the pants correctly.

        Translates the pants garment vertically based on the body's waist
        level and crotch height to ensure proper fit and alignment.

        Parameters:
        -----------
        body : BodyDefinition
            Body measurements object containing body parameters.
            Required parameters:
            - hips_line: Vertical position of the hips line
            - crotch_hip_diff: Vertical difference between crotch and hips
            - computed_waist_level: Vertical position of the waist level (computed property)
        """
        body_crotch_height = body.hips_line - body.crotch_hip_diff
        body_waist_level = body.computed_waist_level

        self.translate_by([0, body_waist_level - body_crotch_height, 0])

    def _propagate_ring_labels(self) -> None:
        """Propagate ring labels for deterministic ARAP detection.

        Labels propagated:
        - HEM: Waist/top edges (for pants, waist is the "hem" anchor for ARAP)
        - LEFT_ANKLE: Left leg bottom edges
        - RIGHT_ANKLE: Right leg bottom edges
        """
        # WAIST/HEM label on waist (TOP interface)
        # We rely on LOWER_INTERFACE being mapped to WAIST in core_types.py
        # This preserves the LOWER_INTERFACE label for constraints
        pass

        # ANKLE labels on bottom edges
        # Check if cuffs are present - if so, propagate to cuff bottom interface
        if self.right.cuff:
            self.right.cuff.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.RIGHT_ANKLE
            )
        else:
            self.right.front.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.RIGHT_ANKLE
            )
            self.right.back.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.RIGHT_ANKLE
            )

        if self.left.cuff:
            self.left.cuff.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.LEFT_ANKLE
            )
        else:
            self.left.front.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.LEFT_ANKLE
            )
            self.left.back.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.LEFT_ANKLE
            )

    def get_garment_metadata(self) -> GarmentMetadata:
        """Get deterministic garment metadata for ARAP processing.

        Returns
        -------
        GarmentMetadata
            Metadata containing category, panel positions, ring connectors,
            and seam paths computed at garment creation time.
        """
        # Panel positions for pants (front/back for each leg)
        panel_positions: dict[str, PanelPosition] = {
            "pant_f_r": PanelPosition.FRONT,
            "pant_b_r": PanelPosition.BACK,
            "pant_f_l": PanelPosition.FRONT,
            "pant_b_l": PanelPosition.BACK,
        }

        # Add cuff panels if present
        if self.right.cuff:
            panel_positions["pant_r_cuff_f"] = PanelPosition.FRONT
            panel_positions["pant_r_cuff_b"] = PanelPosition.BACK

        if self.left.cuff:
            panel_positions["pant_l_cuff_f"] = PanelPosition.FRONT
            panel_positions["pant_l_cuff_b"] = PanelPosition.BACK

        return GarmentMetadata(
            category=GarmentCategory.PANTS,
            panel_positions=panel_positions,
            ring_connectors={},  # Populated during BoxMesh generation
            seam_paths=[],  # Populated during BoxMesh generation
        )

    # =========================================================================
    # ARAPInitializable Protocol Implementation
    # =========================================================================

    def get_garment_category(self) -> GarmentCategory:
        """Return the garment category.

        Returns
        -------
        GarmentCategory
            PANTS category.
        """
        return GarmentCategory.PANTS

    def get_rings(self) -> list[GarmentRing]:
        """Return all rings (closed boundary curves) for pants.

        Pants rings: HEM (waist), LEFT_ANKLE, RIGHT_ANKLE.
        For pants with cuffs, ankle rings come from cuff BOTTOM interfaces.
        Dart edges are NOT included in rings.

        Returns
        -------
        list[GarmentRing]
            All rings with their edges and connectors.
        """
        rings: list[GarmentRing] = []

        # HEM ring (waist) - from TOP interfaces combined
        hem_edges = self._collect_waist_edges()
        if hem_edges and len(hem_edges) > 0:
            conn1 = RingConnector(
                edge=hem_edges[0],
                parameter=0.0,
                ring_type=GarmentRingType.HEM,
            )
            conn2 = RingConnector(
                edge=hem_edges[-1],
                parameter=1.0,
                ring_type=GarmentRingType.HEM,
            )
            rings.append(
                GarmentRing(
                    ring_type=GarmentRingType.HEM,
                    edges=hem_edges,
                    connectors=(conn1, conn2),
                )
            )

        # ANKLE rings
        for side, ankle_type in [
            ("right", GarmentRingType.RIGHT_ANKLE),
            ("left", GarmentRingType.LEFT_ANKLE),
        ]:
            half: PantsHalf = getattr(self, side)
            ankle_edges = self._collect_ankle_edges(half)
            if ankle_edges and len(ankle_edges) > 0:
                conn1 = RingConnector(
                    edge=ankle_edges[0],
                    parameter=0.0,
                    ring_type=ankle_type,
                )
                conn2 = RingConnector(
                    edge=ankle_edges[-1],
                    parameter=1.0,
                    ring_type=ankle_type,
                )
                rings.append(
                    GarmentRing(
                        ring_type=ankle_type,
                        edges=ankle_edges,
                        connectors=(conn1, conn2),
                    )
                )

        return rings

    def _collect_waist_edges(self) -> pyg.EdgeSequence | None:
        """Collect waist ring edges from top interface."""
        if InterfaceName.TOP in self.interfaces:
            return self.interfaces[InterfaceName.TOP].edges
        return None

    def _collect_ankle_edges(self, half: PantsHalf) -> pyg.EdgeSequence | None:
        """Collect ankle ring edges from leg bottom or cuff."""
        if half.cuff:
            # Has cuff - use cuff BOTTOM interface
            if InterfaceName.BOTTOM in half.cuff.interfaces:
                return half.cuff.interfaces[InterfaceName.BOTTOM].edges
        else:
            # No cuff - use combined front/back BOTTOM
            ankle_edges = []
            if InterfaceName.BOTTOM in half.front.interfaces:
                ankle_edges.extend(half.front.interfaces[InterfaceName.BOTTOM].edges)
            if InterfaceName.BOTTOM in half.back.interfaces:
                ankle_edges.extend(half.back.interfaces[InterfaceName.BOTTOM].edges)
            if ankle_edges:
                return pyg.EdgeSequence(*ankle_edges)
        return None

    def _collect_outseam_edges(self, side: str) -> pyg.EdgeSequence | None:
        """Collect edges for the outseam (Waist -> Ankle).

        Ensures strict Descending Y ordering (Top -> Bottom).
        """

        def clone_edge(e):
            return copy.deepcopy(e)

        def reverse_edge(e):
            ne = copy.deepcopy(e)
            ne.reverse()
            return ne

        def _ensure_descending_y(interface):
            corrected_edges = []
            if not interface or not interface.edges:
                return None

            for e in interface.edges:
                # We want Start Y > End Y (Descending)
                if e.start[1] < e.end[1]:
                    corrected_edges.append(reverse_edge(e))
                else:
                    corrected_edges.append(clone_edge(e))
            # Sort Descending by Start Y (Highest first)
            corrected_edges.sort(key=lambda x: x.start[1], reverse=True)
            return pyg.EdgeSequence(*corrected_edges)

        half: PantsHalf = getattr(self, side)
        # Use Front panel's OUTSIDE interface as the canonical path
        if InterfaceName.OUTSIDE in half.front.interfaces:
            seq = _ensure_descending_y(half.front.interfaces[InterfaceName.OUTSIDE])
            if seq:
                edges = list(seq.edges)
                for e in edges:
                    e.panel = half.front

                # If Cuff exists, find the matching side seam
                if half.cuff:
                    leg_end = edges[-1].end
                    # Check Cuff Front RIGHT and LEFT edges
                    candidates = []
                    candidates.append(
                        ("RIGHT", half.cuff.front.interfaces.get(InterfaceName.RIGHT))
                    )
                    candidates.append(
                        ("LEFT", half.cuff.front.interfaces.get(InterfaceName.LEFT))
                    )

                    best_seq = None
                    min_dist = float("inf")

                    for _, cand_int in candidates:
                        if not cand_int:
                            continue
                        cand_seq = _ensure_descending_y(cand_int)
                        if not cand_seq:
                            continue

                        # Distance from Leg End to Cuff Start
                        dist = np.linalg.norm(
                            np.array(leg_end) - np.array(cand_seq.edges[0].start)
                        ).item()
                        if dist < min_dist:
                            min_dist = dist
                            best_seq = cand_seq

                    # If valid match found (tolerance increased for Local Coords mismatch), append
                    if best_seq and min_dist < 300.0:
                        for e in best_seq.edges:
                            e.panel = half.cuff.front
                        edges.extend(best_seq.edges)

                return pyg.EdgeSequence(*edges)
        return None

    def get_ring_connections(self) -> list[RingConnection]:
        """Return all seam paths between rings.

        Pants connections: HEM→LEFT_ANKLE, HEM→RIGHT_ANKLE (side seams).
        NOTE: Left and Right are swapped to correct for body/garment orientation mismatch.
        """
        connections: list[RingConnection] = []

        # 1. Right Outseam (HEM -> RIGHT_ANKLE)
        right_outseam = self._collect_outseam_edges("right")
        if right_outseam:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.HEM,
                    ring_2=GarmentRingType.RIGHT_ANKLE,
                    connector_1=RingConnector(
                        right_outseam[0], 0.0, GarmentRingType.HEM
                    ),
                    connector_2=RingConnector(
                        right_outseam[-1], 1.0, GarmentRingType.RIGHT_ANKLE
                    ),
                    path_edges=right_outseam,
                )
            )

        # 2. Left Outseam (HEM -> LEFT_ANKLE)
        left_outseam = self._collect_outseam_edges("left")
        if left_outseam:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.HEM,
                    ring_2=GarmentRingType.LEFT_ANKLE,
                    connector_1=RingConnector(
                        left_outseam[0], 0.0, GarmentRingType.HEM
                    ),
                    connector_2=RingConnector(
                        left_outseam[-1], 1.0, GarmentRingType.LEFT_ANKLE
                    ),
                    path_edges=left_outseam,
                )
            )

        return connections
