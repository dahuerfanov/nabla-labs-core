import copy
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

import weon_garment_code.pygarment.garmentcode as pyg
from weon_garment_code.config import AttachmentConstraint
from weon_garment_code.garment_programs import weon_sleeves
from weon_garment_code.garment_programs.base_classes import BaseBodicePanel
from weon_garment_code.garment_programs.garment_enums import (
    EdgeLabel,
    InterfaceName,
    PanelAlignment,
)
from weon_garment_code.garment_programs.garment_program_utils import AttachmentHandler
from weon_garment_code.garment_programs.weon_bands import StraightWB
from weon_garment_code.garment_programs.weon_collars import NoPanelsCollar
from weon_garment_code.garment_programs.weon_torso import (
    TorsoBackHalfPanel,
    TorsoFrontHalfPanel,
)
from weon_garment_code.pattern_definitions.body_definition import BodyDefinition
from weon_garment_code.pattern_definitions.shirt_design import ShirtDesign
from weon_garment_code.pattern_definitions.torso_design import TorsoDesign
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


class BodiceFrontHalf(BaseBodicePanel):
    # Dart and fit constants
    BOTTOM_DART_WIDTH_FACTOR: float = 2 / 3
    SIDE_DART_DEPTH_FACTOR: float = 0.75
    BOTTOM_DART_DEPTH_FACTOR: float = 0.9

    # Class attributes
    width: float

    def __init__(
        self, name: str, body: BodyDefinition | dict, torso_design: TorsoDesign
    ) -> None:
        """Initialize half of a fitted bodice front panel.

        Parameters
        ----------
        name : str
            Name of the panel.
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        torso_design : TorsoDesign
            Design parameters dictionary.
        """
        super().__init__(name, torso_design)

        m_bust = body["bust"]
        m_waist = body["waist"]

        # sizes
        bust_point = body["bust_points"] / 2
        front_frac = (body["bust"] - body["back_width"]) / 2 / body["bust"]

        self.width = front_frac * m_bust
        waist = (m_waist - body["waist_back_width"]) / 2
        sh_tan = np.tan(np.deg2rad(body["_shoulder_incl"]))
        shoulder_incl = sh_tan * self.width
        bottom_d_width = (self.width - waist) * self.BOTTOM_DART_WIDTH_FACTOR

        adjustment = sh_tan * (self.width - body["shoulder_w"] / 2)
        max_len = body["waist_over_bust_line"] - adjustment

        # side length is adjusted due to shoulder inclination
        # for the correct sleeve fitting
        fb_diff = (front_frac - (0.5 - front_frac)) * body["bust"]
        back_adjustment = sh_tan * (body["back_width"] / 2 - body["shoulder_w"] / 2)
        side_len = body["waist_line"] - back_adjustment - sh_tan * fb_diff

        self.edges.append(
            pyg.EdgeSeqFactory.from_verts(
                [0, 0],
                [-self.width, 0],
                [-self.width, max_len],
                [0, max_len + shoulder_incl],
            )
        )
        self.edges.close_loop()

        # Side dart
        bust_line = body["waist_line"] - body["_bust_line"]
        side_d_depth = self.SIDE_DART_DEPTH_FACTOR * (
            self.width - bust_point
        )  # NOTE: calculated value
        side_d_width = max_len - side_len
        s_edge, side_interface = self.add_dart(
            pyg.EdgeSeqFactory.dart_shape(side_d_width, side_d_depth),
            self.edges[1],
            offset=bust_line + side_d_width / 2,
        )
        self.edges.substitute(1, s_edge)

        # Take some fabric from the top to match the shoulder width
        s_edge[-1].end[0] += (x_upd := self.width - body["shoulder_w"] / 2)
        s_edge[-1].end[1] += sh_tan * x_upd

        # Bottom dart
        b_edge, b_interface = self.add_dart(
            pyg.EdgeSeqFactory.dart_shape(
                bottom_d_width, self.BOTTOM_DART_DEPTH_FACTOR * bust_line
            ),
            self.edges[0],
            offset=bust_point + bottom_d_width / 2,
        )
        self.edges.substitute(0, b_edge)
        # Take some fabric from side in the bottom (!: after side dart insertion)
        b_edge[-1].end[0] = -(waist + bottom_d_width)

        # Interfaces
        self.interfaces = {
            InterfaceName.OUTSIDE: pyg.Interface(
                self, side_interface
            ),  # side_interface,    # pyp.Interface(self, [side_interface]),  #, self.edges[-3]]),
            InterfaceName.INSIDE: pyg.Interface(self, self.edges[-1]),
            InterfaceName.SHOULDER: pyg.Interface(self, self.edges[-2]),
            InterfaceName.BOTTOM: pyg.Interface(self, b_interface),
            # Reference to the corner for sleeve and collar projections
            InterfaceName.SHOULDER_CORNER: pyg.Interface(
                self, [self.edges[-3], self.edges[-2]]
            ),
            InterfaceName.COLLAR_CORNER: pyg.Interface(
                self, [self.edges[-2], self.edges[-1]]
            ),
        }

        # default placement
        self.translate_by(
            [0, body["height"] - body["head_l"] - max_len - shoulder_incl, 0]
        )


class BodiceBackHalf(BaseBodicePanel):
    """Panel for the back of a basic fitted bodice block."""

    # Dart and fit constants
    SIDE_ADJUSTMENT_THRESHOLD: int = 4
    SIDE_ADJUSTMENT_FACTOR: float = 1 / 6
    BOTTOM_DART_DEPTH_FACTOR: float = 1.0
    SMALL_DART_DEPTH_FACTOR: float = 0.9
    DART_POSITION_MULTIPLIER: float = 0.5

    # Class attributes
    width: float

    def __init__(
        self, name: str, body: BodyDefinition | dict, torso_design: TorsoDesign
    ) -> None:
        """Initialize the back panel of a basic fitted bodice block.

        Parameters
        ----------
        name : str
            Name of the panel.
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        torso_design : TorsoDesign
            Design parameters dictionary.
        """
        super().__init__(name, torso_design)

        # Overall measurements
        self.width = body["back_width"] / 2
        waist = body["waist_back_width"] / 2
        # NOTE: no inclination on the side, since there is not much to begin with
        waist_width = self.width if waist < self.width else waist
        shoulder_incl = (
            sh_tan := np.tan(np.deg2rad(body["_shoulder_incl"]))
        ) * self.width

        # Adjust to make sure length is measured from the shoulder
        # and not the de-fact side of the garment
        back_adjustment = sh_tan * (self.width - body["shoulder_w"] / 2)
        length = body["waist_line"] - back_adjustment

        # Base edge loop
        edge_0 = pyg.CurveEdgeFactory.curve_from_tangents(
            start=[0, shoulder_incl / 4],  # back a little shorter
            end=[-waist_width, 0],
            target_tan0=[-1, 0],
        )
        self.edges.append(edge_0)
        self.edges.append(
            pyg.EdgeSeqFactory.from_verts(
                edge_0.end,
                [
                    -self.width,
                    body["waist_line"] - body["_bust_line"],
                ],  # from the bottom
                [-self.width, length],
                [
                    0,
                    length + shoulder_incl,
                ],  # Add some fabric for the neck (inclination of shoulders)
            )
        )
        self.edges.close_loop()

        # Take some fabric from the top to match the shoulder width
        self.interfaces = {
            InterfaceName.OUTSIDE: pyg.Interface(self, [self.edges[1], self.edges[2]]),
            InterfaceName.INSIDE: pyg.Interface(self, self.edges[-1]),
            InterfaceName.SHOULDER: pyg.Interface(self, self.edges[-2]),
            InterfaceName.BOTTOM: pyg.Interface(self, self.edges[0]),
            # Reference to the corners for sleeve and collar projections
            InterfaceName.SHOULDER_CORNER: pyg.Interface(
                self, pyg.EdgeSequence(self.edges[-3], self.edges[-2])
            ),
            InterfaceName.COLLAR_CORNER: pyg.Interface(
                self, pyg.EdgeSequence(self.edges[-2], self.edges[-1])
            ),
        }

        # Bottom dart as cutout -- for straight line
        if waist < self.get_width(self.edges[2].end[1] - self.edges[2].start[1]):
            w_diff = waist_width - waist
            side_adj = (
                0
                if w_diff < self.SIDE_ADJUSTMENT_THRESHOLD
                else w_diff * self.SIDE_ADJUSTMENT_FACTOR
            )  # NOTE: don't take from sides if the difference is too small
            bottom_d_width = w_diff - side_adj
            bottom_d_width /= 2  # double darts
            bottom_d_depth = self.BOTTOM_DART_DEPTH_FACTOR * (
                length - body["_bust_line"]
            )  # calculated value
            bottom_d_position = body["bum_points"] / 2

            # TODOLOW Avoid hardcoding for matching with the bottoms?
            dist = (
                bottom_d_position * self.DART_POSITION_MULTIPLIER
            )  # Dist between darts -> dist between centers
            b_edge, b_interface = self.add_dart(
                pyg.EdgeSeqFactory.dart_shape(
                    bottom_d_width, self.SMALL_DART_DEPTH_FACTOR * bottom_d_depth
                ),
                self.edges[0],
                offset=bottom_d_position
                + dist / 2
                + bottom_d_width
                + bottom_d_width / 2,
            )
            b_edge, b_interface = self.add_dart(
                pyg.EdgeSeqFactory.dart_shape(bottom_d_width, bottom_d_depth),
                b_edge[0],
                offset=bottom_d_position - dist / 2 + bottom_d_width / 2,
                edge_seq=b_edge,
                int_edge_seq=b_interface,
            )

            self.edges.substitute(0, b_edge)
            self.interfaces[InterfaceName.BOTTOM] = pyg.Interface(self, b_interface)

            # Remove fabric from the sides if the diff is big enough
            b_edge[-1].end[0] += side_adj

        # default placement
        self.translate_by(
            [0, body["height"] - body["head_l"] - length - shoulder_incl, 0]
        )

    def get_width(self, level: float) -> float:
        """Get the width of the panel at a given level.

        Parameters:
        -----------
        level : float
            Vertical level from the top of the panel.

        Returns:
        --------
        float
            Width of the panel at the given level.
        """
        return self.width


class BodiceHalf(pyg.Component):
    """Definition of a half of an upper garment with sleeves and collars"""

    # Panel placement constants
    FRONT_TORSO_TRANSLATION_Z: float = 30
    BACK_TORSO_TRANSLATION_Z: float = -25

    # Sleeve placement constants
    SLEEVE_PLACEMENT_GAP_BASE: float = -1
    SLEEVE_PLACEMENT_GAP_FACTOR: float = 10

    # Class attributes
    shirt_design: ShirtDesign
    ftorso: TorsoFrontHalfPanel
    btorso: TorsoBackHalfPanel
    sleeve: weon_sleeves.Sleeve | None
    collar_comp: Any | None
    _bodice_sleeve_int: pyg.EdgeSequence | None
    _sleeve_placement_gap: float | None

    def __init__(
        self,
        name: str,
        shirt_design: ShirtDesign,
        arm_pose_angle: float,
        fitted: bool = True,
    ) -> None:
        """Define half of an upper garment, including torso, sleeves, and collars.

        Parameters
        ----------
        name : str
            Name of the component (e.g., 'right' or 'left').
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        shirt_design : ShirtDesign
            Shirt design object.
        arm_pose_angle : float
            Arm pose angle.
        fitted : bool, optional
            If True, creates a fitted bodice; otherwise, a looser torso (tee-style).
            Default is True.
        """
        super().__init__(name)

        # Create ShirtDesign from the design dict and store it
        self.shirt_design = shirt_design

        # Torso
        if fitted:
            raise NotImplementedError("Fitted bodice not implemented yet.")
        else:
            self.ftorso = TorsoFrontHalfPanel(
                name=f"{name}_ftorso", shirt_design=shirt_design
            )
            self.ftorso.translate_by([0, 0, self.FRONT_TORSO_TRANSLATION_Z])

            self.btorso = TorsoBackHalfPanel(
                name=f"{name}_btorso", shirt_design=shirt_design
            )
            self.btorso.translate_by([0, 0, self.BACK_TORSO_TRANSLATION_Z])

        # Interfaces
        self.interfaces.update(
            {
                InterfaceName.F_BOTTOM: self.ftorso.interfaces[InterfaceName.BOTTOM],
                InterfaceName.B_BOTTOM: self.btorso.interfaces[InterfaceName.BOTTOM],
                InterfaceName.FRONT_IN: self.ftorso.interfaces[InterfaceName.INSIDE],
                InterfaceName.BACK_IN: self.btorso.interfaces[InterfaceName.INSIDE],
            }
        )

        # Sleeves/collar cuts
        self.sleeve: weon_sleeves.Sleeve | None = None
        self.collar_comp: Any | None = None
        self._bodice_sleeve_int: pyg.EdgeSequence | None = None
        self._sleeve_placement_gap: float | None = None

        if (
            self.shirt_design.torso_design.strapless and fitted
        ):  # NOTE: Strapless design only for fitted tops
            # self.make_strapless(body, design)
            raise NotImplementedError("Strapless design not implemented yet.")
        else:
            # Sleeves and collars
            # NOTE assuming the vertical side is the first argument
            max_cwidth = (
                self.ftorso.interfaces[InterfaceName.SHOULDER_CORNER].edges[0].length()
                - 1
            )  # cm
            min_cwidth = (
                self.shirt_design.torso_design.scye_depth
                - self.shirt_design.torso_design.shoulder_slant
            )
            adjusted_connecting_width = min(
                min_cwidth
                + min_cwidth * self.shirt_design.sleeve_design.connecting_width,
                max_cwidth,
            )
            self.add_sleeves(name, arm_pose_angle, adjusted_connecting_width)

            self.add_collars(name)
            self.stitching_rules.append(
                (
                    self.ftorso.interfaces[InterfaceName.SHOULDER],
                    self.btorso.interfaces[InterfaceName.SHOULDER],
                )
            )  # tops

        # Main connectivity
        self.stitching_rules.append(
            (
                self.ftorso.interfaces[InterfaceName.OUTSIDE],
                self.btorso.interfaces[InterfaceName.OUTSIDE],
            )
        )  # sides

    def add_sleeves(
        self, name: str, arm_pose_angle: float, connecting_width: float
    ) -> None:
        """Add sleeves to the bodice half.

        Parameters
        ----------
        name : str
            Name of the component (e.g., 'right' or 'left').
        design : dict
            Design parameters dictionary.
        arm_pose_angle : float
            Arm pose angle.
        connecting_width: float
            Connecting width of the sleeves.
        """

        self.sleeve = weon_sleeves.Sleeve(
            tag=name,
            shirt_design=self.shirt_design.torso_design,
            sleeve_design=self.shirt_design.sleeve_design,
            front_w=(
                self.shirt_design.torso_design.width_chest
                - self.shirt_design.torso_design.smallest_width
            )
            / 2,
            back_w=(
                self.shirt_design.torso_design.width_chest
                - self.shirt_design.torso_design.smallest_width
            )
            * self.shirt_design.torso_design.back_width
            / self.shirt_design.torso_design.width_chest
            / 2,
            front_hole_edge=self.ftorso.armhole_edge,
            back_hole_edge=self.btorso.armhole_edge,
            adjusted_connecting_width=connecting_width,
        )

        _, f_sleeve_int = pyg.ops.cut_corner(
            self.sleeve.interfaces[InterfaceName.IN_FRONT_SHAPE].edges,
            self.ftorso.interfaces[InterfaceName.SHOULDER_CORNER],
            verbose=self.verbose,
        )
        _, b_sleeve_int = pyg.ops.cut_corner(
            self.sleeve.interfaces[InterfaceName.IN_BACK_SHAPE].edges,
            self.btorso.interfaces[InterfaceName.SHOULDER_CORNER],
            verbose=self.verbose,
        )

        if not self.shirt_design.sleeve_design.sleeveless:
            # Ordering
            bodice_sleeve_int = pyg.Interface.from_multiple(
                f_sleeve_int.reverse(with_edge_dir_reverse=True),
                b_sleeve_int.reverse(),
            )
            self.stitching_rules.append(
                (self.sleeve.interfaces[InterfaceName.IN], bodice_sleeve_int)
            )

            # NOTE: This is a heuristic tuned for arm poses 30 deg-60 deg
            # used in the dataset
            # FIXME Needs a better general solution
            gap = (
                self.SLEEVE_PLACEMENT_GAP_BASE
                - arm_pose_angle / self.SLEEVE_PLACEMENT_GAP_FACTOR
            )
            self.sleeve.place_by_interface(
                self.sleeve.interfaces[InterfaceName.IN],
                bodice_sleeve_int,
                gap=gap,
                alignment=PanelAlignment.TOP,
            )

            # Store armhole interface and gap for re-alignment after rotation
            self._bodice_sleeve_int = bodice_sleeve_int
            self._sleeve_placement_gap = gap

        # Add edge labels
        f_sleeve_int.edges.propagate_label(f"{self.name}_{EdgeLabel.ARMHOLE}")
        b_sleeve_int.edges.propagate_label(f"{self.name}_{EdgeLabel.ARMHOLE}")

    def translate_and_rotate_sleeve(
        self, translation: tuple[float, float, float], angle: float
    ) -> None:
        """Translate and rotate the sleeve piece, then re-align with armhole.

        Parameters
        ----------
        translation : tuple[float, float, float]
            Translation to apply to the sleeve piece.
        angle : float
            Angle to rotate the sleeve piece.
        """
        if self.sleeve is not None:
            self.sleeve.translate_by(translation)
            self.sleeve.rotate_by(R.from_euler("XYZ", [0, 0, angle], degrees=True))

            # Re-align sleeve with armhole after rotation to maintain proper connection
            if (
                self._bodice_sleeve_int is not None
                and not self.shirt_design.sleeve_design.sleeveless
            ):
                self.sleeve.place_by_interface(
                    self.sleeve.interfaces[InterfaceName.IN],
                    self._bodice_sleeve_int,
                    gap=self._sleeve_placement_gap,
                    alignment=PanelAlignment.TOP,
                )

    def add_collars(
        self,
        name: str,
    ) -> None:
        """Add collars to the bodice half.

        Parameters
        ----------
        name : str
            Name of the component (e.g., 'right' or 'left').
        """

        # Front
        self.collar_comp = NoPanelsCollar(name, self.shirt_design.collar_design)

        # Project shape
        _, fc_interface = pyg.ops.cut_corner(
            self.collar_comp.interfaces[InterfaceName.FRONT_PROJ].edges,
            self.ftorso.interfaces[InterfaceName.COLLAR_CORNER],
            verbose=self.verbose,
        )
        _, bc_interface = pyg.ops.cut_corner(
            self.collar_comp.interfaces[InterfaceName.BACK_PROJ].edges,
            self.btorso.interfaces[InterfaceName.COLLAR_CORNER],
            verbose=self.verbose,
        )

        # Add stitches/interfaces
        if InterfaceName.BOTTOM in self.collar_comp.interfaces:
            self.stitching_rules.append(
                (
                    pyg.Interface.from_multiple(fc_interface, bc_interface),
                    self.collar_comp.interfaces[InterfaceName.BOTTOM],
                )
            )

        # Upd front interfaces accordingly
        if InterfaceName.FRONT in self.collar_comp.interfaces:
            self.interfaces[InterfaceName.FRONT_COLLAR] = self.collar_comp.interfaces[
                InterfaceName.FRONT
            ]
            self.interfaces[InterfaceName.FRONT_IN] = pyg.Interface.from_multiple(
                self.ftorso.interfaces[InterfaceName.INSIDE],
                self.interfaces[InterfaceName.FRONT_COLLAR],
            )
        if InterfaceName.BACK in self.collar_comp.interfaces:
            self.interfaces[InterfaceName.BACK_COLLAR] = self.collar_comp.interfaces[
                InterfaceName.BACK
            ]
            self.interfaces[InterfaceName.BACK_IN] = pyg.Interface.from_multiple(
                self.btorso.interfaces[InterfaceName.INSIDE],
                self.interfaces[InterfaceName.BACK_COLLAR],
            )

        # Add edge labels
        fc_interface.edges.propagate_label(EdgeLabel.COLLAR.value)
        bc_interface.edges.propagate_label(EdgeLabel.COLLAR.value)

    def make_strapless(self, body: BodyDefinition | dict, design: dict) -> None:
        """Modify the bodice half to be strapless.

        Parameters
        ----------
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        design : dict
            Design parameters dictionary.
        """

        out_depth = design["sleeve"]["connecting_width"][
            "v"
        ]  # May have been modified in eval_dep_params
        f_in_depth = design["collar"]["f_strapless_depth"][
            "v"
        ]  # May have been modified in eval_dep_params
        b_in_depth = design["collar"]["b_strapless_depth"][
            "v"
        ]  # May have been modified in eval_dep_params

        # Shoulder adjustment for the back
        # TODOLOW Shoulder adj evaluation should be a function
        shoulder_angle = np.deg2rad(body["_shoulder_incl"])
        sleeve_balance = body["_base_sleeve_balance"] / 2
        back_w = self.btorso.get_width(0)
        shoulder_adj = np.tan(shoulder_angle) * (back_w - sleeve_balance)
        out_depth -= shoulder_adj

        # Upd back
        self._adjust_top_level(self.btorso, out_depth, b_in_depth)

        # Front depth determined by ~compensating for lenght difference
        len_back = self.btorso.interfaces[InterfaceName.OUTSIDE].edges.length()
        len_front = self.ftorso.interfaces[InterfaceName.OUTSIDE].edges.length()
        self._adjust_top_level(
            self.ftorso, out_depth, f_in_depth, target_remove=(len_front - len_back)
        )

        # Placement
        # NOTE: The commented line places the top a bit higher, increasing the chanced of correct drape
        # Surcumvented by attachment constraint, so removed for nicer alignment in asymmetric garments
        # self.translate_by([0, out_depth - body['_armscye_depth'] * 0.75, 0])   # adjust for better localisation

        # Add a label
        self.ftorso.interfaces[InterfaceName.SHOULDER].edges.propagate_label(
            EdgeLabel.STRAPLESS_TOP.value
        )
        self.btorso.interfaces[InterfaceName.SHOULDER].edges.propagate_label(
            EdgeLabel.STRAPLESS_TOP.value
        )

    def _adjust_top_level(
        self,
        panel: BaseBodicePanel,
        out_level: float,
        in_level: float,
        target_remove: float | None = None,
    ) -> None:
        """Crop the top of the bodice front/back panel for strapless style.

        Parameters
        ----------
        panel : BaseBodicePanel
            The panel to adjust.
        out_level : float
            Outer depth to remove.
        in_level : float
            Inner depth to remove.
        target_remove : float, optional
            If set, determines the length difference that should be compensated
            after cutting the depth.
        """
        # TODOLOW Should this be the panel's function?

        panel_top = panel.interfaces[InterfaceName.SHOULDER].edges[0]
        min_y = min(panel_top.start[1], panel_top.end[1])

        # Order vertices
        ins, out = panel_top.start, panel_top.end
        if panel_top.start[1] < panel_top.end[1]:
            ins, out = out, ins

        # Inside is a simple vertical line and can be adjusted by chaning Y value
        ins[1] = min_y - in_level

        # Outside could be inclined, so needs further calculations
        outside_edge = panel.interfaces[InterfaceName.OUTSIDE].edges[-1]
        bot, top = outside_edge.start, outside_edge.end
        if bot is out:
            bot, top = top, bot

        if target_remove is not None:
            # Adjust the depth to remove this length exactly
            angle_sin = abs(out[1] - bot[1]) / outside_edge.length()
            curr_remove = out_level / angle_sin
            length_diff = target_remove - curr_remove
            adjustment = length_diff * angle_sin
            out_level += adjustment

        angle_cotan = abs(out[0] - bot[0]) / abs(out[1] - bot[1])
        out[0] -= out_level * angle_cotan
        out[1] = min_y - out_level

    def length(self) -> float:
        """Return the length of the bodice half.

        Returns:
        --------
        float
            Length of the bodice half.
        """
        return self.btorso.length()


class Shirt(pyg.Component):
    """Panel for the front of upper garments with darts to properly fit it to
    the shape"""

    # Class constants
    FRONT_TORSO_TRANSLATION_Z: float = 30.0

    # Class attributes
    shirt_design: ShirtDesign
    right: BodiceHalf
    left: BodiceHalf

    def __init__(
        self, shirt_design: ShirtDesign, arm_pose_angle: float, fitted: bool = False
    ) -> None:
        """Create a shirt garment component.

        Parameters
        ----------
        shirt_design : ShirtDesign
            Shirt design object.
        arm_pose_angle : float
            Arm pose angle, only used for bosy-sleeve init aligment.
        fitted : bool, optional
            If True, creates a fitted shirt; otherwise, a looser fit.
            Default is False.
        """

        name_with_params = f"{self.__class__.__name__}"
        super().__init__(name_with_params)

        # Create ShirtDesign from the design dict and store it
        self.shirt_design = shirt_design

        self.right = BodiceHalf(
            "right", self.shirt_design, arm_pose_angle=arm_pose_angle, fitted=fitted
        )
        self.left = BodiceHalf(
            "left", self.shirt_design, arm_pose_angle=arm_pose_angle, fitted=fitted
        ).mirror()

        self.stitching_rules.append(
            (
                self.right.interfaces[InterfaceName.FRONT_IN],
                self.left.interfaces[InterfaceName.FRONT_IN],
            )
        )
        self.stitching_rules.append(
            (
                self.right.interfaces[InterfaceName.BACK_IN],
                self.left.interfaces[InterfaceName.BACK_IN],
            )
        )

        # Adjust interface ordering for correct connectivity
        self.interfaces = {  # Bottom connection
            InterfaceName.BOTTOM: pyg.Interface.from_multiple(
                self.right.interfaces[InterfaceName.F_BOTTOM].clone().reverse(),
                self.left.interfaces[InterfaceName.F_BOTTOM],
                self.left.interfaces[InterfaceName.B_BOTTOM].clone().reverse(),
                self.right.interfaces[InterfaceName.B_BOTTOM],
            )
        }

        if self.shirt_design.waistband_design.width > 0:
            wb = StraightWB(
                waistband_design=self.shirt_design.waistband_design,
                waist=self.shirt_design.waistband_design.length,
                waist_back=self.shirt_design.waistband_design.length / 2,
                waist_front=self.shirt_design.waistband_design.length / 2,
                rise=1.0,
            )
            wb.place_by_interface(
                wb.interfaces[InterfaceName.TOP],
                self.interfaces[InterfaceName.BOTTOM],
                gap=5,
            )

            # Align Z depth with torso panels dynamically
            # Front
            dz_front = self.right.ftorso.translation[2] - wb.front.translation[2]
            wb.front.translate_by([0, 0, dz_front])

            # Back
            dz_back = self.right.btorso.translation[2] - wb.back.translation[2]
            wb.back.translate_by([0, 0, dz_back])
            # Register waistband as sub-component so its panels are included
            self.subs.append(wb)
            # Stitch front torso halves to front waistband
            self.stitching_rules.append(
                (
                    pyg.Interface.from_multiple(
                        self.right.ftorso.interfaces[InterfaceName.BOTTOM]
                        .clone()
                        .reverse(with_edge_dir_reverse=True),
                        self.left.ftorso.interfaces[InterfaceName.BOTTOM],
                    ),
                    wb.front.interfaces[InterfaceName.TOP],
                )
            )
            # Stitch back torso halves to back waistband
            self.stitching_rules.append(
                (
                    pyg.Interface.from_multiple(
                        self.right.btorso.interfaces[InterfaceName.BOTTOM]
                        .clone()
                        .reverse(with_edge_dir_reverse=True),
                        self.left.btorso.interfaces[InterfaceName.BOTTOM],
                    ),
                    wb.back.interfaces[InterfaceName.TOP]
                    .clone()
                    .reverse(with_edge_dir_reverse=True),
                )
            )
            # Add hem label to the bottom of the waistband
            wb.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.LOWER_INTERFACE.value
            )

            # Update BOTTOM interface to point to the waistband bottom
            self.interfaces[InterfaceName.BOTTOM] = wb.interfaces[InterfaceName.BOTTOM]

        # Propagate ring labels for deterministic ARAP detection
        self._propagate_ring_labels()

    def length(self) -> float:
        """Return the length of the shirt.

        Returns:
        --------
        float
            Length of the shirt.
        """
        return self.right.length()

    @staticmethod
    def get_attachment_constraints() -> list[AttachmentConstraint]:
        """Get the list of attachment constraints for this shirt garment.

        Returns the attachment constraints required for simulating this shirt garment.
        Shirts do not require any attachment constraints by default.

        Returns:
        --------
        list[AttachmentConstraInt]
            An empty list (shirts do not require attachment constraints).
        """
        return []

    def get_vertex_processor_callback(self) -> Callable[[BoxMesh], None]:
        """Get the vertex processor callback for shirt attachment constraints.

        Shirts do not require custom vertex processing, so this returns a default
        callback that uses generic constraint processing.

        Returns:
        --------
        Callable[[BoxMesh], None] | None
            A callback that handles constraints with default behavior.
        """
        return AttachmentHandler.create_default_vertex_processor(Shirt)

    def apply_body_alignment(self, body: BodyDefinition) -> None:
        """Apply body alignment to the shirt.

        Parameters
        ----------
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        """
        # Apply sleeve rotations
        # arm_pose_angle is measured from the spine (vertical), so T-pose is 90 deg.
        # We want to rotate FROM horizontal (90 deg) TO the body angle.
        # Rotation = 90 - angle.
        angle = 90 - body.arm_pose_angle

        self.right.translate_and_rotate_sleeve(translation=(0, 0, 0), angle=angle)

        # Apply to left sleeve (angle negated because left side is mirrored)
        self.left.translate_and_rotate_sleeve(translation=(0, 0, 0), angle=-angle)

        # Position entire shirt relative to body height (torso panels are positioned at Y=0,
        # but should be at body.height - body.head_l to align with body)
        self.translate_by([0, body.height - body.head_l, 0])

    def _propagate_ring_labels(self) -> None:
        """Propagate ring labels for deterministic ARAP detection.

        Labels propagated:
        - HEM: Bottom edges of torso panels or waistband
        - COLLAR: Collar opening edges
        - LEFT_CUFF/RIGHT_CUFF: Sleeve cuff output edges
        """
        # HEM labels (bottom edges)
        if InterfaceName.BOTTOM in self.interfaces:
            self.interfaces[InterfaceName.BOTTOM].edges.propagate_label(
                EdgeLabel.LOWER_INTERFACE.value
            )

        # CUFF labels
        for side in ["right", "left"]:
            bodice = getattr(self, side)
            label = (
                EdgeLabel.RIGHT_CUFF.value
                if side == "right"
                else EdgeLabel.LEFT_CUFF.value
            )

            if bodice.sleeve:
                if (
                    bodice.sleeve.cuff
                    and InterfaceName.BOTTOM in bodice.sleeve.cuff.interfaces
                ):
                    bodice.sleeve.cuff.interfaces[
                        InterfaceName.BOTTOM
                    ].edges.propagate_label(label)
                elif InterfaceName.OUT in bodice.sleeve.interfaces:
                    bodice.sleeve.interfaces[InterfaceName.OUT].edges.propagate_label(
                        label
                    )
            else:
                if hasattr(bodice.ftorso, "armhole_edge"):
                    bodice.ftorso.armhole_edge.label = label
                if hasattr(bodice.btorso, "armhole_edge"):
                    bodice.btorso.armhole_edge.label = label

    def get_garment_metadata(self) -> GarmentMetadata:
        """Get deterministic garment metadata for ARAP processing.

        Returns
        -------
        GarmentMetadata
            Metadata containing category, panel positions, ring connectors,
            and seam paths computed at garment creation time.
        """
        # Panel positions
        # Panel positions
        panel_positions: dict[str, PanelPosition] = {
            "right_ftorso": PanelPosition.FRONT,
            "right_btorso": PanelPosition.BACK,
            "left_ftorso": PanelPosition.FRONT,
            "left_btorso": PanelPosition.BACK,
        }

        # Add Sleeve panels
        for side in ["right", "left"]:
            half: BodiceHalf = getattr(self, side)
            if half.sleeve:
                panel_positions[f"{side}_sleeve_f"] = PanelPosition.FRONT
                panel_positions[f"{side}_sleeve_b"] = PanelPosition.BACK

                # Sleeve Cuffs
                if half.sleeve.cuff:
                    panel_positions[f"{side}_cuff_f"] = PanelPosition.FRONT
                    panel_positions[f"{side}_cuff_b"] = PanelPosition.BACK

        # Add Waistband
        if self.shirt_design.waistband_design.width > 0:
            panel_positions["wb_front"] = PanelPosition.FRONT
            panel_positions["wb_back"] = PanelPosition.BACK

        # Seam paths are computed during BoxMesh generation
        # For now, return empty - will be populated when we have vertex indices
        return GarmentMetadata(
            category=GarmentCategory.SHIRT,
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
            SHIRT category.
        """
        return GarmentCategory.SHIRT

    def get_rings(self) -> list[GarmentRing]:
        """Return all rings (closed boundary curves) for this shirt.

        Shirt rings: COLLAR, LEFT_CUFF, RIGHT_CUFF, HEM.
        For sleeveless shirts, armholes become cuffs.
        Dart edges are NOT included in rings.

        Returns
        -------
        list[GarmentRing]
            All rings with their edges and connectors.
        """
        rings: list[GarmentRing] = []

        # HEM ring - from BOTTOM interface
        if InterfaceName.BOTTOM in self.interfaces:
            hem_edges = self.interfaces[InterfaceName.BOTTOM].edges
            if len(hem_edges) > 0:
                # Connectors at ends (front panel boundaries)
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

        # COLLAR ring - from collar interfaces if available
        # Collar edges come from front/back collar interfaces
        collar_edges = self._collect_collar_edges()
        if collar_edges and len(collar_edges) > 0:
            conn1 = RingConnector(
                edge=collar_edges[0],
                parameter=0.0,
                ring_type=GarmentRingType.COLLAR,
            )
            conn2 = RingConnector(
                edge=collar_edges[-1],
                parameter=1.0,
                ring_type=GarmentRingType.COLLAR,
            )
            rings.append(
                GarmentRing(
                    ring_type=GarmentRingType.COLLAR,
                    edges=collar_edges,
                    connectors=(conn1, conn2),
                )
            )

        # CUFF rings (or armholes for sleeveless)
        for side, cuff_type in [
            ("right", GarmentRingType.RIGHT_CUFF),
            ("left", GarmentRingType.LEFT_CUFF),
        ]:
            half: BodiceHalf = getattr(self, side)
            cuff_edges = self._collect_cuff_edges(half, side)
            if cuff_edges and len(cuff_edges) > 0:
                conn1 = RingConnector(
                    edge=cuff_edges[0],
                    parameter=0.0,
                    ring_type=cuff_type,
                )
                conn2 = RingConnector(
                    edge=cuff_edges[-1],
                    parameter=1.0,
                    ring_type=cuff_type,
                )
                rings.append(
                    GarmentRing(
                        ring_type=cuff_type,
                        edges=cuff_edges,
                        connectors=(conn1, conn2),
                    )
                )

        return rings

    def _collect_collar_edges(self) -> pyg.EdgeSequence | None:
        """Collect collar ring edges from torso panels.

        Uses fuzzy matching to find interfaces with 'collar' or 'neck' in name,
        checking both component-level and sub-panel interfaces.
        """
        collar_edges = []

        for side in ["right", "left"]:
            half: BodiceHalf = getattr(self, side)

            # Helper to find collar edges by checking EDGE LABELS
            # (More robust than interface names which might vary)
            def find_collar_in_component(comp):
                matches = []
                if not hasattr(comp, "interfaces"):
                    return matches

                for interface in comp.interfaces.values():
                    if not interface.edges:
                        continue

                    # Check label of first edge
                    # BoxMesh assigns labels based on edge.label
                    # Use getattr/str to be safe if label is Enum or None
                    lbl = getattr(interface.edges[0], "label", "")
                    lbl_str = str(lbl.value if hasattr(lbl, "value") else lbl).lower()

                    # Keywords: collar, neck. Exclude: corner.
                    if (
                        "collar" in lbl_str or "neck" in lbl_str
                    ) and "corner" not in lbl_str:
                        matches.extend(interface.edges)
                return matches

            # Check Component Level
            found = find_collar_in_component(half)
            if found:
                collar_edges.extend(found)
            else:
                # Check Sub-Panels (ftorso, btorso)
                if getattr(half, "ftorso", None):
                    collar_edges.extend(find_collar_in_component(half.ftorso))
                if getattr(half, "btorso", None):
                    collar_edges.extend(find_collar_in_component(half.btorso))

        if collar_edges:
            return pyg.EdgeSequence(*collar_edges)

        return None

    def _collect_cuff_edges(
        self, half: BodiceHalf, side: str
    ) -> pyg.EdgeSequence | None:
        """Collect cuff ring edges from sleeve or armhole."""
        if half.sleeve and InterfaceName.OUT in half.sleeve.interfaces:
            # Has sleeve - use sleeve output
            return half.sleeve.interfaces[InterfaceName.OUT].edges
        else:
            # Sleeveless - use armhole edges
            armhole_edges = []
            if hasattr(half.ftorso, "armhole_edge"):
                armhole_edges.append(half.ftorso.armhole_edge)
            if hasattr(half.btorso, "armhole_edge"):
                armhole_edges.append(half.btorso.armhole_edge)
            if armhole_edges:
                return pyg.EdgeSequence(*armhole_edges)
        return None

    def _collect_side_seam_edges(self, side: str) -> pyg.EdgeSequence | None:
        """Collect edges for the side seam (HEM -> CUFF/ARMHOLE).

        Path: Waistband Side (opt) -> Torso Side -> Sleeve Inseam (opt) -> Cuff Side (opt)
        Uses Front panels for definitions.
        """
        edges: list[pyg.Edge] = []
        from loguru import logger

        logger.debug(f"Calling latest side seam edges for {side}")

        def _ensure_ascending_y(interface):
            """Ensure edges flow Upwards (Ascending Y) and are sorted."""
            if not interface.edges:
                return interface

            def clone_edge(e):
                return copy.deepcopy(e)

            def reverse_edge(e):
                ne = copy.deepcopy(e)
                ne.reverse()
                return ne

            # 1. Enforce Upward direction for each edge
            corrected_edges = []
            for e in interface.edges:
                # If flowing down (Start Y > End Y), reverse
                if e.start[1] > e.end[1]:
                    corrected_edges.append(reverse_edge(e))
                else:
                    corrected_edges.append(clone_edge(e))

            # 2. Sort by Y-coordinate (Start Y) to form chain Bottom -> Top
            corrected_edges.sort(key=lambda x: x.start[1])

            return pyg.EdgeSequence(*corrected_edges)

        # 1. Waistband side (if present)
        if self.shirt_design.waistband_design.width > 0:
            wb = next((s for s in self.subs if isinstance(s, StraightWB)), None)
            if wb:
                if side == "right":
                    if InterfaceName.RIGHT in wb.front.interfaces:
                        wb_int = _ensure_ascending_y(
                            wb.front.interfaces[InterfaceName.RIGHT]
                        )
                        for e in wb_int.edges:
                            e.panel = wb.front
                        edges.extend(wb_int.edges)
                else:  # left
                    if InterfaceName.LEFT in wb.front.interfaces:
                        wb_int = _ensure_ascending_y(
                            wb.front.interfaces[InterfaceName.LEFT]
                        )
                        for e in wb_int.edges:
                            e.panel = wb.front
                        edges.extend(wb_int.edges)

        # 2. Torso Side (OUTSIDE)
        half: BodiceHalf = getattr(self, side)
        if InterfaceName.OUTSIDE in half.ftorso.interfaces:
            # Ensure Hem->Armhole (Upwards)
            torso_int = _ensure_ascending_y(
                half.ftorso.interfaces[InterfaceName.OUTSIDE]
            )
            for e in torso_int.edges:
                e.panel = half.ftorso
            edges.extend(torso_int.edges)

        def _ensure_sleeve_outward(interface, side):
            """Ensure edges flow AWAY from body (Body -> Wrist) and are sorted."""
            if not interface.edges:
                return interface

            def clone_edge(e):
                return copy.deepcopy(e)

            def reverse_edge(e):
                ne = copy.deepcopy(e)
                ne.reverse()
                return ne

            is_right = side == "right"
            corrected_edges = []

            # 1. Enforce Outward direction for each edge
            for e in interface.edges:
                dx = e.end[0] - e.start[0]
                should_reverse = False
                if is_right:
                    # Right side: Outward is Negative X. Expect dx < 0.
                    # If dx > 0 (Positive), it's Inward. Reverse.
                    if dx > 0:
                        should_reverse = True
                else:
                    # Left side: Outward is Positive X. Expect dx > 0.
                    # If dx < 0 (Negative), it's Inward. Reverse.
                    if dx < 0:
                        should_reverse = True

                if should_reverse:
                    corrected_edges.append(reverse_edge(e))
                else:
                    corrected_edges.append(clone_edge(e))

            # 2. Sort by distance from center (approx Body -> Wrist)
            if is_right:
                # (-20, -40, -60). Sorted by Descending X (Largest/Least Negative first).
                corrected_edges.sort(key=lambda x: x.start[0], reverse=True)
            else:
                # (20, 40, 60). Sorted by Ascending X.
                corrected_edges.sort(key=lambda x: x.start[0])

            return pyg.EdgeSequence(*corrected_edges)

        # 3. Sleeve parts (if present)
        if half.sleeve:
            # 4. Sleeve Inseam (BOTTOM)
            # Ensure Body -> Wrist
            if (
                half.sleeve.f_sleeve
                and InterfaceName.BOTTOM in half.sleeve.f_sleeve.interfaces
            ):
                sleeve_int = _ensure_sleeve_outward(
                    half.sleeve.f_sleeve.interfaces[InterfaceName.BOTTOM], side
                )
                for e in sleeve_int.edges:
                    e.panel = half.sleeve.f_sleeve
                edges.extend(sleeve_int.edges)

            # 5. Cuff Side (if present)
            # Cuff TOP is Sleeve->Wrist (Inseam). Keep order.
            if half.sleeve.cuff:
                cuff_int_name = InterfaceName.TOP  # Inseam (Sleeve -> Wrist)
                if cuff_int_name in half.sleeve.cuff.front.interfaces:
                    cuff_edges = list(
                        half.sleeve.cuff.front.interfaces[cuff_int_name].edges
                    )
                    for e in cuff_edges:
                        e.panel = half.sleeve.cuff.front
                    edges.extend(cuff_edges)

        if edges:
            return pyg.EdgeSequence(*edges)
        return None

    def _collect_shoulder_seam_edges(self, side: str) -> pyg.EdgeSequence | None:
        """Collect edges for the shoulder/top seam (CUFF/ARMHOLE -> COLLAR).

        Path: Cuff Side (opt) -> Sleeve Outseam (opt) -> Torso Shoulder
        Uses Front panels.
        """
        edges: list[pyg.Edge] = []
        half: BodiceHalf = getattr(self, side)

        def clone_edge(e):
            return copy.deepcopy(e)

        def reverse_edge(e):
            ne = copy.deepcopy(e)
            ne.reverse()
            return ne

        def _ensure_inward_x(interface, side):
            """Ensure edges flow INWARD to body (Wrist/Shoulder -> Neck)."""
            if not interface.edges:
                return interface

            is_right = side == "right"
            corrected_edges = []

            # Flow Logic:
            # Right (Neg X): Target is 0. Start is <0. Flow: Increase X. dx > 0.
            # Left (Pos X): Target is 0. Start is >0. Flow: Decrease X. dx < 0.

            for e in interface.edges:
                dx = e.end[0] - e.start[0]
                should_reverse = False
                if is_right:
                    # Right: Expect dx > 0 (Inward). If dx < 0 (Outward), reverse.
                    if dx < 0:
                        should_reverse = True
                else:
                    # Left: Expect dx < 0 (Inward). If dx > 0 (Outward), reverse.
                    if dx > 0:
                        should_reverse = True

                if should_reverse:
                    corrected_edges.append(reverse_edge(e))
                else:
                    corrected_edges.append(clone_edge(e))

            # Sort by distance from center (Outer -> Inner)
            # Right: Start is very negative. Target 0. Sort Ascending X.
            # Left: Start is very positive. Target 0. Sort Descending X.
            if is_right:
                corrected_edges.sort(key=lambda x: x.start[0])
            else:
                corrected_edges.sort(key=lambda x: x.start[0], reverse=True)

            return pyg.EdgeSequence(*corrected_edges)

        # 1. Cuff/Sleeve
        if half.sleeve:
            # 1a. Cuff Side (if present) - Outseam side
            if half.sleeve.cuff:
                cuff_outseam_int = InterfaceName.BOTTOM  # Outseam (Wrist -> Sleeve)
                if cuff_outseam_int in half.sleeve.cuff.front.interfaces:
                    # Use inward logic (Wrist -> Sleeve Body)
                    cuff_int = _ensure_inward_x(
                        half.sleeve.cuff.front.interfaces[cuff_outseam_int], side
                    )
                    for e in cuff_int.edges:
                        e.panel = half.sleeve.cuff.front
                    edges.extend(cuff_int.edges)

            # 1b. Sleeve Outseam (TOP)
            # Shoulder/Body <-> Cuff/Wrist. Ensure Wrist -> Body (Inward).
            if (
                half.sleeve.f_sleeve
                and InterfaceName.TOP in half.sleeve.f_sleeve.interfaces
            ):
                sleeve_int = _ensure_inward_x(
                    half.sleeve.f_sleeve.interfaces[InterfaceName.TOP], side
                )
                for e in sleeve_int.edges:
                    e.panel = half.sleeve.f_sleeve
                edges.extend(sleeve_int.edges)

        # 2. Torso Shoulder
        # Shoulder -> Neck. Ensure Inward.
        if InterfaceName.SHOULDER in half.ftorso.interfaces:
            shoulder_int = _ensure_inward_x(
                half.ftorso.interfaces[InterfaceName.SHOULDER], side
            )
            for e in shoulder_int.edges:
                e.panel = half.ftorso
            edges.extend(shoulder_int.edges)

        if edges:
            return pyg.EdgeSequence(*edges)
        return None

    def get_ring_connections(self) -> list[RingConnection]:
        """Return all seam paths between rings.

        Shirt connections: HEM→COLLAR (front/back seams), HEM→CUFFS (side seams)

        Returns
        -------
        list[RingConnection]
            All ring connections with path edges.
        """
        connections: list[RingConnection] = []

        # 1. Right Side Seam (HEM -> RIGHT_CUFF)
        right_side_path = self._collect_side_seam_edges("right")
        if right_side_path:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.HEM,
                    ring_2=GarmentRingType.RIGHT_CUFF,
                    connector_1=RingConnector(
                        right_side_path[0], 0.0, GarmentRingType.HEM
                    ),
                    connector_2=RingConnector(
                        right_side_path[-1], 1.0, GarmentRingType.RIGHT_CUFF
                    ),
                    path_edges=right_side_path,
                )
            )

        # 2. Left Side Seam (HEM -> LEFT_CUFF)
        left_side_path = self._collect_side_seam_edges("left")
        if left_side_path:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.HEM,
                    ring_2=GarmentRingType.LEFT_CUFF,
                    connector_1=RingConnector(
                        left_side_path[0], 0.0, GarmentRingType.HEM
                    ),
                    connector_2=RingConnector(
                        left_side_path[-1], 1.0, GarmentRingType.LEFT_CUFF
                    ),
                    path_edges=left_side_path,
                )
            )

        # 3. Right Top Seam (RIGHT_CUFF -> COLLAR)
        right_top_path = self._collect_shoulder_seam_edges("right")
        if right_top_path:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.RIGHT_CUFF,
                    ring_2=GarmentRingType.COLLAR,
                    connector_1=RingConnector(
                        right_top_path[0], 0.0, GarmentRingType.RIGHT_CUFF
                    ),
                    connector_2=RingConnector(
                        right_top_path[-1], 1.0, GarmentRingType.COLLAR
                    ),
                    path_edges=right_top_path,
                )
            )

        # 4. Left Top Seam (LEFT_CUFF -> COLLAR)
        left_top_path = self._collect_shoulder_seam_edges("left")
        if left_top_path:
            connections.append(
                RingConnection(
                    ring_1=GarmentRingType.LEFT_CUFF,
                    ring_2=GarmentRingType.COLLAR,
                    connector_1=RingConnector(
                        left_top_path[0], 0.0, GarmentRingType.LEFT_CUFF
                    ),
                    connector_2=RingConnector(
                        left_top_path[-1], 1.0, GarmentRingType.COLLAR
                    ),
                    path_edges=left_top_path,
                )
            )

        return connections


class FittedShirt(Shirt):
    """Creates fitted shirt.

    NOTE: Separate class is used for selection convenience.
    Even though most of the processing is the same (hence implemented with
    the same components except for panels), design parametrization differs
    significantly. With that, we decided to separate the top level names.
    """

    def __init__(
        self,
        shirt_design: ShirtDesign,
        arm_pose_angle: float,
    ) -> None:
        """Initialize a fitted shirt.

        Parameters
        ----------
        body : BodyDefinition | dict
            Body measurements object or dictionary.
        design : dict
            Design parameters dictionary.
        """
        super().__init__(shirt_design, arm_pose_angle, fitted=True)
