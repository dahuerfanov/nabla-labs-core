"""Panels for a straight upper garment (T-shirt).

Note that the code is very similar to Bodice.
"""

import numpy as np
from loguru import logger

import weon_garment_code.pygarment.garmentcode as pyg
from weon_garment_code.garment_programs.base_classes import BaseBodicePanel
from weon_garment_code.garment_programs.garment_enums import InterfaceName
from weon_garment_code.pattern_definitions.shirt_design import ShirtDesign


class TorsoFrontHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)

    Fits to the bust size
    """

    # Class attributes
    width: float

    def __init__(
        self,
        name: str,
        shirt_design: ShirtDesign,
    ) -> None:
        """Create a front half panel for a simple upper garment (e.g. T-Shirt).

        Parameters
        ----------
        name : str
            Name of the panel.
        Shirt_design : ShirtDesign
            Shirt design parameters.
        """
        super().__init__(name, shirt_design.torso_design)

        # sizes
        waist = self.torso_design.width_waist / 2
        bust = self.torso_design.width_chest / 2
        self.width = self.torso_design.width_chest / 2
        max_len = self.torso_design.front_length

        outside_interface_edges = []
        bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            start=[0, -self.torso_design.shirttail_offset],
            end=[-self.torso_design.width_hip / 2, 0],
            target_tan0=np.array([-1, 0]),
        )
        self.edges.append(bottom)
        self._bottom_edge_init = bottom
        last_edge_end = bottom.end

        if (
            self.torso_design.scye_depth + 1
            > self.torso_design.waist_over_bust_line_height
        ):
            logger.debug("Scye depth is greater than waist over bust line height.")
        else:
            right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                start=bottom.end,
                end=[-waist, max_len - self.torso_design.waist_over_bust_line_height],
                target_tan1=np.array([0, 1]),
            )
            self.edges.append(right_bottom)
            outside_interface_edges.append(right_bottom)
            last_edge_end = right_bottom.end

        right_middle = pyg.CurveEdgeFactory.curve_from_tangents(
            start=last_edge_end,
            end=[-bust, max_len - self.torso_design.scye_depth],
            target_tan0=np.array([0, 1]),
        )
        self.edges.append(right_middle)
        last_edge_end = right_middle.end
        outside_interface_edges.append(right_middle)

        self.edges.append(
            pyg.EdgeSeqFactory.from_verts(
                last_edge_end,
                [-bust, max_len - self.torso_design.shoulder_slant],
                [0, max_len],
            )
        )
        self.armhole_edge = self.edges[-2]
        self._outside_edges_init = outside_interface_edges
        self.edges.close_loop()

        self.define_interfaces()

        # default placement
        self.translate_by([0, -max_len, 0])

    def define_interfaces(self) -> None:
        """Define/update interfaces for the torso panel."""
        self.interfaces = {
            InterfaceName.OUTSIDE: pyg.Interface(
                self, pyg.EdgeSequence(*self._outside_edges_init)
            ),
            InterfaceName.INSIDE: pyg.Interface(self, self.edges[-1]),
            InterfaceName.SHOULDER: pyg.Interface(self, self.edges[-2]),
            InterfaceName.BOTTOM: pyg.Interface(self, self._bottom_edge_init),
            # Reference to the corner for sleeve and collar projections
            InterfaceName.SHOULDER_CORNER: pyg.Interface(
                self, [self.edges[-3], self.edges[-2]]
            ),
            InterfaceName.COLLAR_CORNER: pyg.Interface(
                self, [self.edges[-2], self.edges[-1]]
            ),
        }

    def get_width(self, level: float) -> float:
        """Get the width of the panel at a given level"""
        return (
            super().get_width(level)
            + self.width
            - self.neck_to_shoulder_delta_x
            - self.torso_design.neck_width / 2
        )


class TorsoBackHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)"""

    # Class attributes
    width: float

    def __init__(
        self, name: str, shirt_design: ShirtDesign, match_length_to_front: bool = True
    ) -> None:
        """Create a back half panel for a simple upper garment (e.g. T-Shirt).

        Parameters
        ----------
        name : str
            Name of the panel.
        shirt_design : ShirtDesign
            Shirt design parameters.
        match_length_to_front : bool, optional
            Whether to match the back length to the front length.
            Default is True.
        """
        super().__init__(name, shirt_design.torso_design)

        # sizes
        waist = self.torso_design.width_waist / 2
        bust = self.torso_design.width_chest / 2
        self.width = self.torso_design.width_chest / 2
        if not match_length_to_front:
            max_len = self.torso_design.back_length
            bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                start=[0, -self.torso_design.shirttail_offset],
                end=[-self.torso_design.width_hip / 2, 0],
                target_tan0=np.array([-1, 0]),
            )
        else:
            max_len = self.torso_design.front_length
            bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                start=[
                    0,
                    max_len
                    - self.torso_design.back_length
                    - self.torso_design.shirttail_offset,
                ],
                end=[-self.torso_design.width_hip / 2, 0],
                target_tan0=np.array([-1, 0]),
            )

        self.edges.append(bottom)
        self._bottom_edge_init = bottom
        last_edge_end = bottom.end

        outside_interface_edges = []
        if (
            self.torso_design.scye_depth + 1
            > self.torso_design.waist_over_bust_line_height
        ):
            logger.debug("Scye depth is greater than waist over bust line height.")
        else:
            right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                start=bottom.end,
                end=[-waist, max_len - self.torso_design.waist_over_bust_line_height],
                target_tan1=np.array([0, 1]),
            )
            self.edges.append(right_bottom)
            outside_interface_edges.append(right_bottom)
            last_edge_end = right_bottom.end

        right_middle = pyg.CurveEdgeFactory.curve_from_tangents(
            start=last_edge_end,
            end=[
                -bust * self.torso_design.back_width / self.torso_design.width_chest,
                max_len - self.torso_design.scye_depth,
            ],
            target_tan0=np.array([0, 1]),
        )
        self.edges.append(right_middle)
        outside_interface_edges.append(right_middle)
        last_edge_end = right_middle.end

        self.edges.append(
            pyg.EdgeSeqFactory.from_verts(
                last_edge_end,
                [-bust, max_len - self.torso_design.shoulder_slant],
                [0, max_len],
            )
        )
        self.armhole_edge = self.edges[-2]
        self._outside_edges_init = outside_interface_edges
        self.edges.close_loop()

        self.define_interfaces()

        # default placement
        self.translate_by([0, -max_len, 0])

    def define_interfaces(self) -> None:
        """Define/update interfaces for the torso panel."""
        self.interfaces = {
            InterfaceName.OUTSIDE: pyg.Interface(
                self, pyg.EdgeSequence(*self._outside_edges_init)
            ),
            InterfaceName.INSIDE: pyg.Interface(self, self.edges[-1]),
            InterfaceName.SHOULDER: pyg.Interface(self, self.edges[-2]),
            InterfaceName.BOTTOM: pyg.Interface(self, self._bottom_edge_init),
            # Reference to the corner for sleeve and collar projections
            InterfaceName.SHOULDER_CORNER: pyg.Interface(
                self, [self.edges[-3], self.edges[-2]]
            ),
            InterfaceName.COLLAR_CORNER: pyg.Interface(
                self, [self.edges[-2], self.edges[-1]]
            ),
        }

    def get_width(self, level: float) -> float:
        """Get the width of the panel at a given level"""
        return (
            super().get_width(level)
            + self.width
            - self.neck_to_shoulder_delta_x
            - self.torso_design.neck_width / 2
        )
