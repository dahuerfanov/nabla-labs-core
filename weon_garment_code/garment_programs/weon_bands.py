import weon_garment_code.pygarment.garmentcode as pyg
from weon_garment_code.config import AttachmentConstraint
from weon_garment_code.garment_programs.base_classes import BaseBand
from weon_garment_code.garment_programs.garment_enums import InterfaceName
from weon_garment_code.garment_programs.garment_program_utils import AttachmentHandler
from weon_garment_code.pattern_definitions.body_definition import BodyDefinition
from weon_garment_code.pattern_definitions.pants_design import CuffDesign
from weon_garment_code.pattern_definitions.waistband_design import WaistbandDesign


class StraightBandPanel(pyg.Panel):
    """One panel for a panel skirt."""

    def __init__(
        self,
        name: str,
        width: float,
        depth: float,
        match_int_proportion: float | None = None,
    ) -> None:
        super().__init__(name)

        # define edge loop
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], [0, depth], [width, depth], [width, 0], loop=True
        )

        # define interface
        self.interfaces = {
            InterfaceName.RIGHT: pyg.Interface(self, self.edges[0]),
            InterfaceName.TOP: pyg.Interface(
                self,
                self.edges[1],
                ruffle=width / match_int_proportion
                if match_int_proportion is not None
                else 1.0,
            ).reverse(True),
            InterfaceName.LEFT: pyg.Interface(self, self.edges[2]),
            InterfaceName.BOTTOM: pyg.Interface(
                self,
                self.edges[3],
                ruffle=width / match_int_proportion
                if match_int_proportion is not None
                else 1.0,
            ),
        }

        # Default translation
        self.top_center_pivot()
        self.center_x()


class StraightWB(BaseBand):
    """Simple 2 panel waistband."""

    # Panel placement constants
    FRONT_TRANSLATION_Z: int = 20
    BACK_TRANSLATION_Z: int = -15

    # Class attributes
    waistband_design: WaistbandDesign
    waist: float
    waist_back: float
    waist_front: float
    width: float
    rise: float
    front: StraightBandPanel
    back: StraightBandPanel

    def __init__(
        self,
        waistband_design: WaistbandDesign,
        waist: float,
        waist_back: float,
        waist_front: float,
        rise: float = 1.0,
    ) -> None:
        """Initialize a simple 2 panel waistband.

        Parameters:
        -----------
        waistband_design : WaistbandDesign
            Waistband design parameters object.
        waist : float
            Total waist measurement (full circumference).
        waist_back : float
            Back waist width measurement.
        waist_front : float
            Front waist width measurement (waist - waist_back).
        rise : float, optional
            The rise value of the bottoms that the WB is attached to.
            Adapts the shape of the waistband to sit tight on top of the given
            rise level (top measurement). If 1.0 or anything less than waistband width,
            the rise is ignored and the StraightWB is created to sit well on the waist.
            Default is 1.0.
        """
        super().__init__(body=None, design=None, rise=rise)

        # Store design and measurements
        self.waistband_design = waistband_design
        self.waist = waist
        self.waist_back = waist_back
        self.waist_front = waist_front
        self.width = self.waistband_design.width
        self.rise = rise

        self.define_panels()

        # Panels positioned at Y=0, will be moved in apply_body_alignment
        self.front.translate_by([0, 0, self.FRONT_TRANSLATION_Z])
        self.back.translate_by([0, 0, self.BACK_TRANSLATION_Z])

        self.stitching_rules = pyg.Stitches(
            (
                self.front.interfaces[InterfaceName.RIGHT],
                self.back.interfaces[InterfaceName.RIGHT].clone().flip_edges(),
            ),
            (
                self.front.interfaces[InterfaceName.LEFT],
                self.back.interfaces[InterfaceName.LEFT].clone().flip_edges(),
            ),
        )

        self.interfaces = {
            InterfaceName.BOTTOM_F: self.front.interfaces[InterfaceName.BOTTOM],
            InterfaceName.BOTTOM_B: self.back.interfaces[InterfaceName.BOTTOM],
            InterfaceName.TOP_F: self.front.interfaces[InterfaceName.TOP],
            InterfaceName.TOP_B: self.back.interfaces[InterfaceName.TOP],
            InterfaceName.BOTTOM: pyg.Interface.from_multiple(
                self.front.interfaces[InterfaceName.BOTTOM],
                self.back.interfaces[InterfaceName.BOTTOM],
            ),
            InterfaceName.TOP: pyg.Interface.from_multiple(
                self.front.interfaces[InterfaceName.TOP],
                self.back.interfaces[InterfaceName.TOP],
            ),
        }

    def define_panels(self) -> None:
        """Define the front and back panels for the waistband.

        Both panels use the same horizontal width (max of front and back) to ensure
        they appear the same size. The difference in waist measurements is handled
        through ruffles on the interfaces.
        """
        # Use the larger width so both panels are the same size
        panel_width = max(self.waist_front, self.waist_back)

        self.front = StraightBandPanel(
            "wb_front", panel_width, self.width, match_int_proportion=self.waist_front
        )

        self.back = StraightBandPanel(
            "wb_back", panel_width, self.width, match_int_proportion=self.waist_back
        )

    @staticmethod
    def get_attachment_constraints() -> list[AttachmentConstraint]:
        """Get the list of attachment constraints for this waistband garment.

        Returns the attachment constraints required for simulating this waistband garment.
        Waistbands do not require any attachment constraints by default.

        Returns:
        --------
        List[AttachmentConstraint]
            An empty list (waistbands do not require attachment constraints).
        """
        return []

    def get_vertex_processor_callback(self):
        """Get the vertex processor callback for waistband attachment constraints.

        Waistbands do not require custom vertex processing, so this returns a default
        callback that uses generic constraint processing.

        Returns:
        --------
        Callable[[BoxMesh], None] | None
            A callback that handles constraints with default behavior.
        """
        return AttachmentHandler.create_default_vertex_processor(BaseBand)

    def apply_body_alignment(self, body: BodyDefinition) -> None:
        """Apply body alignment to position the waistband correctly.

        Translates the waistband garment vertically based on the body's waist
        level to ensure proper fit and alignment.

        Parameters
        ----------
        body : BodyDefinition
            Body measurements object containing body parameters.
            Required parameters:
            - computed_waist_level: Vertical position of the waist level (computed property)
        """
        waist_level = body.computed_waist_level

        # Position front and back panels at waist level
        self.front.translate_by([0, waist_level, 0])
        self.back.translate_by([0, waist_level, 0])


class CuffBand(BaseBand):
    """Cuff class for sleeves or pants - band-like piece of fabric with optional skirt."""

    # Panel placement constants (same as StraightWB)
    FRONT_TRANSLATION_Z: int = 20
    BACK_TRANSLATION_Z: int = -15

    design: CuffDesign
    front: StraightBandPanel
    back: StraightBandPanel
    vertical: bool

    def __init__(
        self,
        tag: str,
        design: CuffDesign,
        vertical: bool = False,
    ) -> None:
        super().__init__(body=None, design=design, tag=tag)

        self.design = design
        self.vertical = vertical

        # Convention: width = cuff height, length = total circumference
        # Each panel gets half the circumference, full height
        cuff_height = self.design.cuff_width
        half_circumference = self.design.cuff_length / 2

        if self.vertical:
            # Vertical orientation (for sleeves axial alignment)
            # Width = height (axial), Depth = circumference (transverse)
            # Panel is TALL (half_circumference) × NARROW (cuff_height)
            self.front = StraightBandPanel(
                f"{tag}_cuff_f", cuff_height, half_circumference
            )
            self.back = StraightBandPanel(
                f"{tag}_cuff_b", cuff_height, half_circumference
            )

            # Separate panels in Z like waistband does
            self.front.translate_by([0, 0, self.FRONT_TRANSLATION_Z])
            self.back.translate_by([0, 0, self.BACK_TRANSLATION_Z])

            # Stitch front-back via TOP/BOTTOM (short edges = cuff height)
            # Since panel is vertical, TOP/BOTTOM are the short horizontal edges
            self.stitching_rules = pyg.Stitches(
                (
                    self.front.interfaces[InterfaceName.TOP],
                    self.back.interfaces[InterfaceName.TOP],
                ),
                (
                    self.front.interfaces[InterfaceName.BOTTOM],
                    self.back.interfaces[InterfaceName.BOTTOM],
                ),
            )

            # Interfaces mapping for Vertical orientation:
            # - TOP (connects into body/sleeve): Uses the vertical edge (RIGHT - edge 0)
            # - BOTTOM (hem): Uses the other vertical edge (LEFT - edge 2)
            self.interfaces = {
                InterfaceName.BOTTOM: pyg.Interface.from_multiple(
                    self.front.interfaces[InterfaceName.LEFT],
                    self.back.interfaces[InterfaceName.LEFT],
                ),
                InterfaceName.TOP_FRONT: self.front.interfaces[InterfaceName.RIGHT],
                InterfaceName.TOP_BACK: self.back.interfaces[InterfaceName.RIGHT],
                InterfaceName.TOP: pyg.Interface.from_multiple(
                    self.front.interfaces[InterfaceName.RIGHT],
                    self.back.interfaces[InterfaceName.RIGHT],
                ),
            }

        else:
            # Horizontal orientation (standard, e.g. for pants)
            # StraightBandPanel(name, width=horizontal, depth=vertical)
            # Panel is WIDE (half_circumference) × SHORT (cuff_height)
            # TOP interface = horizontal edge (half circumference, stitches to sleeve)
            self.front = StraightBandPanel(
                f"{tag}_cuff_f", half_circumference, cuff_height
            )
            self.back = StraightBandPanel(
                f"{tag}_cuff_b", half_circumference, cuff_height
            )

            # Separate panels in Z like waistband does
            self.front.translate_by([0, 0, self.FRONT_TRANSLATION_Z])
            self.back.translate_by([0, 0, self.BACK_TRANSLATION_Z])

            # Stitch front-back via LEFT/RIGHT (short edges = cuff height)
            self.stitching_rules = pyg.Stitches(
                (
                    self.front.interfaces[InterfaceName.RIGHT],
                    self.back.interfaces[InterfaceName.RIGHT],
                ),
                (
                    self.front.interfaces[InterfaceName.LEFT],
                    self.back.interfaces[InterfaceName.LEFT],
                ),
            )

            self.interfaces = {
                InterfaceName.BOTTOM: pyg.Interface.from_multiple(
                    self.front.interfaces[InterfaceName.BOTTOM],
                    self.back.interfaces[InterfaceName.BOTTOM],
                ),
                InterfaceName.TOP_FRONT: self.front.interfaces[InterfaceName.TOP],
                InterfaceName.TOP_BACK: self.back.interfaces[InterfaceName.TOP],
                InterfaceName.TOP: pyg.Interface.from_multiple(
                    self.front.interfaces[InterfaceName.TOP],
                    self.back.interfaces[InterfaceName.TOP],
                ),
            }
