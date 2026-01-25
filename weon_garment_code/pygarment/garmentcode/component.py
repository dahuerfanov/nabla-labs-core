import numpy as np
from scipy.spatial.transform import Rotation as R

from weon_garment_code.garment_programs.garment_enums import (
    EdgeLabel,
    InterfaceName,
    PanelLabel,
)
from weon_garment_code.pygarment.garmentcode.base import BaseComponent
from weon_garment_code.pygarment.meshgen.arap.core_types import (
    GarmentCategory,
    GarmentMetadata,
    PanelPosition,
)
from weon_garment_code.pygarment.pattern.wrappers import VisPattern


class Component(BaseComponent):
    """Garment element (or whole piece) composed of simpler connected garment
    elements"""

    # TODOLOW Overload copy -- respecting edge sequences -- never had any problems though

    subs: list[BaseComponent]

    def __init__(self, name) -> None:
        super().__init__(name)

        self.subs: list[BaseComponent] = []  # list of generative subcomponents

    def get_garment_metadata(self) -> GarmentMetadata | None:
        """Get deterministic garment metadata for ARAP processing.

        Returns
        -------
        GarmentMetadata | None
            Metadata containing category, panel positions, ring connectors,
            and seam paths computed at garment creation time.
            Returns None by default, should be overridden by specific garment classes.
        """
        return None

    def set_panel_label(self, label: str, overwrite=True):
        """Propagate given label to all sub-panels (in subcomponents)"""
        subs = self._get_subcomponents()
        for sub in subs:
            sub.set_panel_label(label, overwrite)

    def pivot_3D(self):
        """Pivot of a component as a block

        NOTE: The relation of pivots of sub-blocks needs to be
        preserved in any placement operations on components
        """
        mins, maxes = self.bbox3D()
        return np.array(
            ((mins[0] + maxes[0]) / 2, maxes[1], (mins[-1] + maxes[-1]) / 2)
        )

    def length(self):
        """Length of a component in cm

        Defaults the to the vertical length of a 3D bounding box
        * longest_dim -- if set, returns the longest dimention out of the bounding box dimentions
        """
        subs = self._get_subcomponents()
        return sum([s.length() for s in subs]) if subs else 0

    def translate_by(self, delta_vector):
        """Translate component by a vector"""
        for subs in self._get_subcomponents():
            subs.translate_by(delta_vector)
        return self

    def translate_to(self, new_translation):
        """Set panel translation to be exactly that vector"""
        pivot = self.pivot_3D()
        for subs in self._get_subcomponents():
            sub_pivot = subs.pivot_3D()
            subs.translate_to(np.asarray(new_translation) + (sub_pivot - pivot))
        return self

    def rotate_by(self, delta_rotation: R):
        """Rotate component by a given rotation"""
        pivot = self.pivot_3D()
        for subs in self._get_subcomponents():
            # With preserving relationships between components
            rel = subs.pivot_3D() - pivot
            rel_rotated = delta_rotation.apply(rel)
            subs.rotate_by(delta_rotation)
            subs.translate_by(rel_rotated - rel)
        return self

    def rotate_to(self, new_rot):
        # TODOLOW Implement with correct preservation of relative placement
        # of subcomponents
        raise NotImplementedError(
            "Component::ERROR::rotate_to is not supported on component level."
            "Use relative <rotate_by()> method instead"
        )

    def mirror(self, axis=[0, 1]):
        """Swap this component with its mirror image by recursively mirroring
        subcomponents

            Axis specifies 2D axis to swap around: Y axis by default
        """
        for subs in self._get_subcomponents():
            subs.mirror(axis)
        return self

    def assembly(self):
        """Construction process of the garment component

        get serializable representation
        Returns: simulator friendly description of component sewing pattern
        """
        spattern = VisPattern()
        spattern.name = self.name

        subs = self._get_subcomponents()
        if not subs:
            return spattern

        # Simple merge of subcomponent representations
        for sub in subs:
            sub_raw = sub.assembly().pattern

            # simple merge of panels
            spattern.pattern["panels"] = {
                **spattern.pattern["panels"],
                **sub_raw["panels"],
            }

            # of stitches
            spattern.pattern["stitches"] += sub_raw["stitches"]

        spattern.pattern["stitches"] += self.stitching_rules.assembly()
        return spattern

    def bbox3D(self):
        """Evaluate 3D bounding box of the current component"""

        subs = self._get_subcomponents()
        bboxes = [s.bbox3D() for s in subs]

        if not len(subs):
            # Special components without panel geometry -- no bbox defined
            return np.array([[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]])

        mins = np.vstack([b[0] for b in bboxes])
        maxes = np.vstack([b[1] for b in bboxes])

        return mins.min(axis=0), maxes.max(axis=0)

    def is_self_intersecting(self):
        """Check whether the component have self-intersections on panel level"""

        for s in self._get_subcomponents():
            if s.is_self_intersecting():
                return True
        return False

    # Subcomponents
    def _get_subcomponents(self):
        """Unique set of subcomponents defined in the `self.subs` list or as
        attributes of the object"""

        all_attrs = [
            getattr(self, name)
            for name in dir(self)
            if name[:2] != "__" and name[-2:] != "__"
        ]
        return list(
            set(
                [att for att in all_attrs if isinstance(att, BaseComponent)] + self.subs
            )
        )

    def get_vertex_processor_callback(self):
        """
        Get the vertex processor callback for waistband attachment constraints.

        Waistbands do not require custom vertex processing, so this returns a default
        callback that uses generic constraint processing.
        """

        raise NotImplementedError(
            "Component::ERROR::get_vertex_processor_callback is not supported on component level."
        )


class CompositeGarment(Component):
    """A composite garment (e.g. dress, jumpsuit) assembled from multiple parts.

    Combines an upper part (bodice/shirt), optional waistband, and lower part
    (pants/skirt) into a single garment component with unified metadata.
    """

    def __init__(
        self,
        name: str,
        upper: Component | None = None,
        lower: Component | None = None,
        waistband: Component | None = None,
    ) -> None:
        """Initialize composite garment.

        Parameters
        ----------
        name : str
            Name of the garment.
        upper : Component | None
            Upper garment component (e.g. Shirt, FittedShirt).
        lower : Component | None
            Lower garment component (e.g. Pants, SkirtCircle).
        waistband : Component | None
            Waistband component connecting upper and lower.
        """
        super().__init__(name)

        self.upper = upper
        self.lower = lower
        self.waistband = waistband

        # Add subs in display order
        if upper:
            self.subs.append(upper)
            upper.set_panel_label(PanelLabel.BODY, overwrite=False)

        if waistband:
            self.subs.append(waistband)
            waistband.set_panel_label(PanelLabel.BODY, overwrite=False)
            waistband.interfaces[InterfaceName.TOP].edges.propagate_label(
                EdgeLabel.LOWER_INTERFACE
            )

        if lower:
            self.subs.append(lower)
            lower.set_panel_label(PanelLabel.LEG, overwrite=False)

        # Connect components
        self._stitch_components()

    def _stitch_components(self) -> None:
        """Stitch the upper, waistband, and lower components together."""
        # 1. Connect Upper to Waistband
        if self.upper and self.waistband:
            self.waistband.place_by_interface(
                self.waistband.interfaces[InterfaceName.TOP],
                self.upper.interfaces[InterfaceName.BOTTOM],
                gap=5,
            )
            self.stitching_rules.append(
                (
                    self.upper.interfaces[InterfaceName.BOTTOM],
                    self.waistband.interfaces[InterfaceName.TOP],
                )
            )

        # 2. Connect Waistband to Lower
        if self.waistband and self.lower:
            self.lower.place_by_interface(
                self.lower.interfaces[InterfaceName.TOP],
                self.waistband.interfaces[InterfaceName.BOTTOM],
                gap=5,
            )
            self.stitching_rules.append(
                (
                    self.waistband.interfaces[InterfaceName.BOTTOM],
                    self.lower.interfaces[InterfaceName.TOP],
                )
            )
        # 3. Connect Upper to Lower (if no waistband)
        elif self.upper and self.lower:
            self.lower.place_by_interface(
                self.lower.interfaces[InterfaceName.TOP],
                self.upper.interfaces[InterfaceName.BOTTOM],
                gap=5,
            )
            self.stitching_rules.append(
                (
                    self.upper.interfaces[InterfaceName.BOTTOM],
                    self.lower.interfaces[InterfaceName.TOP],
                )
            )
            # Propagate label if no waistband
            self.lower.interfaces[InterfaceName.TOP].edges.propagate_label(
                EdgeLabel.LOWER_INTERFACE
            )
        elif self.lower and not self.waistband:
            # Standalone lower in composite container (edge case)
            self.lower.interfaces[InterfaceName.TOP].edges.propagate_label(
                EdgeLabel.LOWER_INTERFACE
            )

    def get_garment_metadata(self) -> GarmentMetadata | None:
        """Combine metadata from sub-components."""
        panel_positions: dict[str, PanelPosition] = {}
        ring_connectors = {}
        seam_paths = []

        # Determine category based on lower component
        category = GarmentCategory.DRESS  # Default default
        if self.lower:
            meta = self.lower.get_garment_metadata()
            if meta:
                if meta.category == GarmentCategory.PANTS:
                    category = (
                        GarmentCategory.PANTS
                    )  # Treat jumpsuits as pants for ARAP
                elif meta.category == GarmentCategory.SKIRT:
                    category = GarmentCategory.DRESS

        # Combine data
        for comp in [self.upper, self.waistband, self.lower]:
            if comp:
                meta = comp.get_garment_metadata()
                if meta:
                    panel_positions.update(meta.panel_positions)
                    ring_connectors.update(meta.ring_connectors)
                    seam_paths.extend(meta.seam_paths)

        return GarmentMetadata(
            category=category,
            panel_positions=panel_positions,
            ring_connectors=ring_connectors,
            seam_paths=seam_paths,
        )

    def get_vertex_processor_callback(self):
        """Combine vertex processor callbacks from sub-components."""
        callbacks = []
        for comp in [self.upper, self.waistband, self.lower]:
            if comp:
                # Assuming concrete implementation exists or returns None
                try:
                    cb = comp.get_vertex_processor_callback()
                    if cb:
                        callbacks.append(cb)
                except NotImplementedError:
                    pass

        if not callbacks:
            return None

        def composite_callback(box_mesh):
            for cb in callbacks:
                cb(box_mesh)

        return composite_callback
