"""Garment mesh partitioner implementation."""

import numpy as np

from weon_garment_code.pygarment.meshgen.topology.base import MeshPartitioner
from weon_garment_code.pygarment.meshgen.topology.ring_types import (
    GarmentCategory,
    GarmentRingType,
)


class GarmentPartitioner(MeshPartitioner):
    """
    Partitioner for garment meshes (shirts, pants).

    Uses stitch topology to identify front/back seam points.
    """

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        category: GarmentCategory,
        ring_definitions: dict[GarmentRingType, list[int]],
        vertex_panels: list[set[str]],
    ):
        """
        Initialize the garment partitioner.

        Args:
            vertices: (N, 3) vertex positions
            faces: (M, 3) face indices
            category: Garment category (SHIRT or PANTS)
            ring_definitions: Pre-computed ring definitions
            vertex_panels: Per-vertex panel membership
        """
        super().__init__(vertices, faces)
        self.category = category
        self._ring_definitions = ring_definitions
        self.vertex_panels = vertex_panels

    def get_ring_definitions(self) -> dict[GarmentRingType, list[int]]:
        """Get pre-computed ring definitions."""
        return self._ring_definitions

    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        """Get ring connection sequences based on garment category."""
        if self.category == GarmentCategory.SHIRT:
            return [
                [
                    GarmentRingType.HEM,
                    GarmentRingType.LEFT_CUFF,
                    GarmentRingType.COLLAR,
                ],
                [
                    GarmentRingType.HEM,
                    GarmentRingType.RIGHT_CUFF,
                    GarmentRingType.COLLAR,
                ],
            ]
        elif self.category == GarmentCategory.PANTS:
            return [
                [GarmentRingType.HEM, GarmentRingType.LEFT_ANKLE],
                [GarmentRingType.HEM, GarmentRingType.RIGHT_ANKLE],
                [GarmentRingType.LEFT_ANKLE, GarmentRingType.RIGHT_ANKLE],
            ]
        return []

    def get_ring_connectors(self, ring: list[int]) -> list[int]:
        """
        Identify stitch vertices connecting FRONT and BACK panels.

        Uses vertex panel membership to find transition points.
        """
        # Find vertices belonging to multiple panels with front+back tags
        candidates_indices = []
        for i, v in enumerate(ring):
            if v < len(self.vertex_panels):
                panels = self.vertex_panels[v]
                if len(panels) > 1:
                    tags = set()
                    for pname in panels:
                        tags.update(self._analyze_panel_name(pname))
                    if "front" in tags and "back" in tags:
                        candidates_indices.append(i)

        if not candidates_indices:
            return []

        # Group adjacent candidates
        groups: list[list[int]] = []
        current_group = [candidates_indices[0]]
        for x in candidates_indices[1:]:
            if x == current_group[-1] + 1:
                current_group.append(x)
            else:
                groups.append(current_group)
                current_group = [x]
        groups.append(current_group)

        # Handle wrap-around
        if len(groups) > 1 and groups[0][0] == 0 and groups[-1][-1] == len(ring) - 1:
            merged = groups[-1] + groups[0]
            groups[0] = merged
            groups.pop()

        # Extract middle vertex of each group
        connectors = []
        for g in groups:
            mid_idx = len(g) // 2
            idx_in_ring = g[mid_idx]
            connectors.append(ring[idx_in_ring])

        return connectors

    def _analyze_panel_name(self, name: str) -> set[str]:
        """Extract panel tags (front/back/left/right) from name."""
        tags = set()
        n = name.lower()
        if "front" in n or "_f" in n:
            tags.add("front")
        if "back" in n or "_b" in n:
            tags.add("back")
        if "left" in n or "_l" in n:
            tags.add("left")
        if "right" in n or "_r" in n:
            tags.add("right")
        return tags
