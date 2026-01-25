"""ARAPInitializable protocol for garments.

This module defines the protocol that garments must implement to be
compatible with deterministic ARAP initialization.

Usage:
    from weon_garment_code.pygarment.meshgen.arap.arap_protocol import (
        ARAPInitializable,
    )

    class Shirt(BaseGarment, ARAPInitializable):
        def get_rings(self) -> list[GarmentRing]:
            ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from weon_garment_code.pygarment.meshgen.arap.core_types import GarmentCategory

if TYPE_CHECKING:
    from weon_garment_code.pygarment.meshgen.arap.arap_types import (
        GarmentRing,
        RingConnection,
    )


@runtime_checkable
class ARAPInitializable(Protocol):
    """Protocol for garments that provide deterministic ARAP initialization data.

    Garments implementing this protocol provide:
    - Rings: Closed boundary curves (collar, cuffs, hem, ankles)
    - Ring connections: Seam paths between rings

    The data is computed at garment creation time and translated to mesh-level
    during box mesh generation. This eliminates runtime heuristics and ensures
    robust, deterministic ARAP initialization.

    Implementation notes:
    - Rings should NOT include dart edges (internal stitches)
    - Ring connectors should be at front panel boundary points
    - Ring connections should follow panel boundary edges
    - Physical gaps between panels in world-space are handled via metadata:
        providing 'panel_ids' along paths allows the solver to zero-out physical "jumps"
        during distance calculation, maintaining geodesic continuity without
        requiring physical pattern modification.
    """

    def get_rings(self) -> list[GarmentRing]:
        """Return all rings (closed boundary curves) for this garment.

        For shirts: COLLAR, LEFT_CUFF, RIGHT_CUFF, HEM
        For pants: HEM, LEFT_ANKLE, RIGHT_ANKLE

        Returns
        -------
        list[GarmentRing]
            All rings with their edges and connectors
        """
        ...

    def get_ring_connections(self) -> list[RingConnection]:
        """Return all seam paths between rings.

        Each connection defines an ordered path from ring_1 → ring_2.
        Connections should not repeat connectors across different connections.

        Returns
        -------
        list[RingConnection]
            All ring connections with their path edges
        """
        ...

    def get_garment_category(self) -> GarmentCategory:
        """Return the garment category.

        Returns
        -------
        GarmentCategory
            SHIRT, PANTS, SKIRT, DRESS, etc.
        """
        ...
