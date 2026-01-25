"""Strategy pattern for garment category-specific logic.

This module implements the Strategy pattern for handling garment-specific
ARAP behavior. Each strategy knows:
- What ring types to expect for its category
- How to categorize rings from vertex labels (deterministic)
- How to categorize rings from topology (fallback for legacy garments)
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from weon_garment_code.pygarment.meshgen.arap.core_types import (
    RING_LABEL_STR_TO_TYPE,
    GarmentCategory,
    GarmentRingType,
    RingCentroid,
)


class GarmentCategoryStrategy(ABC):
    """Abstract base class for garment category-specific behavior.

    Subclasses must implement:
    - category: The GarmentCategory this strategy handles
    - get_ring_sequences: Ring connection order for seam finding
    - get_expected_ring_types: Ring types this category should have
    - categorize_rings: Fallback topology-based categorization
    """

    @property
    @abstractmethod
    def category(self) -> GarmentCategory:
        """Return the category this strategy handles."""
        ...

    @abstractmethod
    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        """Return the ring sequences for seam finding.

        Each sequence defines a path through rings for seam interpolation.
        For example, a shirt might have [HEM, LEFT_CUFF, COLLAR] as one path.
        """
        ...

    @abstractmethod
    def get_expected_ring_types(self) -> list[GarmentRingType]:
        """Return the ring types expected for this garment category."""
        ...

    def categorize_rings_from_labels(
        self,
        vertex_labels: dict[str, list[int]],
    ) -> dict[GarmentRingType, list[int]]:
        """Deterministic ring categorization from BoxMesh vertex_labels.

        Parameters
        ----------
        vertex_labels : dict[str, list[int]]
            Mapping from label string to vertex indices.

        Returns
        -------
        dict[GarmentRingType, list[int]]
            Mapping from ring type to vertex indices.
        """
        categorized: dict[GarmentRingType, list[int]] = {}
        expected_types = set(self.get_expected_ring_types())

        for label_str, vertices in vertex_labels.items():
            ring_type = RING_LABEL_STR_TO_TYPE.get(label_str)
            if ring_type is not None and ring_type in expected_types:
                if ring_type in categorized:
                    # Merge vertices if same ring type from multiple sources
                    categorized[ring_type].extend(vertices)
                else:
                    categorized[ring_type] = list(vertices)

        return categorized

    @abstractmethod
    def categorize_rings(
        self,
        ring_centroids: list[RingCentroid],
        vertices: np.ndarray,
    ) -> dict[GarmentRingType, list[int]]:
        """Fallback: Categorize detected rings based on their centroids.

        This is the topology-based approach used when vertex labels
        are not available (e.g., legacy or externally-loaded garments).

        Parameters
        ----------
        ring_centroids : list[RingCentroid]
            List of detected rings with their centroids.
        vertices : np.ndarray
            Mesh vertices array.

        Returns
        -------
        dict[GarmentRingType, list[int]]
            Mapping from ring type to vertex indices.
        """
        ...


class ShirtStrategy(GarmentCategoryStrategy):
    """Strategy for shirt garments."""

    @property
    def category(self) -> GarmentCategory:
        return GarmentCategory.SHIRT

    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        return [
            [GarmentRingType.HEM, GarmentRingType.LEFT_CUFF, GarmentRingType.COLLAR],
            [GarmentRingType.HEM, GarmentRingType.RIGHT_CUFF, GarmentRingType.COLLAR],
        ]

    def get_expected_ring_types(self) -> list[GarmentRingType]:
        return [
            GarmentRingType.HEM,
            GarmentRingType.COLLAR,
            GarmentRingType.LEFT_CUFF,
            GarmentRingType.RIGHT_CUFF,
        ]

    def categorize_rings(
        self,
        ring_centroids: list[RingCentroid],
        vertices: np.ndarray,
    ) -> dict[GarmentRingType, list[int]]:
        """Topology-based shirt ring categorization (fallback)."""
        categorized: dict[GarmentRingType, list[int]] = {}

        if not ring_centroids:
            logger.warning("No rings provided for categorization.")
            return categorized

        # Sort by Y (Vertical)
        sorted_rings = sorted(ring_centroids, key=lambda r: r.centroid[1])

        # Collar (Highest Y)
        collar = sorted_rings[-1]
        categorized[GarmentRingType.COLLAR] = collar.vertices

        # Hem (Lowest Y)
        if len(sorted_rings) > 1:
            hem = sorted_rings[0]
            categorized[GarmentRingType.HEM] = hem.vertices

        # Cuffs (Middle - sorted by X)
        if len(sorted_rings) > 2:
            mid_rings = sorted_rings[1:-1]
            mid_rings.sort(key=lambda r: r.centroid[0])

            if len(mid_rings) == 1:
                ring = mid_rings[0]
                if ring.centroid[0] < 0:
                    categorized[GarmentRingType.RIGHT_CUFF] = ring.vertices
                else:
                    categorized[GarmentRingType.LEFT_CUFF] = ring.vertices
            elif len(mid_rings) >= 2:
                # Most negative X -> Right (mirrored pattern layout)
                categorized[GarmentRingType.RIGHT_CUFF] = mid_rings[0].vertices
                # Most positive X -> Left
                categorized[GarmentRingType.LEFT_CUFF] = mid_rings[-1].vertices

        return categorized


class PantsStrategy(GarmentCategoryStrategy):
    """Strategy for pants garments."""

    @property
    def category(self) -> GarmentCategory:
        return GarmentCategory.PANTS

    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        return [
            [GarmentRingType.HEM, GarmentRingType.LEFT_ANKLE],
            [GarmentRingType.HEM, GarmentRingType.RIGHT_ANKLE],
            [GarmentRingType.LEFT_ANKLE, GarmentRingType.RIGHT_ANKLE],
        ]

    def get_expected_ring_types(self) -> list[GarmentRingType]:
        return [
            GarmentRingType.HEM,  # Top of pants (from lower_interface label)
            GarmentRingType.LEFT_ANKLE,
            GarmentRingType.RIGHT_ANKLE,
        ]

    def categorize_rings(
        self,
        ring_centroids: list[RingCentroid],
        vertices: np.ndarray,
    ) -> dict[GarmentRingType, list[int]]:
        """Topology-based pants ring categorization (fallback)."""
        categorized: dict[GarmentRingType, list[int]] = {}

        if not ring_centroids:
            logger.warning("No rings provided for categorization.")
            return categorized

        # Sort by Y (Vertical)
        sorted_rings = sorted(ring_centroids, key=lambda r: r.centroid[1])

        # Waist/Hem (Highest Y)
        waist = sorted_rings[-1]
        categorized[GarmentRingType.HEM] = waist.vertices

        # Ankles (Lower - sorted by X)
        if len(sorted_rings) > 1:
            ankles = sorted_rings[:-1]
            ankles.sort(key=lambda r: r.centroid[0])

            if len(ankles) == 1:
                ring = ankles[0]
                if ring.centroid[0] < 0:
                    categorized[GarmentRingType.RIGHT_ANKLE] = ring.vertices
                else:
                    categorized[GarmentRingType.LEFT_ANKLE] = ring.vertices
            elif len(ankles) >= 2:
                categorized[GarmentRingType.RIGHT_ANKLE] = ankles[0].vertices
                categorized[GarmentRingType.LEFT_ANKLE] = ankles[-1].vertices

        return categorized


class SkirtStrategy(GarmentCategoryStrategy):
    """Strategy for skirt garments."""

    @property
    def category(self) -> GarmentCategory:
        return GarmentCategory.SKIRT

    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        return [
            [GarmentRingType.WAIST, GarmentRingType.SKIRT_HEM],
        ]

    def get_expected_ring_types(self) -> list[GarmentRingType]:
        return [
            GarmentRingType.WAIST,
            GarmentRingType.SKIRT_HEM,
        ]

    def categorize_rings(
        self,
        ring_centroids: list[RingCentroid],
        vertices: np.ndarray,
    ) -> dict[GarmentRingType, list[int]]:
        """Topology-based skirt ring categorization (fallback)."""
        categorized: dict[GarmentRingType, list[int]] = {}

        if not ring_centroids:
            logger.warning("No rings provided for categorization.")
            return categorized

        # Sort by Y (higher = waist, lower = hem)
        sorted_rings = sorted(ring_centroids, key=lambda r: r.centroid[1], reverse=True)

        if len(sorted_rings) >= 1:
            categorized[GarmentRingType.WAIST] = sorted_rings[0].vertices
        if len(sorted_rings) >= 2:
            categorized[GarmentRingType.SKIRT_HEM] = sorted_rings[-1].vertices

        return categorized


class DressStrategy(GarmentCategoryStrategy):
    """Strategy for dress garments.

    Dresses are composite garments with shirt-like top and skirt-like bottom.
    They have collar, cuffs (optional), waist, and skirt hem.
    """

    @property
    def category(self) -> GarmentCategory:
        return GarmentCategory.DRESS

    def get_ring_sequences(self) -> list[list[GarmentRingType]]:
        # Dress has paths from collar through waist to hem
        return [
            [GarmentRingType.COLLAR, GarmentRingType.WAIST, GarmentRingType.SKIRT_HEM],
            [GarmentRingType.LEFT_CUFF, GarmentRingType.WAIST],
            [GarmentRingType.RIGHT_CUFF, GarmentRingType.WAIST],
        ]

    def get_expected_ring_types(self) -> list[GarmentRingType]:
        return [
            GarmentRingType.COLLAR,
            GarmentRingType.WAIST,
            GarmentRingType.SKIRT_HEM,
            GarmentRingType.LEFT_CUFF,
            GarmentRingType.RIGHT_CUFF,
        ]

    def categorize_rings(
        self,
        ring_centroids: list[RingCentroid],
        vertices: np.ndarray,
    ) -> dict[GarmentRingType, list[int]]:
        """Topology-based dress ring categorization (fallback).

        Dress has: collar (top), waist (middle), hem (bottom), optional cuffs (sides).
        """
        categorized: dict[GarmentRingType, list[int]] = {}

        if not ring_centroids:
            logger.warning("No rings provided for categorization.")
            return categorized

        # Sort by Y (Vertical)
        sorted_rings = sorted(ring_centroids, key=lambda r: r.centroid[1])

        # Collar (Highest Y)
        if len(sorted_rings) >= 1:
            collar = sorted_rings[-1]
            categorized[GarmentRingType.COLLAR] = collar.vertices

        # Hem (Lowest Y)
        if len(sorted_rings) >= 2:
            hem = sorted_rings[0]
            categorized[GarmentRingType.SKIRT_HEM] = hem.vertices

        # Waist and cuffs (middle rings)
        if len(sorted_rings) >= 3:
            mid_rings = sorted_rings[1:-1]
            # Find waist: typically widest/largest middle ring
            # For now, use the highest of the middle rings as waist
            sorted_mid = sorted(mid_rings, key=lambda r: r.centroid[1], reverse=True)
            waist = sorted_mid[0]
            categorized[GarmentRingType.WAIST] = waist.vertices

            # Remaining middle rings could be cuffs
            if len(sorted_mid) >= 2:
                cuff_candidates = sorted_mid[1:]
                cuff_candidates.sort(key=lambda r: r.centroid[0])
                if len(cuff_candidates) == 1:
                    ring = cuff_candidates[0]
                    if ring.centroid[0] < 0:
                        categorized[GarmentRingType.RIGHT_CUFF] = ring.vertices
                    else:
                        categorized[GarmentRingType.LEFT_CUFF] = ring.vertices
                elif len(cuff_candidates) >= 2:
                    categorized[GarmentRingType.RIGHT_CUFF] = cuff_candidates[
                        0
                    ].vertices
                    categorized[GarmentRingType.LEFT_CUFF] = cuff_candidates[
                        -1
                    ].vertices

        return categorized


# Strategy registry
_STRATEGY_REGISTRY: dict[GarmentCategory, type[GarmentCategoryStrategy]] = {
    GarmentCategory.SHIRT: ShirtStrategy,
    GarmentCategory.PANTS: PantsStrategy,
    GarmentCategory.SKIRT: SkirtStrategy,
    GarmentCategory.DRESS: DressStrategy,
}


def get_strategy(category: GarmentCategory) -> GarmentCategoryStrategy:
    """Factory function to get the appropriate strategy.

    Parameters
    ----------
    category : GarmentCategory
        The garment category.

    Returns
    -------
    GarmentCategoryStrategy
        Instantiated strategy for the category.

    Raises
    ------
    ValueError
        If no strategy is defined for the category.
    """
    strategy_cls = _STRATEGY_REGISTRY.get(category)
    if strategy_cls is None:
        raise ValueError(f"No strategy defined for category: {category}")
    return strategy_cls()


def register_strategy(
    category: GarmentCategory, strategy_cls: type[GarmentCategoryStrategy]
) -> None:
    """Register a new strategy for a category.

    Parameters
    ----------
    category : GarmentCategory
        The garment category.
    strategy_cls : type[GarmentCategoryStrategy]
        The strategy class to register.
    """
    _STRATEGY_REGISTRY[category] = strategy_cls
