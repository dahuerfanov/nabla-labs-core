"""Core types for ARAP and garment feature recognition.

This module provides the unified source of truth for all garment-related
enumerations and types used across the codebase. It consolidates previously
duplicated enums from garment_programs and meshgen modules.

Usage:
    from weon_garment_code.pygarment.meshgen.arap.core_types import (
        GarmentCategory,
        GarmentRingType,
        BodyRingType,
        RingLabel,
    )
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from weon_garment_code.pygarment.meshgen.arap.arap_types import MeshRingConnection


class GarmentCategory(str, Enum):
    """Garment category types.

    Each category maps to a specific strategy class and factory creation method.
    Extend this enum when adding new garment types to the pipeline.
    """

    SHIRT = "shirt"
    PANTS = "pants"
    SKIRT = "skirt"
    DRESS = "dress"

    @classmethod
    def from_string(cls, value: str) -> "GarmentCategory":
        """Convert a string to GarmentCategory, case-insensitive.

        Parameters
        ----------
        value : str
            The garment type string (e.g., "Pants", "SHIRT", "dress").

        Returns
        -------
        GarmentCategory
            The matching category enum.

        Raises
        ------
        ValueError
            If the value doesn't match any category.
        """
        normalized = value.lower().strip()
        for category in cls:
            if category.value == normalized:
                return category
        raise ValueError(
            f"Unknown garment category: '{value}'. Supported: {[c.value for c in cls]}"
        )


class BodyRingType(Enum):
    """Body ring types using unified naming for transparent association."""

    COLLAR = "collar"
    LEFT_CUFF = "left_cuff"
    RIGHT_CUFF = "right_cuff"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    HEM = "hem"
    WAIST = "waist"


class GarmentRingType(Enum):
    """Garment ring types for ARAP anchor detection."""

    COLLAR = "collar"
    LEFT_CUFF = "left_cuff"
    RIGHT_CUFF = "right_cuff"
    HEM = "hem"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    WAIST = "waist"
    SKIRT_HEM = "skirt_hem"


class RingLabel(str, Enum):
    """Edge labels for ring detection (ARAP pipeline).

    These labels are propagated during garment creation to enable
    deterministic ring detection without topology-based heuristics.
    """

    # Common rings
    HEM = "hem"
    WAIST = "waist"
    COLLAR = "collar"

    # Shirt-specific
    LEFT_CUFF = "left_cuff"
    RIGHT_CUFF = "right_cuff"
    LEFT_ARMHOLE = "left_armhole"
    RIGHT_ARMHOLE = "right_armhole"

    # Pants-specific
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"

    # Skirt/Dress-specific
    SKIRT_HEM = "skirt_hem"


# Mapping from RingLabel to GarmentRingType for deterministic detection
RING_LABEL_TO_TYPE: dict[RingLabel, GarmentRingType] = {
    RingLabel.HEM: GarmentRingType.HEM,
    RingLabel.WAIST: GarmentRingType.WAIST,
    RingLabel.COLLAR: GarmentRingType.COLLAR,
    RingLabel.LEFT_CUFF: GarmentRingType.LEFT_CUFF,
    RingLabel.RIGHT_CUFF: GarmentRingType.RIGHT_CUFF,
    RingLabel.LEFT_ANKLE: GarmentRingType.LEFT_ANKLE,
    RingLabel.RIGHT_ANKLE: GarmentRingType.RIGHT_ANKLE,
    RingLabel.SKIRT_HEM: GarmentRingType.SKIRT_HEM,
}

# String-based lookup for compatibility with vertex_labels dict
RING_LABEL_STR_TO_TYPE: dict[str, GarmentRingType] = {
    label.value: ring_type for label, ring_type in RING_LABEL_TO_TYPE.items()
}
# Add mapping for LOWER_INTERFACE (used in constraints) to HEM (used in ARAP)
RING_LABEL_STR_TO_TYPE["lower_interface"] = GarmentRingType.HEM


@dataclass
class MeshData:
    """Container for mesh geometry."""

    vertices: np.ndarray  # (N, 3) or (N, 2)
    faces: np.ndarray  # (M, 3)


@dataclass
class PartitionResult:
    """Result of mesh partitioning."""

    face_labels: np.ndarray  # (M,) 0=FRONT, 1=BACK
    vertex_labels: np.ndarray  # (N,) 0=FRONT, 1=BACK
    seams: dict[tuple[GarmentRingType, GarmentRingType, int], list[int]]


@dataclass
class RingCentroid:
    """Ring with computed centroid."""

    index: int
    centroid: np.ndarray  # (3,)
    vertices: list[int]


class PanelPosition(str, Enum):
    """Panel position within the garment.

    Used to replace ad-hoc name parsing (_analyze_panel_name).
    Each panel has explicit front/back/left/right designation.
    """

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class SeamPath:
    """Precomputed seam path between two rings.

    Seam paths follow panel outlines, computed at garment creation time.
    This replaces runtime Dijkstra path finding.
    """

    ring_type_1: GarmentRingType
    ring_type_2: GarmentRingType
    connector_1: int  # vertex index at ring_type_1
    connector_2: int  # vertex index at ring_type_2
    path: list[int]  # ordered vertex indices from connector_1 to connector_2
    panel_ids: list[set[str]] | None = None  # Panels triggering disjoint checks


@dataclass
class RingConnectorPair:
    """Pair of connector vertices for a ring.

    Connectors are the two vertices on a ring where seam paths attach.
    These are the vertices at the front/back boundary.
    """

    connector_1: int
    connector_2: int  # opposite side of the ring


@dataclass
class GarmentMetadata:
    """Deterministic garment data computed at creation time.

    This data eliminates ad-hoc heuristics in feature_recognition.py:
    - category: replaces _detect_garment_category
    - panel_positions: replaces _analyze_panel_name
    - ring_connectors: replaces _get_ring_seam_connectors
    - seam_paths: replaces Dijkstra path finding

    After BoxMesh processing, mesh-level ring data is populated:
    - mesh_rings: vertex sequences for each ring
    - mesh_ring_connections: vertex sequences for seam paths
    """

    category: GarmentCategory
    panel_positions: dict[str, PanelPosition]  # panel_name -> position
    ring_connectors: dict[GarmentRingType, RingConnectorPair]
    seam_paths: list[SeamPath]

    # Mesh-level data (populated after BoxMesh.collapse_stitch_vertices)
    mesh_rings: dict[GarmentRingType, list[int]] | None = None
    # Structured connections including panel metadata
    mesh_ring_connections: list["MeshRingConnection"] | None = None
