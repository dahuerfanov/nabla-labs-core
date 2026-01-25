"""Shared types for ARAP and feature recognition.

This module re-exports types from core_types.py for backward compatibility.
New code should import directly from core_types.
"""

# Re-export all types from core_types for backward compatibility
from weon_garment_code.pygarment.meshgen.arap.core_types import (
    RING_LABEL_STR_TO_TYPE,
    RING_LABEL_TO_TYPE,
    BodyRingType,
    GarmentCategory,
    GarmentMetadata,
    GarmentRingType,
    MeshData,
    PanelPosition,
    PartitionResult,
    RingCentroid,
    RingConnectorPair,
    RingLabel,
    SeamPath,
)

__all__ = [
    "BodyRingType",
    "GarmentCategory",
    "GarmentMetadata",
    "GarmentRingType",
    "MeshData",
    "PanelPosition",
    "PartitionResult",
    "RingCentroid",
    "RingConnectorPair",
    "RingLabel",
    "RING_LABEL_STR_TO_TYPE",
    "RING_LABEL_TO_TYPE",
    "SeamPath",
]
