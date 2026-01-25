"""Ring type definitions for mesh partitioning."""

from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class RingType(Protocol):
    """Protocol for ring type enums."""

    @property
    def value(self) -> str: ...


class GarmentRingType(Enum):
    """Ring types for garment meshes."""

    COLLAR = "collar"
    HEM = "hem"
    WAIST = "waist"
    LEFT_CUFF = "left_cuff"
    RIGHT_CUFF = "right_cuff"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class BodyRingType(Enum):
    """Ring types for SMPLX body meshes.

    Uses same names as GarmentRingType for transparent association.
    """

    COLLAR = "collar"  # Neck ring (renamed from NECK)
    LEFT_CUFF = "left_cuff"  # Left wrist (renamed from LEFT_WRIST)
    RIGHT_CUFF = "right_cuff"  # Right wrist (renamed from RIGHT_WRIST)
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    HEM = "hem"  # Virtual: combined ankles for shirts


class GarmentCategory(Enum):
    """Garment category types."""

    SHIRT = "shirt"
    PANTS = "pants"
    UNKNOWN = "unknown"
