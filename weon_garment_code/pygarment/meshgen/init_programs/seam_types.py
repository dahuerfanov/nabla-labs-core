"""Seam types and data structures for garment initialization."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RingType(Enum):
    """Unified ring types for both body and garment meshes.

    Body and garment rings use the same names for transparent association.
    """

    COLLAR = "collar"  # Neck/collar ring
    WAIST = "waist"  # Pants waist ring (deterministic from lower_interface label)
    HEM = "hem"  # Waist/hem ring (virtual for body = combined ankles)
    LEFT_CUFF = "left_cuff"  # Left wrist/cuff
    RIGHT_CUFF = "right_cuff"  # Right wrist/cuff
    LEFT_ANKLE = "left_ankle"  # Left ankle
    RIGHT_ANKLE = "right_ankle"  # Right ankle
    SKIRT_HEM = "skirt_hem"  # Skirt hem


class InitPreference(Enum):
    """Preference for how to position garment seam on body seam.

    When garment seam is shorter than body seam:
    - START: Align to start of body seam (from first ring element)
    - END: Align to end of body seam (towards second ring element)
    - MID: Center on body seam
    """

    START = "start"
    END = "end"
    MID = "mid"


@dataclass
class EdgeDefinition:
    """Defines a seam edge between two rings with interpolation preference.

    Order matters: ring_from is the start, ring_to is the end.
    """

    ring_from: RingType
    ring_to: RingType
    preference: InitPreference
    reset_offset: bool = False  # If True, reset offset before processing this edge

    def reversed(self) -> "EdgeDefinition":
        """Return edge with swapped direction."""
        return EdgeDefinition(
            ring_from=self.ring_to,
            ring_to=self.ring_from,
            preference=self.preference,
        )


@dataclass
class SeamMapping:
    """Result of interpolating garment seam onto body seam.

    Attributes:
        garment_indices: Ordered vertex indices from garment seam
        body_positions: (N, 3) interpolated positions on body seam
        remaining_dist: Distance from last mapped point to body seam end
        scale_factor: 1.0 if garment fits, < 1.0 if compressed
    """

    garment_indices: list[int]
    body_positions: np.ndarray
    remaining_dist: float
    scale_factor: float = 1.0
