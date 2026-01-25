"""Type definitions for stitch identifiers and segmentation.

This module provides edge label enums and stitch/panel identifiers
used in box mesh generation and attachment constraint processing.
"""

from dataclasses import dataclass
from enum import Enum

# Re-export RingLabel from core_types for unified access
from weon_garment_code.pygarment.meshgen.arap.core_types import RingLabel


class EdgeLabel(str, Enum):
    """Enumeration of all edge labels used in box mesh generation.

    This enum provides a centralized definition of all edge labels
    to avoid string literal errors and improve code maintainability.
    """

    # Interface attachment labels
    LOWER_INTERFACE = "lower_interface"
    STRAPLESS_TOP = "strapless_top"

    # Seam labels
    CROTCH_POINT_SEAM = "crotch_point_seam"

    # Vertex labels (for attachment constraints)
    CROTCH = "crotch"
    RIGHT_COLLAR = "right_collar"
    LEFT_COLLAR = "left_collar"

    # Component-specific labels (used with f-strings)
    ARMHOLE = "armhole"
    COLLAR = "collar"

    # Ring detection labels (from RingLabel)
    # Duplicated here for BoxMesh compatibility - prefer RingLabel for new code
    HEM = "hem"
    WAIST = "waist"
    LEFT_CUFF = "left_cuff"
    RIGHT_CUFF = "right_cuff"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    SKIRT_HEM = "skirt_hem"


@dataclass(frozen=True)
class StitchIdentifier:
    """Represents a stitch identifier in the segmentation.

    This class replaces string-based stitch identifiers like "stitch_0"
    with a proper type-safe representation.

    Attributes
    ----------
    stitch_id : int
        The numeric ID of the stitch
    """

    stitch_id: int

    def __str__(self) -> str:
        """Return string representation for serialization."""
        return f"stitch_{self.stitch_id}"

    def __repr__(self) -> str:
        """Return representation for debugging."""
        return f"StitchIdentifier(stitch_id={self.stitch_id})"

    @classmethod
    def from_string(cls, s: str) -> "StitchIdentifier":
        """Create StitchIdentifier from string like 'stitch_0'.

        Parameters
        ----------
        s : str
            String in format 'stitch_<id>'

        Returns
        -------
        StitchIdentifier
            Parsed stitch identifier

        Raises
        ------
        ValueError
            If string format is invalid
        """
        if not s.startswith("stitch_"):
            raise ValueError(f"Invalid stitch identifier format: {s}")
        try:
            stitch_id = int(s.split("_")[-1])
            return cls(stitch_id=stitch_id)
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid stitch identifier format: {s}") from e

    def is_stitch(self) -> bool:
        """Check if this is a stitch identifier (always True for StitchIdentifier)."""
        return True

    def get_panels(self, stitches: list) -> set[str]:
        """Get panel names associated with this stitch.

        Args:
            stitches: List of stitch objects with panel_1 and panel_2 attributes.

        Returns:
            Set of panel names connected by this stitch.
        """
        try:
            s = stitches[self.stitch_id]
            return {s.panel_1, s.panel_2}
        except (IndexError, AttributeError):
            return set()


@dataclass(frozen=True)
class PanelIdentifier:
    """Represents a panel name in the segmentation.

    This class wraps panel names to provide type safety.

    Attributes
    ----------
    panel_name : str
        The name of the panel
    """

    panel_name: str

    def __str__(self) -> str:
        """Return string representation."""
        return self.panel_name

    def __repr__(self) -> str:
        """Return representation for debugging."""
        return f"PanelIdentifier(panel_name={self.panel_name})"

    def is_stitch(self) -> bool:
        """Check if this is a stitch identifier (always False for PanelIdentifier)."""
        return False

    def get_panels(self, stitches: list | None = None) -> set[str]:
        """Get panel name.

        Args:
            stitches: Unused, for interface compatibility.

        Returns:
            Set containing this panel's name.
        """
        return {self.panel_name}


# Type alias for segmentation entries
SegmentationEntry = StitchIdentifier | PanelIdentifier

__all__ = [
    "EdgeLabel",
    "RingLabel",
    "StitchIdentifier",
    "PanelIdentifier",
    "SegmentationEntry",
]
