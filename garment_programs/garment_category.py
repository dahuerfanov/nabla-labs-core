"""Garment category enumeration for the perfect-fit pipeline.

This module provides a centralized enum for garment categories, enabling
maintainable factory patterns and extensibility for future garment types.
"""

from enum import Enum


class GarmentCategory(str, Enum):
    """Supported garment categories for the perfect-fit pipeline.

    Each category maps to a specific factory creation method in GarmentFactory.
    Extend this enum when adding new garment types to the pipeline.
    """

    PANTS = "pants"
    SHIRT = "shirt"
    # Future categories (uncomment when implemented):
    # SKIRT = "skirt"
    # DRESS = "dress"
    # FITTED_SHIRT = "fittedshirt"

    @classmethod
    def from_string(cls, value: str) -> "GarmentCategory":
        """Convert a string to GarmentCategory, case-insensitive.

        Parameters
        ----------
        value : str
            The garment type string (e.g., "Pants", "SHIRT", "fittedshirt").

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
