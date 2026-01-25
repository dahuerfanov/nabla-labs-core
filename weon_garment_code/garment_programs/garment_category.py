"""Garment category enumeration for the perfect-fit pipeline.

This module re-exports GarmentCategory from the unified core_types module
for backward compatibility. New code should import directly from core_types.
"""

# Re-export from core_types for backward compatibility
from weon_garment_code.pygarment.meshgen.arap.core_types import GarmentCategory

__all__ = ["GarmentCategory"]
