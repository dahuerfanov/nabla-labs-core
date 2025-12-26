"""Box mesh generation module.

This module provides classes for generating box meshes from pattern specifications.
"""

from weon_garment_code.pygarment.meshgen.box_mesh_gen.errors import (
    PatternLoadingError,
    MultiStitchingError,
    StitchingError,
    DegenerateTrianglesError,
    NormError,
)
from weon_garment_code.pygarment.meshgen.box_mesh_gen.panel import Panel
from weon_garment_code.pygarment.meshgen.box_mesh_gen.edge import Edge
from weon_garment_code.pygarment.meshgen.box_mesh_gen.seam import Seam
from weon_garment_code.pygarment.meshgen.box_mesh_gen.box_mesh import BoxMesh

__all__ = [
    "PatternLoadingError",
    "MultiStitchingError",
    "StitchingError",
    "DegenerateTrianglesError",
    "NormError",
    "Panel",
    "Edge",
    "Seam",
    "BoxMesh",
]

