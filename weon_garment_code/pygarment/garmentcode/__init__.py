"""
    A Python library for building parametric sewing pattern programs
"""

# Configure logging automatically when package is imported
try:
    import log_config  # noqa: F401
except ImportError:
    # If log_config is not available, use default loguru configuration
    pass

# Building blocks
# Operations
import weon_garment_code.pygarment.garmentcode.operators as ops
import weon_garment_code.pygarment.garmentcode.utils as utils
from weon_garment_code.pygarment.garmentcode.component import Component
from weon_garment_code.pygarment.garmentcode.connector import Stitches
from weon_garment_code.pygarment.garmentcode.edge import (
    CircleEdge,
    CurveEdge,
    Edge,
    EdgeSequence,
)
from weon_garment_code.pygarment.garmentcode.edge_factory import (
    CircleEdgeFactory,
    CurveEdgeFactory,
    EdgeFactory,
    EdgeSeqFactory,
)
from weon_garment_code.pygarment.garmentcode.interface import Interface
from weon_garment_code.pygarment.garmentcode.panel import Panel

# Parameter support
from weon_garment_code.pygarment.garmentcode.params import (
    BodyParametrizationBase,
    DesignSampler,
)

# Errors
from weon_garment_code.pygarment.pattern.core import EmptyPatternError
