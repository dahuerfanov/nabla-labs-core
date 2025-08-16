"""
Nabla Labs Core - A lightweight toolkit for visualizing synthetic human datasets.

This package provides essential visualization tools for synthetic human datasets,
including OpenPose keypoints, body-part segmentation, and 3D bounding boxes.
"""

__version__ = "0.1.0"
__author__ = "Nabla Labs"
__email__ = "contact@nabla-labs.com"

# Import main classes and functions
from .constants import (
    EPSILON,
    NR_OPENPOSE_KEYPOINTS,
    MAX_NR_KEYPOINTS,
    OPENPOSE_BODY25_NAMES,
    OPENPOSE_BODY25_PAIRS,
    PALETTE,
    BODY_PART_COLORS,
    get_body_part_bgr,
    DEFAULT_BODY_PART_PALETTE,
)

from .primitives import (
    draw_openpose_keypoints,
    draw_segmentation_overlay,
    draw_3d_bboxes,
)

from .visualize_dataset import DatasetVisualizer

# Main exports
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Constants
    "EPSILON",
    "NR_OPENPOSE_KEYPOINTS", 
    "MAX_NR_KEYPOINTS",
    "OPENPOSE_BODY25_NAMES",
    "OPENPOSE_BODY25_PAIRS",
    "PALETTE",
    "BODY_PART_COLORS",
    "get_body_part_bgr",
    "DEFAULT_BODY_PART_PALETTE",
    
    # Core functions
    "draw_openpose_keypoints",
    "draw_segmentation_overlay", 
    "draw_3d_bboxes",
    
    # Main classes
    "DatasetVisualizer",
]
