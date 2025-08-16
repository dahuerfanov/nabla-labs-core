#!/usr/bin/env python3
"""
Tests for constants module.
"""

import pytest
import numpy as np

from nabla_labs_core.constants import (
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


class TestConstants:
    """Test constants and utility functions."""
    
    def test_epsilon_value(self):
        """Test EPSILON constant value."""
        assert EPSILON == 1e-6
        assert EPSILON > 0
    
    def test_openpose_constants(self):
        """Test OpenPose-related constants."""
        assert NR_OPENPOSE_KEYPOINTS == 25
        assert MAX_NR_KEYPOINTS == 150
        assert len(OPENPOSE_BODY25_NAMES) == 25
        assert len(OPENPOSE_BODY25_PAIRS) > 0
    
    def test_palette_structure(self):
        """Test color palette structure."""
        assert len(PALETTE) == 12
        for color in PALETTE:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_body_part_colors(self):
        """Test body part color mapping."""
        assert len(BODY_PART_COLORS) > 0
        for part_id, color in BODY_PART_COLORS.items():
            assert isinstance(part_id, int)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_get_body_part_bgr(self):
        """Test get_body_part_bgr function."""
        # Test existing body part
        color = get_body_part_bgr(1)
        assert color in BODY_PART_COLORS.values()
        
        # Test non-existing body part (should generate hash-based color)
        color = get_body_part_bgr(999)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_default_palette(self):
        """Test default body part palette."""
        assert DEFAULT_BODY_PART_PALETTE == BODY_PART_COLORS


class TestOpenPoseStructure:
    """Test OpenPose BODY-25 structure."""
    
    def test_keypoint_names(self):
        """Test that all keypoint names are valid."""
        expected_names = [
            "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
            "left_shoulder", "left_elbow", "left_wrist", "pelvis",
            "right_hip", "right_knee", "right_ankle", "left_hip",
            "left_knee", "left_ankle", "right_eye", "left_eye",
            "right_ear", "left_ear", "left_big_toe", "left_small_toe",
            "left_heel", "right_big_toe", "right_small_toe", "right_heel"
        ]
        
        assert OPENPOSE_BODY25_NAMES == expected_names
    
    def test_keypoint_pairs(self):
        """Test that keypoint pairs are valid indices."""
        for pair in OPENPOSE_BODY25_PAIRS:
            assert len(pair) == 2
            assert all(0 <= idx < 25 for idx in pair)
            assert pair[0] != pair[1]  # No self-connections


if __name__ == "__main__":
    pytest.main([__file__])
