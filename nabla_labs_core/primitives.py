from __future__ import annotations

"""Shared visualisation helpers reused by all visualiser classes.

Adding a helper to draw BODY-25 OpenPose key-points and skeleton so that
dependent packages can use the same, battle-tested implementation.
"""

from typing import Iterable, Tuple, Optional

import cv2
import numpy as np
import math

from .constants import EPSILON, OPENPOSE_BODY25_PAIRS, get_body_part_bgr

__all__ = [
    "draw_openpose_keypoints",
    "draw_segmentation_overlay",
    "draw_3d_bboxes",
]


# ---------------------------------------------------------------------------
# Key-point drawing ----------------------------------------------------------
# ---------------------------------------------------------------------------


def _add_pair_color(a: int, b: int, col: list[int], body_part_colors: dict[tuple[int, int], list[int]]):
    """Register *col* for both (a,b) and (b,a)."""
    body_part_colors[(a, b)] = col
    body_part_colors[(b, a)] = col


def draw_openpose_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,  # (N, 25, 2) or (25, 2)
    *,
    pairs: Iterable[Tuple[int, int]] = OPENPOSE_BODY25_PAIRS,
    radius: int | None = None,
    thickness: int | None = None,
    only_keypoints_in_pairs: bool = True,
    overlay_alpha: float = 0.75,
) -> np.ndarray:
    """Render BODY-25 key-points + skeleton onto *image*.

    Sentinel coordinate value is **-1** (same as original codebase).

    Parameters
    ----------
    image
        3-channel BGR image to draw on.  The returned image is **RGB** as in the
        previous drawing utilities.
    keypoints
        Either a single object ``(25, 2)`` matrix or stacked objects
        ``(N_obj, 25, 2)``.
    pairs
        Iterable of key-point index pairs that form the skeleton.  Defaults to
        :data:`scene.constants.OPENPOSE_BODY25_PAIRS`.
    radius, thickness
        Override adaptive circle radius / line thickness.  When *None* values
        are chosen automatically from image resolution.
    only_keypoints_in_pairs
        If True, only draw keypoints that are in a pair.
    overlay_alpha
        Opacity of the rendered skeleton when composited back onto *image*.
        ``1.0`` keeps the fully opaque OpenPose look, while lower values
        introduce transparency (official OpenPose viewer uses ~0.6).
    """

    # Work on a copy of the image (keep BGR format for OpenCV operations)
    vis = image.copy()

    if keypoints is None or keypoints.size == 0:
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Normalise shape → (N_obj, 25, 2)
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]

    n_obj = keypoints.shape[0]
    h, w = vis.shape[:2]

    # Official OpenPose parameters
    stickwidth = 4  # Official OpenPose stick width
    circle_radius = 4  # Official OpenPose circle radius (fixed size)
    # Override with custom values if provided
    if radius is not None:
        circle_radius = radius
    if thickness is not None:
        stickwidth = thickness

    # Official OpenPose BODY-25 color scheme (BGR format for OpenCV)
    # These are the exact colors from the official OpenPose implementation
    colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
    ]

    # Define specific colors for different body parts (official OpenPose mapping)
    # Color mapping tailored to match user expectations:
    #   • neck ↔ nose           – RED
    #   • neck ↔ both shoulders – BLUE
    #   • right arm chain       – BLUE-ish tones
    #   • left  arm chain       – GREEN-ish tones
    #   • torso / pelvis links  – YELLOW
    #   • legs (already fine)   – keep previous cyan / blue progression

    RED = colors[12]  # RGB red after conversion
    PURPLE = colors[15]  # RGB magenta/violet
    ORANGE = colors[1]  # RGB orange

    # Green gradient (left arm + torso blends)
    GREEN_SHOULDER = colors[4]  # (170,255,0)
    GREEN_UPPER_ARM = colors[5]  # (85,255,0)
    GREEN_LOWER_ARM = colors[6]  # (0,255,0)

    # Blue gradient (right arm + shoulders)
    BLUE_SHOULDER = colors[0]  # pure blue BGR(255,0,0)
    BLUE_UPPER_ARM = colors[1]  # cyan-ish gradient BGR(255,85,0)
    BLUE_LOWER_ARM = colors[2]  # lighter cyan BGR(255,170,0)

    # Torso blends with leg roots
    CENTER_TORSO = colors[8]  # (0,255,170)
    RIGHT_TORSO = colors[7]  # (0,255,85)
    LEFT_TORSO = colors[9]  # (0,255,255)

    # Face gradient colours (unique, not used elsewhere)
    RIGHT_FACE_START = colors[13]  # fuchsia light
    RIGHT_FACE_END = colors[14]  # deeper pink
    LEFT_FACE_START = colors[15]  # purple light
    LEFT_FACE_END = colors[16]  # deeper violet

    body_part_colors: dict[tuple[int, int], list[int]] = {}

    # --- Head & face ---
    _add_pair_color(1, 0, RED, body_part_colors)  # neck ↔ nose

    # Right side (fuchsia gradient)
    _add_pair_color(0, 15, RIGHT_FACE_START, body_part_colors)  # nose ↔ right eye (lighter)
    _add_pair_color(15, 17, RIGHT_FACE_END, body_part_colors)  # right eye ↔ right ear (deeper)

    # Left  side (purple gradient)
    _add_pair_color(0, 16, LEFT_FACE_START, body_part_colors)  # nose ↔ left eye (lighter)
    _add_pair_color(16, 18, LEFT_FACE_END, body_part_colors)  # left eye ↔ left ear (deeper)

    # --- Neck to shoulders (both BLUE) ---
    _add_pair_color(1, 2, BLUE_SHOULDER, body_part_colors)  # neck ↔ right shoulder
    _add_pair_color(1, 5, BLUE_SHOULDER, body_part_colors)  # neck ↔ left  shoulder

    # --- Right arm (blue-ish) ---
    _add_pair_color(2, 3, BLUE_UPPER_ARM, body_part_colors)  # r shoulder ↔ r elbow
    _add_pair_color(3, 4, BLUE_LOWER_ARM, body_part_colors)  # r elbow   ↔ r wrist

    # --- Left arm (green-ish) ---
    _add_pair_color(5, 6, GREEN_SHOULDER, body_part_colors)  # l shoulder ↔ l elbow
    _add_pair_color(6, 7, GREEN_UPPER_ARM, body_part_colors)  # l elbow   ↔ l wrist

    # --- Torso (blended with each leg colour) ---
    _add_pair_color(1, 8, CENTER_TORSO, body_part_colors)  # neck ↔ pelvis
    _add_pair_color(1, 9, RIGHT_TORSO, body_part_colors)  # neck ↔ right hip
    _add_pair_color(1, 12, LEFT_TORSO, body_part_colors)  # neck ↔ left  hip

    # --- Right leg chain (keep previous good colors) ---
    _add_pair_color(8, 9, colors[6], body_part_colors)
    _add_pair_color(9, 10, colors[7], body_part_colors)
    _add_pair_color(10, 11, colors[8], body_part_colors)

    # --- Left leg chain (keep previous good colors) ---
    _add_pair_color(8, 12, colors[9], body_part_colors)
    _add_pair_color(12, 13, colors[10], body_part_colors)
    _add_pair_color(13, 14, colors[11], body_part_colors)

    # --------------------------------------------------
    # Key-point colours  (official OpenPose logic)
    # --------------------------------------------------
    keypoints_in_pairs_set = set()
    for pair in pairs:
        keypoints_in_pairs_set.add(pair[0])
        keypoints_in_pairs_set.add(pair[1])

    for oid in range(n_obj):
        kps = keypoints[oid]

        # Draw skeleton connections using filled ellipses (official OpenPose method)
        for pair_idx, pair in enumerate(pairs):
            i = pair[0]
            j = pair[1]
            if 0 <= i < kps.shape[0] and 0 <= j < kps.shape[0]:
                # Get the two keypoints
                x1, y1 = kps[i, 0], kps[i, 1]
                x2, y2 = kps[j, 0], kps[j, 1]

                # Calculate midpoint and length for ellipse
                mX = (x1 + x2) / 2
                mY = (y1 + y2) / 2
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

                if length > 0:
                    # Calculate angle
                    angle = math.degrees(math.atan2(y1 - y2, x1 - x2))

                    # Create ellipse polygon
                    polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)

                    # Draw filled ellipse with body part color (official OpenPose method)
                    if (i, j) in body_part_colors:
                        color = body_part_colors[(i, j)]
                    else:
                        color = colors[pair_idx % len(colors)]
                    cv2.fillConvexPoly(vis, polygon, color)

        # Draw keypoint circles (official OpenPose method)
        for idx, (x, y) in enumerate(kps):
            if (
                x != -1
                and y != -1
                and 0 <= x < w
                and 0 <= y < h
                and (not only_keypoints_in_pairs or idx in keypoints_in_pairs_set)
            ):
                color = colors[idx % len(colors)]
                cv2.circle(vis, (int(x), int(y)), circle_radius, color, thickness=-1)

    if 0.0 <= overlay_alpha < 1.0:
        vis = cv2.addWeighted(image, 1 - overlay_alpha, vis, overlay_alpha, 0)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Segmentation overlay -------------------------------------------------------
# ---------------------------------------------------------------------------


def _apply_segmentation_color_layer(segmentation: np.ndarray) -> np.ndarray:
    """Create an RGB layer visualising *segmentation*.

    segmentation may be (H,W) int mask or (N_obj,H,W) stack.  Background = 0.
    Returns an RGB uint8 array the same spatial size as the mask.
    """
    if segmentation.ndim == 3:  # (N_obj,H,W)
        seg = np.max(segmentation, axis=0)  # union of objects – frontmost wins later
    else:
        seg = segmentation

    h, w = seg.shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)

    for part_id in np.unique(seg):
        if part_id == 0:
            continue  # background
        mask = seg == part_id
        if np.any(mask):
            layer[mask] = get_body_part_bgr(int(part_id))
    return layer


def draw_segmentation_overlay(
    base_bgr: np.ndarray,
    segmentation_map: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Overlay a segmentation map onto *base_bgr* image.

    Returns an RGB image (like other primitives).
    If *alpha* < 1.0 the overlay is blended; else colors replace pixels fully.
    """
    if base_bgr is None or base_bgr.size == 0:
        raise ValueError("base_bgr image is empty")

    vis_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

    if segmentation_map is None or segmentation_map.size == 0:
        return vis_rgb

    color_layer = _apply_segmentation_color_layer(segmentation_map)

    # Ensure both images have the same dimensions
    if color_layer.shape[:2] != vis_rgb.shape[:2]:
        # Resize color_layer to match vis_rgb dimensions
        color_layer = cv2.resize(color_layer, (vis_rgb.shape[1], vis_rgb.shape[0]))

    # Ensure both images have the same number of channels
    if color_layer.shape[-1] != vis_rgb.shape[-1]:
        if color_layer.shape[-1] == 1:
            color_layer = cv2.cvtColor(color_layer, cv2.COLOR_GRAY2RGB)
        elif vis_rgb.shape[-1] == 1:
            vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_GRAY2RGB)

    if alpha < 1.0:
        vis_rgb = cv2.addWeighted(color_layer, alpha, vis_rgb, 1 - alpha, 0)
    else:
        vis_rgb[color_layer.any(axis=-1)] = color_layer[color_layer.any(axis=-1)]

    return vis_rgb


# ---------------------------------------------------------------------------
# Depth & normal map helpers -------------------------------------------------
# ---------------------------------------------------------------------------


def depth_to_rgb(depth_map: np.ndarray) -> np.ndarray:
    """Convert a depth map to an 8-bit RGB visualisation.

    • Ignores zeros / negatives when computing min-max (treat as invalid).
    • If all valid depths have the same value the output is black.
    """
    if depth_map is None or depth_map.size == 0:
        return np.zeros((100, 100, 3), np.uint8)

    dm = depth_map.copy().astype(np.float32)
    valid = dm > 0
    if not np.any(valid):
        return np.zeros((*dm.shape, 3), np.uint8)

    d_min = float(dm[valid].min())
    d_max = float(dm[valid].max())

    if abs(d_max - d_min) < EPSILON:
        norm = np.zeros_like(dm)
    else:
        norm = (dm - d_min) / (d_max - d_min)

    gray = (norm * 255).astype(np.uint8)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def normal_to_rgb(normal_map: np.ndarray) -> np.ndarray:
    """Convert a normal map (in -1..1 or 0..1 range) to RGB 0-255, blueish style.

    The blue tint is restored, with the blue-red direction inverted compared to the previous version.
    The up/down direction is mapped so that pointing up (Y=+1) is lighter than pointing down (Y=-1).
    """

    if normal_map is None or normal_map.size == 0:
        return np.zeros((100, 100, 3), np.uint8)

    nm = normal_map.copy().astype(np.float32)

    # If values are in [-1,1] shift to [0,2] then /2
    if nm.min() < 0:
        nm = (nm + 1.0) / 2.0

    nm = np.clip(nm, 0.0, 1.0)

    if nm.ndim == 2:
        nm = np.stack([nm] * 3, axis=-1)
    elif nm.shape[-1] == 1:
        nm = np.concatenate([nm] * 3, axis=-1)

    # Blue tint: blue = nm[...,2]*255, green = nm[...,1]*127, red = nm[...,0]*127
    # Invert blue-red direction: blue = 255 - nm[...,2]*255, red = 255 - nm[...,0]*127
    # For green (Y), invert so that up (Y=+1) is lighter, down (Y=0) is darker
    blue = ((1.0 - nm[..., 2]) * 255).astype(np.uint8)
    green = ((1.0 - nm[..., 1]) * 127).astype(np.uint8)  # Invert Y: up=light, down=dark
    red = ((1.0 - nm[..., 0]) * 127).astype(np.uint8)
    bgr = np.stack([blue, green, red], axis=-1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


__all__.extend(["depth_to_rgb", "normal_to_rgb"])


# ---------------------------------------------------------------------------
# 3D Bounding box drawing ----------------------------------------------------
# ---------------------------------------------------------------------------


def draw_3d_bboxes(
    image: np.ndarray,
    bboxes_3d: list[list[tuple[float, float, float]]],
    colors: Optional[list[tuple[int, int, int]]] = None,
    crop_coords: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Draw 3D bounding boxes on an image.

    Bbox coordinates are already in pixel space, so no projection is needed.

    Args:
        image: Input image (BGR format)
        bboxes_3d: List of 3D bounding boxes, each containing 8 corner points as tuples
        alpha: Transparency for the bounding box overlay (unused, kept for compatibility)
        colors: List of BGR colors for each bounding box
        crop_coords: Optional crop coordinates (y_start, y_end, x_start, x_end) to adjust bbox coordinates

    Returns:
        Image with 3D bounding boxes drawn (BGR format)
    """
    result = image.copy()

    if not bboxes_3d:
        return result

    h, w = image.shape[:2]

    # Default colors if not provided
    if colors is None:
        colors = [
            (0, 0, 255),  # Red
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Cyan
            (0, 165, 255),  # Orange
            (128, 0, 128),  # Purple
        ]

    # Adaptive line thickness based on image resolution
    base_size = max(h, w)
    thickness = max(2, int(base_size * 0.002))  # ~0.2% of larger dim

    for obj_idx, bbox_3d in enumerate(bboxes_3d):
        if len(bbox_3d) != 8:
            continue

        # Convert list of tuples to numpy array
        bbox_arr = np.array(bbox_3d, dtype=np.float32)

        # Ensure we have 2D coordinates (bbox coordinates are already in pixels)
        if bbox_arr.shape[1] == 3:
            # 3D coordinates - take only x,y for 2D projection
            corners_2d = bbox_arr[:, :2].astype(np.int32)
        else:
            # Already 2D coordinates
            corners_2d = bbox_arr.astype(np.int32)

        # Adjust coordinates if crop_coords is provided
        if crop_coords is not None:
            y_start, _, x_start, _ = crop_coords
            corners_2d[:, 0] = corners_2d[:, 0] - x_start
            corners_2d[:, 1] = corners_2d[:, 1] - y_start

        # Define the 12 edges of a 3D bounding box
        edges = [
            # Bottom face (z=0)
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top face (z=1)
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical edges connecting bottom to top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        color = colors[obj_idx % len(colors)]
        for start_idx, end_idx in edges:
            pt1 = tuple(corners_2d[start_idx])
            pt2 = tuple(corners_2d[end_idx])
            cv2.line(result, pt1, pt2, color, thickness)

    return result
