#!/usr/bin/env python3
"""
Dataset Visualization Script

This script allows users to visualize the output dataset with different modalities:
- OpenPose keypoints and skeleton
- 3D bounding boxes (cuboids)
- Body-part segmentation
- 2D bounding boxes (calculated from segmentation)

Usage:
    python scripts/visualize_dataset.py <dataset_path> [options]

Example:
    python scripts/visualize_dataset.py output/synthetic_dataset --format coco --modalities openpose,segmentation,bboxes
"""

from pathlib import Path
from typing import Optional, Any
import argparse
import json
import math
import random
import sys

from loguru import logger
from pycocotools import mask as maskUtils
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


from .constants import get_body_part_bgr

from .primitives import (
    draw_openpose_keypoints,
    draw_segmentation_overlay,
    draw_3d_bboxes,
)

class DatasetVisualizer:
    """Visualize dataset with different modalities."""
    
    def __init__(self, dataset_path: Path, format_name: str = "coco"):
        """
        Initialize the dataset visualizer.
        
        Args:
            dataset_path: Path to the dataset directory
            format_name: Annotation format (currently only "coco" supported)
        """
        self.dataset_path = Path(dataset_path)
        self.format_name = format_name
        self._legend_generated = False
        
        # Validate dataset structure
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        self.data_dir = self.dataset_path / "data"
        self.annotations_dir = self.dataset_path / "annotations" / f"{format_name}_format"
        
        # Segmentation visualization now works only with COCO annotations
        logger.info("Segmentation visualization will use COCO annotations (RLE decoding)")
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")
        
        logger.info(f"Dataset loaded: {self.dataset_path}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Annotations directory: {self.annotations_dir}")
    
    def list_samples(self) -> list[str]:
        """List all available samples in the dataset."""
        annotation_files = list(self.annotations_dir.glob("*.json"))
        samples = [f.stem for f in annotation_files]
        return sorted(samples)
    
    def load_annotation(self, sample_name: str) -> dict[str, Any]:
        """Load COCO annotation file for a given sample."""
        json_path = self.annotations_dir / f"{sample_name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
            self._annotation_cache = data

            return data
    
    def load_image(self, sample_name: str) -> Optional[np.ndarray]:
        """Load image for a specific sample."""

        for ext in ['.png', '.jpg', '.jpeg']:
            image_file = self.data_dir / f"{sample_name}{ext}"
            if image_file.exists():
                try:
                    image = cv2.imread(str(image_file))
                    if image is not None:
                        return image

                except Exception as e:
                    logger.error(f"Failed to load image {image_file}: {e}")
        
        logger.error(f"No image found for sample: {sample_name}")
        return None
    
    def calculate_2d_bbox_from_segmentation(self, segmentation: dict[str, Any]) -> Optional[list[float]]:
        """
        Calculate 2D bounding box from segmentation mask.
        
        Args:
            segmentation: COCO segmentation dict with 'counts' and 'size'
            
        Returns:
            List [x, y, width, height] or None if invalid
        """
        try:
            binary_mask = maskUtils.decode(segmentation)
            
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            return [float(x_min), float(y_min), float(width), float(height)]
            
        except ImportError:
            logger.warning("pycocotools not available, cannot calculate 2D bbox from segmentation")
            return None

        except Exception as e:
            logger.error(f"Failed to calculate 2D bbox: {e}")
            return None
    
    def visualize_sample(
        self,
        sample_name: str,
        modalities: list[str],
        output_path: Optional[Path] = None,
        show: bool = True
    ) -> Optional[np.ndarray]:
        """
        Visualize a sample with specified modalities.
        
        Args:
            sample_name: Name of the sample to visualize
            modalities: List of modalities to visualize
            output_path: Optional path to save the visualization
            show: Whether to display the visualization
            
        Returns:
            Visualization image or None if failed
        """
        logger.info(f"Visualizing sample: {sample_name}")
        
        # Load annotation and image
        annotation = self.load_annotation(sample_name)
        if annotation is None:
            return None
        
        image = self.load_image(sample_name)
        if image is None:
            return None
        
        vis_image = image.copy()
        for modality in modalities:
            if modality == "openpose" and "annotations" in annotation:
                vis_image = self._add_openpose_visualization(vis_image, annotation["annotations"])
            
            elif modality == "cuboids" and "annotations" in annotation:
                vis_image = self._add_cuboid_visualization(vis_image, annotation["annotations"])
            
            elif modality == "segmentation" and "annotations" in annotation:
                vis_image = self._add_segmentation_visualization(vis_image, annotation["annotations"])
            
            elif modality == "bboxes" and "annotations" in annotation:
                vis_image = self._add_2d_bbox_visualization(vis_image, annotation["annotations"])
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"Visualization saved to: {output_path}")
        
        if show:
            self._show_visualization(vis_image, sample_name)
        
        return vis_image
    
    def _add_openpose_visualization(self, image: np.ndarray, annotations: list[dict]) -> np.ndarray:
        """Add OpenPose keypoints and skeleton visualization."""
        all_keypoints = []
        cats = self._annotation_cache.get("categories", [])
        skeleton_pairs = [(int(a) - 1, int(b) - 1) for a, b in cats[0]["skeleton"]]
        
        for ann in annotations:
            if "keypoints" in ann and len(ann["keypoints"]) > 0:
                # Convert COCO keypoints format [x1,y1,v1,x2,y2,v2,...] to (N, 25, 2)
                keypoints = ann["keypoints"]
                kps_2d = []
                for i in range(0, len(keypoints), 3):
                    x, y, _ = keypoints[i:i+3]
                    kps_2d.append([x, y])

                all_keypoints.append(kps_2d)

        if all_keypoints:
            keypoints_array = np.array(all_keypoints, dtype=np.float32)
            logger.info(f"Processing {len(all_keypoints)} person instances with keypoints")
            result_rgb = draw_openpose_keypoints(image, keypoints_array, pairs=skeleton_pairs, overlay_alpha=0.8)
            return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        else:
            logger.warning("No valid keypoint data found for OpenPose visualization")
        
        return image
    
    def _add_cuboid_visualization(self, image: np.ndarray, annotations: list[dict]) -> np.ndarray:
        """Add 3D bounding box (cuboid) visualization."""

        bboxes_3d = []
        for ann in annotations:
            bbox_3d = [tuple(corner) for corner in ann["bbox_3d"]]
            bboxes_3d.append(bbox_3d)
        
        if bboxes_3d:
            logger.info(f"Found {len(bboxes_3d)} 3D bounding boxes to visualize")
            return draw_3d_bboxes(image, bboxes_3d)
        else:
            logger.warning("No 3D bounding box data found in annotations")

        return image
    
    def _add_segmentation_visualization(self, image: np.ndarray, annotations: list[dict]) -> np.ndarray:
        """Add body-part segmentation visualization."""

        img_height, img_width = annotations[0]["segmentation"]["size"]
        body_part_masks = np.zeros((img_height, img_width), dtype=np.uint16)
        for ann in annotations:
            parts = ann["segmentation_parts"]
            used_parts = False
            for part in parts:
                mask = self._decode_rle_to_mask({"counts": part["counts"], "size": part["size"]})
                pid = int(part["part_id"])
                body_part_masks[mask > 0] = pid
                used_parts = True

            if used_parts:
                continue

            rle = ann["segmentation"]
            person_mask = self._decode_rle_to_mask(rle)
            body_part_regions = self._create_body_part_regions(person_mask, ann["keypoints"], img_height, img_width)
            update_mask = (body_part_masks == 0) & (body_part_regions > 0)
            body_part_masks[update_mask] = body_part_regions[update_mask]

        result_rgb = draw_segmentation_overlay(image, body_part_masks, alpha=0.6)
        if not self._legend_generated:
            self._generate_body_legend()
            self._legend_generated = True

        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    def _create_body_part_regions(self, person_mask: np.ndarray, keypoints: list[float], height: int, width: int) -> np.ndarray:
        """
        Create body part regions using distance-to-seed assignment per part.

        Seeds = keypoint discs + skeleton sticks for that part.
        For each body pixel, assign the part with minimal distance to its seeds.

        Returns labels in 1..6 (head, torso, r_arm, l_arm, r_leg, l_leg).
        """
        H, W = int(height), int(width)
        body_mask = (person_mask > 0)
        labels = np.zeros((H, W), dtype=np.uint16)
        if not keypoints or len(keypoints) < 6 or not np.any(body_mask):
            return labels

        parts = [
            (1, [0, 15, 16, 17, 18], [(0, 1), (0, 15), (15, 17), (0, 16), (16, 18)]),            # head
            (2, [1, 8, 9, 12],       [(1, 8), (1, 9), (1, 12), (8, 9), (8, 12)]),                 # torso
            (3, [2, 3, 4],           [(2, 3), (3, 4)]),                                           # right arm
            (4, [5, 6, 7],           [(5, 6), (6, 7)]),                                           # left arm
            (5, [9, 10, 11, 22, 23, 24], [(9, 10), (10, 11), (11, 22), (11, 24), (22, 23)]),      # right leg
            (6, [12, 13, 14, 19, 20, 21], [(12, 13), (13, 14), (14, 19), (14, 21), (19, 20)]),    # left leg
        ]

        ys, xs = np.where(body_mask)
        body_h = int(ys.max() - ys.min() + 1)
        body_w = int(xs.max() - xs.min() + 1)
        body_scale = max(body_h, body_w)
        stickwidth = max(3, int(body_scale * 0.02))
        kp_radius = max(3, int(body_scale * 0.03))
        kernel_sz = max(3, (stickwidth // 2) * 2 + 1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz))

        def kp_valid(idx: int) -> bool:
            if idx * 3 + 2 >= len(keypoints):
                return False
            x, y, v = keypoints[idx * 3: idx * 3 + 3]
            return not (v == 0 and x == 0 and y == 0)

        def kp_xy(idx: int) -> tuple[float, float]:
            x, y, v = keypoints[idx * 3: idx * 3 + 3]
            return float(x), float(y)

        # Build per-part seed maps
        seeds = np.zeros((6, H, W), dtype=np.uint8)
        for pi, (_, kp_ids, conn_pairs) in enumerate(parts):
            canvas = np.zeros((H, W), dtype=np.uint8)
            for kid in kp_ids:
                if kp_valid(kid):
                    x, y = kp_xy(kid)
                    cv2.circle(canvas, (int(x), int(y)), kp_radius, 255, thickness=-1)
            for (a, b) in conn_pairs:
                if kp_valid(a) and kp_valid(b):
                    x1, y1 = kp_xy(a)
                    x2, y2 = kp_xy(b)
                    mX = (x1 + x2) / 2.0
                    mY = (y1 + y2) / 2.0
                    length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                    if length <= 0:
                        continue
                    angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                    poly = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                    poly = np.ascontiguousarray(poly, dtype=np.int32)
                    if poly.size > 0:
                        cv2.fillConvexPoly(canvas, poly, 255)
            # Slight dilation for seed robustness and mask to person
            canvas = cv2.dilate(canvas, dilate_kernel, iterations=1)
            canvas[~body_mask] = 0
            seeds[pi] = canvas

        # Build anatomical vertical constraints (allowed masks per part)
        def get_kp_y(idx: int, default_y: float) -> float:
            return kp_xy(idx)[1] if kp_valid(idx) else float(default_y)

        # Defaults from body bbox if kps missing
        y_top = float(ys.min()); y_bottom = float(ys.max())
        y_center = (y_top + y_bottom) * 0.5
        # Neck (1), Pelvis (8), Hips (9,12)
        y_neck = get_kp_y(1, y_top + 0.15 * (y_bottom - y_top))
        hips = [v for v in [get_kp_y(9, y_center), get_kp_y(12, y_center), get_kp_y(8, y_center)] if v is not None]
        y_hips = float(np.median(hips)) if hips else y_center

        row_idx = np.arange(H).reshape(H, 1)
        allowed = [np.zeros((H, W), dtype=bool) for _ in range(6)]
        # Head: up to a bit below neck
        y_head_max = int(max(y_top, min(y_bottom, y_neck + 0.25 * (y_hips - y_neck))))
        allowed[0] = (row_idx <= y_head_max)
        allowed[0] = np.broadcast_to(allowed[0], (H, W))
        # Torso: neck down to slightly below hips
        y_torso_min = int(max(y_top, y_neck - 0.1 * (y_bottom - y_top)))
        y_torso_max = int(min(y_bottom, y_hips + 0.15 * (y_bottom - y_top)))
        allowed[1] = (row_idx >= y_torso_min) & (row_idx <= y_torso_max)
        allowed[1] = np.broadcast_to(allowed[1], (H, W))
        # Right arm / left arm: upper/mid body band
        y_arm_min = int(max(y_top, y_neck - 0.15 * (y_bottom - y_top)))
        y_arm_max = int(min(y_bottom, y_hips + 0.25 * (y_bottom - y_top)))
        allowed[2] = (row_idx >= y_arm_min) & (row_idx <= y_arm_max)
        allowed[2] = np.broadcast_to(allowed[2], (H, W))
        allowed[3] = allowed[2].copy()
        # Legs: from around hips down
        y_leg_min = int(max(y_top, y_hips - 0.05 * (y_bottom - y_top)))
        allowed[4] = (row_idx >= y_leg_min)
        allowed[4] = np.broadcast_to(allowed[4], (H, W))
        allowed[5] = allowed[4].copy()
        # Mask by body
        for i in range(6):
            allowed[i] = allowed[i] & (body_mask.astype(bool))

        # Compute per-part distance maps within body mask
        BIG = 1e6
        distances = np.full((6, H, W), BIG, dtype=np.float32)
        body_uint8 = body_mask.astype(np.uint8)
        for pi in range(6):
            part_seed = seeds[pi]
            # Disallow seeds outside permitted region
            part_seed = np.where(allowed[pi], part_seed, 0).astype(np.uint8)
            if np.any(part_seed):
                # Prepare DT input: 1 for body pixels not seeds, 0 for seeds; outside also 0
                dt_in = np.where(body_uint8 & (part_seed == 0), 1, 0).astype(np.uint8)
                dt = cv2.distanceTransform(dt_in, cv2.DIST_L2, 3)
                # Ignore outside with BIG
                dt[~allowed[pi]] = BIG
                distances[pi] = dt

        # Assign label by minimal distance
        min_idx = np.argmin(distances, axis=0)
        min_val = np.min(distances, axis=0)
        labels[body_mask] = (min_idx[body_mask] + 1).astype(np.uint16)

        # Morphological cleanup to reduce small color islands
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, stickwidth // 2) | 1, max(3, stickwidth // 2) | 1))
        for pid in range(1, 7):
            m = (labels == pid).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, clean_kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, clean_kernel, iterations=1)
            labels[m > 0] = pid

        # Ensure full coverage: assign any unassigned body pixel to torso (2)
        uncovered = body_mask & (labels == 0)
        labels[uncovered] = 2

        return labels
    
    def _generate_body_legend(self):
        """Generate a legend showing body part IDs with their actual colors."""

        body_parts = [
            (1, "rightHand"),
            (2, "rightUpLeg"), 
            (3, "leftArm"),
            (4, "head"),
            (5, "leftEye"),
            (6, "rightEye"),
            (7, "leftLeg"),
            (8, "leftToeBase"),
            (9, "leftFoot"),
            (10, "spine1"),
            (11, "spine2"),
            (12, "leftShoulder"),
            (13, "rightShoulder"),
            (14, "rightFoot"),
            (15, "rightArm"),
            (16, "leftHandIndex1"),
            (17, "rightLeg"),
            (18, "rightHandIndex1"),
            (19, "leftForeArm"),
            (20, "rightForeArm"),
            (21, "neck"),
            (22, "rightToeBase"),
            (23, "spine"),
            (24, "leftUpLeg"),
            (25, "eyeballs"),
            (26, "leftHand"),
            (27, "hips")
        ]
        
        # Create figure
        _, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(0, 22)
        ax.set_ylim(0, 18)
        ax.set_aspect('equal')
        
        # Title
        ax.text(11, 17.5, 'Body Part Segmentation Legend', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Create color patches for each body part
        y_pos = 16
        x_pos = 1
        items_per_row = 5
        
        for i, (part_id, part_name) in enumerate(body_parts):
            b, g, r = get_body_part_bgr(part_id)
            rgb_color = (r/255, g/255, b/255)  # Convert to RGB 0-1 range for matplotlib

            # Calculate position in grid
            row = i // items_per_row
            col = i % items_per_row
            x = x_pos + col * 4
            y = y_pos - row * 2.5
            
            color_patch = patches.Rectangle((x, y - 0.3), 1.5, 0.6, 
                                            facecolor=rgb_color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(color_patch)
            ax.text(x + 2, y, f'{part_id}: {part_name}', 
                    fontsize=10, va='center', fontweight='bold')
            ax.text(x + 2, y - 0.15, f'BGR: ({b},{g},{r})', 
                    fontsize=8, va='center', style='italic')
            color_type = "Predefined" if part_id <= 24 else "Hashed"
            ax.text(x + 2, y - 0.3, f'Type: {color_type}', 
                    fontsize=7, va='center', style='italic', color='gray')

        ax.text(11, 2, 'Body part IDs with their assigned colors', 
                fontsize=12, ha='center', style='italic')
        ax.axis('off')
        plt.tight_layout()

        legend_path = self.dataset_path / "body_segmentation_legend.png"
        plt.savefig(legend_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Body segmentation legend saved to: {legend_path}")

    def _create_region_from_keypoints(self, person_mask: np.ndarray, keypoints: list[float], kp_indices: list[int], region_id: int) -> np.ndarray:
        """
        Create a body part region mask using skeleton sticks (ellipse polygons) between keypoints.
        Intersect with the person mask to keep within silhouette.
        """
        region_mask = np.zeros_like(person_mask, dtype=np.uint16)

        # Collect valid keypoints for the given indices
        valid_kps = []
        for i in kp_indices:
            if i * 3 + 2 < len(keypoints):
                x, y, v = keypoints[i * 3:i * 3 + 3]
                if not (v == 0 and x == 0 and y == 0):
                    valid_kps.append((float(x), float(y)))

        if len(valid_kps) < 2:
            # Not enough signal to define this part; skip
            return np.zeros_like(person_mask, dtype=np.uint16)

        # Draw sticks on a contiguous temporary canvas
        h, w = person_mask.shape[:2]
        canvas = np.zeros((h, w), dtype=np.uint8)
        canvas = np.ascontiguousarray(canvas)
        stickwidth = max(4, int(max(person_mask.shape) * 0.006))
        for a in range(len(valid_kps)):
            for b in range(a + 1, len(valid_kps)):
                x1, y1 = valid_kps[a]
                x2, y2 = valid_kps[b]
                mX = (x1 + x2) / 2.0
                mY = (y1 + y2) / 2.0
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if length <= 0:
                    continue
                angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
                poly = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                poly = np.ascontiguousarray(poly, dtype=np.int32)
                if poly.size > 0:
                    cv2.fillConvexPoly(canvas, poly, 255)

        # Intersect with person mask and set region id
        canvas = (canvas > 0) & (person_mask > 0)
        region_mask[canvas] = region_id
        return region_mask
    
    def _decode_rle_to_mask(self, rle: dict[str, Any]) -> np.ndarray:
        """
        Decode RLE (Run-Length Encoding) to binary mask.
        
        Args:
            rle: RLE dictionary with 'counts' and 'size' keys
            
        Returns:
            Binary mask as numpy array, or None if decoding fails
        """
        return maskUtils.decode(rle).astype(np.uint8)
    
    def _add_2d_bbox_visualization(self, image: np.ndarray, annotations: list[dict]) -> np.ndarray:
        """Add 2D bounding box visualization."""

        vis_image = image.copy()
        for ann in annotations:
            bbox = ann["bbox"]
            x, y, w, h = bbox
            cv2.rectangle(vis_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = f"Person {ann.get('id', '?')}"
            cv2.putText(vis_image, label, (int(x), int(y) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis_image

    def _show_visualization(self, image: np.ndarray, sample_name: str):
        """Display the visualization."""

        height, width = image.shape[:2]
        max_dim = 1200

        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_resized = cv2.resize(image, (new_width, new_height))
        else:
            image_resized = image

        # Display using matplotlib if available (preferred for better visualization)
        try:
            import matplotlib.pyplot as plt
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.title(f"Sample: {sample_name}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except ImportError:
            # Fallback to OpenCV window (image is already in BGR format)
            cv2.imshow(f"Sample: {sample_name}", image_resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def create_visualization_grid(
        self,
        sample_names: list[str],
        modalities: list[str],
        output_path: Optional[Path] = None,
        max_samples: int = 9
    ) -> Optional[np.ndarray]:
        """
        Create a grid visualization of multiple samples.
        
        Args:
            sample_names: List of sample names to visualize
            modalities: List of modalities to apply
            output_path: Optional path to save the grid
            max_samples: Maximum number of samples to show in grid
            
        Returns:
            Grid visualization image or None if failed
        """

        if len(sample_names) > max_samples:
            sample_names = sample_names[:max_samples]
            logger.info(f"Limited to {max_samples} samples for grid visualization")

        grid_size = int(np.ceil(np.sqrt(len(sample_names))))
        first_image = self.load_image(sample_names[0])
        if first_image is None:
            return None

        height, width = first_image.shape[:2]
        target_size = (width, height)

        grid_height = height * grid_size
        grid_width = width * grid_size
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Fill grid with visualizations
        for i, sample_name in enumerate(sample_names):
            row = i // grid_size
            col = i % grid_size
            vis_image = self.visualize_sample(sample_name, modalities, show=False)
            if vis_image is not None:
                vis_resized = cv2.resize(vis_image, target_size)
                y_start = row * height
                y_end = y_start + height
                x_start = col * width
                x_end = x_start + width
                
                grid_image[y_start:y_end, x_start:x_end] = vis_resized

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), grid_image)
            logger.info(f"Grid visualization saved to: {output_path}")
        
        return grid_image


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize dataset with different modalities")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("--format", type=str, default="coco", choices=["coco"], help="Annotation format")
    parser.add_argument("--modalities", type=str, default="openpose,bboxes", 
                       help="Comma-separated list of modalities: openpose,cuboids,segmentation,bboxes")
    parser.add_argument("--sample", type=str, help="Specific sample to visualize")
    parser.add_argument("--output", type=str, help="Output path for visualization")
    parser.add_argument("--grid", action="store_true", help="Create grid visualization of multiple samples")
    parser.add_argument("--list", action="store_true", help="List available samples and exit")
    parser.add_argument("--no-show", action="store_true", help="Don't display visualization")

    args = parser.parse_args()
    
    try:
        visualizer = DatasetVisualizer(args.dataset_path, args.format)

        if args.list:
            samples = visualizer.list_samples()
            print(f"Available samples ({len(samples)}):")
            for sample in samples:
                print(f"  {sample}")
            return

        modalities = [m.strip() for m in args.modalities.split(",")]
        logger.info(f"Visualization modalities: {modalities}")

        if args.sample:
            sample_names = [args.sample]
        else:
            all_samples = visualizer.list_samples()
            if args.grid:
                # Randomly select samples for grid mode
                max_samples = min(9, len(all_samples))
                sample_names = random.sample(all_samples, max_samples)
            else:
                # For single sample mode, still use first ones
                sample_names = all_samples[:9]

        if not sample_names:
            logger.error("No samples found")
            return

        output_path = None
        if args.output:
            output_path = Path(args.output)

        # Visualize
        if args.grid:
            grid_image = visualizer.create_visualization_grid(
                sample_names, modalities, output_path
            )
            if grid_image is not None and not args.no_show:
                visualizer._show_visualization(grid_image, "Grid Visualization")
        else:
            # Single sample visualization
            for sample_name in sample_names:
                visualizer.visualize_sample(
                    sample_name, modalities, output_path, show=not args.no_show
                )

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
