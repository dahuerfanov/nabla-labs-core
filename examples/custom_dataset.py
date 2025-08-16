#!/usr/bin/env python3
"""
Custom dataset integration example for nabla-labs-core.

This script demonstrates how to extend the visualization toolkit
to work with custom dataset formats and annotation structures.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

from nabla_labs_core import (
    draw_openpose_keypoints,
    draw_segmentation_overlay,
    draw_3d_bboxes,
    get_body_part_bgr,
)

class CustomDatasetAdapter:
    """
    Example adapter for custom dataset formats.
    
    This class shows how to integrate custom annotation formats
    with the visualization toolkit.
    """
    
    def __init__(self, dataset_path: str, format_name: str = "custom"):
        """
        Initialize the custom dataset adapter.
        
        Args:
            dataset_path: Path to the dataset directory
            format_name: Name of the custom format
        """
        self.dataset_path = Path(dataset_path)
        self.format_name = format_name
        
        # Validate dataset structure
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    def load_custom_annotation(self, sample_name: str) -> Dict[str, Any]:
        """
        Load custom annotation format.
        
        This is an example - adapt this to your specific format.
        """
        annotation_file = self.dataset_path / "annotations" / f"{sample_name}.json"
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            return json.load(f)
    
    def convert_to_standard_format(self, custom_annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert custom annotation format to standard format.
        
        This method should be customized based on your annotation structure.
        """
        # Example conversion - adapt to your format
        standard_format = {
            "keypoints": [],
            "segmentation": None,
            "bbox_3d": [],
            "bbox_2d": [],
            "metadata": {}
        }
        
        # Convert keypoints if they exist
        if "pose_keypoints" in custom_annotation:
            # Assuming custom format has different keypoint structure
            custom_keypoints = custom_annotation["pose_keypoints"]
            standard_keypoints = self._convert_keypoints(custom_keypoints)
            standard_format["keypoints"] = standard_keypoints
        
        # Convert segmentation if it exists
        if "body_parts" in custom_annotation:
            segmentation = self._convert_segmentation(custom_annotation["body_parts"])
            standard_format["segmentation"] = segmentation
        
        # Convert 3D bounding boxes if they exist
        if "bounding_boxes_3d" in custom_annotation:
            bboxes_3d = self._convert_3d_bboxes(custom_annotation["bounding_boxes_3d"])
            standard_format["bbox_3d"] = bboxes_3d
        
        # Add metadata
        standard_format["metadata"] = {
            "original_format": self.format_name,
            "sample_id": custom_annotation.get("id", "unknown"),
            "timestamp": custom_annotation.get("timestamp", "unknown")
        }
        
        return standard_format
    
    def _convert_keypoints(self, custom_keypoints: List[float]) -> np.ndarray:
        """
        Convert custom keypoint format to BODY-25 format.
        
        Args:
            custom_keypoints: List of keypoint coordinates [x1, y1, conf1, x2, y2, conf2, ...]
        
        Returns:
            numpy array of shape (25, 2) with BODY-25 keypoint format
        """
        # Initialize BODY-25 keypoints with -1 (invisible)
        keypoints = np.full((25, 2), -1, dtype=np.float32)
        
        # Example conversion - adapt based on your keypoint format
        # This assumes custom format has 17 keypoints in COCO order
        if len(custom_keypoints) >= 34:  # 17 keypoints * 2 coordinates
            for i in range(17):
                x = custom_keypoints[i * 2]
                y = custom_keypoints[i * 2 + 1]
                
                # Map to BODY-25 format (example mapping)
                if i < 25:  # Only map if within BODY-25 range
                    keypoints[i] = [x, y]
        
        return keypoints
    
    def _convert_segmentation(self, body_parts: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Convert custom segmentation format to standard format.
        
        Args:
            body_parts: Custom body part segmentation data
        
        Returns:
            numpy array with body part IDs, or None if conversion not possible
        """
        # Example conversion - adapt based on your format
        # This assumes body_parts contains mask data or polygon coordinates
        
        if "mask" in body_parts:
            # If mask is already in numpy format
            return np.array(body_parts["mask"])
        
        elif "polygons" in body_parts:
            # Convert polygon format to mask
            # This is a simplified example
            mask = np.zeros((512, 512), dtype=np.uint8)  # Default size
            # Add polygon rendering logic here
            return mask
        
        return None
    
    def _convert_3d_bboxes(self, bboxes_3d: List[Dict[str, Any]]) -> np.ndarray:
        """
        Convert custom 3D bounding box format to standard format.
        
        Args:
            bboxes_3d: List of custom 3D bounding box definitions
        
        Returns:
            numpy array of shape (N, 7) with [x, y, z, width, height, depth, yaw]
        """
        standard_bboxes = []
        
        for bbox in bboxes_3d:
            # Extract 3D box parameters - adapt to your format
            x = bbox.get("center_x", 0.0)
            y = bbox.get("center_y", 0.0)
            z = bbox.get("center_z", 0.0)
            width = bbox.get("width", 1.0)
            height = bbox.get("height", 1.0)
            depth = bbox.get("depth", 1.0)
            yaw = bbox.get("rotation_y", 0.0)
            
            standard_bboxes.append([x, y, z, width, height, depth, yaw])
        
        return np.array(standard_bboxes) if standard_bboxes else np.empty((0, 7))
    
    def visualize_custom_sample(
        self, 
        sample_name: str, 
        modalities: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Visualize a sample using the custom dataset format.
        
        Args:
            sample_name: Name of the sample to visualize
            modalities: List of visualization modalities to apply
        
        Returns:
            Dictionary mapping modality names to visualization results
        """
        if modalities is None:
            modalities = ["keypoints", "segmentation", "bbox_3d"]
        
        # Load custom annotation
        custom_annotation = self.load_custom_annotation(sample_name)
        
        # Convert to standard format
        standard_annotation = self.convert_to_standard_format(custom_annotation)
        
        # Load image (implement based on your dataset structure)
        image = self._load_image(sample_name)
        if image is None:
            raise ValueError(f"Could not load image for sample: {sample_name}")
        
        results = {}
        
        # Apply requested visualization modalities
        if "keypoints" in modalities and standard_annotation["keypoints"] is not None:
            keypoints = standard_annotation["keypoints"]
            if keypoints.size > 0 and not np.all(keypoints == -1):
                results["keypoints"] = draw_openpose_keypoints(
                    image.copy(), 
                    keypoints,
                    radius=4,
                    thickness=2
                )
        
        if "segmentation" in modalities and standard_annotation["segmentation"] is not None:
            segmentation = standard_annotation["segmentation"]
            results["segmentation"] = draw_segmentation_overlay(
                image.copy(),
                segmentation,
                alpha=0.6
            )
        
        if "bbox_3d" in modalities and len(standard_annotation["bbox_3d"]) > 0:
            # This would require camera intrinsics - simplified here
            bboxes_3d = standard_annotation["bbox_3d"]
            # Create dummy camera matrix for example
            K = np.array([[1000, 0, 512], [0, 1000, 512], [0, 0, 1]])
            results["bbox_3d"] = draw_3d_bboxes(
                image.copy(),
                bboxes_3d,
                K,
                color=(0, 255, 0),
                thickness=2
            )
        
        return results
    
    def _load_image(self, sample_name: str) -> Optional[np.ndarray]:
        """
        Load image for a sample.
        
        Implement this based on your dataset structure.
        """
        # Example implementation - adapt to your format
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = self.dataset_path / "images" / f"{sample_name}{ext}"
            if image_path.exists():
                try:
                    import cv2
                    return cv2.imread(str(image_path))
                except ImportError:
                    print("OpenCV not available for image loading")
                    return None
        
        return None

def create_sample_custom_dataset():
    """
    Create a sample custom dataset structure for demonstration.
    
    This function creates example files to demonstrate the custom adapter.
    """
    dataset_path = Path("sample_custom_dataset")
    dataset_path.mkdir(exist_ok=True)
    
    # Create directory structure
    (dataset_path / "images").mkdir(exist_ok=True)
    (dataset_path / "annotations").mkdir(exist_ok=True)
    
    # Create sample annotation
    sample_annotation = {
        "id": "sample_001",
        "timestamp": "2024-01-01T12:00:00Z",
        "pose_keypoints": [
            200, 100, 0.9,  # nose
            200, 120, 0.8,  # neck
            220, 130, 0.7,  # right shoulder
            240, 150, 0.6,  # right elbow
            260, 170, 0.5,  # right wrist
            # ... add more keypoints as needed
        ],
        "body_parts": {
            "mask": [[1, 1, 2, 2], [1, 1, 2, 2]],  # Simplified mask
            "polygons": {
                "head": [[180, 80], [220, 80], [220, 120], [180, 120]],
                "torso": [[150, 120], [250, 120], [250, 200], [150, 200]]
            }
        },
        "bounding_boxes_3d": [
            {
                "center_x": 200.0,
                "center_y": 200.0,
                "center_z": 0.0,
                "width": 100.0,
                "height": 200.0,
                "depth": 50.0,
                "rotation_y": 0.3
            }
        ]
    }
    
    # Save sample annotation
    with open(dataset_path / "annotations" / "sample_001.json", 'w') as f:
        json.dump(sample_annotation, f, indent=2)
    
    print(f"✅ Sample custom dataset created at: {dataset_path}")
    return str(dataset_path)

def main():
    """Demonstrate custom dataset integration."""
    print("🚀 Custom Dataset Integration Example")
    print("=" * 50)
    
    try:
        # Create sample dataset
        dataset_path = create_sample_custom_dataset()
        
        # Initialize custom adapter
        adapter = CustomDatasetAdapter(dataset_path, format_name="custom_example")
        
        print(f"📁 Custom dataset loaded from: {dataset_path}")
        print("🔧 Available samples: sample_001")
        
        # Visualize sample
        print("\n🎯 Visualizing sample with custom format...")
        results = adapter.visualize_custom_sample(
            "sample_001", 
            modalities=["keypoints", "segmentation", "bbox_3d"]
        )
        
        print(f"✅ Generated visualizations: {list(results.keys())}")
        
        print("\n💡 Key benefits of custom adapter:")
        print("- Integrate any annotation format with the visualization toolkit")
        print("- Maintain consistent API across different dataset types")
        print("- Easy to extend for new modalities and formats")
        print("- Reuse battle-tested visualization functions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This is expected if OpenCV is not available or other dependencies are missing")

if __name__ == "__main__":
    main()
