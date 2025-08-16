#!/usr/bin/env python3
"""
Basic usage examples for nabla-labs-core.

This script demonstrates the core functionality of the visualization toolkit.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# Import the main package
from nabla_labs_core import (
    draw_openpose_keypoints,
    draw_segmentation_overlay,
    draw_3d_bboxes,
    DatasetVisualizer,
    OPENPOSE_BODY25_NAMES,
    get_body_part_bgr,
)

def example_openpose_visualization():
    """Example: Visualize OpenPose keypoints on a synthetic image."""
    print("🎯 Example: OpenPose Keypoint Visualization")
    
    # Create a synthetic image (white background)
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Create synthetic keypoints (BODY-25 format)
    # Format: (x, y) coordinates, -1 for invisible keypoints
    keypoints = np.array([
        [200, 100],  # nose
        [200, 120],  # neck
        [220, 130],  # right shoulder
        [240, 150],  # right elbow
        [260, 170],  # right wrist
        [180, 130],  # left shoulder
        [160, 150],  # left elbow
        [140, 170],  # left wrist
        [200, 200],  # pelvis
        [220, 200],  # right hip
        [220, 250],  # right knee
        [220, 300],  # right ankle
        [180, 200],  # left hip
        [180, 250],  # left knee
        [180, 300],  # left ankle
        [210, 95],   # right eye
        [190, 95],   # left eye
        [215, 90],   # right ear
        [185, 90],   # left ear
        [180, 310],  # left big toe
        [175, 315],  # left small toe
        [180, 320],  # left heel
        [220, 310],  # right big toe
        [225, 315],  # right small toe
        [220, 320],  # right heel
    ])
    
    # Draw keypoints and skeleton
    result = draw_openpose_keypoints(
        image, 
        keypoints,
        radius=5,
        thickness=3
    )
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title("OpenPose BODY-25 Keypoints Visualization")
    plt.axis('off')
    plt.show()
    
    print("✅ OpenPose visualization completed!")

def example_segmentation_visualization():
    """Example: Visualize body-part segmentation overlay."""
    print("🎯 Example: Body-Part Segmentation Visualization")
    
    # Create a synthetic image
    image = np.ones((300, 300, 3), dtype=np.uint8) * 200
    
    # Create synthetic segmentation mask
    # Each pixel value represents a body part ID
    segmentation = np.zeros((300, 300), dtype=np.uint8)
    
    # Draw simple body parts
    # Head (ID: 1)
    cv2.circle(segmentation, (150, 80), 30, 1, -1)
    
    # Torso (ID: 2)
    cv2.rectangle(segmentation, (120, 110), (180, 200), 2, -1)
    
    # Arms (ID: 3, 4)
    cv2.rectangle(segmentation, (80, 120), (110, 180), 3, -1)   # Left arm
    cv2.rectangle(segmentation, (190, 120), (220, 180), 4, -1)  # Right arm
    
    # Legs (ID: 5, 6)
    cv2.rectangle(segmentation, (130, 200), (150, 280), 5, -1)  # Left leg
    cv2.rectangle(segmentation, (150, 200), (170, 280), 6, -1)  # Right leg
    
    # Overlay segmentation on image
    result = draw_segmentation_overlay(
        image, 
        segmentation,
        alpha=0.7
    )
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title("Body-Part Segmentation Visualization")
    plt.axis('off')
    plt.show()
    
    print("✅ Segmentation visualization completed!")

def example_3d_bbox_visualization():
    """Example: Visualize 3D bounding boxes."""
    print("🎯 Example: 3D Bounding Box Visualization")
    
    # Create a synthetic image
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Create synthetic 3D bounding boxes
    # Format: (x, y, z, width, height, depth, yaw)
    bboxes_3d = np.array([
        [200, 200, 0, 100, 200, 50, 0.3],    # Person 1
        [100, 150, 0, 80, 160, 40, -0.2],    # Person 2
    ])
    
    # Camera intrinsic parameters (simplified)
    K = np.array([
        [400, 0, 200],
        [0, 400, 200],
        [0, 0, 1]
    ])
    
    # Draw 3D bounding boxes
    result = draw_3d_bboxes(
        image,
        bboxes_3d,
        K,
        color=(0, 255, 0),
        thickness=2
    )
    
    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result)
    plt.title("3D Bounding Box Visualization")
    plt.axis('off')
    plt.show()
    
    print("✅ 3D bounding box visualization completed!")

def example_dataset_visualization():
    """Example: Using the DatasetVisualizer class."""
    print("🎯 Example: Dataset Visualization")
    
    # This would typically load a real dataset
    # For demonstration, we'll show the expected usage pattern
    
    print("📁 Expected dataset structure:")
    print("dataset/")
    print("├── data/                    # Images")
    print("│   ├── sample1.png")
    print("│   └── sample2.png")
    print("└── annotations/coco_format/ # COCO annotations")
    print("    ├── sample1.json")
    print("    └── sample2.json")
    
    print("\n🔧 Usage pattern:")
    print("visualizer = DatasetVisualizer('path/to/dataset')")
    print("samples = visualizer.list_samples()")
    print("visualizer.visualize_sample(samples[0], modalities=['openpose', 'segmentation'])")
    
    print("\n✅ Dataset visualization example completed!")

def main():
    """Run all examples."""
    print("🚀 Nabla Labs Core - Basic Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_openpose_visualization()
        example_segmentation_visualization()
        example_3d_bbox_visualization()
        example_dataset_visualization()
        
        print("\n🎉 All examples completed successfully!")
        print("\n💡 Tips:")
        print("- Install required dependencies: pip install -r requirements.txt")
        print("- Check the README.md for more detailed documentation")
        print("- Explore the API reference for advanced usage")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the package: pip install -e .")
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
