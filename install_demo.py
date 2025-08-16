#!/usr/bin/env python3
"""
Installation demonstration script for nabla-labs-core.

This script verifies that the package can be imported and used correctly
after installation.
"""

def main():
    """Demonstrate successful package installation."""
    print("🚀 Nabla Labs Core - Installation Verification")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("📦 Testing package imports...")
        from nabla_labs_core import (
            draw_openpose_keypoints,
            draw_segmentation_overlay,
            draw_3d_bboxes,
            DatasetVisualizer,
            OPENPOSE_BODY25_NAMES,
            get_body_part_bgr
        )
        print("✅ All core functions imported successfully!")
        
        # Test constants
        print("\n🔧 Testing constants...")
        print(f"OpenPose keypoints: {len(OPENPOSE_BODY25_NAMES)}")
        print(f"First few keypoints: {OPENPOSE_BODY25_NAMES[:5]}")
        
        # Test utility functions
        print("\n🎨 Testing utility functions...")
        color = get_body_part_bgr(1)
        print(f"Body part 1 color: {color}")
        
        # Test class instantiation
        print("\n🏗️ Testing class instantiation...")
        print("DatasetVisualizer class available for use")
        
        print("\n🎉 Package installation successful!")
        print("\n💡 Next steps:")
        print("1. Check the examples/ directory for usage examples")
        print("2. Read the README.md for detailed documentation")
        print("3. Run 'python examples/basic_usage.py' to see visualizations")
        print("4. Visit the GitHub repository for more information")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Install the package: pip install -e .")
        print("3. Activate your virtual environment if using one")
        print("4. Check that all dependencies are installed")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Please check the error message and ensure proper installation.")

if __name__ == "__main__":
    main()
