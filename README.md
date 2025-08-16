# Nabla Labs Core

A lightweight toolkit for visualizing synthetic datasets with support for multiple annotation formats and visualization modalities.

## 🎯 Purpose

This repository provides essential visualization tools for synthetic datasets, making it easy for researchers and developers to:
- Visualize OpenPose keypoints and skeletal structures
- Display body-part segmentation overlays
- Render 3D bounding boxes and 2D projections
- Support multiple dataset formats (COCO, custom)

## ✨ Features

- **Multi-modal Visualization**: Keypoints, segmentation, bounding boxes
- **Format Agnostic**: Works with COCO annotations and custom formats
- **Professional Rendering**: High-quality visualizations with customizable parameters
- **Lightweight**: Minimal dependencies, focused functionality
- **Extensible**: Easy to integrate with existing pipelines

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nabla-labs/nabla-labs-core.git
cd nabla-labs-core

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from nabla_labs_core import DatasetVisualizer

# Initialize visualizer
visualizer = DatasetVisualizer("path/to/dataset")

# List available samples
samples = visualizer.list_samples()

# Visualize a sample with multiple modalities
visualizer.visualize_sample(
    samples[0], 
    modalities=["openpose", "segmentation", "bboxes"]
)
```

## 📁 Repository Structure

```
nabla-labs-core/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
├── nabla_labs_core/         # Main package
│   ├── __init__.py         # Package initialization
│   ├── constants.py        # Shared constants and utilities
│   ├── primitives.py       # Core visualization primitives
│   └── visualize_dataset.py # Dataset visualization tools
├── examples/                # Usage examples
│   ├── basic_usage.py      # Basic visualization examples
│   └── custom_dataset.py   # Custom dataset integration
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## 🔧 Dependencies

- **Core**: numpy, opencv-python, matplotlib
- **Optional**: pycocotools (for COCO format support)
- **Development**: pytest, black, flake8

## 📖 API Reference

### Core Classes

#### `DatasetVisualizer`
Main class for dataset visualization with support for multiple modalities.

```python
class DatasetVisualizer:
    def __init__(self, dataset_path: str, format_name: str = "coco")
    def list_samples(self) -> List[str]
    def visualize_sample(self, sample_name: str, modalities: List[str])
    def save_visualization(self, sample_name: str, output_path: str)
```

#### `draw_openpose_keypoints`
Render BODY-25 OpenPose keypoints and skeleton onto images.

```python
def draw_openpose_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    pairs: Iterable[Tuple[int, int]] = OPENPOSE_BODY25_PAIRS,
    radius: Optional[int] = None,
    thickness: Optional[int] = None
) -> np.ndarray
```

#### `draw_segmentation_overlay`
Overlay body-part segmentation masks with customizable colors.

```python
def draw_segmentation_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    alpha: float = 0.7,
    color_palette: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray
```

## 🎨 Visualization Examples

### OpenPose Keypoints
![OpenPose Visualization](docs/images/openpose_example.png)

### Body-Part Segmentation
![Segmentation Visualization](docs/images/segmentation_example.png)

### 3D Bounding Boxes
![3D BBox Visualization](docs/images/3d_bbox_example.png)

## 🔄 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/nabla-labs/nabla-labs-core.git
cd nabla-labs-core

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black nabla_labs_core/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/nabla-labs/nabla-labs-core/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nabla-labs/nabla-labs-core/discussions)
- **Documentation**: [Full Documentation](https://nabla-labs-core.readthedocs.io/)

## 🙏 Acknowledgments

- OpenPose team for the BODY-25 keypoint format
- COCO dataset team for annotation format standards
- OpenCV and matplotlib communities for visualization tools

---

**Made with ❤️ by Nabla Labs**
