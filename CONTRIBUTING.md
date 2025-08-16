# Contributing to Nabla Labs Core

Thank you for your interest in contributing to Nabla Labs Core! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

We welcome contributions from the community! Here are several ways you can help:

### 🐛 Report Bugs
- Use the [GitHub Issues](https://github.com/nabla-labs/nabla-labs-core/issues) page
- Provide a clear description of the bug
- Include steps to reproduce, expected vs actual behavior
- Add relevant error messages and system information

### 💡 Suggest Enhancements
- Open a [GitHub Discussion](https://github.com/nabla-labs/nabla-labs-core/discussions)
- Describe the feature you'd like to see
- Explain why it would be useful
- Consider implementation complexity

### 🔧 Submit Code Changes
- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Submit a pull request

## 🚀 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- pip

### Local Development
```bash
# Clone the repository
git clone https://github.com/nabla-labs/nabla-labs-core.git
cd nabla-labs-core

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .
```

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=nabla_labs_core

# Run specific test file
pytest tests/test_constants.py
```

### Code Quality
```bash
# Format code with Black
black nabla_labs_core/ tests/ examples/

# Sort imports
isort nabla_labs_core/ tests/ examples/

# Lint code
flake8 nabla_labs_core/ tests/ examples/

# Type checking
mypy nabla_labs_core/
```

## 📝 Code Style Guidelines

### Python Code
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Use meaningful variable and function names

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions and classes
- Include usage examples in docstrings
- Update examples/ directory for new features

### Testing
- Write tests for new functionality
- Aim for high test coverage
- Use descriptive test names
- Test both success and error cases
- Mock external dependencies when appropriate

## 🔄 Pull Request Process

### Before Submitting
1. Ensure all tests pass locally
2. Run code quality checks
3. Update documentation if needed
4. Add tests for new functionality

### Pull Request Guidelines
1. **Title**: Clear, descriptive title
2. **Description**: Explain what the PR does and why
3. **Related Issues**: Link to relevant issues
4. **Testing**: Describe how you tested the changes
5. **Breaking Changes**: Note any breaking changes

### Review Process
- All PRs require at least one review
- Address review comments promptly
- Maintainers may request changes before merging
- Squash commits before merging (if requested)

## 🏗️ Project Structure

```
nabla-labs-core/
├── nabla_labs_core/          # Main package
│   ├── __init__.py          # Package exports
│   ├── constants.py         # Shared constants
│   ├── primitives.py        # Core visualization functions
│   └── visualize_dataset.py # Dataset visualization tools
├── examples/                 # Usage examples
├── tests/                    # Test suite
├── docs/                     # Documentation
└── requirements*.txt         # Dependencies
```

## 🎯 Areas for Contribution

### High Priority
- Additional dataset format support
- Performance optimizations
- Enhanced visualization options
- Better error handling and validation

### Medium Priority
- Additional visualization modalities
- Export functionality (PDF, video)
- Interactive visualization tools
- Documentation improvements

### Low Priority
- Additional color schemes
- Custom keypoint formats
- Platform-specific optimizations

## 📚 Resources

### Documentation
- [README.md](README.md) - Project overview and usage
- [API Reference](https://nabla-labs-core.readthedocs.io/) - Detailed API documentation
- [Examples](examples/) - Usage examples and tutorials

### External Resources
- [OpenPose Documentation](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [NumPy Documentation](https://numpy.org/doc/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🆘 Getting Help

### Questions and Discussion
- [GitHub Discussions](https://github.com/nabla-labs/nabla-labs-core/discussions)
- [GitHub Issues](https://github.com/nabla-labs/nabla-labs-core/issues)

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the [Contributor Covenant](https://www.contributor-covenant.org/)

## 🙏 Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes
- Documentation acknowledgments
- Project website (if applicable)

---

Thank you for contributing to Nabla Labs Core! 🎉
