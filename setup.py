#!/usr/bin/env python3
"""
Setup script for nabla-labs-core package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="nabla-labs-core",
    version="0.1.0",
    author="Nabla Labs",
    author_email="contact@nabla-labs.com",
    description="A lightweight toolkit for visualizing synthetic human datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nabla-labs/nabla-labs-core",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "computer-vision",
        "dataset-visualization",
        "openpose",
        "segmentation",
        "synthetic-data",
        "human-pose",
        "keypoints",
    ],
    project_urls={
        "Bug Reports": "https://github.com/nabla-labs/nabla-labs-core/issues",
        "Source": "https://github.com/nabla-labs/nabla-labs-core",
        "Documentation": "https://nabla-labs-core.readthedocs.io/",
    },
)
