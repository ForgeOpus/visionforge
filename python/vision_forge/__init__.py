"""
VisionForge - Visual Neural Network Builder

A local desktop application for building deep learning architectures visually.
Export production-ready PyTorch or TensorFlow code with one click.
"""

__version__ = "0.1.0"
__author__ = "ForgeOpus"
__license__ = "BSD-3-Clause"

from .server import create_app

__all__ = ["create_app", "__version__"]
