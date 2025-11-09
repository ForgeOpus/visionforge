"""
TensorFlow node definitions package.
Currently mirrors PyTorch implementations.
Future: Implement TensorFlow-specific behaviors where they diverge.
"""

# Re-export PyTorch nodes for now (they work identically)
# As TensorFlow-specific requirements emerge, replace with separate implementations

from ..pytorch.linear import LinearNode
from ..pytorch.conv2d import Conv2DNode

__all__ = [
    'LinearNode',
    'Conv2DNode',
]

# Note: These are currently identical to PyTorch definitions.
# Create separate implementations in this directory when TensorFlow-specific
# logic is needed.
