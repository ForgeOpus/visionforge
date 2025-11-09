from __future__ import annotations

from typing import Dict, Optional

from ..specs import Framework


class TensorShape(dict):
    """Lightweight tensor shape representation for shape inference."""

    @property
    def dims(self):
        return self.get("dims", [])


def compute_conv2d_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    if not input_shape:
        return None

    dims = input_shape.get("dims", [])
    if len(dims) != 4:
        return None

    if framework is Framework.PYTORCH:
        batch, channels, height, width = dims
        kernel_size = int(config.get("kernel_size", 3))
        stride = int(config.get("stride", 1))
        padding = int(config.get("padding", 0))
        dilation = int(config.get("dilation", 1))

        out_channels_value = config.get("out_channels")
        out_channels = int(out_channels_value) if out_channels_value is not None else int(channels)

        effective_kernel = dilation * (kernel_size - 1) + 1
        out_height = (height + 2 * padding - effective_kernel) // stride + 1
        out_width = (width + 2 * padding - effective_kernel) // stride + 1

        return TensorShape({
            "dims": [int(batch), out_channels, out_height, out_width],
            "description": "Convolved feature map",
        })

    batch, height, width, channels = dims
    kernel_size = int(config.get("kernel_size", 3))
    stride = int(config.get("strides", 1))
    padding = config.get("padding", "valid")

    filters_value = config.get("filters")
    out_filters = int(filters_value) if filters_value is not None else int(channels)

    if padding == "same":
        out_height = (height + stride - 1) // stride
        out_width = (width + stride - 1) // stride
    else:
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1

    return TensorShape({
        "dims": [int(batch), out_height, out_width, out_filters],
        "description": "Convolved feature map (NHWC)",
    })


def compute_linear_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for Linear/Dense layers."""
    if not input_shape:
        return None

    dims = input_shape.get("dims", [])
    if len(dims) != 2:
        return None

    batch = dims[0]
    
    if framework is Framework.PYTORCH:
        out_features_value = config.get("out_features")
        if out_features_value is None:
            return None
        out_features = int(out_features_value)
    else:  # TensorFlow
        units_value = config.get("units")
        if units_value is None:
            return None
        out_features = int(units_value)

    return TensorShape({
        "dims": [int(batch), out_features],
        "description": "Linear transformation output",
    })


def compute_flatten_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for Flatten layers."""
    if not input_shape:
        return None

    dims = input_shape.get("dims", [])
    if len(dims) < 2:
        return None

    batch = dims[0]
    
    if framework is Framework.PYTORCH:
        # NCHW format
        features = 1
        for dim in dims[1:]:
            features *= int(dim)
    else:  # TensorFlow
        # NHWC format
        features = 1
        for dim in dims[1:]:
            features *= int(dim)

    return TensorShape({
        "dims": [int(batch), features],
        "description": "Flattened tensor",
    })


def compute_maxpool_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for MaxPool layers."""
    if not input_shape:
        return None

    dims = input_shape.get("dims", [])
    if len(dims) != 4:
        return None

    if framework is Framework.PYTORCH:
        batch, channels, height, width = dims
        kernel_size = int(config.get("kernel_size", 2))
        stride_value = config.get("stride")
        stride = int(stride_value) if stride_value is not None else kernel_size
        padding = int(config.get("padding", 0))

        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1

        return TensorShape({
            "dims": [int(batch), int(channels), out_height, out_width],
            "description": "Pooled feature map",
        })
    else:  # TensorFlow
        batch, height, width, channels = dims
        pool_size = int(config.get("pool_size", 2))
        stride_value = config.get("strides")
        strides = int(stride_value) if stride_value is not None else pool_size
        padding = config.get("padding", "valid")

        if padding == "same":
            out_height = (height + strides - 1) // strides
            out_width = (width + strides - 1) // strides
        else:  # valid
            out_height = (height - pool_size) // strides + 1
            out_width = (width - pool_size) // strides + 1

        return TensorShape({
            "dims": [int(batch), out_height, out_width, int(channels)],
            "description": "Pooled feature map (NHWC)",
        })


def compute_concat_output(
    input_shapes: list[Optional[TensorShape]],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for Concatenate layers."""
    if not input_shapes or len(input_shapes) < 2:
        return None

    # Filter out None shapes
    valid_shapes = [s for s in input_shapes if s is not None]
    if len(valid_shapes) < 2:
        return None

    first_dims = valid_shapes[0].get("dims", [])
    if not first_dims:
        return None

    axis = int(config.get("axis", -1))
    ndim = len(first_dims)
    
    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis

    # Verify all shapes match except at concat axis
    for shape in valid_shapes[1:]:
        dims = shape.get("dims", [])
        if len(dims) != ndim:
            return None
        for i in range(ndim):
            if i != axis and dims[i] != first_dims[i]:
                return None

    # Compute concatenated dimension
    concat_dim = sum(int(s.get("dims", [])[axis]) for s in valid_shapes)
    
    result_dims = list(first_dims)
    result_dims[axis] = concat_dim

    return TensorShape({
        "dims": result_dims,
        "description": f"Concatenated at axis {axis}",
    })


def compute_add_output(
    input_shapes: list[Optional[TensorShape]],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for Add (element-wise addition) layers."""
    if not input_shapes or len(input_shapes) < 2:
        return None

    valid_shapes = [s for s in input_shapes if s is not None]
    if len(valid_shapes) < 2:
        return None

    # All shapes must match exactly for element-wise addition
    first_dims = valid_shapes[0].get("dims", [])
    for shape in valid_shapes[1:]:
        dims = shape.get("dims", [])
        if dims != first_dims:
            return None

    return TensorShape({
        "dims": first_dims,
        "description": "Element-wise sum",
    })


def compute_batchnorm_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for BatchNorm layers (preserves input shape)."""
    if not input_shape:
        return None

    return TensorShape({
        "dims": input_shape.get("dims", []),
        "description": "Batch normalized",
    })


def compute_dropout_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for Dropout layers (preserves input shape)."""
    if not input_shape:
        return None

    return TensorShape({
        "dims": input_shape.get("dims", []),
        "description": "Dropout applied",
    })


def compute_activation_output(
    input_shape: Optional[TensorShape],
    config: Dict[str, int],
    framework: Framework,
) -> Optional[TensorShape]:
    """Compute output shape for activation layers (preserves input shape)."""
    if not input_shape:
        return None

    return TensorShape({
        "dims": input_shape.get("dims", []),
        "description": "Activation applied",
    })
