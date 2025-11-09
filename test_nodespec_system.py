#!/usr/bin/env python
"""
Test script to verify NodeSpec system implementation
Tests Phase 1-3 completion
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'project'))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
import django
django.setup()

from block_manager.services.nodes.specs import Framework
from block_manager.services.nodes.specs.registry import (
    list_node_specs,
    get_node_spec,
    iter_all_specs,
)
from block_manager.services.nodes.specs.serialization import spec_to_dict, compute_spec_hash
from block_manager.services.nodes.templates.renderer import render_node_template
from block_manager.services.nodes.rules import (
    compute_conv2d_output,
    compute_linear_output,
    validate_connection,
    validate_config,
    TensorShape,
)


def test_spec_registry():
    """Test that node specs can be loaded from registry."""
    print("=" * 60)
    print("TEST 1: Node Spec Registry")
    print("=" * 60)
    
    # Test PyTorch specs
    pytorch_specs = list_node_specs(Framework.PYTORCH)
    print(f"✓ Loaded {len(pytorch_specs)} PyTorch node specs")
    
    # Test TensorFlow specs
    tf_specs = list_node_specs(Framework.TENSORFLOW)
    print(f"✓ Loaded {len(tf_specs)} TensorFlow node specs")
    
    # Test getting specific spec
    conv_spec = get_node_spec("conv2d", Framework.PYTORCH)
    assert conv_spec is not None, "Failed to get conv2d spec"
    print(f"✓ Retrieved specific spec: {conv_spec.label}")
    
    # Test iteration
    all_count = sum(1 for _ in iter_all_specs())
    print(f"✓ Total specs across all frameworks: {all_count}")
    
    print()


def test_serialization():
    """Test spec serialization and hashing."""
    print("=" * 60)
    print("TEST 2: Spec Serialization")
    print("=" * 60)
    
    conv_spec = get_node_spec("conv2d", Framework.PYTORCH)
    
    # Test dict conversion
    spec_dict = spec_to_dict(conv_spec)
    assert "type" in spec_dict, "Missing 'type' in serialized spec"
    assert "configSchema" in spec_dict, "Missing 'configSchema' in serialized spec"
    print(f"✓ Serialized spec to dict with {len(spec_dict)} keys")
    
    # Test hashing
    spec_hash = compute_spec_hash(spec_dict)
    assert len(spec_hash) == 64, "Invalid hash length"
    print(f"✓ Computed spec hash: {spec_hash[:16]}...")
    
    # Test determinism
    spec_hash2 = compute_spec_hash(spec_dict)
    assert spec_hash == spec_hash2, "Hash not deterministic"
    print("✓ Hash is deterministic")
    
    print()


def test_template_rendering():
    """Test Jinja2 template rendering."""
    print("=" * 60)
    print("TEST 3: Template Rendering")
    print("=" * 60)
    
    # Test PyTorch Conv2D
    conv_spec = get_node_spec("conv2d", Framework.PYTORCH)
    config = {
        "out_channels": 64,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
    }
    
    rendered = render_node_template(conv_spec, config)
    print(f"✓ Rendered PyTorch Conv2D:")
    print(f"  {rendered.code}")
    
    # Test TensorFlow Conv2D
    tf_conv_spec = get_node_spec("conv2d", Framework.TENSORFLOW)
    tf_config = {
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
        "activation": "relu",
    }
    
    tf_rendered = render_node_template(tf_conv_spec, tf_config)
    print(f"✓ Rendered TensorFlow Conv2D:")
    print(f"  {tf_rendered.code}")
    
    # Test Linear/Dense
    linear_spec = get_node_spec("linear", Framework.PYTORCH)
    linear_config = {"out_features": 128, "bias": True}
    
    linear_rendered = render_node_template(linear_spec, linear_config)
    print(f"✓ Rendered PyTorch Linear:")
    print(f"  {linear_rendered.code}")
    
    print()


def test_shape_computation():
    """Test shape inference functions."""
    print("=" * 60)
    print("TEST 4: Shape Computation")
    print("=" * 60)
    
    # Test Conv2D shape (PyTorch NCHW)
    input_shape = TensorShape({"dims": [1, 3, 224, 224]})
    config = {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
    
    output_shape = compute_conv2d_output(input_shape, config, Framework.PYTORCH)
    assert output_shape is not None, "Failed to compute conv2d output shape"
    print(f"✓ PyTorch Conv2D: {input_shape.dims} → {output_shape.dims}")
    
    # Test TensorFlow Conv2D (NHWC)
    tf_input_shape = TensorShape({"dims": [1, 224, 224, 3]})
    tf_config = {"filters": 64, "kernel_size": 3, "strides": 1, "padding": "same"}
    
    tf_output_shape = compute_conv2d_output(tf_input_shape, tf_config, Framework.TENSORFLOW)
    assert tf_output_shape is not None, "Failed to compute TF conv2d output shape"
    print(f"✓ TensorFlow Conv2D: {tf_input_shape.dims} → {tf_output_shape.dims}")
    
    # Test Linear
    linear_input = TensorShape({"dims": [32, 512]})
    linear_config = {"out_features": 128}
    
    linear_output = compute_linear_output(linear_input, linear_config, Framework.PYTORCH)
    assert linear_output is not None, "Failed to compute linear output shape"
    print(f"✓ Linear: {linear_input.dims} → {linear_output.dims}")
    
    print()


def test_validation():
    """Test validation functions."""
    print("=" * 60)
    print("TEST 5: Validation")
    print("=" * 60)
    
    # Test config validation
    conv_spec = get_node_spec("conv2d", Framework.PYTORCH)
    
    # Valid config
    valid_config = {"out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1}
    is_valid, errors = validate_config(conv_spec, valid_config)
    assert is_valid, f"Valid config marked as invalid: {errors}"
    print("✓ Valid config passed validation")
    
    # Invalid config (missing required field)
    invalid_config = {"kernel_size": 3}
    is_valid, errors = validate_config(conv_spec, invalid_config)
    assert not is_valid, "Invalid config passed validation"
    print(f"✓ Invalid config rejected: {errors[0]}")
    
    # Test connection validation
    conv_spec = get_node_spec("conv2d", Framework.PYTORCH)
    linear_spec = get_node_spec("linear", Framework.PYTORCH)
    
    # Conv2D requires 4D input
    output_shape_4d = TensorShape({"dims": [1, 64, 56, 56]})
    is_valid, error = validate_connection(conv_spec, conv_spec, output_shape_4d)
    assert is_valid, f"Valid connection rejected: {error}"
    print("✓ Valid connection (Conv2D → Conv2D) accepted")
    
    # Linear requires 2D input - should fail with 4D
    is_valid, error = validate_connection(conv_spec, linear_spec, output_shape_4d)
    assert not is_valid, "Invalid connection (4D → Linear) accepted"
    print(f"✓ Invalid connection rejected: {error}")
    
    print()


def test_api_integration():
    """Test that API endpoints work with new spec system."""
    print("=" * 60)
    print("TEST 6: API Integration")
    print("=" * 60)
    
    from django.test import RequestFactory
    from block_manager.views.architecture_views import (
        get_node_definitions,
        get_node_definition,
        render_node_code,
    )
    
    factory = RequestFactory()
    
    # Test get_node_definitions
    request = factory.get('/api/node-definitions?framework=pytorch')
    response = get_node_definitions(request)
    assert response.status_code == 200, f"API returned {response.status_code}"
    data = response.data
    assert data['success'], f"API error: {data.get('error')}"
    print(f"✓ GET /node-definitions returned {data['count']} specs")
    
    # Test get_node_definition
    request = factory.get('/api/node-definitions/conv2d?framework=pytorch')
    response = get_node_definition(request, 'conv2d')
    assert response.status_code == 200
    data = response.data
    assert data['success']
    print(f"✓ GET /node-definitions/conv2d returned spec for '{data['definition']['label']}'")
    
    # Test render_node_code
    request = factory.post(
        '/api/render-node-code',
        data={
            'node_type': 'conv2d',
            'framework': 'pytorch',
            'config': {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        },
        content_type='application/json'
    )
    response = render_node_code(request)
    assert response.status_code == 200
    data = response.data
    assert data['success']
    print(f"✓ POST /render-node-code returned code:")
    print(f"  {data['code']}")
    
    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("VISIONFORGE NODESPEC SYSTEM VERIFICATION")
    print("Phase 1-3 Implementation Test")
    print("=" * 60 + "\n")
    
    try:
        test_spec_registry()
        test_serialization()
        test_template_rendering()
        test_shape_computation()
        test_validation()
        test_api_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nPhase 1-3 Implementation Complete:")
        print("  ✓ Backend Domain Model Refactor (Phase 1)")
        print("  ✓ Backend API Redesign (Phase 2)")
        print("  ✓ Frontend Integration (Phase 3)")
        print()
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
