"""
Verification script to test PyTorch node implementations
Run this to verify all 17 PyTorch nodes are properly registered
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from block_manager.services.nodes.registry import NodeRegistry
from block_manager.services.nodes.base import Framework


def verify_pytorch_nodes():
    """Verify all PyTorch nodes are registered and functional"""
    
    registry = NodeRegistry()
    
    expected_nodes = [
        'input',
        'dataloader',
        'linear',
        'conv2d',
        'conv1d',
        'conv3d',
        'flatten',
        'dropout',
        'batchnorm2d',
        'maxpool2d',
        'avgpool2d',
        'adaptiveavgpool2d',
        'lstm',
        'gru',
        'embedding',
        'concat',
        'add',
    ]
    
    print("=" * 60)
    print("PyTorch Node Registry Verification")
    print("=" * 60)
    
    all_nodes = registry.get_all_node_definitions(Framework.PYTORCH)
    print(f"\nTotal nodes registered: {len(all_nodes)}")
    print(f"Expected nodes: {len(expected_nodes)}")
    
    print("\n" + "-" * 60)
    print("Registered Nodes:")
    print("-" * 60)
    
    registered_types = []
    for node in all_nodes:
        metadata = node.metadata
        registered_types.append(metadata.type)
        print(f"✓ {metadata.type:20s} | {metadata.label:20s} | {metadata.category}")
    
    print("\n" + "-" * 60)
    print("Missing Nodes:")
    print("-" * 60)
    
    missing = set(expected_nodes) - set(registered_types)
    if missing:
        for node_type in missing:
            print(f"✗ {node_type}")
    else:
        print("✓ All expected nodes are registered!")
    
    print("\n" + "-" * 60)
    print("Extra Nodes:")
    print("-" * 60)
    
    extra = set(registered_types) - set(expected_nodes)
    if extra:
        for node_type in extra:
            print(f"+ {node_type}")
    else:
        print("No unexpected nodes found.")
    
    print("\n" + "=" * 60)
    print("Testing Node Functionality")
    print("=" * 60)
    
    # Test a few nodes
    test_nodes = ['linear', 'conv2d', 'flatten', 'lstm', 'concat']
    
    for node_type in test_nodes:
        node = registry.get_node_definition(node_type, Framework.PYTORCH)
        if node:
            print(f"\n✓ {node_type}:")
            print(f"  - Config fields: {len(node.config_schema)}")
            print(f"  - Category: {node.metadata.category}")
            print(f"  - Description: {node.metadata.description}")
            
            # Test shape computation with dummy config
            if node_type == 'linear':
                from block_manager.services.nodes.base import TensorShape
                test_shape = TensorShape(dims=[32, 128], description="Test")
                output = node.compute_output_shape(test_shape, {"out_features": 64})
                if output:
                    print(f"  - Shape computation: [32, 128] -> {output.dims}")
        else:
            print(f"\n✗ {node_type}: NOT FOUND")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    success = len(missing) == 0
    print(f"Status: {'✓ PASS' if success else '✗ FAIL'}")
    print(f"Registered: {len(all_nodes)}/{len(expected_nodes)}")
    
    return success


if __name__ == "__main__":
    success = verify_pytorch_nodes()
    sys.exit(0 if success else 1)
