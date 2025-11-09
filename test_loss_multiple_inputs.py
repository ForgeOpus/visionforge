"""
Test the Loss Node with Multiple Inputs

This test verifies that the Loss node spec has been updated to support
multiple named input ports based on the loss function type.
"""

from block_manager.services.nodes.specs.pytorch import LOSS_SPEC
from block_manager.services.nodes.specs.models import InputPortSpec


def test_loss_spec_has_input_ports():
    """Test that LOSS_SPEC has input_ports defined"""
    print("Testing Loss Spec Configuration...\n")
    
    # Check that input_ports exist
    assert LOSS_SPEC.input_ports is not None, "LOSS_SPEC should have input_ports defined"
    print(f"✓ Loss spec has {len(LOSS_SPEC.input_ports)} default input ports")
    
    # Check default input ports
    for port in LOSS_SPEC.input_ports:
        assert isinstance(port, InputPortSpec), f"Port {port} should be InputPortSpec instance"
        print(f"  - {port.label} ({port.id}): {port.description}")
    
    print()


def test_loss_spec_allows_multiple_inputs():
    """Test that LOSS_SPEC allows multiple inputs"""
    print("Testing Multiple Inputs Support...\n")
    
    assert LOSS_SPEC.allows_multiple_inputs is True, "LOSS_SPEC should allow multiple inputs"
    print("✓ Loss spec allows multiple inputs")
    print()


def test_loss_spec_has_port_configs():
    """Test that LOSS_SPEC has input port configurations in metadata"""
    print("Testing Input Port Configurations...\n")
    
    assert "input_ports_config" in LOSS_SPEC.metadata, "LOSS_SPEC should have input_ports_config in metadata"
    
    port_configs = LOSS_SPEC.metadata["input_ports_config"]
    
    # Test standard 2-input losses
    standard_losses = ["cross_entropy", "mse", "mae", "bce", "nll", "kl_div"]
    for loss_type in standard_losses:
        assert loss_type in port_configs, f"{loss_type} should be in port_configs"
        ports = port_configs[loss_type]
        assert len(ports) == 2, f"{loss_type} should have 2 input ports"
        print(f"✓ {loss_type:15s}: {len(ports)} inputs - {', '.join(ports)}")
    
    # Test triplet loss (3 inputs)
    assert "triplet" in port_configs, "triplet should be in port_configs"
    triplet_ports = port_configs["triplet"]
    assert len(triplet_ports) == 3, "triplet should have 3 input ports"
    assert triplet_ports == ["anchor", "positive", "negative"], "triplet should have anchor, positive, negative"
    print(f"✓ {'triplet':15s}: {len(triplet_ports)} inputs - {', '.join(triplet_ports)}")
    
    # Test contrastive loss (3 inputs)
    assert "contrastive" in port_configs, "contrastive should be in port_configs"
    contrastive_ports = port_configs["contrastive"]
    assert len(contrastive_ports) == 3, "contrastive should have 3 input ports"
    assert contrastive_ports == ["input1", "input2", "label"], "contrastive should have input1, input2, label"
    print(f"✓ {'contrastive':15s}: {len(contrastive_ports)} inputs - {', '.join(contrastive_ports)}")
    
    print()


def test_loss_spec_config_options():
    """Test that LOSS_SPEC has all loss type options"""
    print("Testing Loss Type Options...\n")
    
    # Get loss_type config field
    loss_type_field = None
    for field in LOSS_SPEC.config_schema:
        if field.name == "loss_type":
            loss_type_field = field
            break
    
    assert loss_type_field is not None, "loss_type config field should exist"
    
    # Check that we have the expected loss types
    expected_types = ["cross_entropy", "mse", "mae", "bce", "triplet", "contrastive", "nll", "kl_div"]
    option_values = [opt.value for opt in loss_type_field.options]
    
    for loss_type in expected_types:
        assert loss_type in option_values, f"{loss_type} should be in loss_type options"
        print(f"✓ {loss_type} is available")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("LOSS NODE MULTIPLE INPUTS TEST")
    print("=" * 60)
    print()
    
    try:
        test_loss_spec_has_input_ports()
        test_loss_spec_allows_multiple_inputs()
        test_loss_spec_has_port_configs()
        test_loss_spec_config_options()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print()
        print("Summary:")
        print("- Loss node supports multiple named input ports")
        print("- Different loss functions have appropriate input configurations")
        print("- Standard losses (MSE, Cross Entropy, etc.) have 2 inputs")
        print("- Triplet and Contrastive losses have 3 inputs")
        print("- System is extensible for new loss types")
        
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        raise
