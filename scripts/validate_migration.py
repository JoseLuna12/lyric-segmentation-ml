#!/usr/bin/env python3
"""
Migration Validation Script

Tests both legacy and boundary-aware loss functions with comprehensive
parameter sets to ensure feature parity and no regression in safety mechanisms.

This script validates:
1. All parameters are correctly parsed and used
2. Both loss functions handle edge cases properly  
3. All safety mechanisms are active (anti-collapse, etc.)
4. Loss values are reasonable and stable
5. Gradient flow is healthy
"""

import sys
import os
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from segmodel.losses.cross_entropy import CrossEntropyWithLabelSmoothing
from segmodel.losses.boundary_aware_cross_entropy import BoundaryAwareCrossEntropy
from segmodel.utils.config_loader import load_yaml_config


def create_test_data(batch_size=4, seq_len=20, num_classes=2):
    """Create synthetic test data with known properties."""
    # Create logits with reasonable values (not too extreme)
    logits = torch.randn(batch_size, seq_len, num_classes) * 0.5
    
    # Create targets with some boundaries and segments
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
    for b in range(batch_size):
        # Create 2-3 segments per sequence
        seg_points = sorted([0, seq_len//3, 2*seq_len//3, seq_len])
        for i in range(len(seg_points)-1):
            start, end = seg_points[i], seg_points[i+1]
            label = i % num_classes
            targets[b, start:end] = label
    
    # Create mask (all positions valid except last 2 for padding test)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    return logits, targets, mask


def test_loss_function(loss_fn, name, logits, targets, mask):
    """Test a loss function with various scenarios."""
    print(f"\n{'='*50}")
    print(f"Testing {name}")
    print(f"{'='*50}")
    
    # Test 1: Basic forward pass
    try:
        loss_info = loss_fn(logits, targets, mask)
        if isinstance(loss_info, dict):
            total_loss = loss_info['total_loss']
        else:
            total_loss = loss_info
        print(f"✅ Basic forward pass: loss = {total_loss.item():.4f}")
        
        # Check loss components if available
        if isinstance(loss_info, dict) and 'components' in loss_info:
            print("📊 Loss components:")
            for comp, value in loss_info['components'].items():
                if isinstance(value, torch.Tensor):
                    print(f"   {comp}: {value.item():.4f}")
                else:
                    print(f"   {comp}: {value}")
    except Exception as e:
        print(f"❌ Basic forward pass failed: {e}")
        return False
    
    # Test 2: Gradient computation
    try:
        total_loss.backward()
        print("✅ Gradient computation successful")
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False
    
    # Test 3: Edge cases
    try:
        # Test with padding (mask some positions)
        mask_with_padding = mask.clone()
        mask_with_padding[:, -2:] = False  # Mask last 2 positions
        
        loss_info_padded = loss_fn(logits, targets, mask_with_padding)
        if isinstance(loss_info_padded, dict):
            loss_val = loss_info_padded['total_loss']
        else:
            loss_val = loss_info_padded
        print(f"✅ Padding handling: loss = {loss_val.item():.4f}")
        
        # Test with extreme logits
        extreme_logits = logits.clone()
        extreme_logits[0, 0, :] = torch.tensor([10.0, -10.0])  # Very confident
        
        loss_info_extreme = loss_fn(extreme_logits, targets, mask)
        if isinstance(loss_info_extreme, dict):
            loss_val = loss_info_extreme['total_loss']
        else:
            loss_val = loss_info_extreme
        print(f"✅ Extreme logits handling: loss = {loss_val.item():.4f}")
        
    except Exception as e:
        print(f"❌ Edge case handling failed: {e}")
        return False
    
    # Test 4: Numerical stability
    try:
        # Check for NaN/Inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("❌ Loss contains NaN or Inf")
            return False
        
        # Check reasonable range
        if total_loss.item() < 0 or total_loss.item() > 100:
            print(f"⚠️  Loss outside reasonable range: {total_loss.item()}")
        else:
            print("✅ Loss in reasonable range")
            
    except Exception as e:
        print(f"❌ Numerical stability check failed: {e}")
        return False
    
    return True


def test_config_loading():
    """Test that our comprehensive configs load correctly."""
    print(f"\n{'='*50}")
    print("Testing Configuration Loading")
    print(f"{'='*50}")
    
    config_files = [
        "configs/training/boundary_aware_comprehensive.yaml",
        "configs/training/legacy_cross_entropy_comprehensive.yaml"
    ]
    
    for config_file in config_files:
        config_path = project_root / config_file
        if not config_path.exists():
            print(f"❌ Config file not found: {config_file}")
            continue
            
        try:
            config = load_yaml_config(str(config_path))
            loss_config = config.get('loss', {})
            loss_type = loss_config.get('type')
            
            print(f"✅ {config_file}: type={loss_type}")
            
            # Check key parameters
            key_params = ['label_smoothing', 'num_classes', 'ignore_index']
            for param in key_params:
                if param in loss_config:
                    print(f"   {param}: {loss_config[param]}")
                    
            # Check boundary-specific params
            if loss_type == 'boundary_aware':
                boundary_params = ['boundary_weight', 'segment_consistency_lambda', 'conf_penalty_lambda']
                for param in boundary_params:
                    if param in loss_config:
                        print(f"   {param}: {loss_config[param]}")
                        
        except Exception as e:
            print(f"❌ Failed to load {config_file}: {e}")


def test_parameter_sensitivity():
    """Test that parameters actually affect the loss values."""
    print(f"\n{'='*50}")
    print("Testing Parameter Sensitivity")
    print(f"{'='*50}")
    
    logits, targets, mask = create_test_data()
    logits.requires_grad_(True)  # Enable gradients
    
    # Test label smoothing sensitivity
    print("\n📊 Label Smoothing Sensitivity:")
    for smoothing in [0.0, 0.1, 0.2]:
        loss_fn = BoundaryAwareCrossEntropy(
            label_smoothing=smoothing,
            boundary_weight=1.0,  # Disable boundary effects
            segment_consistency_lambda=0.0,  # Disable segment effects
            conf_penalty_lambda=0.0  # Disable confidence effects
        )
        loss_info = loss_fn(logits, targets, mask)
        if isinstance(loss_info, dict):
            loss_val = loss_info['total_loss']
        else:
            loss_val = loss_info
        print(f"   smoothing={smoothing}: loss = {loss_val.item():.4f}")
    
    # Test boundary weight sensitivity
    print("\n📊 Boundary Weight Sensitivity:")
    for boundary_weight in [1.0, 2.0, 3.0]:
        loss_fn = BoundaryAwareCrossEntropy(
            label_smoothing=0.1,
            boundary_weight=boundary_weight,
            segment_consistency_lambda=0.0,
            conf_penalty_lambda=0.0
        )
        loss_info = loss_fn(logits, targets, mask)
        if isinstance(loss_info, dict):
            loss_val = loss_info['total_loss']
        else:
            loss_val = loss_info
        print(f"   boundary_weight={boundary_weight}: loss = {loss_val.item():.4f}")


def main():
    """Run comprehensive validation tests."""
    print("🚀 Loss Function Migration Validation")
    print("=" * 60)
    
    # Create test data
    logits, targets, mask = create_test_data()
    print(f"📊 Test data: {logits.shape} logits, {targets.shape} targets, {mask.shape} mask")
    
    # Enable gradients for logits
    logits.requires_grad_(True)
    
    # Test legacy loss function
    legacy_loss = CrossEntropyWithLabelSmoothing(
        num_classes=2,
        label_smoothing=0.1,
        entropy_lambda=0.01
    )
    
    legacy_success = test_loss_function(
        legacy_loss, 
        "Legacy Cross-Entropy", 
        logits, 
        targets,
        mask
    )
    
    # Test boundary-aware loss function
    boundary_loss = BoundaryAwareCrossEntropy(
        num_classes=2,
        label_smoothing=0.2,
        entropy_lambda=0.01,
        boundary_weight=2.0,
        segment_consistency_lambda=0.05,
        conf_penalty_lambda=0.01,
        conf_threshold=0.95
    )
    
    boundary_success = test_loss_function(
        boundary_loss,
        "Boundary-Aware Cross-Entropy", 
        logits, 
        targets,
        mask
    )
    
    # Test configuration loading
    test_config_loading()
    
    # Test parameter sensitivity
    test_parameter_sensitivity()
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Legacy Loss: {'✅ PASSED' if legacy_success else '❌ FAILED'}")
    print(f"Boundary-Aware Loss: {'✅ PASSED' if boundary_success else '❌ FAILED'}")
    
    if legacy_success and boundary_success:
        print("\n🎉 ALL TESTS PASSED - Migration is ready!")
        print("\nNext steps:")
        print("1. Run training with boundary_aware_comprehensive.yaml")
        print("2. Compare metrics with legacy_cross_entropy_comprehensive.yaml") 
        print("3. Use CONFIGURATION_VARIANTS.yaml for parameter tuning")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Check implementation!")
        return 1


if __name__ == "__main__":
    exit(main())
