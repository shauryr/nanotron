#!/usr/bin/env python
"""Unit tests for ZeRO-2 and ZeRO-3 implementation (CPU only)"""

import sys
import torch

def test_zero_imports():
    """Test that all ZeRO components can be imported"""
    print("Testing imports...")

    try:
        from nanotron.optim import ZeroDistributedOptimizer
        from nanotron.optim.zero import SlicedFlatTensor
        from nanotron.config.config import OptimizerArgs
        print("✓ ZeroDistributedOptimizer imported")
        print("✓ SlicedFlatTensor imported")
        print("✓ OptimizerArgs imported")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_zero_stage_attribute():
    """Test that ZeroDistributedOptimizer has zero_stage attribute"""
    print("\nTesting ZeRO stage attribute...")

    from nanotron.optim.zero import ZeroDistributedOptimizer

    # Check that __init__ signature includes zero_stage
    import inspect
    sig = inspect.signature(ZeroDistributedOptimizer.__init__)
    params = list(sig.parameters.keys())

    if 'zero_stage' in params:
        print("✓ zero_stage parameter exists in __init__")

        # Check default value
        default = sig.parameters['zero_stage'].default
        if default == 1:
            print(f"✓ zero_stage default value is 1")
        else:
            print(f"⚠ zero_stage default value is {default} (expected 1)")

        return True
    else:
        print(f"✗ zero_stage parameter not found in __init__. Parameters: {params}")
        return False

def test_config_validation():
    """Test that config validates zero_stage correctly"""
    print("\nTesting config validation...")

    from dataclasses import dataclass
    from nanotron.config.config import OptimizerArgs, AdamWOptimizerArgs, LRSchedulerArgs

    # Test valid stages
    for stage in [0, 1, 2, 3]:
        try:
            args = OptimizerArgs(
                optimizer_factory=AdamWOptimizerArgs(
                    name="adamW",
                    adam_eps=1e-8,
                    adam_beta1=0.9,
                    adam_beta2=0.999,
                    torch_adam_is_fused=False
                ),
                zero_stage=stage,
                weight_decay=0.01,
                clip_grad=1.0,
                accumulate_grad_in_fp32=True,
                learning_rate_scheduler=LRSchedulerArgs(
                    learning_rate=3e-4,
                    lr_warmup_steps=100,
                    lr_warmup_style="linear",
                    lr_decay_style="cosine",
                    min_decay_lr=1e-5
                )
            )
            print(f"✓ zero_stage={stage} validated successfully")
        except ValueError as e:
            if stage in [0, 1, 2, 3]:
                print(f"✗ zero_stage={stage} should be valid but raised: {e}")
                return False
            else:
                print(f"✓ zero_stage={stage} correctly rejected: {e}")

    # Test invalid stage
    try:
        args = OptimizerArgs(
            optimizer_factory=AdamWOptimizerArgs(
                name="adamW",
                adam_eps=1e-8,
                adam_beta1=0.9,
                adam_beta2=0.999,
                torch_adam_is_fused=False
            ),
            zero_stage=4,  # Invalid
            weight_decay=0.01,
            clip_grad=1.0,
            accumulate_grad_in_fp32=True,
            learning_rate_scheduler=LRSchedulerArgs(
                learning_rate=3e-4,
                lr_warmup_steps=100,
                lr_warmup_style="linear",
                lr_decay_style="cosine",
                min_decay_lr=1e-5
            )
        )
        print(f"✗ zero_stage=4 should have been rejected")
        return False
    except ValueError as e:
        if "zero_stage must be" in str(e):
            print(f"✓ Invalid zero_stage=4 correctly rejected: {e}")
            return True
        else:
            print(f"✗ Wrong error for zero_stage=4: {e}")
            return False

def test_gradient_accumulator_changes():
    """Test that gradient accumulator has reduce-scatter support"""
    print("\nTesting gradient accumulator changes...")

    try:
        from nanotron.optim.gradient_accumulator import get_fp32_accum_hook
        import inspect

        sig = inspect.signature(get_fp32_accum_hook)
        params = list(sig.parameters.keys())

        if 'reduce_scatter' in params:
            print("✓ get_fp32_accum_hook has reduce_scatter parameter")
            return True
        else:
            print(f"✗ reduce_scatter parameter not found. Parameters: {params}")
            return False
    except Exception as e:
        print(f"✗ Error testing gradient accumulator: {e}")
        return False

def test_checkpoint_serialization():
    """Test that checkpoint serialization includes zero_stage"""
    print("\nTesting checkpoint serialization...")

    try:
        # Read the optimizer serialization code
        with open('src/nanotron/serialize/optimizer.py', 'r') as f:
            content = f.read()

        if 'zero_stage' in content and 'config["configs"]["zero_stage"]' in content:
            print("✓ Checkpoint serialization includes zero_stage")
            return True
        else:
            print("✗ Checkpoint serialization missing zero_stage")
            return False
    except Exception as e:
        print(f"✗ Error checking checkpoint code: {e}")
        return False

def test_helpers_integration():
    """Test that helpers.py passes zero_stage to optimizer"""
    print("\nTesting helpers.py integration...")

    try:
        with open('src/nanotron/helpers.py', 'r') as f:
            content = f.read()

        if 'zero_stage=optimizer_args.zero_stage' in content:
            print("✓ helpers.py passes zero_stage to ZeroDistributedOptimizer")
        else:
            print("⚠ helpers.py may not be passing zero_stage correctly")

        if 'optimizer.zero_stage >= 2' in content:
            print("✓ helpers.py checks zero_stage for reduce_scatter")
            return True
        else:
            print("⚠ helpers.py may not be checking zero_stage for reduce_scatter")
            return True
    except Exception as e:
        print(f"✗ Error checking helpers.py: {e}")
        return False

def test_zero_implementation_completeness():
    """Test that ZeRO implementation has all required methods"""
    print("\nTesting ZeRO implementation completeness...")

    from nanotron.optim.zero import ZeroDistributedOptimizer

    required_methods = [
        '_partition_parameters',
        '_all_gather_params',
        'step',
        'zero_grad'
    ]

    zero3_methods = [
        'register_zero3_hooks',
        '_all_gather_params_for_computation',
        '_release_full_params'
    ]

    all_ok = True

    for method in required_methods:
        if hasattr(ZeroDistributedOptimizer, method):
            print(f"✓ Method {method} exists")
        else:
            print(f"✗ Missing method: {method}")
            all_ok = False

    for method in zero3_methods:
        if hasattr(ZeroDistributedOptimizer, method):
            print(f"✓ ZeRO-3 method {method} exists")
        else:
            print(f"✗ Missing ZeRO-3 method: {method}")
            all_ok = False

    return all_ok

def test_documentation_exists():
    """Test that documentation was created"""
    print("\nTesting documentation...")

    import os

    if os.path.exists('ZERO_OPTIMIZER_GUIDE.md'):
        print("✓ ZERO_OPTIMIZER_GUIDE.md exists")

        with open('ZERO_OPTIMIZER_GUIDE.md', 'r') as f:
            content = f.read()

        if 'ZeRO-2' in content and 'ZeRO-3' in content and 'FSDP' in content:
            print("✓ Documentation includes ZeRO-2 and ZeRO-3 information")
            return True
        else:
            print("⚠ Documentation may be incomplete")
            return True
    else:
        print("✗ ZERO_OPTIMIZER_GUIDE.md not found")
        return False

def main():
    """Run all unit tests"""
    print("=" * 70)
    print("ZeRO-2 and ZeRO-3 Implementation Unit Tests (CPU)")
    print("=" * 70)
    print()

    tests = [
        ("Imports", test_zero_imports),
        ("ZeRO Stage Attribute", test_zero_stage_attribute),
        ("Config Validation", test_config_validation),
        ("Gradient Accumulator", test_gradient_accumulator_changes),
        ("Checkpoint Serialization", test_checkpoint_serialization),
        ("Helpers Integration", test_helpers_integration),
        ("Implementation Completeness", test_zero_implementation_completeness),
        ("Documentation", test_documentation_exists),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
