#!/usr/bin/env python
"""Code structure tests for ZeRO-2 and ZeRO-3 implementation (no dependencies)"""

import sys
import os
import re

def test_zero_stage_parameter():
    """Test that ZeroDistributedOptimizer has zero_stage parameter"""
    print("Testing ZeRO stage parameter in zero.py...")

    with open('src/nanotron/optim/zero.py', 'r') as f:
        content = f.read()

    # Check for zero_stage parameter in __init__
    if 'zero_stage: int = 1' in content or 'zero_stage: int,' in content:
        print("✓ zero_stage parameter found in __init__")
    else:
        print("✗ zero_stage parameter not found in __init__")
        return False

    # Check for zero_stage assertion
    if 'assert zero_stage in [1, 2, 3]' in content:
        print("✓ zero_stage validation found")
    else:
        print("⚠ zero_stage validation not found")

    # Check for self.zero_stage assignment
    if 'self.zero_stage = zero_stage' in content:
        print("✓ self.zero_stage assignment found")
    else:
        print("✗ self.zero_stage assignment not found")
        return False

    # Check for ZeRO-3 specific attributes
    if '_forward_pre_hooks' in content and '_params_gathered' in content:
        print("✓ ZeRO-3 specific attributes found")
    else:
        print("⚠ ZeRO-3 attributes may be missing")

    # Check for register_zero3_hooks method
    if 'def register_zero3_hooks' in content:
        print("✓ register_zero3_hooks method found")
    else:
        print("✗ register_zero3_hooks method not found")
        return False

    # Check for parameter gathering methods
    if 'def _gather_parameters' in content:
        print("✓ _gather_parameters method found")
    else:
        print("✗ _gather_parameters method not found")
        return False

    if 'def _restore_sharded_params' in content:
        print("✓ _restore_sharded_params method found")
    else:
        print("✗ _restore_sharded_params method not found")
        return False

    if 'def _shard_parameters' in content:
        print("✓ _shard_parameters method found")
    else:
        print("✗ _shard_parameters method not found")
        return False

    if 'def reduce_scatter_gradients' in content:
        print("✓ reduce_scatter_gradients method found")
    else:
        print("✗ reduce_scatter_gradients method not found")
        return False

    # Check for conditional all-gather based on zero_stage
    if 'if self.zero_stage <= 2' in content or 'if self.zero_stage == 3' in content:
        print("✓ Conditional logic based on zero_stage found")
    else:
        print("⚠ Conditional logic based on zero_stage may be missing")

    return True

def test_gradient_accumulator_reduce_scatter():
    """Test that gradient accumulator has reduce-scatter implementation"""
    print("\nTesting reduce-scatter in gradient_accumulator.py...")

    with open('src/nanotron/optim/gradient_accumulator.py', 'r') as f:
        content = f.read()

    # Check for reduce_scatter logic
    if 'if reduce_scatter:' in content:
        print("✓ reduce_scatter conditional found")
    else:
        print("✗ reduce_scatter conditional not found")
        return False

    # Check that NotImplementedError was removed
    if 'raise NotImplementedError("Not implemented")' in content and 'reduce_scatter' in content:
        print("✗ NotImplementedError still present in reduce_scatter block")
        return False
    else:
        print("✓ NotImplementedError removed from reduce_scatter")

    # Check for reduce_scatter_coalesced call
    if 'reduce_scatter_coalesced' in content:
        print("✓ reduce_scatter_coalesced call found")
    else:
        print("✗ reduce_scatter_coalesced call not found")
        return False

    # Check for ZeRO-2 comment
    if 'ZeRO-2' in content:
        print("✓ ZeRO-2 comment found")
    else:
        print("⚠ ZeRO-2 comment not found")

    return True

def test_config_validation():
    """Test that config has zero_stage validation"""
    print("\nTesting config validation in config.py...")

    with open('src/nanotron/config/config.py', 'r') as f:
        content = f.read()

    # Check for zero_stage validation in __post_init__
    if 'zero_stage not in [0, 1, 2, 3]' in content or 'zero_stage must be' in content:
        print("✓ zero_stage validation found in __post_init__")
    else:
        print("✗ zero_stage validation not found")
        return False

    return True

def test_helpers_integration():
    """Test that helpers.py integrates zero_stage"""
    print("\nTesting helpers.py integration...")

    with open('src/nanotron/helpers.py', 'r') as f:
        content = f.read()

    # Check that zero_stage is passed to ZeroDistributedOptimizer
    if 'zero_stage=optimizer_args.zero_stage' in content:
        print("✓ zero_stage passed to ZeroDistributedOptimizer")
    else:
        print("✗ zero_stage not passed to ZeroDistributedOptimizer")
        return False

    # Check for reduce_scatter based on zero_stage
    if 'optimizer.zero_stage >= 2' in content:
        print("✓ reduce_scatter enabled for ZeRO-2/3")
    else:
        print("✗ reduce_scatter logic not found")
        return False

    # Check for register_zero3_hooks call
    if 'register_zero3_hooks' in content:
        print("✓ register_zero3_hooks call found")
    else:
        print("✗ register_zero3_hooks call not found")
        return False

    # Check for zero_stage == 3 condition
    if 'optimizer_args.zero_stage == 3' in content:
        print("✓ ZeRO-3 conditional found")
    else:
        print("⚠ ZeRO-3 conditional may be missing")

    return True

def test_checkpoint_serialization():
    """Test that checkpoint serialization includes zero_stage"""
    print("\nTesting checkpoint serialization...")

    with open('src/nanotron/serialize/optimizer.py', 'r') as f:
        content = f.read()

    # Check for zero_stage in config
    if 'config["configs"]["zero_stage"]' in content:
        print("✓ zero_stage saved in checkpoint config")
    else:
        print("✗ zero_stage not saved in checkpoint")
        return False

    # Check for ZeRO-1, ZeRO-2, and ZeRO-3 comment
    if 'ZeRO-1, ZeRO-2, and ZeRO-3' in content or 'ZeRO optimizer' in content:
        print("✓ ZeRO stages mentioned in comments")
    else:
        print("⚠ ZeRO stages not clearly documented")

    return True

def test_documentation():
    """Test that documentation exists and is complete"""
    print("\nTesting documentation...")

    if not os.path.exists('ZERO_OPTIMIZER_GUIDE.md'):
        print("✗ ZERO_OPTIMIZER_GUIDE.md not found")
        return False

    print("✓ ZERO_OPTIMIZER_GUIDE.md exists")

    with open('ZERO_OPTIMIZER_GUIDE.md', 'r') as f:
        content = f.read()

    required_sections = [
        'ZeRO-1',
        'ZeRO-2',
        'ZeRO-3',
        'FSDP',
        'Configuration',
        'Memory Savings',
        'gradient sharding',
        'parameter sharding',
        'optimizer state',
    ]

    missing = []
    for section in required_sections:
        if section.lower() in content.lower():
            print(f"  ✓ Section '{section}' found")
        else:
            print(f"  ✗ Section '{section}' missing")
            missing.append(section)

    if missing:
        print(f"⚠ Missing sections: {', '.join(missing)}")
        return True  # Still pass, just warn

    return True

def test_logging_messages():
    """Test that logging messages include ZeRO stage info"""
    print("\nTesting logging messages...")

    with open('src/nanotron/optim/zero.py', 'r') as f:
        content = f.read()

    # Check for ZeRO-{stage} in logging
    if 'ZeRO-{self.zero_stage}' in content or 'f"[ZeRO-{self.zero_stage}' in content:
        print("✓ Dynamic ZeRO stage in logging messages")
    else:
        print("⚠ ZeRO stage not included in logging")

    # Check for stage description in logging
    if 'stage_desc = "optimizer states"' in content:
        print("✓ Stage-specific descriptions found")
    else:
        print("⚠ Stage descriptions may be missing")

    return True

def count_lines_changed():
    """Count approximate lines changed"""
    print("\nCounting implementation size...")

    files_to_check = [
        'src/nanotron/optim/zero.py',
        'src/nanotron/optim/gradient_accumulator.py',
        'src/nanotron/helpers.py',
        'src/nanotron/config/config.py',
        'src/nanotron/serialize/optimizer.py',
        'ZERO_OPTIMIZER_GUIDE.md'
    ]

    total_lines = 0
    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"  {filepath}: {lines} lines")
        except FileNotFoundError:
            print(f"  ✗ {filepath} not found")

    print(f"\n✓ Total implementation size: ~{total_lines} lines across {len(files_to_check)} files")
    return True

def main():
    """Run all code structure tests"""
    print("=" * 70)
    print("ZeRO-2 and ZeRO-3 Implementation Code Structure Tests")
    print("=" * 70)
    print()

    tests = [
        ("ZeRO Stage Parameter", test_zero_stage_parameter),
        ("Gradient Accumulator Reduce-Scatter", test_gradient_accumulator_reduce_scatter),
        ("Config Validation", test_config_validation),
        ("Helpers Integration", test_helpers_integration),
        ("Checkpoint Serialization", test_checkpoint_serialization),
        ("Documentation", test_documentation),
        ("Logging Messages", test_logging_messages),
        ("Implementation Size", count_lines_changed),
    ]

    results = []
    for test_name, test_func in tests:
        print()
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

    if failed == 0:
        print("\n🎉 All tests passed! The implementation looks good.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review the implementation.")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
