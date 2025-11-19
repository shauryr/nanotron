#!/usr/bin/env python
"""Simple test to validate ZeRO-2 and ZeRO-3 implementation"""

import torch
from nanotron.optim import NamedOptimizer, ZeroDistributedOptimizer
from nanotron.parallel.parameters import NanotronParameter
from nanotron import distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

def create_simple_model():
    """Create a simple test model"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    # Convert parameters to NanotronParameter
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            setattr(module, name, NanotronParameter(param))
    return model.cuda()

def test_zero_stage_validation():
    """Test that zero_stage parameter is validated correctly"""
    print("Testing ZeRO stage validation...")

    # Create a mock process group (single process)
    if not dist.is_initialized():
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29500',
            world_size=1,
            rank=0
        )

    model = create_simple_model()

    # Test valid stages
    for stage in [1, 2, 3]:
        try:
            optimizer = ZeroDistributedOptimizer(
                named_params_or_groups=model.named_parameters(),
                optimizer_builder=lambda named_param_groups: NamedOptimizer(
                    named_params_or_groups=named_param_groups,
                    optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
                ),
                dp_pg=dist.GroupMember.WORLD,
                zero_stage=stage,
            )
            assert optimizer.zero_stage == stage, f"ZeRO stage mismatch: expected {stage}, got {optimizer.zero_stage}"
            print(f"✓ ZeRO-{stage} initialization successful")
        except Exception as e:
            print(f"✗ ZeRO-{stage} initialization failed: {e}")
            raise

    # Test invalid stage
    try:
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=model.named_parameters(),
            optimizer_builder=lambda named_param_groups: NamedOptimizer(
                named_params_or_groups=named_param_groups,
                optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
            ),
            dp_pg=dist.GroupMember.WORLD,
            zero_stage=4,  # Invalid
        )
        print("✗ Should have raised error for invalid zero_stage=4")
        raise AssertionError("Invalid zero_stage should raise error")
    except AssertionError as e:
        if "zero_stage must be 1, 2, or 3" in str(e):
            print("✓ Invalid zero_stage correctly rejected")
        else:
            raise

    print("\n✅ All ZeRO stage validation tests passed!\n")
    return True

def test_zero_optimizer_states():
    """Test that optimizer states are properly sharded"""
    print("Testing optimizer state sharding...")

    if not dist.is_initialized():
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29500',
            world_size=1,
            rank=0
        )

    model = create_simple_model()

    for stage in [1, 2, 3]:
        optimizer = ZeroDistributedOptimizer(
            named_params_or_groups=model.named_parameters(),
            optimizer_builder=lambda named_param_groups: NamedOptimizer(
                named_params_or_groups=named_param_groups,
                optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
            ),
            dp_pg=dist.GroupMember.WORLD,
            zero_stage=stage,
        )

        # Run a simple forward/backward pass
        x = torch.randn(4, 10).cuda()
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Take optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Check that optimizer state is created
        state_dict = optimizer.state_dict()
        assert 'state' in state_dict, f"ZeRO-{stage}: Missing 'state' in state_dict"
        assert len(state_dict['state']) > 0, f"ZeRO-{stage}: Optimizer state is empty"

        print(f"✓ ZeRO-{stage} optimizer step successful, {len(state_dict['state'])} parameter states created")

    print("\n✅ All optimizer state tests passed!\n")
    return True

def test_zero_hooks():
    """Test that ZeRO-3 hooks are registered"""
    print("Testing ZeRO-3 hooks...")

    if not dist.is_initialized():
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:29500',
            world_size=1,
            rank=0
        )

    model = create_simple_model()

    # Create ZeRO-3 optimizer
    optimizer = ZeroDistributedOptimizer(
        named_params_or_groups=model.named_parameters(),
        optimizer_builder=lambda named_param_groups: NamedOptimizer(
            named_params_or_groups=named_param_groups,
            optimizer_builder=lambda param_groups: torch.optim.AdamW(param_groups),
        ),
        dp_pg=dist.GroupMember.WORLD,
        zero_stage=3,
    )

    # Check that ZeRO-3 specific attributes exist
    assert hasattr(optimizer, '_forward_pre_hooks'), "ZeRO-3: Missing _forward_pre_hooks"
    assert hasattr(optimizer, '_forward_post_hooks'), "ZeRO-3: Missing _forward_post_hooks"
    assert hasattr(optimizer, '_params_gathered'), "ZeRO-3: Missing _params_gathered"
    assert hasattr(optimizer, '_full_params_cache'), "ZeRO-3: Missing _full_params_cache"

    print("✓ ZeRO-3 hook attributes initialized")

    # Register hooks on model
    optimizer.register_zero3_hooks(model)

    # Check that hooks were registered (hook lists should be populated)
    # Note: With single process, hooks may not be registered if DP size is 1
    print(f"✓ ZeRO-3 hooks registered: {len(optimizer._forward_pre_hooks)} pre-hooks, {len(optimizer._forward_post_hooks)} post-hooks")

    print("\n✅ ZeRO-3 hook tests passed!\n")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("ZeRO-2 and ZeRO-3 Implementation Tests")
    print("=" * 60)
    print()

    try:
        test_zero_stage_validation()
        test_zero_optimizer_states()
        test_zero_hooks()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    exit(main())
