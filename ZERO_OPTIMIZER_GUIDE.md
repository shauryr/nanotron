# ZeRO Optimizer Guide

This guide explains how to use ZeRO (Zero Redundancy Optimizer) stages 1-3 in Nanotron.

## Overview

ZeRO is a memory optimization technique for distributed training that partitions optimizer states, gradients, and model parameters across data parallel processes. This implementation supports:

- **ZeRO-1**: Partitions optimizer states only
- **ZeRO-2**: Partitions optimizer states + gradients (uses reduce-scatter)
- **ZeRO-3**: Partitions optimizer states + gradients + parameters (FSDP-style)

Reference: [ZeRO Paper](https://arxiv.org/abs/1910.02054v3)

## Memory Savings

For a model with N parameters and data parallelism degree D:

| ZeRO Stage | Optimizer States | Gradients | Parameters | Memory/GPU |
|------------|------------------|-----------|------------|------------|
| Stage 0    | Replicated       | Replicated| Replicated | ~4N        |
| Stage 1    | Partitioned (÷D) | Replicated| Replicated | ~4N - 3N/D |
| Stage 2    | Partitioned (÷D) | Partitioned (÷D) | Replicated | ~4N - 4N/D |
| Stage 3    | Partitioned (÷D) | Partitioned (÷D) | Partitioned (÷D) | ~4N/D |

*Note: Assumes mixed-precision training with Adam optimizer (2 states per param)*

## Configuration

Set the `zero_stage` parameter in your optimizer configuration:

```yaml
optimizer:
  zero_stage: 2  # Options: 0 (disabled), 1, 2, or 3
  optimizer_factory:
    name: adamW
    adam_eps: 1.0e-08
    adam_beta1: 0.9
    adam_beta2: 0.95
    torch_adam_is_fused: true
  weight_decay: 0.01
  clip_grad: 1.0
  accumulate_grad_in_fp32: true
  learning_rate_scheduler:
    learning_rate: 3.0e-4
    lr_warmup_steps: 100
    lr_warmup_style: linear
    lr_decay_style: cosine
    min_decay_lr: 1.0e-5
```

## Usage Examples

### ZeRO-1 (Optimizer State Sharding)
```yaml
optimizer:
  zero_stage: 1
  # ... rest of config
```
- Best for: Small to medium models where gradient memory is manageable
- Memory savings: ~12 bytes per parameter (for Adam)
- Communication: All-reduce gradients, all-gather parameters after optimizer step

### ZeRO-2 (+ Gradient Sharding)
```yaml
optimizer:
  zero_stage: 2
  accumulate_grad_in_fp32: true  # Recommended with ZeRO-2
  # ... rest of config
```
- Best for: Medium to large models where gradient memory is significant
- Memory savings: ~16 bytes per parameter (for Adam with FP32 grads)
- Communication: Reduce-scatter gradients, all-gather parameters after optimizer step

### ZeRO-3 (+ Parameter Sharding / FSDP)
```yaml
optimizer:
  zero_stage: 3
  accumulate_grad_in_fp32: true  # Recommended with ZeRO-3
  # ... rest of config
```
- Best for: Very large models that don't fit in GPU memory with ZeRO-2
- Memory savings: ~16+ bytes per parameter (shards everything)
- Communication: All-gather parameters on-demand during forward/backward, reduce-scatter gradients

## Implementation Details

### ZeRO-1
- Optimizer states (momentum, variance) are partitioned evenly across data parallel ranks
- Each rank only stores and updates its assigned parameter slices
- After optimizer step, all-gather synchronizes updated parameters

### ZeRO-2
- Builds on ZeRO-1, adds gradient sharding
- Uses reduce-scatter instead of all-reduce for gradient synchronization
- Each rank only accumulates gradients for its assigned parameter slices
- Requires `accumulate_grad_in_fp32: true` for best performance

### ZeRO-3
- Builds on ZeRO-2, adds parameter sharding
- Parameters remain sharded in memory when not in use
- Registers hooks on each module to:
  - All-gather parameters before forward/backward computation
  - Release full parameters after computation
- Only sharded parameter slices remain in memory between operations

## Checkpointing

All ZeRO stages use the same checkpoint format:
- Each data parallel rank saves its sharded optimizer states
- Checkpoint includes `param_name_to_dp_rank_offsets` for reconstruction
- Can load checkpoints with different tensor parallelism degrees
- ZeRO stage is saved in checkpoint metadata

## Performance Considerations

1. **Communication Overhead**
   - ZeRO-1: One all-gather per optimizer step
   - ZeRO-2: One reduce-scatter per backward pass
   - ZeRO-3: Additional all-gathers during forward/backward

2. **Gradient Accumulation**
   - Always use `accumulate_grad_in_fp32: true` with ZeRO-2/3
   - This ensures numerical stability with sharded gradients

3. **Scaling**
   - ZeRO-1/2: Scales well with large batch sizes
   - ZeRO-3: May have higher communication overhead, use for very large models

## Compatibility

- ✅ Works with tensor parallelism (TP)
- ✅ Works with pipeline parallelism (PP)
- ✅ Works with DDP
- ✅ Works with FP32 gradient accumulation
- ✅ Works with gradient clipping
- ✅ Works with activation checkpointing

## Troubleshooting

### OOM (Out of Memory) Errors
- Try increasing ZeRO stage: 0 → 1 → 2 → 3
- Enable FP32 gradient accumulation: `accumulate_grad_in_fp32: true`
- Reduce micro batch size
- Increase data parallelism degree

### Slow Training
- ZeRO-3 has higher communication overhead
- Consider using ZeRO-2 if model fits in memory
- Ensure high-speed interconnect (InfiniBand, NVLink)

### Checkpoint Loading Issues
- Ensure checkpoint was saved with same model architecture
- Check that `param_name_to_dp_rank_offsets` exists in checkpoint config
- Verify ZeRO stage compatibility

## Example Training Command

```bash
torchrun --nproc_per_node=8 run_train.py --config-file config_zero3.yaml
```

With config:
```yaml
parallelism:
  dp: 8
  tp: 1
  pp: 1

optimizer:
  zero_stage: 3  # Use ZeRO-3 for maximum memory savings
  accumulate_grad_in_fp32: true
```

## References

- [ZeRO Paper](https://arxiv.org/abs/1910.02054v3)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
