# Multi-view execution modes

This package supports two multi-view execution strategies through one build-time
flag:

- `execution_mode="sequential"`: original per-view / per-object path.
  For two views and three objects per view this is approximately `2E + 6 x B1`.
- `execution_mode="fixed_batch"`: batched view encoding plus one fixed all-view /
  all-object propagation batch. With two views and three slots per view this is
  `E(B=2) + tracking(B=6)`.

Choose the mode when the predictor is built. Do not change `predictor.execution_mode`
on a live compiled predictor; build a new predictor instead. `torch.compile` creates
shape specializations for the selected batch regime.

## Build

```python
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)

predictor = build_efficienttam_video_predictor(
    config_file="configs/efficienttam/efficienttam_s_512x512.yaml",
    ckpt_path="checkpoints/efficienttam_s_512x512.pt",
    device="cuda",
    vos_optimized=True,
    execution_mode="fixed_batch",  # "sequential" | "fixed_batch"
    fixed_num_views=2,
    max_objects_per_view=3,
)

print(predictor.execution_summary())
```

`fixed_num_views` and `max_objects_per_view` define the compiled shape only for
`fixed_batch`. They are ignored by the sequential propagation path.

## Unified integration API

Create one inference state per view, then seed the real objects on the detector /
conditioning frame using the normal `add_new_mask` or point APIs.

```python
states = [state_cam0, state_cam1]

# After all real objects for this refresh frame have been seeded:
predictor.prepare_multiview_states(
    states,
    conditioning_frame_idx=0,
)

# Ordinary propagation frames:
results = predictor.propagate_multiview_step(
    states,
    frame_idx=1,
)
```

`results` has one dictionary per view:

```python
{
    "view_idx": 0,
    "frame_idx": 1,
    "obj_ids": [...],
    "video_res_masks": ...,       # real objects only
    "num_real_objects": 3,
    "num_dummy_objects": 0,
    "execution_mode": "fixed_batch",
}
```

The calling code is identical for both execution modes.

## Fixed-batch constraints

`fixed_batch` requires:

1. exactly `fixed_num_views` synchronized view states;
2. no more than `max_objects_per_view` real objects in any view;
3. all real objects in a refresh to be seeded on the same conditioning frame;
4. aligned temporal history across all fixed slots.

Missing object slots are automatically padded with empty-mask dummy objects.
Dummy objects participate in the fixed compiled batch but are removed from returned
`obj_ids` and `video_res_masks`.

For a detector refresh / newly appearing object, re-seed the desired real objects on
the same refresh frame and prepare a fresh aligned set of states rather than changing
the compiled batch size mid-stream.

## Sequential mode

`sequential` preserves the original semantics. For each view, the first object on a
new frame computes that view's image features; subsequent objects reuse the cached
features. Every object then runs a separate B=1 tracking call. No dummy slots are
created.

The original single-video API (`propagate_in_video`) and the explicit fixed-batch APIs
(`prepare_fixed_multiview_states`, `propagate_fixed_multiview_step`) are retained for
backward compatibility, but new integration code should prefer the unified multi-view
API above.

## Compile notes

The VOS-optimized path keeps the stable settings established during the B>1 work:

- image encoder: compiled / max-autotune when enabled;
- memory encoder: `max-autotune`, `dynamic=False`;
- memory attention: `default`, `dynamic=False` (avoids the B>1 Inductor fusion bug);
- prompt encoder and mask decoder: `max-autotune`, `dynamic=False`;
- `use_max_autotune=False` switches the memory encoder, prompt encoder, and mask decoder to `mode="default"` for faster compile/startup experiments; `memory_attention` remains `default` in both cases.
- Dynamo recompile limit is raised to at least 16 because temporal memory length grows
  through a finite set of static specializations during warmup;
- invalid max-autotune candidates that exceed the GPU shared-memory limit are pruned
  through `TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM=1` in the builder.

Do not include the first compile/specialization frames in realtime latency statistics.

## Asynchronous corrected-reference primitive

For a sparse detector running asynchronously, EfficientTAM can keep persistent
image-feature snapshots and later correct the current tracker state directly from
an older detector frame without replaying every intermediate frame.

```python
# Every synchronized RGB bundle can be encoded once and stored in an external
# ring buffer. The snapshot owns cloned GPU tensors and is safe across later
# torch.compile / CUDAGraph steps.
snapshot_t = predictor.snapshot_multiview_image_features(
    states,
    frame_idx=t,
)

# Ordinary tracking can reuse the same snapshot, so the encoder is not run twice.
results_t = predictor.propagate_multiview_step(
    states,
    frame_idx=t,
    image_feature_snapshot=snapshot_t,
)

# When an asynchronous detector result for historical frame x arrives:
results_now = predictor.correct_multiview_from_reference(
    states,
    reference_feature_snapshot=feature_ring[x],
    reference_masks=masks_x,  # one N x H x W entry per view, real-object order
    current_frame_idx=t_now,
    current_feature_snapshot=feature_ring[t_now],
)
```

The correction path intentionally discards stale intermediate non-conditioning
memories. It builds a corrected conditioning memory on frame `x`, directly
infers frame `t_now`, and replaces the live history with:

```text
cond:     x (corrected detector mask)
non-cond: t_now (new direct prediction)
```

The following frame uses the ordinary `propagate_multiview_step()` API again.
Both `sequential` and `fixed_batch` modes are supported. In fixed-batch mode,
reference masks are supplied only for real objects; dummy slots are padded with
zero masks internally and the compiled batch size remains unchanged.

The feature snapshot ring-buffer policy is intentionally not implemented inside
EfficientTAM. The application owns retention/eviction. One snapshot's storage can
be inspected with:

```python
bytes_per_snapshot = predictor.multiview_feature_snapshot_nbytes(snapshot_t)
```

A standalone smoke/benchmark is included at:

```text
efficient_track_anything/tests/test_direct_reference_correction.py
```
