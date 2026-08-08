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
- Dynamo recompile limit is raised to at least 16 because temporal memory length grows
  through a finite set of static specializations during warmup;
- invalid max-autotune candidates that exceed the GPU shared-memory limit are pruned
  through `TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM=1` in the builder.

Do not include the first compile/specialization frames in realtime latency statistics.
