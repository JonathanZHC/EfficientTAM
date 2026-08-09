#!/usr/bin/env python3
"""Small two-view smoke/benchmark for asynchronous direct-reference correction.

The test builds synthetic synchronized RGB videos, keeps a persistent EfficientTAM
image-feature snapshot for every live frame, and compares:

  direct: feature[x] + corrected mask[x] + feature[t] -> mask[t]
  replay: corrected mask[x] -> x+1 -> ... -> t

It also propagates one more ordinary frame after the direct correction to verify
that both paths return to the same normal multi-view propagation API.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/efficienttam/efficienttam_s_512x512.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/efficienttam_s_512x512.pt",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("fixed_batch", "sequential"),
        default="fixed_batch",
    )
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--max-objects-per-view", type=int, default=3)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--reference-frame", type=int, default=2)
    parser.add_argument("--current-frame", type=int, default=6)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--vos-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=20,
        help="Repeats for steady-state feature-cache microbenchmarks.",
    )
    return parser.parse_args()


def cuda_ms(function, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    result = function(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end)), 1000.0 * (time.perf_counter() - wall_start)


def object_box(view_idx: int, obj_idx: int, frame_idx: int, width: int, height: int):
    box_w = max(42, width // 9)
    box_h = max(54, height // 7)
    x0 = 55 + obj_idx * 155 + frame_idx * (7 + obj_idx * 2) + view_idx * 22
    y0 = 70 + obj_idx * 120 + frame_idx * (3 + obj_idx) + view_idx * 8
    x0 %= max(1, width - box_w - 8)
    y0 %= max(1, height - box_h - 8)
    return x0, y0, x0 + box_w, y0 + box_h


def synthetic_frame_and_masks(
    view_idx: int,
    frame_idx: int,
    width: int,
    height: int,
    num_objects: int,
):
    image = Image.new(
        "RGB",
        (width, height),
        (32 + 8 * view_idx, 38, 46 + 5 * view_idx),
    )
    draw = ImageDraw.Draw(image)
    masks = []
    palette = [
        (220, 70, 65),
        (70, 210, 105),
        (70, 120, 235),
    ]
    for obj_idx in range(num_objects):
        box = object_box(view_idx, obj_idx, frame_idx, width, height)
        draw.rounded_rectangle(box, radius=10, fill=palette[obj_idx % len(palette)])
        mask = np.zeros((height, width), dtype=np.bool_)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = True
        masks.append(mask)
    return image, masks


def make_synthetic_videos(
    root: Path,
    views: int,
    frames: int,
    width: int,
    height: int,
    num_objects: int,
):
    video_dirs = []
    gt_masks = {}
    for view_idx in range(views):
        view_dir = root / f"view_{view_idx}"
        view_dir.mkdir(parents=True, exist_ok=True)
        video_dirs.append(view_dir)
        for frame_idx in range(frames):
            image, masks = synthetic_frame_and_masks(
                view_idx,
                frame_idx,
                width,
                height,
                num_objects,
            )
            image.save(view_dir / f"{frame_idx}.jpg", quality=95)
            gt_masks[(view_idx, frame_idx)] = masks
    return video_dirs, gt_masks


def make_states(predictor, video_dirs):
    return [
        predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        for video_dir in video_dirs
    ]


def masks_for_frame(gt_masks, views: int, frame_idx: int):
    return [gt_masks[(view_idx, frame_idx)] for view_idx in range(views)]


def seed_states(predictor, states, masks_per_view, frame_idx: int):
    for state, masks in zip(states, masks_per_view):
        for obj_idx, mask in enumerate(masks):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_idx + 1,
                mask=mask,
            )
    predictor.prepare_multiview_states(
        states,
        conditioning_frame_idx=frame_idx,
    )


def result_iou(a_results, b_results):
    values = []
    for a_view, b_view in zip(a_results, b_results):
        a = a_view["video_res_masks"] > 0
        b = b_view["video_res_masks"] > 0
        if a.shape != b.shape:
            raise RuntimeError(f"result shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}")
        for obj_idx in range(a.shape[0]):
            aa = a[obj_idx]
            bb = b[obj_idx]
            inter = torch.logical_and(aa, bb).sum().item()
            union = torch.logical_or(aa, bb).sum().item()
            values.append(1.0 if union == 0 else inter / union)
    return values


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This test requires CUDA")
    if args.views != 2:
        raise ValueError("This simple test currently expects --views 2")
    if not 1 <= args.objects <= args.max_objects_per_view:
        raise ValueError("--objects must be in [1, max_objects_per_view]")
    if args.reference_frame < 1:
        raise ValueError("--reference-frame must be >= 1")
    if args.current_frame <= args.reference_frame:
        raise ValueError("--current-frame must be after --reference-frame")
    if args.timing_repeats < 3:
        raise ValueError("--timing-repeats must be >= 3")

    next_frame = args.current_frame + 1
    num_frames = next_frame + 1
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Pass --checkpoint with the real path."
        )

    print(
        f"Building EfficientTAM: mode={args.execution_mode}, views={args.views}, "
        f"real_objects/view={args.objects}, fixed_slots/view={args.max_objects_per_view}"
    )
    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=str(checkpoint),
        device="cuda",
        vos_optimized=args.vos_optimized,
        apply_postprocessing=False,
        execution_mode=args.execution_mode,
        fixed_num_views=args.views,
        max_objects_per_view=args.max_objects_per_view,
        hydra_overrides_extra=[
            "++model.fill_hole_area=0",
            "++model.non_overlap_masks_for_mem_enc=false",
        ],
    )

    with tempfile.TemporaryDirectory(prefix="efficienttam_direct_ref_") as tmp:
        root = Path(tmp)
        video_dirs, gt_masks = make_synthetic_videos(
            root,
            args.views,
            num_frames,
            args.width,
            args.height,
            args.objects,
        )

        # --------------------------------------------------------------
        # Warm every measured topology before timing. CUDA-event timing can
        # otherwise include CPU-side torch.compile/CUDAGraph specialization
        # as a long GPU-idle interval between the start/end events.
        # --------------------------------------------------------------
        print("Warming normal temporal path...")
        warm_normal_states = make_states(predictor, video_dirs)
        seed_states(
            predictor,
            warm_normal_states,
            masks_for_frame(gt_masks, args.views, 0),
            frame_idx=0,
        )
        for frame_idx in range(1, next_frame + 1):
            warm_snapshot = predictor.snapshot_multiview_image_features(
                warm_normal_states,
                frame_idx,
            )
            predictor.propagate_multiview_step(
                warm_normal_states,
                frame_idx,
                image_feature_snapshot=warm_snapshot,
            )
            del warm_snapshot
        torch.cuda.synchronize()
        del warm_normal_states
        gc.collect()

        print("Warming direct-reference correction path...")
        warm_direct_states = make_states(predictor, video_dirs)
        seed_states(
            predictor,
            warm_direct_states,
            masks_for_frame(gt_masks, args.views, 0),
            frame_idx=0,
        )
        warm_ref = predictor.snapshot_multiview_image_features(
            warm_direct_states,
            args.reference_frame,
        )
        warm_cur = predictor.snapshot_multiview_image_features(
            warm_direct_states,
            args.current_frame,
        )
        predictor.correct_multiview_from_reference(
            warm_direct_states,
            warm_ref,
            masks_for_frame(
                gt_masks,
                args.views,
                args.reference_frame,
            ),
            args.current_frame,
            warm_cur,
            False,
        )
        warm_next = predictor.snapshot_multiview_image_features(
            warm_direct_states,
            next_frame,
        )
        predictor.propagate_multiview_step(
            warm_direct_states,
            next_frame,
            image_feature_snapshot=warm_next,
        )
        torch.cuda.synchronize()
        del warm_direct_states, warm_ref, warm_cur, warm_next
        gc.collect()

        print("Warming replay baseline path...")
        warm_replay_states = make_states(predictor, video_dirs)
        seed_states(
            predictor,
            warm_replay_states,
            masks_for_frame(
                gt_masks,
                args.views,
                args.reference_frame,
            ),
            frame_idx=args.reference_frame,
        )
        for frame_idx in range(args.reference_frame + 1, next_frame + 1):
            predictor.propagate_multiview_step(
                warm_replay_states,
                frame_idx,
            )
        torch.cuda.synchronize()
        del warm_replay_states
        gc.collect()
        torch.cuda.empty_cache()

        # --------------------------------------------------------------
        # Feature-cache microbenchmark. Measure encoder-only, clone-only,
        # and the public encoder+snapshot path separately.
        # --------------------------------------------------------------
        print("Benchmarking feature-cache overhead...")
        feature_bench_states = make_states(predictor, video_dirs)
        bench_frame = min(1, next_frame)
        frame_indices = [bench_frame] * args.views

        # One persistent snapshot supplies stable tensors for clone-only timing.
        persistent_bench_snapshot = predictor.snapshot_multiview_image_features(
            feature_bench_states,
            bench_frame,
        )
        torch.cuda.synchronize()

        encoder_only_gpu = []
        clone_only_gpu = []
        snapshot_total_gpu = []

        def encode_only_once():
            # Avoid keeping graph-managed outputs alive across logical steps.
            predictor._mark_cudagraph_step_begin()
            predictor.cache_image_features_batched(
                feature_bench_states,
                frame_indices,
            )
            for state in feature_bench_states:
                state["cached_features"].clear()

        for _ in range(args.timing_repeats):
            _, gpu_ms, _ = cuda_ms(encode_only_once)
            encoder_only_gpu.append(gpu_ms)

            cloned, gpu_ms, _ = cuda_ms(
                predictor._clone_backbone_output_persistent,
                persistent_bench_snapshot["backbone_out"],
            )
            clone_only_gpu.append(gpu_ms)
            del cloned

            snapshot, gpu_ms, _ = cuda_ms(
                predictor.snapshot_multiview_image_features,
                feature_bench_states,
                bench_frame,
            )
            snapshot_total_gpu.append(gpu_ms)
            del snapshot

        torch.cuda.synchronize()
        del feature_bench_states, persistent_bench_snapshot
        gc.collect()
        torch.cuda.empty_cache()
        print("Warmup complete. Starting measured run.\n")

        # --------------------------------------------------------------
        # Live branch: ordinary tracking until t, then direct correction.
        # --------------------------------------------------------------
        live_states = make_states(predictor, video_dirs)
        seed_states(
            predictor,
            live_states,
            masks_for_frame(gt_masks, args.views, 0),
            frame_idx=0,
        )

        snapshots = {}
        normal_cycle_gpu = []
        normal_cycle_wall = []
        snapshot_gpu = []
        propagation_gpu = []

        for frame_idx in range(1, args.current_frame):
            snapshot, enc_gpu, _ = cuda_ms(
                predictor.snapshot_multiview_image_features,
                live_states,
                frame_idx,
            )
            (_, prop_gpu, prop_wall) = cuda_ms(
                predictor.propagate_multiview_step,
                live_states,
                frame_idx,
                False,
                snapshot,
            )
            snapshots[frame_idx] = snapshot
            snapshot_gpu.append(enc_gpu)
            propagation_gpu.append(prop_gpu)
            normal_cycle_gpu.append(enc_gpu + prop_gpu)
            normal_cycle_wall.append(prop_wall)

        reference_snapshot = snapshots[args.reference_frame]
        current_snapshot, current_encode_gpu, _ = cuda_ms(
            predictor.snapshot_multiview_image_features,
            live_states,
            args.current_frame,
        )
        direct_results, direct_gpu, direct_wall = cuda_ms(
            predictor.correct_multiview_from_reference,
            live_states,
            reference_snapshot,
            masks_for_frame(gt_masks, args.views, args.reference_frame),
            args.current_frame,
            current_snapshot,
            False,
        )

        next_snapshot, next_encode_gpu, _ = cuda_ms(
            predictor.snapshot_multiview_image_features,
            live_states,
            next_frame,
        )
        direct_next_results, direct_next_prop_gpu, _ = cuda_ms(
            predictor.propagate_multiview_step,
            live_states,
            next_frame,
            False,
            next_snapshot,
        )

        # --------------------------------------------------------------
        # Replay baseline: corrected frame x, then every intermediate frame.
        # --------------------------------------------------------------
        replay_states = make_states(predictor, video_dirs)
        torch.cuda.synchronize()
        replay_seed_wall_start = time.perf_counter()
        replay_seed_start = torch.cuda.Event(enable_timing=True)
        replay_seed_end = torch.cuda.Event(enable_timing=True)
        replay_seed_start.record()
        seed_states(
            predictor,
            replay_states,
            masks_for_frame(gt_masks, args.views, args.reference_frame),
            frame_idx=args.reference_frame,
        )
        replay_seed_end.record()
        torch.cuda.synchronize()
        replay_seed_gpu = float(replay_seed_start.elapsed_time(replay_seed_end))
        replay_seed_wall = 1000.0 * (time.perf_counter() - replay_seed_wall_start)

        replay_results = None
        replay_prop_gpu_total = 0.0
        replay_prop_wall_total = 0.0
        for frame_idx in range(args.reference_frame + 1, args.current_frame + 1):
            replay_results, gpu_ms, wall_ms = cuda_ms(
                predictor.propagate_multiview_step,
                replay_states,
                frame_idx,
            )
            replay_prop_gpu_total += gpu_ms
            replay_prop_wall_total += wall_ms

        replay_next_results, replay_next_gpu, _ = cuda_ms(
            predictor.propagate_multiview_step,
            replay_states,
            next_frame,
        )

        iou_current = result_iou(direct_results, replay_results)
        iou_next = result_iou(direct_next_results, replay_next_results)

        snapshot_bytes = predictor.multiview_feature_snapshot_nbytes(
            reference_snapshot
        )
        encoder_only_median = float(np.median(encoder_only_gpu))
        clone_only_median = float(np.median(clone_only_gpu))
        snapshot_total_median = float(np.median(snapshot_total_gpu))
        snapshot_minus_encoder = snapshot_total_median - encoder_only_median

        print("\n=== Persistent feature cache ===")
        print(f"one B={args.views} snapshot: {snapshot_bytes / 2**20:.2f} MiB")
        print(f"32-frame ring estimate:    {32 * snapshot_bytes / 2**20:.2f} MiB")
        print(f"encoder-only median:        {encoder_only_median:.3f} ms GPU")
        print(f"clone-only median:          {clone_only_median:.3f} ms GPU")
        print(f"encoder + snapshot median:  {snapshot_total_median:.3f} ms GPU")
        print(
            "snapshot - encoder delta:  "
            f"{snapshot_minus_encoder:.3f} ms GPU "
            "(noisy cross-run estimate)"
        )

        print("\n=== Live normal path with cached features ===")
        normal_cycle_median = None
        if snapshot_gpu:
            snapshot_median = float(np.median(snapshot_gpu))
            propagation_median = float(np.median(propagation_gpu))
            normal_cycle_median = float(np.median(normal_cycle_gpu))
            print(f"image encode + persistent clone median: {snapshot_median:.3f} ms GPU")
            print(f"tracking-only propagation median:       {propagation_median:.3f} ms GPU")
            print(f"combined normal cycle median:           {normal_cycle_median:.3f} ms GPU")
            print(
                "normal propagation samples:          "
                + ", ".join(f"{value:.3f}" for value in propagation_gpu)
                + " ms"
            )

        print("\n=== Direct corrected-reference path ===")
        print(f"current image encode + clone: {current_encode_gpu:.3f} ms GPU")
        print(f"direct correction x->t:       {direct_gpu:.3f} ms GPU / {direct_wall:.3f} ms wall")
        print(f"direct boundary total:        {current_encode_gpu + direct_gpu:.3f} ms GPU")
        print(f"next normal frame:            {next_encode_gpu + direct_next_prop_gpu:.3f} ms GPU")

        print("\n=== Full replay baseline ===")
        print(f"reseed corrected frame x:     {replay_seed_gpu:.3f} ms GPU / {replay_seed_wall:.3f} ms wall")
        print(f"replay x+1..t:                {replay_prop_gpu_total:.3f} ms GPU / {replay_prop_wall_total:.3f} ms wall")
        print(f"reseed + replay total:        {replay_seed_gpu + replay_prop_gpu_total:.3f} ms GPU")
        print(f"next normal frame:            {replay_next_gpu:.3f} ms GPU")

        print("\n=== Mask agreement: direct vs replay ===")
        print(f"current frame t: min={min(iou_current):.6f}, mean={np.mean(iou_current):.6f}")
        print(f"next frame t+1: min={min(iou_next):.6f}, mean={np.mean(iou_next):.6f}")

        direct_next_total = next_encode_gpu + direct_next_prop_gpu
        if (
            normal_cycle_median is not None
            and normal_cycle_median > max(100.0, 5.0 * direct_next_total)
        ):
            print(
                "\nWARNING: normal-path timing is still inconsistent with the "
                "post-correction steady-state frame. This usually means an "
                "unwarmed compile/CUDAGraph specialization remains."
            )

        print("\nPASS: direct correction returned the normal multi-view result format and")
        print("      ordinary propagation succeeded on the following frame.")

        del live_states, replay_states, snapshots
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
