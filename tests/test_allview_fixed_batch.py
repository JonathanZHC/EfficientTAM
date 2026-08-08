#!/usr/bin/env python3
"""
Correctness + performance benchmark for fixed-shape all-view/all-object EfficientTAM propagation.

The sequential B=1 reference and the fixed B=(views * slots) path are benchmarked
in separate Python worker processes so torch.compile(dynamic=False) does not keep
specializing one memory_attention.forward between B=1 and B=N in the same process.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM",
    "1",
)

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--max-objects-per-view", type=int, default=4)
    p.add_argument("--real-objects-per-view", type=str, default="3,2")
    p.add_argument("--warmup-frames", type=int, default=8)
    p.add_argument("--benchmark-frames", type=int, default=8)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--device", default="cuda")
    p.add_argument("--vos-optimized", action="store_true")
    p.add_argument("--min-mask-iou", type=float, default=0.985)

    # Internal worker arguments.
    p.add_argument("--worker", choices=["sequential", "batched"], default=None)
    p.add_argument("--video-root", default=None)
    p.add_argument("--result-path", default=None)
    return p.parse_args()


def parse_real_counts(args):
    counts = [int(x) for x in args.real_objects_per_view.split(",")]
    if len(counts) != args.views:
        raise ValueError("--real-objects-per-view must contain one count per view")
    if any(n < 0 or n > args.max_objects_per_view for n in counts):
        raise ValueError("Each real object count must be in [0, max_objects_per_view]")
    return counts


def make_video(folder: Path, view_idx: int, num_frames: int, h: int, w: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(num_frames):
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[..., 0] = ((xx + 13 * t + 31 * view_idx) % 255).astype(np.uint8)
        base[..., 1] = ((yy + 7 * t + 53 * view_idx) % 255).astype(np.uint8)
        base[..., 2] = (((xx // 2 + yy // 3) + 11 * t) % 255).astype(np.uint8)
        Image.fromarray(base).save(folder / f"{t:05d}.jpg", quality=92)


def make_seed_masks(num_objects: int, h: int, w: int, view_idx: int):
    masks = []
    box_w = max(24, w // 10)
    box_h = max(24, h // 9)
    for obj_idx in range(num_objects):
        mask = torch.zeros(h, w, dtype=torch.bool)
        x0 = 30 + obj_idx * (box_w + 25) + view_idx * 11
        y0 = 35 + obj_idx * (box_h + 18) + view_idx * 9
        x0 = min(x0, w - box_w - 1)
        y0 = min(y0, h - box_h - 1)
        mask[y0:y0 + box_h, x0:x0 + box_w] = True
        masks.append(mask)
    return masks


def init_state_set(predictor, video_dirs, real_counts, max_objects, pad_fixed: bool):
    states = []
    for view_idx, (video_dir, real_count) in enumerate(zip(video_dirs, real_counts)):
        state = predictor.init_state(
            str(video_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
        masks = make_seed_masks(
            real_count,
            state["video_height"],
            state["video_width"],
            view_idx,
        )
        for obj_idx, mask in enumerate(masks):
            predictor.add_new_mask(
                state,
                frame_idx=0,
                obj_id=f"view{view_idx}_obj{obj_idx}",
                mask=mask,
            )
        states.append(state)

    if pad_fixed:
        predictor.prepare_fixed_multiview_states(states, conditioning_frame_idx=0)
        for view_idx, state in enumerate(states):
            assert predictor._get_obj_num(state) == max_objects
            assert state["fixed_batch_real_obj_count"] == real_counts[view_idx]
    else:
        for state in states:
            predictor.propagate_in_video_preflight(state)
    return states


@torch.inference_mode()
def sequential_reference_step(predictor, states, frame_idx: int):
    predictor.cache_image_features_batched(states, [frame_idx] * len(states))
    results = []

    for state in states:
        pred_masks = []
        real_count = predictor._get_obj_num(state)

        for obj_idx in range(real_count):
            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()

            obj_output_dict = state["output_dict_per_obj"][obj_idx]
            current_out, pred = predictor._run_single_frame_inference(
                inference_state=state,
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out
            state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": False}
            pred_masks.append(pred)

        if real_count:
            real_pred = torch.cat(pred_masks, dim=0)
            _, video_masks = predictor._get_orig_video_res_output(state, real_pred)
        else:
            video_masks = torch.empty(
                (0, 1, state["video_height"], state["video_width"]),
                device=state["device"],
            )

        results.append(video_masks)

    return results


@torch.inference_mode()
def allview_batch_step(predictor, states, frame_idx: int):
    out = predictor.propagate_fixed_multiview_step(
        states,
        frame_idx=frame_idx,
        reverse=False,
    )
    return [v["video_res_masks"] for v in out]


def timed_step(fn, predictor, states, frame_idx: int):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    wall0 = time.perf_counter()
    start.record()
    out = fn(predictor, states, frame_idx)
    end.record()
    torch.cuda.synchronize()

    wall_ms = (time.perf_counter() - wall0) * 1000.0
    gpu_ms = start.elapsed_time(end)
    return out, float(gpu_ms), float(wall_ms)


def cpu_copy_outputs(outputs):
    return [x.detach().float().cpu().contiguous() for x in outputs]


def run_worker(args):
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.video_root is None or args.result_path is None:
        raise ValueError("Worker requires --video-root and --result-path.")

    real_counts = parse_real_counts(args)
    fixed_batch = args.views * args.max_objects_per_view

    print("\n" + "=" * 78)
    print(f"WORKER: {args.worker.upper()}")
    print("=" * 78)
    print(f"views                : {args.views}")
    print(f"max objects/view     : {args.max_objects_per_view}")
    print(f"fixed tracking batch : {fixed_batch}")
    print(f"real objects/view    : {real_counts}")

    # One worker sees one batch regime only, so 16 is enough for finite
    # memory-history specializations without masking accidental shape mixing.
    torch._dynamo.config.recompile_limit = 16

    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
        vos_optimized=args.vos_optimized,
        fixed_num_views=args.views,
        max_objects_per_view=args.max_objects_per_view,
    )

    # Benchmark tracking core; avoid optional custom connected-component warnings.
    if hasattr(predictor, "fill_hole_area"):
        predictor.fill_hole_area = 0

    root = Path(args.video_root)
    video_dirs = [root / f"view_{i}" for i in range(args.views)]

    states = init_state_set(
        predictor,
        video_dirs,
        real_counts,
        args.max_objects_per_view,
        pad_fixed=(args.worker == "batched"),
    )

    step_fn = sequential_reference_step if args.worker == "sequential" else allview_batch_step

    print("Warmup/compile...")
    for frame_idx in range(1, 1 + args.warmup_frames):
        step_fn(predictor, states, frame_idx)
    torch.cuda.synchronize()

    gpu_times = []
    wall_times = []
    saved_outputs = []

    first_bench = 1 + args.warmup_frames
    total_frames = 1 + args.warmup_frames + args.benchmark_frames

    print("Benchmark...")
    for frame_idx in range(first_bench, total_frames):
        out, gpu_ms, wall_ms = timed_step(step_fn, predictor, states, frame_idx)
        gpu_times.append(gpu_ms)
        wall_times.append(wall_ms)
        saved_outputs.append(cpu_copy_outputs(out))

    payload = {
        "worker": args.worker,
        "gpu_times_ms": gpu_times,
        "wall_times_ms": wall_times,
        "gpu_mean_ms": sum(gpu_times) / len(gpu_times),
        "wall_mean_ms": sum(wall_times) / len(wall_times),
        "outputs": saved_outputs,
        "model_resolution": int(predictor.image_size),
    }
    torch.save(payload, args.result_path)

    print(f"{args.worker} mean GPU  : {payload['gpu_mean_ms']:.3f} ms")
    print(f"{args.worker} mean wall : {payload['wall_mean_ms']:.3f} ms")


def binary_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a > 0
    bb = b > 0
    inter = (aa & bb).sum().item()
    union = (aa | bb).sum().item()
    return 1.0 if union == 0 else inter / union


def compare_output_sequences(reference_frames, batched_frames):
    if len(reference_frames) != len(batched_frames):
        raise AssertionError("Benchmark frame count mismatch.")

    ious = []
    max_abs = 0.0
    mean_abs = []

    for ref_frame, bat_frame in zip(reference_frames, batched_frames):
        if len(ref_frame) != len(bat_frame):
            raise AssertionError("View count mismatch.")

        for ref_view, bat_view in zip(ref_frame, bat_frame):
            if ref_view.shape != bat_view.shape:
                raise AssertionError(
                    f"Output shape mismatch: {tuple(ref_view.shape)} vs {tuple(bat_view.shape)}"
                )

            for obj_idx in range(ref_view.shape[0]):
                ious.append(binary_iou(ref_view[obj_idx], bat_view[obj_idx]))

            diff = (ref_view.float() - bat_view.float()).abs()
            if diff.numel():
                max_abs = max(max_abs, float(diff.max().item()))
                mean_abs.append(float(diff.mean().item()))

    return (
        min(ious) if ious else 1.0,
        sum(ious) / len(ious) if ious else 1.0,
        max_abs,
        sum(mean_abs) / len(mean_abs) if mean_abs else 0.0,
    )


def worker_command(args, worker: str, video_root: Path, result_path: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--views", str(args.views),
        "--max-objects-per-view", str(args.max_objects_per_view),
        "--real-objects-per-view", args.real_objects_per_view,
        "--warmup-frames", str(args.warmup_frames),
        "--benchmark-frames", str(args.benchmark_frames),
        "--height", str(args.height),
        "--width", str(args.width),
        "--device", args.device,
        "--min-mask-iou", str(args.min_mask_iou),
        "--worker", worker,
        "--video-root", str(video_root),
        "--result-path", str(result_path),
    ]
    if args.vos_optimized:
        cmd.append("--vos-optimized")
    return cmd


def main_parent(args):
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    real_counts = parse_real_counts(args)
    total_frames = 1 + args.warmup_frames + args.benchmark_frames
    fixed_batch = args.views * args.max_objects_per_view

    print("EfficientTAM fixed all-view benchmark")
    print(f"  views                : {args.views}")
    print(f"  max objects/view     : {args.max_objects_per_view}")
    print(f"  fixed tracking batch : {fixed_batch}")
    print(f"  real objects/view    : {real_counts}")
    print("  benchmark isolation  : separate B=1 and B=N worker processes")

    with tempfile.TemporaryDirectory(prefix="etam_fixed_multiview_") as td:
        root = Path(td)
        video_root = root / "videos"
        video_root.mkdir(parents=True, exist_ok=True)

        for view_idx in range(args.views):
            make_video(
                video_root / f"view_{view_idx}",
                view_idx,
                total_frames,
                args.height,
                args.width,
            )

        seq_result = root / "sequential.pt"
        bat_result = root / "batched.pt"

        print("\nLaunching sequential B=1 worker...")
        subprocess.run(
            worker_command(args, "sequential", video_root, seq_result),
            check=True,
        )

        print("\nLaunching fixed all-view worker...")
        subprocess.run(
            worker_command(args, "batched", video_root, bat_result),
            check=True,
        )

        seq = torch.load(seq_result, map_location="cpu", weights_only=False)
        bat = torch.load(bat_result, map_location="cpu", weights_only=False)

        min_iou, mean_iou, max_abs, mean_abs = compare_output_sequences(
            seq["outputs"],
            bat["outputs"],
        )

        seq_gpu = float(seq["gpu_mean_ms"])
        bat_gpu = float(bat["gpu_mean_ms"])
        seq_wall = float(seq["wall_mean_ms"])
        bat_wall = float(bat["wall_mean_ms"])

        print("\n" + "=" * 78)
        print("FIXED ALL-VIEW / ALL-OBJECT BATCH RESULT")
        print("=" * 78)
        print(f"Model resolution              : {bat['model_resolution']}x{bat['model_resolution']}")
        print(f"Views                         : {args.views}")
        print(f"Max slots per view            : {args.max_objects_per_view}")
        print(f"Total fixed tracking batch    : {fixed_batch}")
        print(f"Real objects per view         : {real_counts}")
        print(f"Benchmark frames              : {args.benchmark_frames}")
        print()
        print(f"Current real-only B=1 GPU/frame: {seq_gpu:.3f} ms")
        print(f"All-view B={fixed_batch} GPU/frame       : {bat_gpu:.3f} ms")
        print(f"GPU speedup                   : {seq_gpu / bat_gpu:.3f}x")
        print()
        print(f"Current real-only wall/frame  : {seq_wall:.3f} ms")
        print(f"All-view wall/frame           : {bat_wall:.3f} ms")
        print(f"Wall speedup                  : {seq_wall / bat_wall:.3f}x")
        print()
        print(f"Minimum real-mask IoU         : {min_iou:.6f}")
        print(f"Mean real-mask IoU            : {mean_iou:.6f}")
        print(f"Maximum abs logit error       : {max_abs:.6f}")
        print(f"Mean abs logit error          : {mean_abs:.6f}")
        print("=" * 78)

        if min_iou < args.min_mask_iou:
            raise AssertionError(
                f"Correctness failed: min IoU={min_iou:.6f} < {args.min_mask_iou:.6f}"
            )

        print("PASS: fixed all-view/all-object batching is functional.")


def main():
    args = parse_args()
    if args.worker is None:
        main_parent(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()