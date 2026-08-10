#!/usr/bin/env python3
"""Compare EfficientTAM global torch.compile default vs max-autotune.

The flag under test lives on EfficientTAMBase and controls every compiled
submodule that previously used ``mode="max-autotune"``:
  - image_encoder.forward
  - memory_encoder.forward
  - sam_prompt_encoder.forward
  - sam_mask_decoder.forward

memory_attention intentionally remains ``mode="default"`` in both cases
because that is the known-stable configuration for the fixed-batch path.

Examples
--------
Compare both modes in fresh Python processes:

  python tests/benchmark_vos_compile_mode.py \
      --checkpoint checkpoints/efficienttam_s_512x512.pt \
      --execution-mode fixed_batch \
      --max-objects-per-view 3 \
      --objects 2

Run only one mode:

  python tests/benchmark_vos_compile_mode.py ... --mode default
  python tests/benchmark_vos_compile_mode.py ... --mode max-autotune

For a stricter startup/compile-time comparison, isolate the TorchInductor disk
cache for each child process:

  python tests/benchmark_vos_compile_mode.py ... --fresh-compile-cache

The steady-state CUDA timings are valid with or without a fresh compile cache.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)

RESULT_PREFIX = "__EFFICIENTTAM_COMPILE_BENCH_RESULT__="


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="configs/efficienttam/efficienttam_s_512x512.yaml",
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/efficienttam_s_512x512.pt",
    )
    p.add_argument(
        "--mode",
        choices=("both", "default", "max-autotune"),
        default="both",
        help="'both' launches two clean Python child processes.",
    )
    p.add_argument(
        "--execution-mode",
        choices=("fixed_batch", "sequential"),
        default="fixed_batch",
    )
    p.add_argument("--views", type=int, default=2)
    p.add_argument("--max-objects-per-view", type=int, default=3)
    p.add_argument("--objects", type=int, default=2)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--measure-frames", type=int, default=100)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--fresh-compile-cache",
        action="store_true",
        help=(
            "Use a separate empty TORCHINDUCTOR_CACHE_DIR per mode. This makes "
            "startup/compile wall time more meaningful but can make both runs slow."
        ),
    )
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


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
    palette = [(220, 70, 65), (70, 210, 105), (70, 120, 235)]
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
    seed_masks = []
    for view_idx in range(views):
        view_dir = root / f"view_{view_idx}"
        view_dir.mkdir(parents=True, exist_ok=True)
        video_dirs.append(view_dir)
        first_masks = None
        for frame_idx in range(frames):
            image, masks = synthetic_frame_and_masks(
                view_idx, frame_idx, width, height, num_objects
            )
            image.save(view_dir / f"{frame_idx}.jpg", quality=95)
            if frame_idx == 0:
                first_masks = masks
        seed_masks.append(first_masks)
    return video_dirs, seed_masks


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


def seed_states(predictor, states, seed_masks):
    for state, masks in zip(states, seed_masks):
        for obj_idx, mask in enumerate(masks):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_idx + 1,
                mask=mask,
            )
    predictor.prepare_multiview_states(states, conditioning_frame_idx=0)


def one_normal_cycle(predictor, states, frame_idx: int):
    start = torch.cuda.Event(enable_timing=True)
    after_encode = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    snapshot = predictor.snapshot_multiview_image_features(states, frame_idx)
    after_encode.record()
    predictor.propagate_multiview_step(
        states,
        frame_idx,
        image_feature_snapshot=snapshot,
    )
    end.record()
    end.synchronize()

    return {
        "encode_gpu_ms": float(start.elapsed_time(after_encode)),
        "propagate_gpu_ms": float(after_encode.elapsed_time(end)),
        "cycle_gpu_ms": float(start.elapsed_time(end)),
        "cycle_wall_ms": 1000.0 * (time.perf_counter() - wall_start),
    }


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def run_child(args: argparse.Namespace) -> dict:
    if args.mode not in ("default", "max-autotune"):
        raise ValueError("child mode must be default or max-autotune")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.views <= 0:
        raise ValueError("--views must be > 0")
    if not 1 <= args.objects <= args.max_objects_per_view:
        raise ValueError("--objects must be in [1, max_objects_per_view]")
    if args.warmup_frames < 1 or args.measure_frames < 1:
        raise ValueError("warmup and measure frame counts must be >= 1")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_frames = 1 + args.warmup_frames + args.measure_frames
    use_max_autotune = args.mode == "max-autotune"

    print("\n" + "=" * 78)
    print(f"EfficientTAM global compile benchmark: {args.mode}")
    print("=" * 78)
    print(
        f"execution={args.execution_mode}, views={args.views}, "
        f"real_objects/view={args.objects}, fixed_slots/view={args.max_objects_per_view}, "
        f"warmup={args.warmup_frames}, measured={args.measure_frames}"
    )
    print(
        "Flag scope: image_encoder + memory_encoder + sam_prompt_encoder + "
        "sam_mask_decoder; memory_attention stays default."
    )

    with tempfile.TemporaryDirectory(prefix=f"efficienttam_compile_{args.mode}_") as tmp:
        root = Path(tmp)
        video_dirs, seed_masks = make_synthetic_videos(
            root=root,
            views=args.views,
            frames=total_frames,
            width=args.width,
            height=args.height,
            num_objects=args.objects,
        )

        build_t0 = time.perf_counter()
        predictor = build_efficienttam_video_predictor(
            config_file=args.config,
            ckpt_path=str(checkpoint),
            device="cuda",
            vos_optimized=True,
            apply_postprocessing=False,
            execution_mode=args.execution_mode,
            fixed_num_views=args.views,
            max_objects_per_view=args.max_objects_per_view,
            use_max_autotune=use_max_autotune,
            hydra_overrides_extra=[
                "++model.fill_hole_area=0",
                "++model.non_overlap_masks_for_mem_enc=false",
            ],
        )
        build_wall_s = time.perf_counter() - build_t0

        actual_flag = bool(getattr(predictor, "use_max_autotune", None))
        if actual_flag != use_max_autotune:
            raise RuntimeError(
                f"predictor flag mismatch: requested={use_max_autotune}, actual={actual_flag}"
            )
        expected_compile_mode = "max-autotune" if use_max_autotune else "default"
        actual_compile_mode = getattr(predictor, "compile_optimization_mode", None)
        if actual_compile_mode != expected_compile_mode:
            raise RuntimeError(
                "predictor compile mode mismatch: "
                f"requested={expected_compile_mode}, actual={actual_compile_mode}"
            )
        print(f"Effective compile mode: {actual_compile_mode}")

        states = make_states(predictor, video_dirs)

        # This wall interval intentionally includes seeding and all lazy torch.compile
        # work triggered by the first temporal shapes.
        warmup_t0 = time.perf_counter()
        seed_states(predictor, states, seed_masks)
        for frame_idx in range(1, 1 + args.warmup_frames):
            one_normal_cycle(predictor, states, frame_idx)
        torch.cuda.synchronize()
        warmup_wall_s = time.perf_counter() - warmup_t0

        records = []
        start_frame = 1 + args.warmup_frames
        end_frame = start_frame + args.measure_frames
        for frame_idx in range(start_frame, end_frame):
            records.append(one_normal_cycle(predictor, states, frame_idx))

        result = {
            "mode": args.mode,
            "use_max_autotune": use_max_autotune,
            "compile_optimization_mode": actual_compile_mode,
            "execution_mode": args.execution_mode,
            "views": args.views,
            "objects": args.objects,
            "max_objects_per_view": args.max_objects_per_view,
            "warmup_frames": args.warmup_frames,
            "measure_frames": args.measure_frames,
            "build_wall_s": build_wall_s,
            "warmup_wall_s": warmup_wall_s,
            "encode_gpu_ms": stats([x["encode_gpu_ms"] for x in records]),
            "propagate_gpu_ms": stats([x["propagate_gpu_ms"] for x in records]),
            "cycle_gpu_ms": stats([x["cycle_gpu_ms"] for x in records]),
            "cycle_wall_ms": stats([x["cycle_wall_ms"] for x in records]),
            "samples_cycle_gpu_ms": [float(x["cycle_gpu_ms"]) for x in records],
        }

        print("\nSteady-state results")
        print(f"  build model:       {build_wall_s:.3f} s wall")
        print(f"  seed + warmup:     {warmup_wall_s:.3f} s wall")
        print(
            "  encoder B=views:   "
            f"{result['encode_gpu_ms']['median']:.3f} ms median GPU "
            f"(p95 {result['encode_gpu_ms']['p95']:.3f})"
        )
        print(
            "  propagation:       "
            f"{result['propagate_gpu_ms']['median']:.3f} ms median GPU "
            f"(p95 {result['propagate_gpu_ms']['p95']:.3f})"
        )
        print(
            "  normal cycle:      "
            f"{result['cycle_gpu_ms']['median']:.3f} ms median GPU "
            f"(p95 {result['cycle_gpu_ms']['p95']:.3f})"
        )
        print(
            "  normal wall cycle: "
            f"{result['cycle_wall_ms']['median']:.3f} ms median "
            f"(p95 {result['cycle_wall_ms']['p95']:.3f})"
        )

        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return result


def child_command(args: argparse.Namespace, mode: str):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_child",
        "--mode",
        mode,
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--execution-mode",
        args.execution_mode,
        "--views",
        str(args.views),
        "--max-objects-per-view",
        str(args.max_objects_per_view),
        "--objects",
        str(args.objects),
        "--warmup-frames",
        str(args.warmup_frames),
        "--measure-frames",
        str(args.measure_frames),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    return cmd


def launch_child(args: argparse.Namespace, mode: str) -> dict:
    env = os.environ.copy()
    cache_tmp = None
    if args.fresh_compile_cache:
        cache_tmp = tempfile.TemporaryDirectory(prefix=f"torchinductor_{mode}_")
        env["TORCHINDUCTOR_CACHE_DIR"] = cache_tmp.name
        print(f"[{mode}] isolated TORCHINDUCTOR_CACHE_DIR={cache_tmp.name}")

    proc = subprocess.Popen(
        child_command(args, mode),
        cwd=str(Path.cwd()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    result = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX) :])
    returncode = proc.wait()
    if cache_tmp is not None:
        cache_tmp.cleanup()
    if returncode != 0:
        raise RuntimeError(f"{mode} child exited with code {returncode}")
    if result is None:
        raise RuntimeError(f"{mode} child did not emit a result")
    return result


def print_comparison(max_result: dict, default_result: dict):
    max_cycle = max_result["cycle_gpu_ms"]["median"]
    def_cycle = default_result["cycle_gpu_ms"]["median"]
    max_encode = max_result["encode_gpu_ms"]["median"]
    def_encode = default_result["encode_gpu_ms"]["median"]
    max_prop = max_result["propagate_gpu_ms"]["median"]
    def_prop = default_result["propagate_gpu_ms"]["median"]

    encode_delta = def_encode - max_encode
    cycle_delta = def_cycle - max_cycle
    prop_delta = def_prop - max_prop
    encode_pct = 100.0 * encode_delta / max_encode if max_encode else float("nan")
    cycle_pct = 100.0 * cycle_delta / max_cycle if max_cycle else float("nan")
    prop_pct = 100.0 * prop_delta / max_prop if max_prop else float("nan")
    startup_saved = max_result["warmup_wall_s"] - default_result["warmup_wall_s"]

    print("\n" + "=" * 78)
    print("DEFAULT vs MAX-AUTOTUNE")
    print("=" * 78)
    print(
        f"max-autotune normal cycle median : {max_cycle:.3f} ms GPU "
        f"(p95 {max_result['cycle_gpu_ms']['p95']:.3f})"
    )
    print(
        f"default normal cycle median      : {def_cycle:.3f} ms GPU "
        f"(p95 {default_result['cycle_gpu_ms']['p95']:.3f})"
    )
    print(
        f"default cycle penalty            : {cycle_delta:+.3f} ms "
        f"({cycle_pct:+.2f}%)"
    )
    print()
    print(f"max-autotune encoder median      : {max_encode:.3f} ms GPU")
    print(f"default encoder median           : {def_encode:.3f} ms GPU")
    print(
        f"default encoder penalty          : {encode_delta:+.3f} ms "
        f"({encode_pct:+.2f}%)"
    )
    print()
    print(f"max-autotune propagation median  : {max_prop:.3f} ms GPU")
    print(f"default propagation median       : {def_prop:.3f} ms GPU")
    print(
        f"default propagation penalty      : {prop_delta:+.3f} ms "
        f"({prop_pct:+.2f}%)"
    )
    print()
    print(f"max-autotune seed+warmup wall    : {max_result['warmup_wall_s']:.3f} s")
    print(f"default seed+warmup wall         : {default_result['warmup_wall_s']:.3f} s")
    print(f"startup time saved by default    : {startup_saved:+.3f} s")
    if startup_saved > 0 and cycle_delta > 0:
        break_even_frames = startup_saved * 1000.0 / cycle_delta
        print(f"rough break-even                 : {break_even_frames:.0f} normal frames")
    print()
    print(
        "Interpretation: a positive penalty means default is slower at steady state; "
        "a positive startup saving means default finished lazy compile/warmup sooner."
    )
    if not np.isfinite(startup_saved):
        return
    print(
        "Note: startup timing is only a clean compile comparison when "
        "--fresh-compile-cache is used."
    )


def main():
    args = parse_args()
    if args._child:
        run_child(args)
        return

    if args.mode in ("default", "max-autotune"):
        # Keep single-mode use simple and in-process.
        run_child(args)
        return

    # Separate processes are important: each mode gets a newly constructed model and
    # independent torch.compile wrappers instead of inheriting state from the other.
    max_result = launch_child(args, "max-autotune")
    default_result = launch_child(args, "default")
    print_comparison(max_result=max_result, default_result=default_result)


if __name__ == "__main__":
    main()
