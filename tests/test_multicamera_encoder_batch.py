from __future__ import annotations

import argparse
import time

import torch

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "configs/efficienttam/"
            "efficienttam_ti_512x512.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "efficienttam_ti_512x512.pt"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--vos-optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=2e-2,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
    )

    return parser.parse_args()


def make_fake_state(
    image: torch.Tensor,
    device: torch.device,
):
    """
    Minimal EfficientTAM state required by
    cache_image_features_batched().
    """

    return {
        "images": [image],
        "device": device,
        "cached_features": {},
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_cuda(
    fn,
    device,
    warmup,
    runs,
):
    for _ in range(warmup):
        fn()

    synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        for _ in range(runs):
            fn()

        end.record()

        torch.cuda.synchronize(device)

        return start.elapsed_time(end) / runs

    start_time = time.perf_counter()

    for _ in range(runs):
        fn()

    return (
        (time.perf_counter() - start_time)
        * 1000.0
        / runs
    )


def compare_backbone(
    reference,
    candidate,
    atol,
    rtol,
):
    keys = (
        "backbone_fpn",
        "vision_pos_enc",
    )

    max_abs_error = 0.0

    for key in keys:
        reference_list = reference[key]
        candidate_list = candidate[key]

        assert len(reference_list) == len(candidate_list)

        for level, (ref, cand) in enumerate(
            zip(reference_list, candidate_list)
        ):
            # Position encoding may be stored once for the whole batch.
            if ref.shape != cand.shape:
                raise AssertionError(
                    f"{key}[{level}] shape mismatch: "
                    f"{tuple(ref.shape)} vs "
                    f"{tuple(cand.shape)}"
                )

            diff = (
                ref.float()
                - cand.float()
            ).abs()

            max_abs_error = max(
                max_abs_error,
                float(diff.max().item()),
            )

            torch.testing.assert_close(
                cand.float(),
                ref.float(),
                atol=atol,
                rtol=rtol,
            )

    return max_abs_error


def main():
    args = parse_args()

    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError(
            "This benchmark is intended for CUDA."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Building EfficientTAM...")
    print(f"  config      : {args.config}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {device}")
    print(f"  batch size  : {args.batch_size}")
    print(
        f"  vos optimized: "
        f"{args.vos_optimized}"
    )

    predictor = build_efficienttam_video_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device,
        vos_optimized=args.vos_optimized,
    )

    predictor.eval()

    image_size = int(predictor.image_size)

    print(
        f"EfficientTAM model resolution: "
        f"{image_size}x{image_size}"
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(1234)

    images = [
        torch.randn(
            3,
            image_size,
            image_size,
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(args.batch_size)
    ]

    states = [
        make_fake_state(
            image=image,
            device=device,
        )
        for image in images
    ]

    autocast_context = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    )

    with torch.inference_mode(), autocast_context:

        # -------------------------------------------------------------
        # 1. Correctness test
        # -------------------------------------------------------------

        print("\n[1/3] Computing sequential B=1 references...")

        sequential_outputs = []

        for image in images:
            image_gpu = (
                image
                .to(device)
                .float()
                .unsqueeze(0)
            )

            out = predictor.forward_image(
                image_gpu
            )

            sequential_outputs.append(out)

        synchronize(device)

        print("[2/3] Computing batched encoder output...")

        predictor.cache_image_features_batched(
            inference_states=states,
            frame_indices=[
                0
                for _ in range(args.batch_size)
            ],
        )

        synchronize(device)

        global_max_error = 0.0

        for camera_idx in range(args.batch_size):
            _, cached_backbone = (
                states[camera_idx][
                    "cached_features"
                ][0]
            )

            error = compare_backbone(
                reference=sequential_outputs[
                    camera_idx
                ],
                candidate=cached_backbone,
                atol=args.atol,
                rtol=args.rtol,
            )

            global_max_error = max(
                global_max_error,
                error,
            )

            print(
                f"  camera {camera_idx}: "
                f"max abs error = {error:.6g}"
            )

        print(
            "Correctness test PASSED. "
            f"Global max abs error = "
            f"{global_max_error:.6g}"
        )

        # -------------------------------------------------------------
        # 2. Cache test
        # -------------------------------------------------------------

        print("\n[3/3] Verifying normal _get_image_feature() uses cache...")

        original_forward_image = predictor.forward_image

        forward_calls = 0

        def counted_forward_image(*args, **kwargs):
            nonlocal forward_calls
            forward_calls += 1
            return original_forward_image(
                *args,
                **kwargs,
            )

        predictor.forward_image = counted_forward_image

        for state in states:
            predictor._get_image_feature(
                inference_state=state,
                frame_idx=0,
                batch_size=1,
            )

        assert forward_calls == 0, (
            "_get_image_feature() unexpectedly reran the encoder "
            f"{forward_calls} time(s)"
        )

        predictor.forward_image = original_forward_image

        print("Cache reuse test PASSED.")

        # -------------------------------------------------------------
        # 3. Performance benchmark
        # -------------------------------------------------------------

        def sequential_encoder():
            for image in images:
                image_gpu = (
                    image
                    .to(device)
                    .float()
                    .unsqueeze(0)
                )

                predictor.forward_image(
                    image_gpu
                )

        def batched_encoder():
            predictor.cache_image_features_batched(
                inference_states=states,
                frame_indices=[
                    0
                    for _ in range(args.batch_size)
                ],
            )

        print(
            "\nBenchmarking encoder..."
        )

        sequential_ms = benchmark_cuda(
            fn=sequential_encoder,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )

        batched_ms = benchmark_cuda(
            fn=batched_encoder,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )

    speedup = (
        sequential_ms / batched_ms
    )

    pair_fps = (
        1000.0 / batched_ms
    )

    print()
    print("=" * 72)
    print("MULTI-CAMERA ENCODER BATCH RESULT")
    print("=" * 72)

    print(
        f"Model resolution       : "
        f"{image_size}x{image_size}"
    )

    print(
        f"Camera batch size      : "
        f"{args.batch_size}"
    )

    print(
        f"Sequential B=1 x "
        f"{args.batch_size:<2}   : "
        f"{sequential_ms:.3f} ms"
    )

    print(
        f"Batched B="
        f"{args.batch_size:<2}          : "
        f"{batched_ms:.3f} ms"
    )

    print(
        f"Encoder speedup        : "
        f"{speedup:.3f}x"
    )

    print(
        f"Batched camera-pair FPS: "
        f"{pair_fps:.2f} Hz"
    )

    print("=" * 72)

    if batched_ms < sequential_ms:
        print(
            "PASS: batched encoder is faster "
            "than two sequential encoder calls."
        )
    else:
        print(
            "WARNING: batching did not improve latency "
            "on this configuration."
        )


if __name__ == "__main__":
    main()