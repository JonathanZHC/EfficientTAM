# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

# RTX 50-series / TorchInductor: prune max-autotune candidates that exceed
# the device shared-memory limit before benchmarking them. setdefault keeps
# an explicit shell override possible. This must be set before importing torch.
os.environ.setdefault(
    "TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM",
    "1",
)

import efficient_track_anything

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

if os.path.isdir(
    os.path.join(efficient_track_anything.__path__[0], "efficient_track_anything")
):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the EfficientTAM repository "
    )

# Working on putting efficient track anything models on Facebook Hugging Face Hub.
# This is just for demonstration.
# Please download efficient track anything models from https://huggingface.co/yunyangx/efficient-track-anything.
# and use build_efficienttam/build_efficienttam_video_predictor for loading them.
HF_MODEL_ID_TO_FILENAMES = {
    "facebook/efficienttam_s": (
        "configs/efficienttam/efficienttam_s.yaml",
        "efficienttam_s.pt",
    ),
    "facebook/efficienttam_s_512x512": (
        "configs/efficienttam/efficienttam_s_512x512.yaml",
        "efficienttam_s_512x512.pt",
    ),
    "facebook/efficienttam_s_1": (
        "configs/efficienttam/efficienttam_s_1.yaml",
        "efficienttam_s_1.pt",
    ),
    "facebook/efficienttam_s_2": (
        "configs/efficienttam/efficienttam_s_2.yaml",
        "efficienttam_s_2.pt",
    ),
    "facebook/efficienttam_ti": (
        "configs/efficienttam/efficienttam_ti.yaml",
        "efficienttam_ti.pt",
    ),
    "facebook/efficienttam_ti_512x512": (
        "configs/efficienttam/efficienttam_ti_512x512.yaml",
        "efficienttam_ti_512x512.pt",
    ),
    "facebook/efficienttam_ti_1": (
        "configs/efficienttam/efficienttam_ti_1.yaml",
        "efficienttam_ti_1.pt",
    ),
    "facebook/efficienttam_ti_2": (
        "configs/efficienttam/efficienttam_ti_2.yaml",
        "efficienttam_ti_2.pt",
    ),
}


def build_efficienttam(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    use_max_autotune=True,
    **kwargs,
):
    hydra_overrides_extra = list(hydra_overrides_extra or [])
    hydra_overrides_extra.append(
        f"++model.use_max_autotune={str(bool(use_max_autotune)).lower()}"
    )

    if apply_postprocessing:
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_efficienttam_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=None,
    apply_postprocessing=True,
    vos_optimized=False,
    execution_mode="sequential",
    fixed_num_views=2,
    max_objects_per_view=4,
    use_max_autotune=True,
    **kwargs,
):
    hydra_overrides_extra = list(hydra_overrides_extra or [])

    execution_mode = str(execution_mode).strip().lower()
    valid_execution_modes = {"sequential", "fixed_batch"}
    if execution_mode not in valid_execution_modes:
        raise ValueError(
            f"Unsupported execution_mode={execution_mode!r}. "
            f"Expected one of {sorted(valid_execution_modes)}."
        )

    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).major < 8:
        print("Disable torch compile due to unsupported GPU.")
        hydra_overrides_extra.append("++model.compile_image_encoder=False")
        vos_optimized = False

    hydra_overrides = [
        "++model._target_=efficient_track_anything.efficienttam_video_predictor.EfficientTAMVideoPredictor",
        f"++model.execution_mode={execution_mode}",
        f"++model.fixed_num_views={int(fixed_num_views)}",
        f"++model.max_objects_per_view={int(max_objects_per_view)}",
        f"++model.use_max_autotune={str(bool(use_max_autotune)).lower()}",
    ]
    if vos_optimized:
        hydra_overrides = [
            "++model._target_=efficient_track_anything.efficienttam_video_predictor.EfficientTAMVideoPredictorVOS",
            "++model.compile_image_encoder=True",  # Let efficienttam_base handle this
            f"++model.execution_mode={execution_mode}",
            f"++model.fixed_num_views={int(fixed_num_views)}",
            f"++model.max_objects_per_view={int(max_objects_per_view)}",
            f"++model.use_max_autotune={str(bool(use_max_autotune)).lower()}",
        ]

    if apply_postprocessing:
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    if ckpt_path is not None:
        _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_efficienttam_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_efficienttam(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_efficienttam_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_efficienttam_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint successfully")
