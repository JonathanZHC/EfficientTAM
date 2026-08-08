# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F

from efficient_track_anything.modeling.efficienttam_base import (
    EfficientTAMBase,
    NO_OBJ_SCORE,
)
from efficient_track_anything.utils.misc import (
    concat_points,
    fill_holes_in_mask_scores,
    load_video_frames,
)

from tqdm import tqdm


EXECUTION_MODE_SEQUENTIAL = "sequential"
EXECUTION_MODE_FIXED_BATCH = "fixed_batch"
VALID_EXECUTION_MODES = {
    EXECUTION_MODE_SEQUENTIAL,
    EXECUTION_MODE_FIXED_BATCH,
}


class EfficientTAMVideoPredictor(EfficientTAMBase):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # Multi-view execution strategy. Choose once when building the predictor.
        # "sequential": per-view image encoding + per-object B=1 propagation.
        # "fixed_batch": all views are encoded together and all fixed object slots
        # are propagated in one stable batch.
        execution_mode=EXECUTION_MODE_SEQUENTIAL,
        # Fixed-shape multi-view propagation hyperparameters. They are only used by
        # execution_mode="fixed_batch" and intentionally live at model level so
        # torch.compile sees one stable tracking batch size.
        fixed_num_views=2,
        max_objects_per_view=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        self.execution_mode = str(execution_mode).strip().lower()
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(
                f"Unsupported execution_mode={self.execution_mode!r}. "
                f"Expected one of {sorted(VALID_EXECUTION_MODES)}."
            )

        self.fixed_num_views = int(fixed_num_views)
        self.max_objects_per_view = int(max_objects_per_view)
        if self.fixed_num_views <= 0:
            raise ValueError("fixed_num_views must be > 0")
        if self.max_objects_per_view <= 0:
            raise ValueError("max_objects_per_view must be > 0")

    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "EfficientTAMVideoPredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (EfficientTAMVideoPredictor): The loaded model.
        """
        from efficient_track_anything.build_efficienttam import (
            build_efficienttam_video_predictor_hf,
        )

        efficienttam_model = build_efficienttam_video_predictor_hf(model_id, **kwargs)
        return efficienttam_model

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # We always allow adding new objects (including after tracking starts).
        allow_new_object = True
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["frames_tracked_per_obj"][obj_idx] = {}
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how EfficientTAM is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def add_new_points(self, *args, **kwargs):
        """Deprecated method. Please use `add_new_points_or_box` instead."""
        return self.add_new_points_or_box(*args, **kwargs)

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        consolidate_at_video_res=False,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask

        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Check and make sure that every object has received input points or masks.
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError(
                "No input points or masks are provided for any object; please add inputs first."
            )

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                # Separately consolidate conditioning and non-conditioning temp outputs
                storage_key = (
                    "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                )
                # Find all the frames that contain temporary outputs for any objects
                # (these should be the frames that have just received clicks for mask inputs
                # via `add_new_points_or_box` or `add_new_mask`)
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder on the temporary outputs (if the memory feature is missing)
                    if out["maskmem_features"] is None:
                        high_res_masks = torch.nn.functional.interpolate(
                            out["pred_masks"].to(inference_state["device"]),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            high_res_masks=high_res_masks,
                            object_score_logits=out["object_score_logits"],
                            # these frames are what the user interacted with
                            is_mask_from_pts=True,
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc

                    obj_output_dict[storage_key][frame_idx] = out
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )

                # clear temporary outputs in `temp_output_dict_per_obj`
                obj_temp_output_dict[storage_key].clear()

            # check and make sure that every object has received input points or masks
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            if len(obj_output_dict["cond_frame_outputs"]) == 0:
                obj_id = self._obj_idx_to_id(inference_state, obj_idx)
                raise RuntimeError(
                    f"No input points or masks are provided for object id {obj_id}; please add inputs first."
                )
            # edge case: if an output is added to "cond_frame_outputs", we remove any prior
            # output on the same frame in "non_cond_frame_outputs"
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    if self.clear_non_cond_mem_around_input:
                        # clear non-conditioning memory of the surrounding frames
                        self._clear_obj_non_cond_mem_around_input(
                            inference_state, frame_idx, obj_idx
                        )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj[obj_idx] = pred_masks

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks


    # ---------------------------------------------------------------------
    # Unified multi-view execution API
    # ---------------------------------------------------------------------

    @property
    def fixed_tracking_batch_size(self):
        """Return the compiled object batch size used by fixed-batch mode."""
        return self.fixed_num_views * self.max_objects_per_view

    def execution_summary(self):
        """Return a small integration-friendly description of the execution mode."""
        summary = {
            "execution_mode": self.execution_mode,
            "fixed_num_views": self.fixed_num_views,
            "max_objects_per_view": self.max_objects_per_view,
        }
        if self.execution_mode == EXECUTION_MODE_FIXED_BATCH:
            summary["tracking_batch_size"] = self.fixed_tracking_batch_size
        return summary

    @staticmethod
    def _mark_cudagraph_step_begin():
        """Mark one logical inference step when PyTorch CUDAGraphs are active."""
        compiler = getattr(torch, "compiler", None)
        marker = getattr(compiler, "cudagraph_mark_step_begin", None)
        if marker is not None:
            marker()

    @torch.inference_mode()
    def prepare_multiview_states(
        self,
        inference_states,
        conditioning_frame_idx=0,
    ):
        """
        Prepare a list of view states for the selected execution strategy.

        Call this once after seeding the real objects on the conditioning frame.
        The same integration code can then call ``propagate_multiview_step`` for
        either execution mode.
        """
        if not inference_states:
            raise ValueError("inference_states must contain at least one view")

        if self.execution_mode == EXECUTION_MODE_FIXED_BATCH:
            self.prepare_fixed_multiview_states(
                inference_states,
                conditioning_frame_idx=conditioning_frame_idx,
            )
        else:
            for state in inference_states:
                self.propagate_in_video_preflight(state)
                state["multiview_prepared"] = True
                state["multiview_execution_mode"] = self.execution_mode

        return inference_states

    @torch.inference_mode()
    def propagate_sequential_multiview_step(
        self,
        inference_states,
        frame_idx,
        reverse=False,
    ):
        """
        Original multi-view execution path: per-view encoder + per-object B=1.

        With two views and three objects per view this corresponds to the old
        ``2E + 6B`` structure. Image features are cached inside each view state,
        so the first object of a view runs the encoder and the remaining objects
        reuse that view's cached features.
        """
        results = []

        for view_idx, state in enumerate(inference_states):
            if not state.get("multiview_prepared", False):
                raise RuntimeError(
                    "Call prepare_multiview_states(...) once after seeding "
                    "objects and before multi-view propagation."
                )

            obj_ids = list(state["obj_ids"])
            num_objects = self._get_obj_num(state)
            pred_masks_per_obj = []

            for obj_idx in range(num_objects):
                obj_output_dict = state["output_dict_per_obj"][obj_idx]

                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                    pred_masks = current_out["pred_masks"].to(
                        state["device"], non_blocking=True
                    )
                    if self.clear_non_cond_mem_around_input:
                        self._clear_obj_non_cond_mem_around_input(
                            state, frame_idx, obj_idx
                        )
                else:
                    # Each B=1 object propagation is one logical compiled/CUDAGraph
                    # invocation. Explicitly mark the boundary to avoid graph-managed
                    # outputs being reused across object calls.
                    self._mark_cudagraph_step_begin()
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = current_out

                state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }
                pred_masks_per_obj.append(pred_masks)

            if pred_masks_per_obj:
                all_pred_masks = (
                    torch.cat(pred_masks_per_obj, dim=0)
                    if len(pred_masks_per_obj) > 1
                    else pred_masks_per_obj[0]
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    state, all_pred_masks
                )
            else:
                video_res_masks = torch.empty(
                    (0, 1, state["video_height"], state["video_width"]),
                    device=state["device"],
                )

            results.append(
                {
                    "view_idx": view_idx,
                    "frame_idx": frame_idx,
                    "obj_ids": obj_ids,
                    "video_res_masks": video_res_masks,
                    "num_real_objects": num_objects,
                    "num_dummy_objects": 0,
                    "execution_mode": EXECUTION_MODE_SEQUENTIAL,
                }
            )

        return results

    @torch.inference_mode()
    def propagate_multiview_step(
        self,
        inference_states,
        frame_idx,
        reverse=False,
    ):
        """Propagate one synchronized multi-view frame using the configured mode."""
        if self.execution_mode == EXECUTION_MODE_FIXED_BATCH:
            return self.propagate_fixed_multiview_step(
                inference_states=inference_states,
                frame_idx=frame_idx,
                reverse=reverse,
            )
        return self.propagate_sequential_multiview_step(
            inference_states=inference_states,
            frame_idx=frame_idx,
            reverse=reverse,
        )

    @torch.inference_mode()
    def propagate_multiview(
        self,
        inference_states,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Generator wrapper around :meth:`propagate_multiview_step`."""
        if not inference_states:
            return

        if max_frame_num_to_track is None:
            max_frame_num_to_track = min(
                state["num_frames"] for state in inference_states
            )

        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track + 1, 0)
            processing_order = range(start_frame_idx, end - 1, -1)
        else:
            last_frame = min(
                min(state["num_frames"] for state in inference_states) - 1,
                start_frame_idx + max_frame_num_to_track - 1,
            )
            processing_order = range(start_frame_idx, last_frame + 1)

        for frame_idx in processing_order:
            yield frame_idx, self.propagate_multiview_step(
                inference_states=inference_states,
                frame_idx=frame_idx,
                reverse=reverse,
            )


    # ---------------------------------------------------------------------
    # Fixed-shape all-view / all-object propagation
    # ---------------------------------------------------------------------

    def _fixed_batch_slot_order(self, inference_states):
        """Return view-major (view_idx, obj_idx) slot order."""
        if len(inference_states) != self.fixed_num_views:
            raise ValueError(
                f"Expected exactly {self.fixed_num_views} views, got "
                f"{len(inference_states)}. The number of views is part of the "
                "compiled batch shape."
            )
        slots = []
        for view_idx, state in enumerate(inference_states):
            n = self._get_obj_num(state)
            if n != self.max_objects_per_view:
                raise RuntimeError(
                    f"View {view_idx} has {n} object slots, expected exactly "
                    f"{self.max_objects_per_view}. Call "
                    "pad_state_to_fixed_object_slots(...) before propagation."
                )
            for obj_idx in range(self.max_objects_per_view):
                slots.append((view_idx, obj_idx))
        return slots

    @torch.inference_mode()
    def pad_state_to_fixed_object_slots(
        self,
        inference_state,
        frame_idx=0,
        view_idx=0,
        dummy_prefix="__efficienttam_dummy__",
    ):
        """
        Pad one view to ``max_objects_per_view`` using real empty-mask objects.

        Dummy slots are created once, on the same conditioning frame as the real
        objects. They subsequently propagate like normal objects, but callers omit
        them from returned results. This keeps every slot's temporal history and
        tensor shape aligned, which is what makes a single fixed B graph possible.
        """
        current_n = self._get_obj_num(inference_state)
        if current_n > self.max_objects_per_view:
            raise RuntimeError(
                f"View {view_idx} already has {current_n} objects but "
                f"max_objects_per_view={self.max_objects_per_view}."
            )

        if "fixed_batch_real_obj_ids" not in inference_state:
            inference_state["fixed_batch_real_obj_ids"] = list(
                inference_state["obj_ids"]
            )
            inference_state["fixed_batch_real_obj_count"] = current_n
            inference_state["fixed_batch_dummy_obj_ids"] = []
        else:
            # Padding is idempotent. Do not reinterpret already-created dummy slots
            # as real detections on a later call.
            current_n = self._get_obj_num(inference_state)

        if current_n == self.max_objects_per_view:
            return list(inference_state.get("fixed_batch_dummy_obj_ids", []))

        # For exact batched/sequential equivalence, all real objects should have
        # been initialized on this same conditioning frame. This is the natural
        # setup when a detector refresh seeds all instances together.
        for obj_idx in range(inference_state["fixed_batch_real_obj_count"]):
            has_prompt = (
                frame_idx in inference_state["mask_inputs_per_obj"][obj_idx]
                or frame_idx in inference_state["point_inputs_per_obj"][obj_idx]
            )
            if not has_prompt:
                raise RuntimeError(
                    f"Real object slot {obj_idx} in view {view_idx} was not seeded "
                    f"on frame {frame_idx}. Fixed batching requires all slots to "
                    "share the same conditioning/history topology."
                )

        dummy_mask = torch.zeros(
            inference_state["video_height"],
            inference_state["video_width"],
            dtype=torch.bool,
        )
        dummy_ids = inference_state["fixed_batch_dummy_obj_ids"]
        while self._get_obj_num(inference_state) < self.max_objects_per_view:
            slot_idx = self._get_obj_num(inference_state)
            dummy_id = f"{dummy_prefix}:view={view_idx}:slot={slot_idx}"
            suffix = 0
            while dummy_id in inference_state["obj_id_to_idx"]:
                suffix += 1
                dummy_id = (
                    f"{dummy_prefix}:view={view_idx}:slot={slot_idx}:{suffix}"
                )
            self.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=dummy_id,
                mask=dummy_mask,
            )
            dummy_ids.append(dummy_id)
        return list(dummy_ids)

    @torch.inference_mode()
    def prepare_fixed_multiview_states(
        self,
        inference_states,
        conditioning_frame_idx=0,
    ):
        """Pad all views, finalize conditioning memories, and validate alignment."""
        if len(inference_states) != self.fixed_num_views:
            raise ValueError(
                f"Expected {self.fixed_num_views} inference states, got "
                f"{len(inference_states)}"
            )

        num_frames = {state["num_frames"] for state in inference_states}
        if len(num_frames) != 1:
            raise RuntimeError(
                "Fixed multi-view batching requires synchronized views with the "
                "same num_frames."
            )

        for view_idx, state in enumerate(inference_states):
            self.pad_state_to_fixed_object_slots(
                state,
                frame_idx=conditioning_frame_idx,
                view_idx=view_idx,
            )
            self.propagate_in_video_preflight(state)
            state["fixed_batch_prepared"] = True
            state["multiview_prepared"] = True
            state["multiview_execution_mode"] = EXECUTION_MODE_FIXED_BATCH

        self._validate_fixed_batch_history_alignment(inference_states)
        return inference_states

    def _validate_fixed_batch_history_alignment(self, inference_states):
        """
        Require identical conditioning/non-conditioning frame keys for all slots.

        The underlying EfficientTAM memory attention is batched over objects but
        does not carry a per-batch-element memory padding mask. Keeping slot history
        topology aligned gives exact semantics without introducing approximate
        memory padding for newly-created objects.
        """
        slots = self._fixed_batch_slot_order(inference_states)
        reference = None
        for view_idx, obj_idx in slots:
            out = inference_states[view_idx]["output_dict_per_obj"][obj_idx]
            signature = (
                tuple(sorted(out["cond_frame_outputs"].keys())),
                tuple(sorted(out["non_cond_frame_outputs"].keys())),
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                raise RuntimeError(
                    "Fixed all-view batching requires aligned memory history across "
                    "all object slots. Re-seed all slots together on the same detector "
                    "refresh frame (real masks + dummy zero masks) before using this "
                    "fast path."
                )
        return reference

    @staticmethod
    def _cat_fixed_batch_frame_outputs(frame_outputs):
        """Pack B=1 compact outputs into one B=N compact output."""
        packed = {}
        tensor_keys = (
            "maskmem_features",
            "pred_masks",
            "obj_ptr",
            "object_score_logits",
        )
        for key in tensor_keys:
            vals = [out.get(key, None) for out in frame_outputs]
            if all(v is None for v in vals):
                packed[key] = None
            elif any(v is None for v in vals):
                raise RuntimeError(
                    f"Cannot batch frame outputs: key '{key}' is missing for only "
                    "some slots."
                )
            else:
                packed[key] = torch.cat(vals, dim=0)

        pos_vals = [out.get("maskmem_pos_enc", None) for out in frame_outputs]
        if all(v is None for v in pos_vals):
            packed["maskmem_pos_enc"] = None
        elif any(v is None for v in pos_vals):
            raise RuntimeError(
                "Cannot batch frame outputs: maskmem_pos_enc is missing for only "
                "some slots."
            )
        else:
            num_levels = len(pos_vals[0])
            if any(len(v) != num_levels for v in pos_vals):
                raise RuntimeError("maskmem_pos_enc level count differs across slots")
            packed["maskmem_pos_enc"] = [
                torch.cat([v[level] for v in pos_vals], dim=0)
                for level in range(num_levels)
            ]
        return packed

    def _pack_fixed_batch_output_dict(self, inference_states):
        """Pack all view/object histories into one output_dict with batch B=V*S."""
        self._validate_fixed_batch_history_alignment(inference_states)
        slots = self._fixed_batch_slot_order(inference_states)
        packed = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            first_state, first_obj = slots[0]
            first_dict = inference_states[first_state]["output_dict_per_obj"][
                first_obj
            ][storage_key]
            for hist_frame_idx in sorted(first_dict.keys()):
                frame_outputs = [
                    inference_states[view_idx]["output_dict_per_obj"][obj_idx][
                        storage_key
                    ][hist_frame_idx]
                    for view_idx, obj_idx in slots
                ]
                packed[storage_key][hist_frame_idx] = (
                    self._cat_fixed_batch_frame_outputs(frame_outputs)
                )
        return packed

    @torch.inference_mode()
    def _get_fixed_multiview_features(self, inference_states, frame_idx):
        """
        Encode all views together, then expand each view feature to its fixed slots.
        """
        frame_indices = [frame_idx] * len(inference_states)
        self.cache_image_features_batched(inference_states, frame_indices)

        per_view = []
        for state in inference_states:
            features = self._get_image_feature(
                state,
                frame_idx=frame_idx,
                batch_size=self.max_objects_per_view,
            )
            per_view.append(features)

        feat_sizes = per_view[0][4]
        for features in per_view[1:]:
            if features[4] != feat_sizes:
                raise RuntimeError("Feature sizes differ across views")

        num_levels = len(per_view[0][2])
        current_vision_feats = [
            torch.cat([features[2][level] for features in per_view], dim=1)
            for level in range(num_levels)
        ]
        current_vision_pos_embeds = [
            torch.cat([features[3][level] for features in per_view], dim=1)
            for level in range(num_levels)
        ]
        return current_vision_feats, current_vision_pos_embeds, feat_sizes

    @torch.inference_mode()
    def propagate_fixed_multiview_step(
        self,
        inference_states,
        frame_idx,
        reverse=False,
    ):
        """
        Propagate every object slot in every view in one fixed-size tracking batch.

        Example: fixed_num_views=2 and max_objects_per_view=4 => one B=8
        ``track_step`` call. If a view contains only 2 or 3 real detections, its
        remaining slots are real zero-mask dummy objects created during preparation.

        Returns a list with one result dict per view. Dummy slots are omitted from
        obj_ids and video_res_masks.
        """
        if self.non_overlap_masks_for_mem_enc:
            raise RuntimeError(
                "propagate_fixed_multiview_step requires "
                "non_overlap_masks_for_mem_enc=False. Global B contains multiple "
                "independent views, so cross-batch non-overlap would be incorrect."
            )

        for state in inference_states:
            if not state.get("fixed_batch_prepared", False):
                raise RuntimeError(
                    "Call prepare_fixed_multiview_states(...) once after seeding "
                    "the real object masks and before fixed-batch propagation."
                )

        self._validate_fixed_batch_history_alignment(inference_states)
        slots = self._fixed_batch_slot_order(inference_states)

        # This fast path is intended for ordinary propagation frames. Conditioning
        # frames are handled during detector refresh / preparation.
        for view_idx, obj_idx in slots:
            obj_out = inference_states[view_idx]["output_dict_per_obj"][obj_idx]
            if frame_idx in obj_out["cond_frame_outputs"]:
                raise RuntimeError(
                    f"Frame {frame_idx} is a conditioning frame for at least one "
                    "slot. Fixed batch propagation should start on the next frame."
                )

        current_vision_feats, current_vision_pos_embeds, feat_sizes = (
            self._get_fixed_multiview_features(inference_states, frame_idx)
        )
        packed_output_dict = self._pack_fixed_batch_output_dict(inference_states)
        total_batch = self.fixed_tracking_batch_size

        self._mark_cudagraph_step_begin()
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=packed_output_dict,
            num_frames=inference_states[0]["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )

        if current_out["pred_masks"].shape[0] != total_batch:
            raise RuntimeError(
                f"Unexpected batched output size {current_out['pred_masks'].shape[0]}, "
                f"expected {total_batch}"
            )

        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )

        results = []
        for view_idx, state in enumerate(inference_states):
            view_start = view_idx * self.max_objects_per_view
            # Cache one B=1 positional-encoding copy per inference session, matching
            # the storage format used by the original predictor.
            raw_view_pos = None
            if current_out["maskmem_pos_enc"] is not None:
                raw_view_pos = [
                    x[view_start : view_start + 1]
                    for x in current_out["maskmem_pos_enc"]
                ]
                shared_view_pos = self._get_maskmem_pos_enc(
                    state, {"maskmem_pos_enc": raw_view_pos}
                )
            else:
                shared_view_pos = None

            for obj_idx in range(self.max_objects_per_view):
                batch_idx = view_start + obj_idx
                storage_device = state["storage_device"]

                maskmem_features = current_out["maskmem_features"]
                if maskmem_features is not None:
                    slot_mem = maskmem_features[
                        batch_idx : batch_idx + 1
                    ].to(torch.bfloat16)
                    slot_mem = slot_mem.to(storage_device, non_blocking=True)
                else:
                    slot_mem = None

                slot_pred = pred_masks_gpu[
                    batch_idx : batch_idx + 1
                ].to(storage_device, non_blocking=True)
                slot_ptr = current_out["obj_ptr"][batch_idx : batch_idx + 1]
                slot_score = current_out["object_score_logits"][
                    batch_idx : batch_idx + 1
                ]

                compact_out = {
                    "maskmem_features": slot_mem,
                    "maskmem_pos_enc": shared_view_pos,
                    "pred_masks": slot_pred,
                    "obj_ptr": slot_ptr,
                    "object_score_logits": slot_score,
                }
                state["output_dict_per_obj"][obj_idx][
                    "non_cond_frame_outputs"
                ][frame_idx] = compact_out
                state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                    "reverse": reverse
                }

            real_count = state.get(
                "fixed_batch_real_obj_count", self.max_objects_per_view
            )
            real_ids = list(state.get("fixed_batch_real_obj_ids", []))
            real_pred_masks = pred_masks_gpu[
                view_start : view_start + real_count
            ]
            if real_count > 0:
                _, video_res_masks = self._get_orig_video_res_output(
                    state, real_pred_masks
                )
            else:
                video_res_masks = torch.empty(
                    (0, 1, state["video_height"], state["video_width"]),
                    device=state["device"],
                    dtype=pred_masks_gpu.dtype,
                )
            results.append(
                {
                    "view_idx": view_idx,
                    "frame_idx": frame_idx,
                    "obj_ids": real_ids,
                    "video_res_masks": video_res_masks,
                    "num_real_objects": real_count,
                    "num_dummy_objects": self.max_objects_per_view - real_count,
                    "execution_mode": EXECUTION_MODE_FIXED_BATCH,
                }
            )

        return results

    @torch.inference_mode()
    def propagate_fixed_multiview(
        self,
        inference_states,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Explicit fixed-batch generator kept for backward compatibility."""
        if max_frame_num_to_track is None:
            max_frame_num_to_track = min(
                state["num_frames"] for state in inference_states
            )

        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track + 1, 0)
            processing_order = range(start_frame_idx, end - 1, -1)
        else:
            last_frame = min(
                min(state["num_frames"] for state in inference_states) - 1,
                start_frame_idx + max_frame_num_to_track - 1,
            )
            processing_order = range(start_frame_idx, last_frame + 1)

        for frame_idx in processing_order:
            yield frame_idx, self.propagate_fixed_multiview_step(
                inference_states=inference_states,
                frame_idx=frame_idx,
                reverse=reverse,
            )

    @torch.inference_mode()
    def clear_all_prompts_in_frame(
        self, inference_state, frame_idx, obj_id, need_output=True
    ):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        if out is not None:
            # The frame is not a conditioning frame anymore since it's not receiving inputs,
            # so we "downgrade" its output (if exists) to a non-conditioning frame output.
            obj_output_dict["non_cond_frame_outputs"][frame_idx] = out
            inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

        if not need_output:
            return
        # Finally, output updated masks per object (after removing the inputs above)
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["frames_tracked_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["frames_tracked_per_obj"].values():
            v.clear()
    
    @staticmethod
    def _slice_batched_backbone_output(
        backbone_out,
        batch_idx: int,
        batch_size: int,
    ):
        """Return one B=1 view from a batched backbone output."""

        if not 0 <= batch_idx < batch_size:
            raise IndexError(
                f"batch_idx={batch_idx} outside batch_size={batch_size}"
            )

        sliced_backbone_fpn = []

        for level, feat in enumerate(backbone_out["backbone_fpn"]):
            if feat.shape[0] != batch_size:
                raise RuntimeError(
                    f"backbone_fpn[{level}] has batch={feat.shape[0]}, "
                    f"expected {batch_size}; shape={tuple(feat.shape)}"
                )

            sliced_backbone_fpn.append(
                feat[batch_idx : batch_idx + 1]
            )

        sliced_vision_pos_enc = []

        for level, pos in enumerate(backbone_out["vision_pos_enc"]):
            if pos.shape[0] == batch_size:
                sliced_pos = pos[batch_idx : batch_idx + 1]

            elif pos.shape[0] == 1:
                # Position encoding can be shared by all images.
                sliced_pos = pos

            else:
                raise RuntimeError(
                    f"vision_pos_enc[{level}] has batch={pos.shape[0]}, "
                    f"expected 1 or {batch_size}; "
                    f"shape={tuple(pos.shape)}"
                )

            sliced_vision_pos_enc.append(sliced_pos)

        return {
            "backbone_fpn": sliced_backbone_fpn,
            "vision_pos_enc": sliced_vision_pos_enc,
        }

    @torch.inference_mode()
    def cache_image_features_batched(
        self,
        inference_states,
        frame_indices,
    ):
        """
        Batch the image encoder across independent videos/cameras.

        Only image encoding is batched. Each inference_state keeps its own
        temporal memories, object states, and tracking outputs.
        """

        if not inference_states:
            raise ValueError("inference_states must not be empty")

        if len(inference_states) != len(frame_indices):
            raise ValueError(
                "inference_states and frame_indices must have equal length"
            )

        batch_size = len(inference_states)
        device = self.device

        images = []
        expected_shape = None

        for state_idx, (inference_state, frame_idx) in enumerate(
            zip(inference_states, frame_indices)
        ):
            image = inference_state["images"][frame_idx]

            if image.ndim != 3:
                raise RuntimeError(
                    f"Expected image [3,H,W], got {tuple(image.shape)} "
                    f"for state {state_idx}"
                )

            image = image.to(
                device,
                non_blocking=True,
            ).float()

            if expected_shape is None:
                expected_shape = tuple(image.shape)
            elif tuple(image.shape) != expected_shape:
                raise RuntimeError(
                    "All camera images must have the same model-input shape. "
                    f"Expected {expected_shape}, "
                    f"got {tuple(image.shape)}"
                )

            images.append(image)

        # B cameras:
        # [B, 3, H, W]
        image_batch = torch.stack(images, dim=0)

        # ONE image encoder invocation for all cameras.
        backbone_batch = self.forward_image(image_batch)

        for batch_idx, (inference_state, frame_idx) in enumerate(
            zip(inference_states, frame_indices)
        ):
            backbone_single = self._slice_batched_backbone_output(
                backbone_out=backbone_batch,
                batch_idx=batch_idx,
                batch_size=batch_size,
            )

            image_single = image_batch[
                batch_idx : batch_idx + 1
            ]

            # Match the existing `_get_image_feature()` cache format exactly.
            inference_state["cached_features"] = {
                frame_idx: (
                    image_single,
                    backbone_single,
                )
            }

        return image_batch, backbone_batch

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.reset_state(inference_state)
            return inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])
        _map_keys(inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            non_cond_frame_outputs = obj_output_dict["non_cond_frame_outputs"]
            for t in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_frame_outputs.pop(t, None)


class EfficientTAMVideoPredictorVOS(EfficientTAMVideoPredictor):
    """Optimized for the VOS setting"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._compile_all_components()

    def _compile_all_components(self):
        print("Compiling all components for VOS setting. First time may be very slow.")

        # memory_attention is specialized with dynamic=False because the dynamic
        # B>1 path currently triggers a TorchInductor fusion/codegen failure. Its
        # temporal-memory length still grows during startup, so keep enough static
        # specializations for the finite warmup sequence.
        try:
            current_limit = int(torch._dynamo.config.recompile_limit)
            if current_limit < 16:
                torch._dynamo.config.recompile_limit = 16
        except (AttributeError, TypeError, ValueError):
            pass

        self.memory_encoder.forward = torch.compile(
            self.memory_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.memory_attention.forward = torch.compile(
            self.memory_attention.forward,
            mode="default",
            fullgraph=True,
            dynamic=False,  # Num. of memories varies
        )

        self.sam_prompt_encoder.forward = torch.compile(
            self.sam_prompt_encoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        self.sam_mask_decoder.forward = torch.compile(
            self.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

    def forward_image(self, img_batch: torch.Tensor):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the backbone features and pos encoding to enable compilation.
        """
        backbone_out = self.image_encoder(img_batch)
        # Clone to help torch.compile
        for i in range(len(backbone_out["backbone_fpn"])):
            backbone_out["backbone_fpn"][i] = backbone_out["backbone_fpn"][i].clone()
            backbone_out["vision_pos_enc"][i] = backbone_out["vision_pos_enc"][
                i
            ].clone()
        return backbone_out

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the outputs of prompt_encoder and mask_decoder to enable compilation.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 2, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,  # use antialias for downsampling
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        # Clone image_pe and the outputs of sam_prompt_encoder
        # to enable compilation
        sparse_embeddings = sparse_embeddings.clone()
        dense_embeddings = dense_embeddings.clone()
        image_pe = self.sam_prompt_encoder.get_dense_pe().clone()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        # Clone the output of sam_mask_decoder
        # to enable compilation
        low_res_multimasks = low_res_multimasks.clone()
        ious = ious.clone()
        sam_output_tokens = sam_output_tokens.clone()
        object_score_logits = object_score_logits.clone()

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """
        Identical to the corresponding method in the parent (EfficientTAMVideoPredictor), but
        cloning the memories and their pos enc to enable compilation.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        # Clone the feats and pos_enc to enable compilation
        maskmem_features = maskmem_out["vision_features"].clone()
        maskmem_pos_enc = [m.clone() for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc
