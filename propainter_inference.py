from dataclasses import dataclass, field
from typing import Optional
import sys

import numpy as np
from numpy.typing import NDArray

import torch

from comfy import model_management

from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import (
    RecurrentFlowCompleteNet,
)
from .utils.model_utils import Models
from .model.propainter import InpaintGenerator


@dataclass
class ProPainterConfig:
    ref_stride: int
    neighbor_length: int
    subvideo_length: int
    raft_iter: int
    fp16: str
    video_length: int
    device: torch.device
    process_size: tuple[int, int]
    use_half: bool = field(init=False)

    def __post_init__(self) -> None:
        """Initialize use-half."""
        self.use_half = self.fp16 == "enable"
        if self.device == torch.device("cpu"):
            self.use_half = False


def get_ref_index(
    mid_neighbor_id: int,
    neighbor_ids: list[int],
    config: ProPainterConfig,
    ref_num: int = -1,
) -> list[int]:
    """Calculate reference frame indices using exponential-distance sampling.

    Samples outward from mid_neighbor_id with exponentially growing gaps
    (ref_stride × 1, × 2, × 4, × 8, …). This ensures nearby frames — which
    carry the most relevant temporal context — are always included, while very
    distant frames are sampled sparsely or skipped entirely. Compared to uniform
    sampling, this bounds the number of reference frames logarithmically with
    video length, significantly reducing VRAM usage in feature_propagation() for
    long videos with negligible quality impact.
    """
    ref_index = []
    seen = set(neighbor_ids)

    if ref_num == -1:
        # Short video path: sample the whole video exponentially
        gap = config.ref_stride
        while True:
            neg = mid_neighbor_id - gap
            pos = mid_neighbor_id + gap
            added_any = False
            if neg >= 0 and neg not in seen:
                ref_index.append(neg)
                seen.add(neg)
                added_any = True
            if pos < config.video_length and pos not in seen:
                ref_index.append(pos)
                seen.add(pos)
                added_any = True
            if not added_any and neg < 0 and pos >= config.video_length:
                break
            if neg < 0 and pos >= config.video_length:
                break
            gap *= 2
    else:
        # Long video path: sample up to ref_num refs with exponential spacing
        gap = config.ref_stride
        while len(ref_index) < ref_num:
            neg = mid_neighbor_id - gap
            pos = mid_neighbor_id + gap
            if neg < 0 and pos >= config.video_length:
                break
            if neg >= 0 and neg not in seen:
                ref_index.append(neg)
                seen.add(neg)
                if len(ref_index) >= ref_num:
                    break
            if pos < config.video_length and pos not in seen:
                ref_index.append(pos)
                seen.add(pos)
            gap *= 2

    return ref_index


def _update_progress(pbar, phase_offset: float, phase_weight: float,
                     step: int, total_steps: int, phase_name: str = "") -> None:
    """Update ComfyUI progress bar and console with weighted phase progress."""
    phase_progress = step / max(total_steps, 1)
    overall = phase_offset + phase_progress * phase_weight
    percent = int(overall * 100)

    if pbar is not None:
        pbar.update_absolute(percent, 100)

    suffix = f" - {phase_name}" if phase_name else ""
    msg = f"[ProPainter] Progress: {percent:>3}% ({step}/{total_steps}){suffix}"
    print(msg, file=sys.stderr, flush=True)


def compute_flow(
    raft_model: RAFT_bi, frames: torch.Tensor, config: ProPainterConfig,
    pbar=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward and backward optical flows using the RAFT model."""
    if frames.size(dim=-1) <= 640:
        short_clip_len = 12
    elif frames.size(dim=-1) <= 720:
        short_clip_len = 8
    elif frames.size(dim=-1) <= 1280:
        short_clip_len = 4
    else:
        short_clip_len = 2

    # Phase weights: optical flow = 0-25%
    PHASE_OFFSET = 0.0
    PHASE_WEIGHT = 0.25

    # use fp32 for RAFT
    # NOTE: Each RAFT chunk produces flow tensors that must NOT accumulate on
    # the GPU — for long videos this causes linear VRAM growth and eventual OOM.
    # We immediately move each chunk's output to CPU and only bring the
    # concatenated result back to the GPU after the loop.
    if frames.size(dim=1) > short_clip_len:
        gt_flows_f_list, gt_flows_b_list = [], []
        chunks = list(range(0, config.video_length, short_clip_len))
        total_chunks = len(chunks)
        for chunk_idx, chunck in enumerate(chunks):
            end_f = min(config.video_length, chunck + short_clip_len)
            if chunck == 0:
                flows_f, flows_b = raft_model(
                    frames[:, chunck:end_f], iters=config.raft_iter
                )
            else:
                flows_f, flows_b = raft_model(
                    frames[:, chunck - 1 : end_f], iters=config.raft_iter
                )

            # Offload to CPU immediately to prevent accumulation on GPU VRAM.
            # Peak GPU usage becomes bounded by model + one chunk, not all chunks.
            gt_flows_f_list.append(flows_f.cpu())
            gt_flows_b_list.append(flows_b.cpu())
            del flows_f, flows_b
            torch.cuda.empty_cache()
            _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT,
                             chunk_idx + 1, total_chunks, "RAFT Flow")

        # Concatenate on CPU then move back to device as a single allocation.
        gt_flows_f = torch.cat(gt_flows_f_list, dim=1).to(frames.device)
        gt_flows_b = torch.cat(gt_flows_b_list, dim=1).to(frames.device)
        del gt_flows_f_list, gt_flows_b_list
        gt_flows_bi = (gt_flows_f, gt_flows_b)
    else:
        gt_flows_bi = raft_model(frames, iters=config.raft_iter)
        torch.cuda.empty_cache()
        _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT, 1, 1, "RAFT Flow")

    return gt_flows_bi


def complete_flow(
    recurrent_flow_model: RecurrentFlowCompleteNet,
    flows_tuple: tuple[torch.Tensor, torch.Tensor],
    flow_masks: torch.Tensor,
    subvideo_length: int,
    pbar=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete and refine optical flows using a recurrent flow completion model.

    This function processes optical flows in chunks if the total length exceeds the specified
    subvideo length. It uses a recurrent model to complete and refine the flows, combining
    forward and backward flows into bidirectional flows.
    """
    # Phase weights: flow completion = 25-45%
    PHASE_OFFSET = 0.25
    PHASE_WEIGHT = 0.20

    flow_length = flows_tuple[0].size(dim=1)
    if flow_length > subvideo_length:
        pred_flows_f_list, pred_flows_b_list = [], []
        pad_len = 5
        chunks = list(range(0, flow_length, subvideo_length))
        total_chunks = len(chunks)
        for chunk_idx, f in enumerate(chunks):
            s_f = max(0, f - pad_len)
            e_f = min(flow_length, f + subvideo_length + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(flow_length, f + subvideo_length)
            pred_flows_bi_sub, _ = recurrent_flow_model.forward_bidirect_flow(
                (flows_tuple[0][:, s_f:e_f], flows_tuple[1][:, s_f:e_f]),
                flow_masks[:, s_f : e_f + 1],
            )
            pred_flows_bi_sub = recurrent_flow_model.combine_flow(
                (flows_tuple[0][:, s_f:e_f], flows_tuple[1][:, s_f:e_f]),
                pred_flows_bi_sub,
                flow_masks[:, s_f : e_f + 1],
            )

            # Offload chunk flows to CPU to avoid accumulating VRAM on long videos
            pred_flows_f_list.append(
                pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e].cpu()
            )
            pred_flows_b_list.append(
                pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e].cpu()
            )
            del pred_flows_bi_sub
            torch.cuda.empty_cache()
            _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT,
                             chunk_idx + 1, total_chunks, "Flow Complete")

        # Concatenate on CPU then move back to device as a single allocation
        pred_flows_f = torch.cat(pred_flows_f_list, dim=1).to(flows_tuple[0].device)
        pred_flows_b = torch.cat(pred_flows_b_list, dim=1).to(flows_tuple[0].device)
        del pred_flows_f_list, pred_flows_b_list

        pred_flows_bi = (pred_flows_f, pred_flows_b)

    else:
        pred_flows_bi, _ = recurrent_flow_model.forward_bidirect_flow(
            flows_tuple, flow_masks
        )
        pred_flows_bi = recurrent_flow_model.combine_flow(
            flows_tuple, pred_flows_bi, flow_masks
        )

        torch.cuda.empty_cache()
        _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT, 1, 1, "Flow Complete")

    return pred_flows_bi


def image_propagation(
    inpaint_model: InpaintGenerator,
    frames: torch.Tensor,
    masks_dilated: torch.Tensor,
    prediction_flows: tuple[torch.Tensor, torch.Tensor],
    config: ProPainterConfig,
    pbar=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate inpainted images across video frames.

    If the video length exceeds a defined threshold, the process is segmented and handled in chunks.
    """
    # Phase weights: image propagation = 45-60%
    PHASE_OFFSET = 0.45
    PHASE_WEIGHT = 0.15

    process_width, process_height = config.process_size
    subvideo_length_img_prop = min(
        100, config.subvideo_length
    )  # ensure a minimum of 100 frames for image propagation
    if config.video_length > subvideo_length_img_prop:
        updated_frames_list, updated_masks_list = [], []
        pad_len = 10
        chunks = list(range(0, config.video_length, subvideo_length_img_prop))
        total_chunks = len(chunks)
        for chunk_idx, f in enumerate(chunks):
            s_f = max(0, f - pad_len)
            e_f = min(config.video_length, f + subvideo_length_img_prop + pad_len)
            pad_len_s = max(0, f) - s_f
            pad_len_e = e_f - min(config.video_length, f + subvideo_length_img_prop)
            b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
            pred_flows_bi_sub = (
                prediction_flows[0][:, s_f : e_f - 1],
                prediction_flows[1][:, s_f : e_f - 1],
            )
            # Compute masked frames per-chunk rather than precomputing for whole video
            masked_frames_chunk = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f])
            prop_imgs_sub, updated_local_masks_sub = inpaint_model.img_propagation(
                masked_frames_chunk,
                pred_flows_bi_sub,
                masks_dilated[:, s_f:e_f],
                "nearest",
            )
            updated_frames_sub = (
                masked_frames_chunk
                + prop_imgs_sub.view(b, t, 3, process_height, process_width)
                * masks_dilated[:, s_f:e_f]
            )
            del masked_frames_chunk
            updated_masks_sub = updated_local_masks_sub.view(
                b, t, 1, process_height, process_width
            )

            # Offload chunk results to CPU to avoid accumulating VRAM on long videos
            updated_frames_list.append(
                updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e].cpu()
            )
            updated_masks_list.append(
                updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e].cpu()
            )
            del updated_frames_sub, updated_masks_sub
            torch.cuda.empty_cache()
            _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT,
                             chunk_idx + 1, total_chunks, "Image Prop")

        # Concatenate on CPU then move back to device as a single allocation
        updated_frames = torch.cat(updated_frames_list, dim=1).to(frames.device)
        updated_masks = torch.cat(updated_masks_list, dim=1).to(frames.device)
        del updated_frames_list, updated_masks_list
    else:
        b, t, _, _, _ = masks_dilated.size()
        masked_frames = frames * (1 - masks_dilated)
        prop_imgs, updated_local_masks = inpaint_model.img_propagation(
            masked_frames, prediction_flows, masks_dilated, "nearest"
        )
        updated_frames = (
            masked_frames
            + prop_imgs.view(b, t, 3, process_height, process_width) * masks_dilated
        )
        del masked_frames
        updated_masks = updated_local_masks.view(b, t, 1, process_height, process_width)
        torch.cuda.empty_cache()
        _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT, 1, 1, "Image Prop")

    return updated_frames, updated_masks


def feature_propagation(
    inpaint_model: InpaintGenerator,
    updated_frames: torch.Tensor,
    updated_masks: torch.Tensor,
    masks_dilated: torch.Tensor,
    prediction_flows: tuple[torch.Tensor, torch.Tensor],
    original_frames: torch.Tensor | list[NDArray],
    config: ProPainterConfig,
    pbar=None,
) -> list[NDArray]:
    """Propagate inpainted features across video frames using high-precision blending."""
    # Phase weights: feature propagation = 60-100%
    PHASE_OFFSET = 0.60
    PHASE_WEIGHT = 0.40

    process_width, process_height = config.process_size

    # Select precision based on config
    acc_dtype = torch.float16 if config.use_half else torch.float32
    
    # Accumulation buffers on CPU for memory safety
    accum_frames = torch.zeros((config.video_length, process_height, process_width, 3), dtype=acc_dtype)
    accum_counts = torch.zeros((config.video_length, 1, 1, 1), dtype=acc_dtype)

    neighbor_stride = config.neighbor_length // 2
    ref_num = (
        config.subvideo_length // config.ref_stride
        if config.video_length > config.subvideo_length
        else -1
    )

    iterations = list(range(0, config.video_length, neighbor_stride))
    total_iterations = len(iterations)

    # Pre-compute reference indices
    all_ref_ids = {}
    for f in iterations:
        neighbor_ids = list(
            range(
                max(0, f - neighbor_stride),
                min(config.video_length, f + neighbor_stride + 1),
            )
        )
        all_ref_ids[f] = get_ref_index(f, neighbor_ids, config, ref_num)

    for iter_idx, f in enumerate(iterations):
        neighbor_ids = list(
            range(
                max(0, f - neighbor_stride),
                min(config.video_length, f + neighbor_stride + 1),
            )
        )
        ref_ids = all_ref_ids[f]
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        if config.use_half:
            selected_masks = selected_masks.half()
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            prediction_flows[0][:, neighbor_ids[:-1], :, :, :],
            prediction_flows[1][:, neighbor_ids[:-1], :, :, :],
        )
        with torch.no_grad():
            l_t = len(neighbor_ids)

            pred_img = inpaint_model(
                selected_imgs,
                selected_pred_flows_bi,
                selected_masks,
                selected_update_masks,
                l_t,
            )

            pred_img = pred_img.view(-1, 3, process_height, process_width)
            # Normalization to [0, 1] floating point
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).to(acc_dtype)

            # Vectorized compositing for this neighbor group
            n_ids = len(neighbor_ids)
            mask_batch = (
                masks_dilated[0, neighbor_ids, :, :, :]
                .cpu()
                .permute(0, 2, 3, 1)
                .to(acc_dtype)
            )
            
            # Process each frame in the neighbor group
            for i, idx in enumerate(neighbor_ids):
                # Get the high-precision background
                if isinstance(original_frames, torch.Tensor):
                    # Use the original float tensor directly
                    bg = original_frames[idx].cpu().to(acc_dtype)
                else:
                    # Fallback for outpainting or cases where tensor isn't available
                    bg = torch.from_numpy(original_frames[idx].astype(np.float32) / 255.0).to(acc_dtype)

                # Composite the prediction using the mask
                # If mask is 0 (original), we get exactly 'bg'.
                # If mask is 1 (predicted), we get exactly 'pred_img[i]'.
                img = pred_img[i] * mask_batch[i] + bg * (1 - mask_batch[i])
                
                # Rolling Average Formula: avg = avg + (new - avg) / n
                # This ensures we don't bit-crush the image and gives equal weight to all passes.
                count = accum_counts[idx] + 1
                accum_frames[idx] += (img - accum_frames[idx]) / count
                accum_counts[idx] = count

        torch.cuda.empty_cache()
        _update_progress(pbar, PHASE_OFFSET, PHASE_WEIGHT,
                         iter_idx + 1, total_iterations, "Feature Prop")

    # Final conversion to uint8 for compatibility with the handle_output logic
    # but now it only happens ONCE at the very end of all blending.
    composed_frames = []
    for i in range(config.video_length):
        frame = (accum_frames[i] * 255.0).clamp(0, 255).to(torch.uint8).numpy()
        composed_frames.append(frame)

    return composed_frames


def process_inpainting(
    models: Models,
    frames: torch.Tensor,
    flow_masks: torch.Tensor,
    masks_dilated: torch.Tensor,
    config: ProPainterConfig,
    pbar=None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Apply inpainting on video using recurrent flow and ProPainter model."""
    device = config.device

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config.use_half):
            # Phase 1: Compute optical flow (RAFT always uses fp32)
            gt_flows_bi = compute_flow(models.raft_model, frames, config, pbar=pbar)
    
            # Offload frames to CPU while flow completion runs (not needed until image_propagation)
            frames_cpu = frames.to("cpu", non_blocking=True)
            del frames
            model_management.soft_empty_cache()
    
            if config.use_half:
                flow_masks, masks_dilated = (
                    flow_masks.half(),
                    masks_dilated.half(),
                )
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
    
            # Phase 2: Complete flow
            pred_flows_bi = complete_flow(
                models.flow_model, gt_flows_bi, flow_masks, config.subvideo_length,
                pbar=pbar,
            )
    
            # Free ground-truth flows and flow_masks — only predicted flows needed going forward
            del gt_flows_bi
            del flow_masks
            torch.cuda.empty_cache()
            model_management.soft_empty_cache()
    
            # Bring frames back to device for image propagation
            frames = frames_cpu.to(device, non_blocking=True)
            if config.use_half:
                frames = frames.half()
            del frames_cpu
    
            # Phase 3: Image propagation
            updated_frames, updated_masks = image_propagation(
                models.inpaint_model, frames, masks_dilated, pred_flows_bi, config,
                pbar=pbar,
            )

    return updated_frames, updated_masks, pred_flows_bi
