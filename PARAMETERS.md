Here's how each parameter works in the ProPainter pipeline:

---

### `raft_iter` (default: 20)

**Where:** [compute_flow()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:78:0-126:22) → passed to [raft_model(frames, iters=raft_iter)](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/utils/model_utils.py:26:0-32:21)

This controls the number of **iterative refinement steps** inside the RAFT optical flow model. RAFT works by making an initial flow estimate and then refining it through a recurrent GRU update loop. More iterations = more accurate flow, but slower.

- **Lower values (5–10):** Faster, acceptable for small/simple motions
- **Default (20):** Good balance for most videos
- **Higher values (30+):** Diminishing returns — RAFT typically converges well by ~20 iterations

This parameter has **negligible VRAM impact** — it's purely compute time. During each iteration RAFT holds intermediate GRU states on GPU, but since `test_mode=True` is always used, only the final flow is returned and intermediates are released. The cost is purely speed, not memory.

---

### `subvideo_length` (default: 80)

**Where:** [complete_flow()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:129:0-193:24), [image_propagation()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:196:0-273:40), and indirectly in [feature_propagation()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:276:0-368:26) (for `ref_num` calculation)

This is the **chunking window** for the flow completion and image propagation phases. When a video has more frames than `subvideo_length`, both phases split the video into overlapping chunks of this size (with padding of 5–10 frames for smooth transitions at boundaries).

- **Larger values:** Better temporal consistency (the model "sees" more context), but **higher VRAM** since it processes more frames at once
- **Smaller values:** Lower VRAM but potentially visible seams at chunk boundaries (mitigated by the overlap padding)
- **Also controls `ref_num`:** When `video_length > subvideo_length`, the feature propagation phase limits reference frames to `subvideo_length // ref_stride` instead of using all available reference frames

Think of it as: *"how many frames can I afford to process at once?"*

---

### `neighbor_length` (default: 10)

**Where:** [feature_propagation()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:276:0-368:26) — determines the **local attention window**

This is the most **VRAM-critical parameter for the feature propagation phase** (~60–100% of the pipeline). It controls how many neighboring frames are fed into the [InpaintGenerator](cci:2://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/model/propainter.py:293:0-452:21) transformer at each step:

```python
neighbor_stride = neighbor_length // 2    # = 5 by default
# At each step, neighbor_ids = frames within ±neighbor_stride of current position
```

The `InpaintGenerator.forward()` receives `neighbor_ids + ref_ids` frames. The `neighbor_ids` are the **local frames** (passed through the transformer's attention mechanism), while `ref_ids` are **reference frames** (used as encoder context only, not through the attention layers).

- **Larger values:** Better inpainting quality because the model sees more temporal context for each frame. But VRAM grows significantly because:
  - `selected_imgs` tensor: `[1, neighbor_length + ref_count, 3, H, W]`
  - The transformer inside [InpaintGenerator](cci:2://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/model/propainter.py:293:0-452:21) computes self-attention over `neighbor_length` local tokens
  - The encoder processes all `neighbor_length + ref_count` frames at once
- **Smaller values:** Less VRAM, but the model has less temporal context, which can cause flickering or inconsistency in the inpainted region

The stride is `neighbor_length // 2`, meaning frames are processed with **50% overlap** — each frame gets composited 2x on average (blended 50/50), which smooths transitions.

> **Note:** `neighbor_length` only affects the *feature propagation* phase. It has no effect on `compute_flow()` or `complete_flow()`.

---

### `ref_stride` (default: 10)

**Where:** [get_ref_index()](cci:1://file:///c:/ComfyUI_portable/ComfyUI/custom_nodes/ComfyUI_ProPainter_Nodes/propainter_inference.py:37:0-59:20) — determines **reference frame spacing**

This controls the minimum gap of **reference frames** — frames outside the local `neighbor_ids` window that provide global temporal context to the inpainting model.

#### Sampling Strategy: Exponential Distance

Reference frames are sampled with **exponentially growing gaps** outward from the current frame position:

```
gaps: ref_stride × 1, × 2, × 4, × 8, × 16, …
```

For example, with `ref_stride=10` and current frame at 100 in a 200-frame video:
- Nearby refs: 90, 110 (gap=10)
- Next tier: 80, 120 (gap=20)
- Next tier: 60, 140 (gap=40)
- Next tier: 20, 180 (gap=80)
- Done (outer refs would exceed bounds)

This means:
- **Nearby frames** are always included — they carry the most relevant temporal context
- **Distant frames** are sampled sparsely — they contribute diminishing information for inpainting a region far from them in time
- **Total ref count grows logarithmically** with video length, not linearly

- **Smaller values (e.g., 5):** Denser initial sampling → more reference frames → higher VRAM per step
- **Larger values (e.g., 20):** Sparser initial sampling → fewer reference frames → less VRAM
- **Interaction with `neighbor_length`:** Both neighbor and ref frames are fed into the encoder simultaneously. VRAM scales with `neighbor_length + ref_count`.
- **Interaction with `subvideo_length`:** When `video_length > subvideo_length`, `ref_num = subvideo_length // ref_stride` caps the maximum number of reference frames per step.

---

### ⚠️ Important: `compute_flow()` Memory Pattern for Long Videos

The optical flow phase (RAFT, Phase 1) has a VRAM behavior **independent of all four parameters above**. RAFT processes frames in small fixed-size chunks (`short_clip_len = 12` for 616-pixel-wide videos). Each chunk's output flows are immediately offloaded to CPU, so **GPU peak VRAM during flow computation is bounded by model size + one chunk** regardless of total video length.

Without this CPU-offload, accumulated flow tensors would stay on GPU throughout the loop, causing linear VRAM growth with frame count and OOM errors on longer videos (e.g., >~350 frames at 616×320 on 6 GB VRAM).

---

### Summary Table

| Parameter | Controls | VRAM Impact | Quality Impact | Speed Impact |
|---|---|---|---|---|
| `raft_iter` | Flow refinement iterations | Negligible | Low (diminishing after ~20) | Linear |
| `subvideo_length` | Chunk size for flow completion + image propagation | **High** | Medium (chunk boundaries) | Moderate |
| `neighbor_length` | Local attention window in feature propagation | **Highest** (feature prop phase) | **High** (temporal coherence) | Moderate |
| `ref_stride` | Minimum gap for reference frame sampling (exponential) | Medium | Medium (global context) | Low |

**Tuning priority for 6 GB VRAM:**
1. If OOM during flow phase → verify the CPU-offload pattern is in place (code-level fix, not a user parameter)
2. If OOM during feature propagation → lower `neighbor_length` first (biggest impact)
3. Then raise `ref_stride` (fewer refs per step)
4. Then lower `subvideo_length` (affects flow completion + image propagation quality)
5. `raft_iter` is pure speed — lower it last