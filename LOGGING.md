# Logging and Progress Reporting Architecture

This document describes the logging and progress reporting strategy in `ComfyUI_ProPainter_Nodes` and explains why this approach is required when integrating with automated rendering and task execution pipelines (e.g., Flamenco workers, background runners, and headless ComfyUI wrappers).

---

## 1. Background & Problem Statement

When ComfyUI is executed as a subprocess inside an automated pipeline or render farm worker (such as Flamenco), console output behaves differently than in an interactive terminal session:

1. **Subprocess Stream Routing**:
   - Automated runners and wrapper scripts (such as `stages_flamenco_worker.py` and `[ComfyUI-local]`) typically capture and forward `sys.stderr` to the pipeline log or task monitor.
   - Standard `print(...)` in Python targets `sys.stdout` by default. Depending on how the runner captures subprocess output, `stdout` might be ignored, suppressed, or handled separately.

2. **I/O Stream Buffering**:
   - In non-interactive environments (piped stdout/stderr), Python enables block buffering by default.
   - Output from fast or intermediate phases gets held in memory buffers and is not immediately visible to external monitors, or can be lost entirely if the subprocess terminates before flushing.

3. **Logging Framework Inconsistencies**:
   - Python's standard `logging` library defaults to writing via a `StreamHandler` to `sys.stderr`.
   - Mixing `logging.info(...)` and `print(...)` across custom nodes causes inconsistent stream routing.
   - Upstream library boilerplate (such as `get_root_logger` in `misc.py`) introduced redundant logger handlers, custom formatting strings, and potential handler leaks.

4. **Progress Visibility in Later Pipeline Phases**:
   - In earlier versions, progress reporting in later phases (Flow Completion, Image Propagation, and Feature Propagation) either failed to display or stalled because progress calls were trapped behind buffering or nested inside conditional blocks without immediate stream flushes.

---

## 2. The Solution: Direct, Unbuffered `stderr` Reporting

To ensure 100% reliable logging across local interactive GUI sessions, headless API runs, and remote pipeline workers, this codebase implements a **wrapper-free, unbuffered stderr pattern**:

```python
import sys

print(f"[ProPainter] ...", file=sys.stderr, flush=True)
```

### Key Rules Implemented

1. **Direct `sys.stderr` Target**:
   - All informative console outputs and progress updates are explicitly written to `sys.stderr`.
   - This ensures capture by pipeline monitoring wrappers (like `[ComfyUI-local]`) which listen on stderr for real-time progress.

2. **Guaranteed Immediate Flushing (`flush=True`)**:
   - Every single `print` statement includes `flush=True` to immediately dispatch output through the pipe, preventing buffered stalls.

3. **No Redundant `logging` Wrappers**:
   - Removed all `import logging`, `logging.info(...)`, and unused root logger factories (such as `get_root_logger()` in `model/misc.py`).
   - Clean, lightweight, and eliminates collisions with ComfyUI's internal logging configuration.

4. **Dual Progress Reporting (UI + Console)**:
   - In `propainter_inference.py`, `_update_progress()` simultaneously updates:
     1. ComfyUI's frontend progress bar via `pbar.update_absolute(...)` (when `pbar` is provided).
     2. The console / pipeline log via `print(msg, file=sys.stderr, flush=True)`.

---

## 3. Progress Breakdown Across Inference Phases

The ProPainter pipeline is split into four distinct computational phases with weighted progress allocations spanning 0% to 100%:

| Phase | Percentage Range | Function | Log Identifier | Description |
|---|---|---|---|---|
| **1. Optical Flow** | `0% – 25%` | `compute_flow()` | `RAFT Flow` | RAFT calculates bidirectional optical flow across video frame chunks. |
| **2. Flow Completion** | `25% – 45%` | `complete_flow()` | `Flow Complete` | Recurrent flow completion network refines and inpaints optical flows. |
| **3. Image Propagation** | `45% – 60%` | `image_propagation()` | `Image Prop` | Warps and propagates pixel data along completed flow vectors. |
| **4. Feature Propagation** | `60% – 100%` | `feature_propagation()` | `Feature Prop` | Inpainting transformer model performs multi-frame temporal feature propagation and blend compositing. |

### Progress Log Format

```text
[ProPainter] Progress:  XX% (current_step/total_steps) - <Phase Name>
```

**Example output stream:**
```text
pid=14416 > [ComfyUI-local] [ProPainter] Starting Inpaint Node execution...
pid=14416 > [ComfyUI-local] [ProPainter] Loading RAFT model...
pid=14416 > [ComfyUI-local] [ProPainter] Progress:   0% (1/26) - RAFT Flow
...
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  25% (26/26) - RAFT Flow
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  27% (1/8) - Flow Complete
...
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  45% (8/8) - Flow Complete
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  46% (1/8) - Image Prop
...
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  60% (8/8) - Image Prop
pid=14416 > [ComfyUI-local] [ProPainter] Progress:  60% (1/45) - Feature Prop
...
pid=14416 > [ComfyUI-local] [ProPainter] Progress: 100% (45/45) - Feature Prop
```

---

## 4. Affected Files & Module Roles

- **[`propainter_inference.py`](propainter_inference.py)**:
  - Houses `_update_progress()` which computes the global weighted percentage and outputs to `sys.stderr` with `flush=True`.
- **[`propainter_nodes.py`](propainter_nodes.py)**:
  - Node entry points (`ProPainterInpaint`, `ProPainterOutpaint`) reporting execution start.
- **[`utils/model_utils.py`](utils/model_utils.py)**:
  - Model initialization, caching status, and model path resolution logs.
- **[`utils/download_utils.py`](utils/download_utils.py)**:
  - Model directory verification, cache checks, and download status.
- **[`model/misc.py`](model/misc.py)**:
  - Removed dead upstream `get_root_logger()` and associated `logging` constructs.
