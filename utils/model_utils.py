from dataclasses import dataclass
import sys

import torch

from ..model.modules.flow_comp_raft import RAFT_bi
from ..model.propainter import InpaintGenerator

from ..model.recurrent_flow_completion import (
    RecurrentFlowCompleteNet,
)
from ..utils.download_utils import download_model


@dataclass
class Models:
    raft_model: RAFT_bi
    flow_model: RecurrentFlowCompleteNet
    inpaint_model: InpaintGenerator


PRETRAIN_MODEL_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

# Module-level cache: keyed by (device_str, use_half_bool) to avoid
# reloading all 3 models from disk on every node execution.
_model_cache: dict[tuple[str, bool], Models] = {}


def load_raft_model(device: torch.device) -> RAFT_bi:
    """Loads the RAFT bi-directional model."""
    print("[ProPainter] Loading RAFT model...", file=sys.stderr, flush=True)
    model_path = download_model(PRETRAIN_MODEL_URL, "raft-things.pth")
    print(f"[ProPainter] RAFT model path: {model_path}", file=sys.stderr, flush=True)
    raft_model = RAFT_bi(str(model_path), device)
    return raft_model


def load_recurrent_flow_model(device: torch.device) -> RecurrentFlowCompleteNet:
    """Loads the Recurrent Flow Completion Network model."""
    print("[ProPainter] Loading recurrent flow model...", file=sys.stderr, flush=True)
    model_path = download_model(PRETRAIN_MODEL_URL, "recurrent_flow_completion.pth")
    print(f"[ProPainter] Recurrent flow model path: {model_path}", file=sys.stderr, flush=True)
    flow_model = RecurrentFlowCompleteNet(str(model_path))
    for p in flow_model.parameters():
        p.requires_grad = False
    flow_model.to(device)
    flow_model.eval()
    return flow_model


def load_inpaint_model(device: torch.device) -> InpaintGenerator:
    """Loads the Inpaint Generator model."""
    print("[ProPainter] Loading inpaint model...", file=sys.stderr, flush=True)
    model_path = download_model(PRETRAIN_MODEL_URL, "ProPainter.pth")
    print(f"[ProPainter] Inpaint model path: {model_path}", file=sys.stderr, flush=True)
    inpaint_model = InpaintGenerator(model_path=str(model_path)).to(device)
    inpaint_model.eval()
    return inpaint_model


def initialize_models(device: torch.device, use_half: str) -> Models:
    """Return initialized inference models, using cache when possible."""
    use_half_bool = use_half == "enable"
    cache_key = (str(device), use_half_bool)

    # CUDNN benchmark and channels_last removed for VRAM stability

    if cache_key in _model_cache:
        print(f"[ProPainter] Using cached models for {cache_key}", file=sys.stderr, flush=True)
        return _model_cache[cache_key]

    print("[ProPainter] initialize_models starting (loading from disk)...", file=sys.stderr, flush=True)
    raft_model = load_raft_model(device)
    flow_model = load_recurrent_flow_model(device)
    inpaint_model = load_inpaint_model(device)

    # Channels last removed for stability

    if use_half_bool:
        print("[ProPainter] Half precision enabled...", file=sys.stderr, flush=True)
        # raft_model = raft_model.half()
        flow_model = flow_model.half()
        inpaint_model = inpaint_model.half()

    models = Models(raft_model, flow_model, inpaint_model)
    _model_cache[cache_key] = models

    print("[ProPainter] initialize_models complete (cached for reuse).", file=sys.stderr, flush=True)
    return models
