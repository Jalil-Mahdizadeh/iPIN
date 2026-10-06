"""Strict offline ESM C loader; deliberately keeps FP32 trainable parameters.

This is the original SDK model, not the future v4 PPI/chain-aware adapter.
"""
from pathlib import Path
import torch
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer

DEFAULT_WEIGHTS = Path("/opt/esmc/assets/esmc-600m/esmc_600m_2024_12_v0.pth")


def load_model(weights=DEFAULT_WEIGHTS, device="cpu"):
    model = ESMC(d_model=1152, n_heads=18, n_layers=36,
                 tokenizer=EsmSequenceTokenizer(), use_flash_attn=False)
    state = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=torch.float32).eval()
