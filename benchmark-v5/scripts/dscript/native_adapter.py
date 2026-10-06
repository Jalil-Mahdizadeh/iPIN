"""Frozen original D-SCRIPT: length-safe indices and bounded contact-map memory.

No learned weights, residue coverage or pooling rules are changed. Preprojected
FP32 residues are an inference cache of the original learned linear/ReLU layer.
The upstream ModelInteraction.map_predict implementation remains the score path.
"""
import hashlib
import json
from pathlib import Path
import torch
from dscript.models.interaction import DSCRIPTModel
from common import ORIGINAL_SHA, sha


class LengthSafeOriginal(DSCRIPTModel):
    tile_area = 1_000_000

    def extend_positions(self, length):
        if length > self.xx.numel():
            self.xx = torch.nn.Parameter(torch.arange(length, device=self.xx.device), requires_grad=False)

    def embed(self, residues):
        if residues.ndim != 3 or residues.shape[0] != 1:
            raise ValueError("Native score pooling requires one unpadded pair at a time")
        if residues.shape[-1] == 100:
            return residues
        if residues.shape[-1] == 6165:
            return super().embed(residues)
        raise ValueError("Expected original 6165-D residues or their cached 100-D projection")

    def cpred(self, z0, z1, embed_foldseek=False, f0=None, f1=None):
        if embed_foldseek or f0 is not None or f1 is not None:
            raise ValueError("This adapter is the original sequence-only model, not TT3D")
        n, m = z0.shape[1], z1.shape[1]
        if self.tile_area is None or n * m <= self.tile_area:
            return super().cpred(z0, z1)
        a, b = self.embed(z0), self.embed(z1)
        halo = self.contact.conv.kernel_size[0] // 2
        step = max(1, self.tile_area // m - 2 * halo)
        result = torch.empty((1, 1, n, m), device=a.device, dtype=a.dtype)
        # Each local 7x7 convolution receives its complete three-residue halo.
        # Only real global edges use upstream zero padding. No residues are cropped.
        for start in range(0, n, step):
            stop = min(n, start + step)
            left, right = max(0, start - halo), min(n, stop + halo)
            hidden = self.contact.cmap(a[:, left:right], b)
            local = self.contact.predict(hidden)
            result[:, :, start:stop] = local[:, :, start-left:stop-left]
        return result

    def predict(self, z0, z1, *args, **kwargs):
        self.extend_positions(max(z0.shape[1], z1.shape[1]))
        return super().predict(z0, z1, *args, **kwargs)


def create(device="cuda", max_length=7570, tile_area=1_000_000):
    weights = Path("/opt/dscript/weights")
    path = weights / "dscript_human_v1.pt"
    if sha(path) != ORIGINAL_SHA:
        raise RuntimeError("Original checkpoint changed")
    config = json.loads((weights / "human_v1_hf/config.json").read_text())
    config["use_cuda"] = str(device).startswith("cuda")
    model = LengthSafeOriginal(**config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    model.tile_area = tile_area
    model.extend_positions(max_length)
    return model


def learned_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name == "xx":
            continue  # Integer positions are the sole extended nonlearned state.
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
