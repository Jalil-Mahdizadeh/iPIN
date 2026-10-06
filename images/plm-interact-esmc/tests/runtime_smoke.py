"""Container qualification on synthetic amino-acid strings, not a PPI experiment."""
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from esmc_runtime import DEFAULT_WEIGHTS, load_model

EXPECTED_WEIGHTS = "8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324"


def pair_ids(tokenizer, a, b):
    aa = tokenizer.encode(a, add_special_tokens=False)
    bb = tokenizer.encode(b, add_special_tokens=False)
    assert len(aa) == len(a) and len(bb) == len(b)
    assert tokenizer.unk_token_id not in aa + bb
    return [tokenizer.cls_token_id, *aa, tokenizer.eos_token_id,
            *bb, tokenizer.eos_token_id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    versions = {name: md.version(name) for name in (
        "esm", "torch", "transformers", "tokenizers", "numpy", "biotite", "biotraj"
    )}
    assert platform.machine() == "aarch64"
    assert versions["esm"] == "3.2.3"
    assert versions["torch"] == "2.8.0a0+34c6371d24.nv25.8"
    assert versions["transformers"] == "4.48.1"
    assert versions["tokenizers"] == "0.21.4"
    assert versions["numpy"] == "1.26.4"
    tokenizer = EsmSequenceTokenizer()
    assert (tokenizer.cls_token_id, tokenizer.pad_token_id,
            tokenizer.eos_token_id, tokenizer.mask_token_id) == (0, 1, 2, 32)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows = [pair_ids(tokenizer, aa * 2, aa[::-1]),
            pair_ids(tokenizer, aa + "ACDE", aa[:15])]
    length = ((max(map(len, rows)) + 7) // 8) * 8
    tokens = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = torch.tensor(row)
    result = {
        "architecture": platform.machine(), "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "pair_layout": "[CLS] A [EOS] B [EOS]",
        "synthetic_pair_lengths_with_specials": list(map(len, rows)),
        "cpu_imports_and_tokenization": "passed",
        "production_trainer_qualified": False,
        "chain_aware_adapter_qualified": False,
        "long_context_qualified": False,
        "four_gpu_resume_qualified": False,
    }
    if args.gpu:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        with args.weights.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        assert digest == EXPECTED_WEIGHTS
        model = load_model(args.weights, "cuda")
        assert all(p.dtype == torch.float32 for p in model.parameters())
        result["weights_sha256"] = digest
        result["strict_pretrained_load"] = "passed"
        result["total_parameters"] = sum(p.numel() for p in model.parameters())
        model.sequence_head.requires_grad_(False)
        tokens = tokens.cuda()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), \
                sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = model(tokens)
            assert output.embeddings.shape == (2, length, 1152)
            assert torch.isfinite(output.embeddings).all()
            assert torch.isfinite(output.sequence_logits).all()
        result["pretrained_bf16_inference"] = "passed"
        del output
        model.train()
        head = torch.nn.Linear(1152, 1).cuda()
        parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=2e-5, foreach=False)
        with torch.autocast("cuda", dtype=torch.bfloat16), \
                sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            output = model(tokens)
            logits = head(output.embeddings[:, 0]).squeeze(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.], device="cuda"))
        assert torch.isfinite(loss)
        loss.backward()
        gradients = {}
        for name in ("embed.weight", "transformer.blocks.0.attn.layernorm_qkv.1.weight",
                     "transformer.blocks.35.attn.layernorm_qkv.1.weight"):
            grad = dict(model.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all()
            norm = float(grad.norm())
            assert norm > 0
            gradients[name] = norm
        before = head.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(head.weight, before)
        assert all(torch.isfinite(p).all() for p in parameters)
        assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
        torch.cuda.synchronize()
        result.update(
            gpu=torch.cuda.get_device_name(),
            compute_capability=list(torch.cuda.get_device_capability()),
            gpu_total_memory_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
            encoder_gradient_norms=gradients,
            synthetic_loss=float(loss.detach()),
            backward_and_disposable_adamw_step="passed",
            optimizer_state_dtype="float32",
            sdpa_backend="EFFICIENT_ATTENTION",
            peak_allocated_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
            note="Synthetic runtime check only; updated weights are discarded and no candidate checkpoint is saved.",
        )
    else:
        tiny = ESMC(64, 4, 1, tokenizer, use_flash_attn=False).eval()
        with torch.no_grad():
            output = tiny(tokens)
        assert torch.isfinite(output.embeddings).all()
        result["tiny_cpu_forward"] = "passed"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    result["status"] = "passed"
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
