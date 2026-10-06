# ESM C 600M runtime for the v4 proposal

Prepared on 2026-10-01 for ARM64 Arrhenius GH200 nodes. This directory contains a separate ESM C image and its build inputs. The existing ESM2/native image is unchanged.

**Status: built and runtime-tested successfully on the GH200.** Image: `plm-interact-esmc-arm64-v1.sif`, **13,578,153,984 bytes**. SHA-256: `ed6aeac781502090632bc2a705300b161b6a65ccfbcf6c7cf64d8d61d8e5b5ef`. [Digest](manifests/sif.sha256), [final GPU check](manifests/sif-gpu-smoke.json), [validation summary](manifests/validation.json).

## Scope and versions

This is an offline **sequence-backbone runtime**. It includes the original ESM C 600M weights, the official unmodified SDK and a small strict-loading helper. The separately versioned standard and chain-aware PPI adapters and trainer live in [retrain-v4](../../retrain-v4/README.md), outside the SIF.

| Component | Pinned identity |
| --- | --- |
| Base SIF | `../plm-interact/plm-interact-native-arm64-v1.sif` |
| Base SHA-256 | `e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2` |
| Architecture / Python | aarch64 / 3.12.3 |
| Torch | `2.8.0a0+34c6371d24.nv25.8`, inherited unchanged |
| CUDA runtime | 13.0, inherited unchanged |
| ESM SDK | `esm==3.2.3` |
| Transformers / Tokenizers | 4.48.1 / 0.21.4 |
| NumPy | 1.26.4, inherited unchanged |
| Model repository / revision | `biohub/esmc-600m-2024-12` / `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c` |
| Weight SHA-256 | `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324` |
| Original weight size | 2,300,275,866 bytes |
| Loaded SDK model | 36 layers, width 1152, 18 heads; 575,036,992 parameters including its sequence decoder |

The current SDK release requires a different Torch stack. This older official SDK supports the original checkpoint without changing the already working NVIDIA Torch/CUDA runtime. The precise model identity is in [model.json](manifests/model.json); its digest matches the Hub revision's upstream LFS record.

The new image installs the additions in `/opt/esmc/venv` with inherited base packages. It removes the native PLM-interact package and `/opt/plm_interact` assets **only from the new image's build root**; these are unnecessary for ESM C and impose a conflicting Transformers requirement. The source SIF is never modified. Use the original image for ESM2/native experiments.

The SDK declares `torchtext`, but no Python file in its wheel references it. This unused dependency is deliberately omitted: there is no corresponding ARM wheel, and installing an unrelated Torch version to satisfy it would defeat the runtime choice. This is a scoped ESM C runtime, not a claim that every feature in the SDK is supported. [Source audit](manifests/sdk-source-audit.json)

Biotite needs Biotraj, which also lacks a suitable published ARM wheel. The unmodified `biotraj==1.2.2` source was downloaded with its PyPI hash verified and built locally. Its source and compiled wheel are retained. [Source identity](manifests/biotraj-source.json), [build log](logs/biotraj-build.log). An initial build with an incomplete build environment produced incorrect version metadata; that wheel is excluded from `wheelhouse` and from the image. The successful build used its declared isolated build dependencies; their exact versions are in the log. Runtime NumPy remains 1.26.4.

All 23 added wheels are pinned by version and SHA-256 in [requirements.lock](build/requirements.lock), with download/source records in [wheels.json](manifests/wheels.json). The base SIF digest pins inherited packages. Package licensing files remain in the installed distributions and wheels; the original model card is in [assets](assets/esmc-600m/README.md).

## Use

From the workspace root, verify the image and run its embedded CPU check:

```bash
(cd images/plm-interact-esmc && sha256sum -c manifests/sif.sha256)
apptainer exec --cleanenv images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif \
    python /opt/esmc/tests/runtime_smoke.py
```

For a full pretrained-model GPU check on an allocated GPU:

```bash
apptainer exec --nv --cleanenv images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif \
    python /opt/esmc/tests/runtime_smoke.py --gpu
```

This second command runs inference and one disposable optimizer step on two short synthetic amino-acid pairs. It saves no trained checkpoint and does not read any experimental train/validation/test split.

Inside the image, `from esmc_runtime import load_model; model = load_model(device="cuda")` strictly loads the embedded original checkpoint with `weights_only=True`. The model starts in evaluation mode with FP32 parameters; the caller explicitly selects training mode and autocast. No Hub downloads or dynamically fetched code are needed. The loader uses SDK SDPA attention (`use_flash_attn=False`); the GPU check explicitly selects PyTorch's efficient SDPA backend.

The unmodified SDK forward returns all layers' hidden states and MLM logits. It is suitable for the short runtime check; v4 implements and checks a final-embedding-only adapter for long-pair training. Do not pass A/B chain identifiers into the SDK's `sequence_id` argument: this argument masks cross-ID attention. The v4 adapter handles chain metadata separately.

For distributed workers, use **`python -m torch.distributed.run`** inside the image. The inherited `/usr/local/bin/torchrun` executable has a base-interpreter shebang and bypasses `/opt/esmc/venv`, so its workers cannot find the ESM C SDK. The v4 training wrapper uses the module entry point; its worker-environment check verifies the interpreter and SDK imports. This requires no image or package change.

## Validation record and limits

The development environment passed CPU import/tokenization/tiny-forward checks and strict full-weight loading, BF16 inference, finite encoder gradients and an FP32-state AdamW step on the interactive GH200. [Development CPU report](manifests/development-cpu-smoke.json), [development GPU report](manifests/development-gpu-smoke.json).

The final SIF repeated the full GPU check successfully with `--cleanenv --containall`, the host working-directory/administrator bind paths disabled, and only the result directory explicitly bound. It used the embedded `/opt/esmc` code and pretrained weights. The cluster disables PID-namespace virtualization; the relevant check here is that model code, dependencies and weights load from the image. [Final GPU report](manifests/sif-gpu-smoke.json), [execution log](logs/sif-gpu-smoke.log), [build-time CPU test](logs/sif-build.log).

All 23 added distribution versions match the hash lock. `pip check` reports exactly the documented missing, unused `torchtext` dependency and no other conflicts. [Dependency check](manifests/dependency-check.json), [installed distributions](manifests/distributions.json). The final image's ESM C labels were verified, and the original source SIF was hashed again after the build and found unchanged. [Base image verification](manifests/base.json), [image metadata](manifests/sif-inspect.json).

The synthetic GPU check peaked at **8.99 GiB** allocated memory. Its 63/42-token inputs do not estimate production memory or throughput. It saved no candidate checkpoint. The temporary intermediate SIF with inherited native-model labels was removed after the corrected final image passed; the build-attempt logs remain.

These runtime checks do **not** qualify the future PPI head/embedding adapter, chain-aware attention, longest real pairs or four-GPU interrupted/resumed training. Those finite engineering checks are specified in [the v4 proposal](../../improvment-proposal-v4.md). No v4 production job is submitted by this directory's tools.

## Rebuild and provenance

The saved inputs support an offline package/image build from the pinned existing base SIF:

```bash
bash images/plm-interact-esmc/build/build_image.sh \
    > images/plm-interact-esmc/logs/sif-build.log 2>&1
```

The script verifies base, model and wheel hashes and refuses to overwrite an existing output image. It uses a temporary build root on node-local `/tmp`, cleans that root on exit, and records the resulting SIF hash and inspection metadata. Gzip level 1 with eight workers limits assembly time without changing the installed files. The recipe is [plm-interact-esmc-arm64-v1.def](build/plm-interact-esmc-arm64-v1.def). Apptainer filesystem timestamps may change on rebuilding; the recorded digest identifies this concrete image rather than promising byte-identical SIF reconstruction.

`build/runtime-venv` is a development-only environment used to resolve/test packages. It is not copied into the image and is not needed to run it. `build/prepare_wheelhouse.py` preserves the successful dependency resolution in a version/hash lock; the installed image uses that lock with offline `--no-deps --require-hashes` installation. Earlier failed dependency/build attempts remain in `logs` for provenance and are not evidence of a passing final image.
