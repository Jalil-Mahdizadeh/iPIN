PLM-interact runtime image copied from
`/nobackup/proj/disk/theo-storage/personal/jalil/iPIN-OpenPPI/benchmark/containers/`.

The image is `plm-interact-native-arm64-v1.sif` (13,101,785,088 bytes), built for
Linux ARM64/aarch64 and previously qualified on GH200 according to the source
container documentation. It contains PLMinteract 0.1.1, Transformers 4.40.1,
the original 650M humanV11 checkpoint, dependencies, tokenizer configuration,
and the pinned upstream source.

Bundled locations inside the image:

- Checkpoint: `/opt/plm_interact/assets/humanV11/pytorch_model.bin`
- ESM-2 configuration/tokenizer: `/opt/plm_interact/assets/esm2_650m/`
- Upstream source: `/opt/plm_interact/upstream/`
- Dependency provenance: `/opt/plm_interact/downloads.json`

Image SHA-256:
`e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2`

Verify the local files from this directory:

```bash
sha256sum -c SHA256SUMS
```

The `manifests/` directory preserves the original download and inspection
records. `manifests/source-sif.sha256` retains the original source pathname;
use the top-level `SHA256SUMS` to verify this copy.

The `build-reference/` directory preserves the original recipe, build script,
download script, and build log. Those scripts refer to the original
iPIN-OpenPPI repository layout and its parent image. They are build records;
the copied SIF already contains the runtime assets.
