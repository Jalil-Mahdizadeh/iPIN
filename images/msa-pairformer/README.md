# Frozen MSA Pairformer runtime

`msa-pairformer-arm64-v1.sif` extends the existing native PLM-interact ARM64 SIF without modifying it. It adds pinned Python wheels, MSA Pairformer source commit `875363570df1ae484cf725aba382790444223005`, and weights snapshot `7563e77a87536b5572f91683c39073ef348639a4`. The runtime uses the upstream standard PyTorch triangle updates, avoiding an unqualified CUDA-12 cuEquivariance installation in the CUDA-13 base.

The only upstream source patch initializes `CUEQUIVARIANCE_AVAILABLE=False` before the CUDA availability check so CPU imports work. It changes no GPU tensor operation or model parameter. See `manifests/local-patches.json`.

[runtime.def](runtime.def) and [build.sh](build.sh) describe the build. Wheels are hash-pinned in [manifests/requirements.txt](manifests/requirements.txt) (copied to the build lock); source, weights and wheels are covered by `manifests/build-files.sha256`. The final SIF hash and inspection metadata are recorded separately. Upstream source and weight licenses are preserved inside the image.

The image's build test strictly loads all 111,365,468 parameters on CPU. GPU qualification is separate under `pilot-vx/qualification/`. The pilot does not evaluate either released structural contact head; its primary readout is fitted on Bernett TRAIN only.
