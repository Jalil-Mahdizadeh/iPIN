#!/usr/bin/env bash
set -euo pipefail
umask 077
vx_image_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
vx_project=$(cd -- "$vx_image_root/../.." && pwd -P)
cd "$vx_project"
vx_image="$vx_image_root/msa-pairformer-arm64-v1.sif"
test ! -e "$vx_image"
if test -e "$vx_image_root/manifests/requirements-hashed.lock"; then
    cmp "$vx_image_root/manifests/requirements.txt" "$vx_image_root/manifests/requirements-hashed.lock"
else
    cp "$vx_image_root/manifests/requirements.txt" "$vx_image_root/manifests/requirements-hashed.lock"
fi
test "$(sha256sum images/plm-interact/plm-interact-native-arm64-v1.sif | cut -d ' ' -f1)" = e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2
(cd "$vx_image_root" && sha256sum -c manifests/build-files.sha256)
vx_build_tmp=$(mktemp -d /tmp/ipin-vx-build.XXXXXXXX)
trap 'rm -rf -- "$vx_build_tmp"' EXIT
mkdir -p "$vx_build_tmp/cache" "$vx_build_tmp/tmp"
export APPTAINER_CACHEDIR="$vx_build_tmp/cache"
export APPTAINER_TMPDIR="$vx_build_tmp/tmp"
export TMPDIR="$vx_build_tmp/tmp"
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
apptainer build --fakeroot --mksquashfs-args '-processors 8 -Xcompression-level 1' "$vx_image" "$vx_image_root/runtime.def"
apptainer inspect --json "$vx_image" > "$vx_image_root/manifests/sif-inspect.json"
sha256sum "$vx_image" > "$vx_image_root/manifests/sif.sha256"
chmod a-w "$vx_image"
