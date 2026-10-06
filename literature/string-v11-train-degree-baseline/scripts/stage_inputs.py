"""Archive the exact released V11 inputs, validating previously recorded hashes."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def digest(path, compressed=False):
    h = hashlib.sha256()
    opener = gzip.open if compressed else open
    with opener(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    provenance = ROOT / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    previous = PROJECT / "benchmark-v5/provenance/human-releases-exposure.json"
    native_origin = PROJECT / "bechmark-v5-nonhuman/provenance/native-human-exposure-inputs.json"
    origin = {x["sha256"]: x["url"] for x in json.loads(native_origin.read_text())}
    audit = json.loads(previous.read_text())
    records = []
    for split in audit["sources"]:
        for kind, source_name, suffix in [("native", "native_file", "csv"),
                                          ("tuna", "tuna_dictionary", "dictionary.tsv"),
                                          ("tuna", "tuna_interactions", "interactions.tsv")]:
            source = split[source_name]
            path = Path(source["path"])
            dest = ROOT / "inputs" / kind / f"{split['split']}.{suffix}.gz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert path.stat().st_size == source["bytes"]
            assert digest(path) == source["sha256"], str(path)
            if not dest.exists():
                temp = dest.with_suffix(".tmp")
                with path.open("rb") as src, temp.open("wb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as out:
                        shutil.copyfileobj(src, out, 8 * 1024 * 1024)
                assert digest(temp, compressed=True) == source["sha256"]
                temp.replace(dest)
            assert digest(dest, compressed=True) == source["sha256"]
            url = origin.get(source["sha256"])
            if kind == "tuna":
                rel = str(path).split("/TUnA/", 1)[1]
                url = "https://raw.githubusercontent.com/Wang-lab-UCSD/TUnA/b5bda8fee261a4f27821738db995cf5883dcd133/" + rel
            records.append({"model_source": kind, "split": split["split"], "kind": source_name,
                            "archive": str(dest.relative_to(ROOT)), "source_path": str(path),
                            "source_url": url, "uncompressed_bytes": source["bytes"],
                            "uncompressed_sha256": source["sha256"],
                            "archive_bytes": dest.stat().st_size, "archive_sha256": digest(dest)})
            print("Verified and archived", dest.relative_to(ROOT), flush=True)
    copied = []
    for path in [previous, native_origin,
                 PROJECT / "literature/string-v12.5-review-v6/tuna-seed47-config.yaml",
                 PROJECT / "literature/string-v12.5-review-v6/legacy-degree-baseline.json"]:
        dest = provenance / path.name
        shutil.copyfile(path, dest)
        copied.append({"path": str(dest.relative_to(ROOT)), "source_path": str(path), "sha256": digest(dest)})
    manifest = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "dataset": "Released STRING v11/D-SCRIPT human TRAIN and validation used by humanV11 and TUnA human seed 47",
                "native_dataset_revision": "19dbed25184466fda5341b5ba70c2560ade1bf85",
                "tuna_repository_revision": "b5bda8fee261a4f27821738db995cf5883dcd133",
                "native_model_revision": "e86e392dec13dd0c23252c94947b04a7a9821b0e",
                "tuna_model_revision": "eaec69cafc574d984119079220e75b11ffcb7c09",
                "tuna_model_path": "x-species/TUnA_seed47/model",
                "inputs": records, "copied_provenance": copied,
                "storage": "Lossless gzip copies; uncompressed bytes and SHA256 exactly match the released source files."}
    (provenance / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
