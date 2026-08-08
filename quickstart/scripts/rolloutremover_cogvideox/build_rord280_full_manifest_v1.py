#!/usr/bin/env python3
"""Freeze a strict full-280 CogKit RR cache manifest from the indexed RORD view."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_INDEX_MANIFEST_SHA256 = (
    "841c5b6f24ed0f1b03420dace35843c356fd803ab24ca1f6f3ef0bc52bbc1de4"
)
EXPECTED_MAPPING_SHA256 = (
    "1177ee7b49f7ec85e2fa85d3cddfc7597ef860a10c25cf6e45dc62808e0f8422"
)
EXPECTED_COUNT = 280


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def inspect_mask(path: Path) -> tuple[list[int], float, tuple[int, int]]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    values = sorted(int(value) for value in np.unique(gray))
    if not set(values).issubset({0, 255}) or not {0, 255}.issubset(values):
        raise ValueError(f"mask must contain binary background and target: {path} {values}")
    return values, float((gray > 127).mean()), (gray.shape[1], gray.shape[0])


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if not args.dataset_root.is_dir() or not args.index_manifest.is_file():
        raise FileNotFoundError("dataset root or index manifest is missing")
    manifest_sha256 = sha256_file(args.index_manifest)
    if manifest_sha256 != EXPECTED_INDEX_MANIFEST_SHA256:
        raise ValueError(f"index manifest hash drift: {manifest_sha256}")
    payload = json.loads(args.index_manifest.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT:
        raise ValueError("index manifest must contain exactly 280 records")
    recorded_mapping = payload.get("mapping_sha256")
    if recorded_mapping is not None and recorded_mapping != EXPECTED_MAPPING_SHA256:
        raise ValueError(f"mapping hash drift: {recorded_mapping}")

    manifest_rows = []
    mask_fractions = []
    for index, record in enumerate(records):
        sample_id = f"{index:08d}"
        if record.get("index") != index or record.get("indexed_id") != sample_id:
            raise ValueError(f"non-canonical index record at {index}")
        files = record.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"missing files mapping for {sample_id}")
        source = Path(str(files["img"]["adapter"]))
        gt = Path(str(files["gt"]["adapter"]))
        mask = Path(str(files["mask"]["adapter"]))
        for role, path in (("source", source), ("gt", gt), ("mask", mask)):
            if not path.is_file():
                raise FileNotFoundError(f"missing {role} for {sample_id}: {path}")
            path.absolute().relative_to(args.dataset_root.absolute())
        with Image.open(source) as source_image, Image.open(gt) as gt_image:
            source_size = source_image.size
            gt_size = gt_image.size
        mask_values, mask_fraction, mask_size = inspect_mask(mask)
        if source_size != (960, 540) or gt_size != source_size or mask_size != source_size:
            raise ValueError(
                f"geometry mismatch for {sample_id}: "
                f"source={source_size} gt={gt_size} mask={mask_size}"
            )
        mask_fractions.append(mask_fraction)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "source": str(source),
                "gt": str(gt),
                "mask_sam": str(mask),
                "mask_check": str(mask),
                "dataset_id": "RORD-280-indexed-v1",
                "mask_convention": "white-target-identity",
                "mask_values": mask_values,
                "mask_fraction": mask_fraction,
                "role": "post-selection-exploratory-full280",
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_dir = args.output_root / "manifests"
    audit_dir = args.output_root / "audit"
    manifest_dir.mkdir()
    audit_dir.mkdir()
    manifest_path = manifest_dir / "rord280_full.jsonl"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "status": "passed",
        "dataset_id": "RORD-280-indexed-v1",
        "scientific_scope": "post-selection success-conditioned exploratory diagnostic",
        "dataset_root": str(args.dataset_root),
        "index_manifest": str(args.index_manifest),
        "index_manifest_sha256": manifest_sha256,
        "expected_mapping_sha256": EXPECTED_MAPPING_SHA256,
        "count": len(manifest_rows),
        "id_range": [manifest_rows[0]["sample_id"], manifest_rows[-1]["sample_id"]],
        "geometry": "960x540",
        "stored_mask_convention": "white-target",
        "model_mask_convention": "white-target",
        "mask_transform": "identity",
        "mask_fraction_min": min(mask_fractions),
        "mask_fraction_max": max(mask_fractions),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    audit_path = audit_dir / "full_manifest_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
