#!/usr/bin/env python3
"""Restore CogKit 720x480 letterboxed predictions and audit RORD-280 pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_COUNT = 280
EXPECTED_INDEX_MANIFEST_SHA256 = (
    "841c5b6f24ed0f1b03420dace35843c356fd803ab24ca1f6f3ef0bc52bbc1de4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--canvas-width", type=int, default=720)
    parser.add_argument("--canvas-height", type=int, default=480)
    return parser.parse_args()


def prediction_map(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in root.glob("shard*/*/prediction.png"):
        sample_id = path.parent.name
        if sample_id in mapping:
            raise ValueError(f"duplicate prediction for {sample_id}")
        mapping[sample_id] = path
    return mapping


def restore_letterbox(
    prediction: Image.Image,
    original_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    if prediction.size != canvas_size:
        raise ValueError(f"prediction canvas mismatch: {prediction.size} != {canvas_size}")
    original_width, original_height = original_size
    scale = min(canvas_height / original_height, canvas_width / original_width)
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))
    left = (canvas_width - content_width) // 2
    top = (canvas_height - content_height) // 2
    crop = prediction.crop((left, top, left + content_width, top + content_height))
    return crop.resize(original_size, Image.Resampling.BILINEAR)


def letterbox_crop_box(
    original_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    canvas_width, canvas_height = canvas_size
    original_width, original_height = original_size
    scale = min(canvas_height / original_height, canvas_width / original_width)
    content_width = max(1, int(round(original_width * scale)))
    content_height = max(1, int(round(original_height * scale)))
    left = (canvas_width - content_width) // 2
    top = (canvas_height - content_height) // 2
    return left, top, left + content_width, top + content_height


def validate_mask(path: Path, expected_size: tuple[int, int]) -> None:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
        size = image.size
    values = set(int(value) for value in np.unique(mask))
    if size != expected_size or not values.issubset({0, 255}) or not {0, 255}.issubset(values):
        raise ValueError(f"invalid precise mask {path}: size={size} values={sorted(values)}")


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    if not args.inference_root.is_dir() or not args.dataset_root.is_dir():
        raise FileNotFoundError("inference or dataset root is missing")
    if sha256_file(args.index_manifest) != EXPECTED_INDEX_MANIFEST_SHA256:
        raise ValueError("RORD-280 index manifest hash drift")
    index_payload = json.loads(args.index_manifest.read_text(encoding="utf-8"))
    records = index_payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT:
        raise ValueError("expected exactly 280 index records")
    predictions = prediction_map(args.inference_root)
    expected_ids = {f"{index:08d}" for index in range(EXPECTED_COUNT)}
    if set(predictions) != expected_ids:
        raise ValueError(
            f"prediction key mismatch: missing={sorted(expected_ids - set(predictions))} "
            f"extra={sorted(set(predictions) - expected_ids)}"
        )

    prediction_dir = args.output_root / "prediction"
    gt_dir = args.output_root / "gt"
    mask_dir = args.output_root / "mask"
    prediction_dir.mkdir(parents=True, exist_ok=False)
    gt_dir.mkdir()
    mask_dir.mkdir()
    audit_rows = []
    for index, record in enumerate(records):
        sample_id = f"{index:08d}"
        if record.get("index") != index or record.get("indexed_id") != sample_id:
            raise ValueError(f"index record drift at {sample_id}")
        files = record["files"]
        gt = Path(str(files["gt"]["adapter"]))
        mask = Path(str(files["mask"]["adapter"]))
        source = Path(str(files["img"]["adapter"]))
        for path in (source, gt, mask):
            if not path.is_file():
                raise FileNotFoundError(path)
        with Image.open(source) as source_image, Image.open(gt) as gt_image:
            original_size = source_image.size
            if original_size != (960, 540) or gt_image.size != original_size:
                raise ValueError(f"source/GT geometry mismatch for {sample_id}")
        validate_mask(mask, original_size)
        with Image.open(predictions[sample_id]) as image:
            restored = restore_letterbox(
                image.convert("RGB"),
                original_size,
                (args.canvas_width, args.canvas_height),
            )
        prediction_output = prediction_dir / f"{sample_id}.png"
        restored.save(prediction_output)
        gt_output = gt_dir / f"{sample_id}.png"
        mask_output = mask_dir / f"{sample_id}.png"
        os.symlink(str(gt.resolve(strict=True)), gt_output)
        os.symlink(str(mask.resolve(strict=True)), mask_output)
        audit_rows.append(
            {
                "sample_id": sample_id,
                "source_prediction": str(predictions[sample_id]),
                "restored_prediction": str(prediction_output),
                "restored_prediction_sha256": sha256_file(prediction_output),
                "gt": str(gt_output),
                "mask": str(mask_output),
                "canvas_size": [args.canvas_width, args.canvas_height],
                "content_crop": list(
                    letterbox_crop_box(
                        original_size,
                        (args.canvas_width, args.canvas_height),
                    )
                ),
                "output_size": list(original_size),
            }
        )

    audit = {
        "schema_version": 1,
        "status": "passed",
        "count": len(audit_rows),
        "dataset_id": "RORD-280-indexed-v1",
        "index_manifest_sha256": EXPECTED_INDEX_MANIFEST_SHA256,
        "prediction_semantics": "pure CogVideoX VAE decode; no blending or paste-back",
        "stored_mask_convention": "white-target",
        "mask_transform": "identity",
        "geometry_transform": "crop 720x405 content from 720x480 letterbox, then RGB bilinear restore to 960x540",
        "records": audit_rows,
    }
    audit_path = args.output_root / "prepare_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "count": len(audit_rows), "audit": str(audit_path)}))


if __name__ == "__main__":
    main()
