#!/usr/bin/env python3
"""Audit RORD-280 against the actual CogVideoX training population and freeze 12 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


PHASH_SIZE = 8
PHASH_HIGHFREQ_FACTOR = 4
PHASH_THRESHOLD = 4
SELECTION_SALT = "RolloutRemover-CogVideoX5B-RORD280-selection12-v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dct_matrix(size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.float64)[:, None]
    cols = np.arange(size, dtype=np.float64)[None, :]
    matrix = np.cos(np.pi * (2 * cols + 1) * rows / (2 * size))
    matrix[0] *= np.sqrt(1.0 / size)
    matrix[1:] *= np.sqrt(2.0 / size)
    return matrix


_PHASH_INPUT_SIZE = PHASH_SIZE * PHASH_HIGHFREQ_FACTOR
_DCT_MATRIX = dct_matrix(_PHASH_INPUT_SIZE)


def decoded_rgb_features(path: Path) -> tuple[str, int]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        exact_header = f"RGB\0{width}x{height}\0".encode("ascii")
        exact = hashlib.sha256(exact_header + rgb.tobytes()).hexdigest()
        gray = rgb.convert("L").resize(
            (_PHASH_INPUT_SIZE, _PHASH_INPUT_SIZE), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT_MATRIX @ pixels @ _DCT_MATRIX.T
    low = dct[:PHASH_SIZE, :PHASH_SIZE].reshape(-1)
    median = float(np.median(low[1:]))
    bits = low > median
    phash = 0
    for bit_index, value in enumerate(bits):
        if bool(value):
            phash |= 1 << bit_index
    return exact, phash


class HammingIndex:
    """Exact radius index for 64-bit hashes using threshold+1 disjoint chunks."""

    def __init__(self, values: Iterable[int], threshold: int) -> None:
        self.threshold = threshold
        chunk_count = threshold + 1
        base, extra = divmod(64, chunk_count)
        widths = [base + (1 if i < extra else 0) for i in range(chunk_count)]
        self.chunks: list[tuple[int, int]] = []
        shift = 0
        for width in widths:
            self.chunks.append((shift, (1 << width) - 1))
            shift += width
        self.buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
        for value in set(values):
            for index, (chunk_shift, mask) in enumerate(self.chunks):
                self.buckets[(index, (value >> chunk_shift) & mask)].add(value)

    def nearest_within(self, query: int) -> int | None:
        candidates: set[int] = set()
        for index, (shift, mask) in enumerate(self.chunks):
            candidates.update(self.buckets.get((index, (query >> shift) & mask), ()))
        distances = [(query ^ value).bit_count() for value in candidates]
        admitted = [distance for distance in distances if distance <= self.threshold]
        return min(admitted) if admitted else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    return parser.parse_args()


def load_training_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("sample_id", "source", "gt"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"training manifest line {line_number} missing {key}")
            rows.append(row)
    if len(rows) != 64505:
        raise ValueError(f"expected 64505 training rows, found {len(rows)}")
    return rows


def feature_many(paths: list[Path], workers: int) -> dict[Path, tuple[str, int]]:
    unique = sorted(set(paths), key=str)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(decoded_rgb_features, unique))
    return dict(zip(unique, values))


def mask_fraction(path: Path) -> tuple[float, list[int], tuple[int, int]]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    unique = sorted(int(value) for value in np.unique(gray))
    if not set(unique).issubset({0, 255}):
        raise ValueError(f"mask is not binary 0/255: {path} values={unique[:20]}")
    return float((gray > 127).mean()), unique, (int(gray.shape[1]), int(gray.shape[0]))


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"output root already exists: {args.output_root}")
    if not args.dataset_root.is_dir() or not args.index_manifest.is_file():
        raise FileNotFoundError("RORD-280 dataset root or index manifest is missing")

    training_rows = load_training_rows(args.training_manifest)
    training_jobs = []
    for row in training_rows:
        training_jobs.extend((Path(str(row["source"])), Path(str(row["gt"]))))
    if any(not path.is_file() for path in training_jobs):
        raise FileNotFoundError("at least one training source/GT path is missing")
    training_features = feature_many(training_jobs, args.workers)
    training_exact = {value[0] for value in training_features.values()}
    training_phash_index = HammingIndex(
        (value[1] for value in training_features.values()), PHASH_THRESHOLD
    )

    index_payload = json.loads(args.index_manifest.read_text(encoding="utf-8"))
    index_records = index_payload.get("records")
    if not isinstance(index_records, list) or len(index_records) != 280:
        raise ValueError("index manifest must contain exactly 280 records")

    candidate_rows = []
    candidate_jobs = []
    for expected_index, index_record in enumerate(index_records):
        if index_record.get("index") != expected_index:
            raise ValueError(f"non-contiguous index manifest at {expected_index}")
        sample_id = index_record.get("indexed_id")
        if sample_id != f"{expected_index:08d}":
            raise ValueError(f"unexpected indexed_id at {expected_index}: {sample_id!r}")
        files = index_record.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"missing files mapping for {sample_id}")
        source = Path(str(files["img"]["adapter"]))
        gt = Path(str(files["gt"]["adapter"]))
        mask = Path(str(files["mask"]["adapter"]))
        expected_parent = args.dataset_root.resolve()
        for path in (source, gt, mask):
            try:
                path.resolve(strict=True).relative_to(expected_parent)
            except ValueError:
                # Adapter entries are symlinks by design; require the link itself to
                # live under the frozen indexed view even when its target does not.
                path.absolute().relative_to(expected_parent)
        for path in (source, gt, mask):
            if not path.is_file():
                raise FileNotFoundError(path)
        source_size = Image.open(source).size
        gt_size = Image.open(gt).size
        fraction, mask_values, mask_size = mask_fraction(mask)
        if source_size != gt_size or source_size != mask_size:
            raise ValueError(f"geometry mismatch for {sample_id}")
        row = {
            "sample_id": sample_id,
            "source": source,
            "gt": gt,
            "mask": mask,
            "mask_fraction": fraction,
            "mask_values": mask_values,
            "geometry": list(source_size),
        }
        candidate_rows.append(row)
        candidate_jobs.extend((source, gt))
    candidate_features = feature_many(candidate_jobs, args.workers)

    exact_collisions = []
    near_collisions = []
    unseen = []
    for row in candidate_rows:
        rejected = False
        for role in ("source", "gt"):
            exact, phash = candidate_features[row[role]]
            if exact in training_exact:
                exact_collisions.append({"sample_id": row["sample_id"], "role": role})
                rejected = True
            distance = training_phash_index.nearest_within(phash)
            if distance is not None:
                near_collisions.append(
                    {"sample_id": row["sample_id"], "role": role, "distance": distance}
                )
                rejected = True
        if not rejected:
            unseen.append(row)
    if len(unseen) < 12:
        raise RuntimeError(f"only {len(unseen)} RORD-280 cases pass overlap audit")

    ranked = sorted(unseen, key=lambda row: (row["mask_fraction"], row["sample_id"]))
    quartiles: list[list[dict[str, object]]] = [[], [], [], []]
    for rank, row in enumerate(ranked):
        quartiles[min(3, rank * 4 // len(ranked))].append(row)
    selected = []
    for quartile_index, rows in enumerate(quartiles):
        ordered = sorted(
            rows,
            key=lambda row: sha256_bytes(
                f"{SELECTION_SALT}\nq{quartile_index}\n{row['sample_id']}".encode("ascii")
            ),
        )
        selected.extend(ordered[:3])
    selected.sort(key=lambda row: row["sample_id"])

    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_dir = args.output_root / "manifests"
    audit_dir = args.output_root / "audit"
    manifest_dir.mkdir()
    audit_dir.mkdir()
    manifest_path = manifest_dir / "selection12.jsonl"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            payload = {
                "sample_id": row["sample_id"],
                "source": str(row["source"]),
                "gt": str(row["gt"]),
                "mask_sam": str(row["mask"]),
                "mask_check": str(row["mask"]),
                "selection_role": "checkpoint-selection-only",
                "mask_fraction": row["mask_fraction"],
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    report = {
        "schema_version": 1,
        "status": "passed" if len(unseen) == 280 else "passed_with_exclusions",
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "training_record_count": len(training_rows),
        "training_unique_source_gt_path_count": len(training_features),
        "candidate_dataset": "RORD-280-indexed-v1",
        "candidate_index_manifest": str(args.index_manifest),
        "candidate_index_manifest_sha256": sha256_file(args.index_manifest),
        "candidate_count": len(candidate_rows),
        "overlap_policy": {
            "roles": "source and GT jointly, including cross-role",
            "exact": "SHA256(RGB\\0{width}x{height}\\0 + decoded RGB bytes)",
            "phash": "RGB->L; LANCZOS 32x32; orthonormal DCT; low 8x8; median excluding DC",
            "phash_hamming_threshold": PHASH_THRESHOLD,
        },
        "exact_collision_occurrences": exact_collisions,
        "near_collision_occurrences": near_collisions,
        "unseen_candidate_count": len(unseen),
        "selection_policy": "four mask-area rank quartiles; three SHA256-salted IDs per quartile",
        "selection_salt": SELECTION_SALT,
        "selection_count": len(selected),
        "selected": [
            {
                "sample_id": row["sample_id"],
                "mask_fraction": row["mask_fraction"],
                "geometry": row["geometry"],
            }
            for row in selected
        ],
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "scientific_boundary": "success-conditioned internal checkpoint diagnostic only",
    }
    report_path = audit_dir / "overlap_and_selection.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
