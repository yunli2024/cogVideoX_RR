#!/usr/bin/env python3
"""Compute checkpoint-selection diagnostics and contact sheets for CogKit RR."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="STEP=ROOT",
        help="Checkpoint step and completed inference root; repeat once per checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet-rows", type=int, default=4)
    return parser.parse_args()


def parse_runs(values: list[str]) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values:
        step_text, separator, root_text = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value: {value!r}")
        step = int(step_text)
        root = Path(root_text)
        if step in runs:
            raise ValueError(f"duplicate checkpoint step: {step}")
        if not root.is_dir():
            raise FileNotFoundError(root)
        runs[step] = root
    if len(runs) < 2:
        raise ValueError("at least two checkpoint runs are required")
    return dict(sorted(runs.items()))


def sample_ids(root: Path) -> set[str]:
    return {
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "prediction.png").is_file()
    }


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray > 127


def region_mae(left: np.ndarray, right: np.ndarray, region: np.ndarray) -> float:
    if left.shape != right.shape or left.shape[:2] != region.shape:
        raise ValueError("image/mask geometry mismatch")
    if not region.any():
        raise ValueError("empty metric region")
    error = np.abs(left.astype(np.float32) - right.astype(np.float32)).mean(axis=2)
    return float(error[region].mean())


def psnr(left: np.ndarray, right: np.ndarray) -> float:
    error = left.astype(np.float64) - right.astype(np.float64)
    mse = float(np.mean(error * error))
    return float("inf") if mse == 0 else float(10.0 * math.log10((255.0**2) / mse))


def boundary_ring(mask: np.ndarray, size: int = 11) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    dilated = np.asarray(pil.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8) > 0
    eroded = np.asarray(pil.filter(ImageFilter.MinFilter(size)), dtype=np.uint8) > 0
    return np.logical_and(dilated, np.logical_not(eroded))


def checkpoint_rows(runs: dict[int, Path]) -> tuple[list[dict[str, object]], list[str]]:
    sets = {step: sample_ids(root) for step, root in runs.items()}
    reference = next(iter(sets.values()))
    if not reference:
        raise ValueError("no completed prediction directories found")
    for step, values in sets.items():
        if values != reference:
            raise ValueError(
                f"checkpoint {step} sample set mismatch: "
                f"missing={sorted(reference - values)} extra={sorted(values - reference)}"
            )

    rows: list[dict[str, object]] = []
    for sample_id in sorted(reference):
        for step, root in runs.items():
            sample = root / sample_id
            prediction = rgb(sample / "prediction.png")
            source = rgb(sample / "source.png")
            gt = rgb(sample / "gt.png")
            mask = binary_mask(sample / "mask_condition.png")
            ring = boundary_ring(mask)
            inside = region_mae(prediction, gt, mask)
            source_inside = region_mae(source, gt, mask)
            rows.append(
                {
                    "sample_id": sample_id,
                    "checkpoint": step,
                    "mask_fraction": float(mask.mean()),
                    "inside_mae_gt": inside,
                    "inside_improvement_vs_source": source_inside - inside,
                    "boundary_mae_gt": region_mae(prediction, gt, ring),
                    "outside_mae_source": region_mae(prediction, source, ~mask),
                    "full_psnr_gt": psnr(prediction, gt),
                }
            )
    return rows, sorted(reference)


def aggregates(rows: list[dict[str, object]], steps: list[int]) -> list[dict[str, object]]:
    by_sample: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    wins = Counter()
    for sample_rows in by_sample.values():
        best = min(sample_rows, key=lambda row: float(row["inside_mae_gt"]))
        wins[int(best["checkpoint"])] += 1

    output = []
    for step in steps:
        selected = [row for row in rows if int(row["checkpoint"]) == step]
        payload: dict[str, object] = {
            "checkpoint": step,
            "count": len(selected),
            "inside_win_count": wins[step],
        }
        for name in (
            "inside_mae_gt",
            "inside_improvement_vs_source",
            "boundary_mae_gt",
            "outside_mae_source",
            "full_psnr_gt",
        ):
            values = np.asarray([float(row[name]) for row in selected], dtype=np.float64)
            payload[f"{name}_mean"] = float(values.mean())
            payload[f"{name}_median"] = float(np.median(values))
            payload[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(payload)
    return output


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labeled_tile(path: Path, label: str, width: int = 280) -> Image.Image:
    with Image.open(path) as image:
        tile = image.convert("RGB")
    height = round(width * tile.height / tile.width)
    tile = tile.resize((width, height), Image.Resampling.LANCZOS)
    tile = ImageOps.expand(tile, border=(0, 24, 0, 0), fill="white")
    ImageDraw.Draw(tile).text((4, 4), label, fill="black")
    return tile


def save_contact_sheets(
    output_dir: Path,
    runs: dict[int, Path],
    ids: list[str],
    rows_per_sheet: int,
) -> None:
    first_root = next(iter(runs.values()))
    for sheet_index, start in enumerate(range(0, len(ids), rows_per_sheet), start=1):
        row_images = []
        for sample_id in ids[start : start + rows_per_sheet]:
            base = first_root / sample_id
            cells = [
                labeled_tile(base / "source.png", f"{sample_id} source"),
                labeled_tile(base / "mask_condition.png", "mask"),
                labeled_tile(base / "gt.png", "GT"),
            ]
            cells.extend(
                labeled_tile(root / sample_id / "prediction.png", f"ckpt {step}")
                for step, root in runs.items()
            )
            canvas = Image.new(
                "RGB",
                (sum(cell.width for cell in cells), max(cell.height for cell in cells)),
                "white",
            )
            x = 0
            for cell in cells:
                canvas.paste(cell, (x, 0))
                x += cell.width
            row_images.append(canvas)
        sheet = Image.new(
            "RGB",
            (max(row.width for row in row_images), sum(row.height for row in row_images)),
            "white",
        )
        y = 0
        for row in row_images:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(output_dir / f"contact_sheet_{sheet_index}.jpg", quality=94)


def write_summary(path: Path, aggregate: list[dict[str, object]]) -> None:
    lines = [
        "# CogKit selection12 checkpoint diagnostics",
        "",
        "All region errors are RGB MAE on the common decoded 720x480 canvas; lower is better.",
        "Full-frame PSNR is diagnostic only and is not the primary checkpoint criterion.",
        "",
        "| ckpt | n | inside MAE median | boundary MAE median | outside MAE median | inside wins | PSNR mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {checkpoint} | {count} | {inside_mae_gt_median:.4f} | "
            "{boundary_mae_gt_median:.4f} | {outside_mae_source_median:.4f} | "
            "{inside_win_count} | {full_psnr_gt_mean:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Selection boundary: reject severe redraw/collapse first; prioritize inside-mask and boundary quality; use outside-mask fidelity as a guardrail.",
            "These runs must be labelled with their actual scheduler and are not final RORD-280 benchmark evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = parse_runs(args.run)
    if args.sheet_rows <= 0:
        raise ValueError("--sheet-rows must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, ids = checkpoint_rows(runs)
    aggregate = aggregates(rows, list(runs))
    save_csv(args.output_dir / "per_case_metrics.csv", rows)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.output_dir / "SUMMARY.md", aggregate)
    save_contact_sheets(args.output_dir, runs, ids, args.sheet_rows)
    print(json.dumps({"status": "completed", "count": len(ids), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
