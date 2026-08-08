#!/usr/bin/env python3
"""Fail-closed artifact audit for the frozen CogKit RR RORD-280 full run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image


RUN_ID = "run-20260807-221601-77609ae1"
EXPECTED_COUNT = 280
EXPECTED_MANIFEST_SHA256 = (
    "fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a"
)
EXPECTED_INDEX_SHA256 = (
    "841c5b6f24ed0f1b03420dace35843c356fd803ab24ca1f6f3ef0bc52bbc1de4"
)
EXPECTED_LORA_SHA256 = (
    "db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1"
)
EXPECTED_TRAINING_CONTRACT_SHA256 = (
    "4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1"
)
EXPECTED_SCHEDULER_SHA256 = (
    "247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5"
)
SHARD_SHA256 = {
    0: "2a127c0e70da8ff7a4cf7c949a29e18c109afbb2f3425199159f16ccb3111538",
    1: "49b48c01fab6cc333aa8cff7cb5a620613eb75c22275973f29bfcb4c35af1f0a",
    2: "30e74450ed42712766a609dfddf1c1850e848d363ac4c30bd489dde2894ba1b5",
    3: "5c2a2b6d491d86705716e2c4dd5fcc02d4ee8d5beda4b32d370f77ebf3956ceb",
}
METRICS = ["psnr", "ssim", "lpips", "fid", "cmmd", "as", "cfd", "remove"]
SUMMARY_KEYS = {"PSNR", "SSIM", "LPIPS", "FID", "CMMD", "AS", "CFD", "ReMOVE"}
EXPECTED_EVALUATOR_HASHES = {
    "AestheticScore.py": "a9a321917c6c29887df8ec89038c46344f29f372ed9af6dde6dc1aa99d39558f",
    "CFD.py": "e3a1f1e2addd70c0629b54a9c83cde5a642b385aabfe39236ab7581c9e1a9776",
    "CMMD.py": "3d81ae41970db3a3e2231484d0c375aae5a54805b75b504120d0754db87c2b9e",
    "FID.py": "f8c214c01ecf716bf105c5e0638417811a8b898a2fd00e1a5985eade1776ff1c",
    "LPIPS.py": "10a75f8e088036c841f00711da78bc24a8d9d5b56b96dbdfccc6dd7f373bd5c1",
    "PSNR.py": "9b4e1c1fe28a29bdcf3db5d8278d1d3a496ce9ab07420478912606219668e4d4",
    "ReMOVE.py": "4d53a8a24e09d53d743964534fecded3a067ea8a2834372d59b5dec45fbb5051",
    "SSIM.py": "4a4784b09a39eb5ed54f352e45a3483de48efcda739d7e538144a03501191edc",
    "spatial_protocol.py": "fd04b2301d6a15b62a96a6cbf38b4e627a88d66af1a6a44dc438cbed8c9f0f31",
    "unify_eval.py": "386c7054ab9aaee1bb3450250ea7292bdb8c142ff689dae4a35f19f8b2e35adf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("inference", "prepared", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def expected_shard_ids(shard: int) -> list[str]:
    start = shard * 70
    return [f"{index:08d}" for index in range(start, start + 70)]


def audit_inference(run_root: Path) -> dict[str, Any]:
    require_equal(run_root.name, RUN_ID, "run id")
    inference_root = run_root / "inference"
    all_ids: list[str] = []
    shard_results = []
    for shard in range(4):
        expected_ids = expected_shard_ids(shard)
        shard_root = inference_root / f"shard{shard}"
        contract_path = shard_root / "inference_contract.json"
        manifest_path = shard_root / "inference_manifest.jsonl"
        log_path = run_root / "logs" / f"shard{shard}.log"
        for path in (contract_path, manifest_path, log_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        contract = load_json(contract_path)
        exact_contract = {
            "schema_version": 2,
            "model_path": "/mnt/cpfs/jiachengliu/pretrained_models/CogVideoX-5b",
            "lora_sha256": EXPECTED_LORA_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "training_contract_sha256": EXPECTED_TRAINING_CONTRACT_SHA256,
            "height": 480,
            "width": 720,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "use_dynamic_cfg": False,
            "seed": 42,
            "dtype": "bf16",
            "mask_condition_kind": "mask_check",
            "save_trajectory": False,
            "save_raw_latents": False,
            "pure_output": True,
            "paste_back": False,
            "output_blending": False,
            "gt_used_as_model_input": False,
        }
        for key, expected in exact_contract.items():
            require_equal(contract.get(key), expected, f"shard{shard} contract.{key}")
        require_equal(contract.get("sample_ids"), expected_ids, f"shard{shard} sample ids")
        sample_ids_file = contract.get("sample_ids_file")
        if not isinstance(sample_ids_file, dict):
            raise ValueError(f"shard{shard} sample_ids_file is not an object")
        require_equal(
            sample_ids_file.get("sha256"),
            SHARD_SHA256[shard],
            f"shard{shard} sample-id hash",
        )
        scheduler = contract.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ValueError(f"shard{shard} scheduler is not an object")
        exact_scheduler = {
            "class_name": "CogVideoXDPMScheduler",
            "config_sha256": EXPECTED_SCHEDULER_SHA256,
            "kind": "dpm",
            "num_train_timesteps": 1000,
            "prediction_type": "v_prediction",
            "timestep_spacing": "trailing",
        }
        for key, expected in exact_scheduler.items():
            require_equal(scheduler.get(key), expected, f"shard{shard} scheduler.{key}")

        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require_equal(len(rows), 70, f"shard{shard} manifest count")
        require_equal([row.get("sample_id") for row in rows], expected_ids, f"shard{shard} manifest ids")
        for row, sample_id in zip(rows, expected_ids, strict=True):
            sample_dir = shard_root / sample_id
            for filename in (
                "prediction.png",
                "metadata.json",
                "source.png",
                "gt.png",
                "mask_condition.png",
                "mask_prediction.png",
            ):
                if not (sample_dir / filename).is_file():
                    raise FileNotFoundError(sample_dir / filename)
            if (sample_dir / "trajectory").exists() or (sample_dir / "raw_f7_f13_latents.pt").exists():
                raise ValueError(f"forbidden full-run trajectory/raw latent artifact: {sample_dir}")
            with Image.open(sample_dir / "prediction.png") as image:
                require_equal(image.size, (720, 480), f"{sample_id} prediction size")
            for key, expected in {
                "schema_version": 2,
                "sample_id": sample_id,
                "trajectory_frames": [],
                "raw_latents": None,
                "pure_output": True,
                "paste_back": False,
                "output_blending": False,
                "gt_used_as_model_input": False,
            }.items():
                require_equal(row.get(key), expected, f"{sample_id} manifest.{key}")
            metadata = load_json(sample_dir / "metadata.json")
            require_equal(metadata, row, f"{sample_id} metadata/manifest equality")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if '"status": "completed"' not in log_text or '"count": 70' not in log_text:
            raise ValueError(f"shard{shard} log lacks terminal completion record")
        all_ids.extend(expected_ids)
        shard_results.append(
            {
                "shard": shard,
                "count": 70,
                "contract_sha256": sha256_file(contract_path),
                "manifest_sha256": sha256_file(manifest_path),
                "log_sha256": sha256_file(log_path),
            }
        )
    require_equal(all_ids, [f"{index:08d}" for index in range(EXPECTED_COUNT)], "full sample ids")
    return {"count": EXPECTED_COUNT, "shards": shard_results}


def audit_prepared(run_root: Path) -> dict[str, Any]:
    ready = run_root / "eval_ready"
    audit_path = ready / "prepare_audit.json"
    audit = load_json(audit_path)
    require_equal(audit.get("status"), "passed", "prepare status")
    require_equal(audit.get("count"), EXPECTED_COUNT, "prepare count")
    require_equal(audit.get("index_manifest_sha256"), EXPECTED_INDEX_SHA256, "prepare index hash")
    require_equal(audit.get("stored_mask_convention"), "white-target", "mask convention")
    require_equal(audit.get("mask_transform"), "identity", "mask transform")
    records = audit.get("records")
    if not isinstance(records, list):
        raise ValueError("prepare records must be a list")
    require_equal(len(records), EXPECTED_COUNT, "prepare record count")
    expected_ids = [f"{index:08d}" for index in range(EXPECTED_COUNT)]
    require_equal([record.get("sample_id") for record in records], expected_ids, "prepare ids")
    for record, sample_id in zip(records, expected_ids, strict=True):
        prediction = ready / "prediction" / f"{sample_id}.png"
        gt = ready / "gt" / f"{sample_id}.png"
        mask = ready / "mask" / f"{sample_id}.png"
        for path in (prediction, gt, mask):
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                require_equal(image.size, (960, 540), f"{sample_id} prepared size")
        if not gt.is_symlink() or not mask.is_symlink():
            raise ValueError(f"{sample_id} GT/mask must be preserved source symlinks")
        require_equal(record.get("canvas_size"), [720, 480], f"{sample_id} canvas size")
        require_equal(record.get("content_crop"), [0, 37, 720, 442], f"{sample_id} crop")
        require_equal(record.get("output_size"), [960, 540], f"{sample_id} output size")
        require_equal(
            record.get("restored_prediction_sha256"),
            sha256_file(prediction),
            f"{sample_id} restored hash",
        )
    return {"count": EXPECTED_COUNT, "prepare_audit_sha256": sha256_file(audit_path)}


def audit_evaluation(run_root: Path) -> dict[str, Any]:
    output = run_root / "eval_outputs" / "all8_full_frame"
    summary_path = output / "summary.json"
    pairwise_path = output / "pairwise_metrics.csv"
    summary = load_json(summary_path)
    metadata = summary.get("metadata")
    metrics = summary.get("summary")
    pairs = summary.get("pairs")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict) or not isinstance(pairs, list):
        raise ValueError("evaluation summary has invalid top-level structure")
    require_equal(metadata.get("pair_count"), EXPECTED_COUNT, "eval pair count")
    require_equal(metadata.get("selected_metrics"), METRICS, "eval metrics")
    require_equal(metadata.get("device"), "cuda:0", "eval device")
    require_equal(metadata.get("spatial_protocol"), "full_frame", "eval spatial protocol")
    require_equal(metadata.get("protocol_manifest"), None, "eval protocol manifest")
    require_equal(metadata.get("input_size_counts"), {"960x540": EXPECTED_COUNT}, "eval input sizes")
    require_equal(metadata.get("config", {}).get("crop_border"), 0, "eval crop border")
    require_equal(metadata.get("config", {}).get("fid_batch_size"), 50, "FID batch size")
    require_equal(metadata.get("config", {}).get("cmmd_batch_size"), 32, "CMMD batch size")
    require_equal(metadata.get("evaluator_sha256"), EXPECTED_EVALUATOR_HASHES, "evaluator hashes")
    require_equal(set(metrics), SUMMARY_KEYS, "eval summary keys")
    values: dict[str, float] = {}
    for name in sorted(SUMMARY_KEYS):
        item = metrics[name]
        require_equal(item.get("status"), "ok", f"{name} status")
        value = item.get("value") if name in {"FID", "CMMD"} else item.get("mean")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} is not finite: {value!r}")
        values[name] = float(value)
        if name not in {"FID", "CMMD"}:
            require_equal(item.get("count"), EXPECTED_COUNT, f"{name} count")
    require_equal(len(pairs), EXPECTED_COUNT, "eval pairs")
    require_equal([pair.get("index") for pair in pairs], list(range(EXPECTED_COUNT)), "eval pair indices")
    with pairwise_path.open("r", encoding="utf-8", newline="") as handle:
        pairwise_rows = list(csv.DictReader(handle))
    require_equal(len(pairwise_rows), EXPECTED_COUNT, "pairwise CSV row count")
    require_equal(
        [row.get("key") for row in pairwise_rows],
        [str(index) for index in range(EXPECTED_COUNT)],
        "pairwise CSV keys",
    )
    return {
        "count": EXPECTED_COUNT,
        "metrics": values,
        "summary_sha256": sha256_file(summary_path),
        "pairwise_sha256": sha256_file(pairwise_path),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.run_root.is_dir():
        raise FileNotFoundError(args.run_root)
    result = {
        "schema_version": 1,
        "status": "passed",
        "run_id": RUN_ID,
        "stage": args.stage,
    }
    if args.stage == "inference":
        result.update(audit_inference(args.run_root))
    elif args.stage == "prepared":
        result.update(audit_prepared(args.run_root))
    else:
        result.update(audit_evaluation(args.run_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
