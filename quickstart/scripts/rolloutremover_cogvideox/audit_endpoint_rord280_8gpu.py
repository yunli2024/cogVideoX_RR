#!/usr/bin/env python3
"""Fail-closed audit for one EndpointRemover 8x35 RORD-280 inference run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


EXPECTED_COUNT = 280
SHARD_COUNT = 8
SHARD_SIZE = 35
EXPECTED_MANIFEST_SHA256 = "fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a"
EXPECTED_LORA_SHA256 = "324bc5fd791d2a874c3bd482643a8fd2b2f32a7c0a29e691e77cd880edcac135"
EXPECTED_TRAINING_CONTRACT_SHA256 = "c676fe9b741245b70d26d95d46a24c6363ec49be36a9c928e8d488f220c709ee"
EXPECTED_SCHEDULER_SHA256 = "247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
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


def expected_ids(shard: int) -> list[str]:
    start = shard * SHARD_SIZE
    return [f"{index:08d}" for index in range(start, start + SHARD_SIZE)]


def expected_seed(sample_id: str) -> int:
    payload = f"cogvideox_endpoint_inference_v1\0{42}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def main() -> None:
    args = parse_args()
    if args.run_root.name != args.run_id:
        raise ValueError("run root and run id disagree")
    if args.output.exists():
        raise FileExistsError(args.output)

    all_ids: list[str] = []
    shard_results: list[dict[str, Any]] = []
    topology = {
        "total_frames": 3,
        "frame_roles": ["mask_condition", "source_condition", "final_target"],
        "clean_condition_indices": [0, 1],
        "supervised_frame_indices": [2],
        "constructs_intermediate_rollout": False,
    }
    scheduler = {
        "kind": "dpm",
        "class_name": "CogVideoXDPMScheduler",
        "config_sha256": EXPECTED_SCHEDULER_SHA256,
        "prediction_type": "v_prediction",
        "timestep_spacing": "trailing",
        "num_train_timesteps": 1000,
    }

    for shard in range(SHARD_COUNT):
        ids = expected_ids(shard)
        root = args.run_root / "inference" / f"shard{shard}"
        contract_path = root / "inference_contract.json"
        manifest_path = root / "inference_manifest.jsonl"
        log_path = args.run_root / "logs" / f"shard{shard}.log"
        for path in (contract_path, manifest_path, log_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        contract = load_json(contract_path)
        for key, expected in {
            "schema_version": 2,
            "task": "CogVideoX5B-EndpointRemover-cached-inference",
            "inference_policy": "cogkit_native_condition_clamped_endpoint_v1",
            "seed_policy": "sha256_global_seed_plus_sample_id_endpoint_v1",
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
            "save_states": False,
            "pure_output": True,
            "paste_back": False,
            "output_blending": False,
            "gt_used_as_model_input": False,
        }.items():
            require_equal(contract.get(key), expected, f"shard{shard} contract.{key}")
        require_equal(contract.get("sample_ids"), ids, f"shard{shard} contract.sample_ids")
        for key, expected in topology.items():
            require_equal(contract.get("topology", {}).get(key), expected, f"shard{shard} topology.{key}")
        for key, expected in scheduler.items():
            require_equal(contract.get("scheduler", {}).get(key), expected, f"shard{shard} scheduler.{key}")

        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require_equal(len(rows), SHARD_SIZE, f"shard{shard} manifest count")
        require_equal([row.get("sample_id") for row in rows], ids, f"shard{shard} manifest ids")
        for row, sample_id in zip(rows, ids, strict=True):
            sample_dir = root / sample_id
            for filename in ("prediction.png", "metadata.json", "source.png", "gt.png", "mask_condition.png"):
                if not (sample_dir / filename).is_file():
                    raise FileNotFoundError(sample_dir / filename)
            if (sample_dir / "states").exists():
                raise ValueError(f"forbidden state artifacts: {sample_dir}")
            for filename in ("prediction.png", "source.png", "gt.png", "mask_condition.png"):
                with Image.open(sample_dir / filename) as image:
                    require_equal(image.size, (720, 480), f"{sample_id}/{filename} size")
                    require_equal(image.mode, "RGB", f"{sample_id}/{filename} mode")
                    image.verify()
            for key, expected in {
                "schema_version": 2,
                "sample_id": sample_id,
                "sample_seed": expected_seed(sample_id),
                "generated_states": [],
                "pure_output": True,
                "paste_back": False,
                "output_blending": False,
                "gt_used_as_model_input": False,
            }.items():
                require_equal(row.get(key), expected, f"{sample_id} manifest.{key}")
            require_equal(load_json(sample_dir / "metadata.json"), row, f"{sample_id} metadata equality")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if '"status": "completed"' not in log_text or f'"count": {SHARD_SIZE}' not in log_text:
            raise ValueError(f"shard{shard} log lacks terminal completion")
        for forbidden in ("Traceback", "CUDA out of memory", "nan", "NCCL error"):
            if forbidden.lower() in log_text.lower():
                raise ValueError(f"shard{shard} log contains {forbidden!r}")
        all_ids.extend(ids)
        shard_results.append({
            "shard": shard,
            "count": SHARD_SIZE,
            "contract_sha256": sha256_file(contract_path),
            "manifest_sha256": sha256_file(manifest_path),
            "log_sha256": sha256_file(log_path),
        })

    require_equal(all_ids, [f"{index:08d}" for index in range(EXPECTED_COUNT)], "full ids")
    require_equal(len(list((args.run_root / "inference").glob("shard*/*/prediction.png"))), EXPECTED_COUNT, "prediction count")
    result = {
        "schema_version": 1,
        "status": "passed",
        "run_id": args.run_id,
        "stage": "endpoint-inference",
        "count": EXPECTED_COUNT,
        "shards": shard_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
