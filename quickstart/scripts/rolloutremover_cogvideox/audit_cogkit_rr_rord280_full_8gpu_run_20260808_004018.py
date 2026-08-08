#!/usr/bin/env python3
"""Fail-closed inference audit for one frozen 8x35 CogKit RR RORD-280 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


RUN_ID = "run-20260808-004018-149f154b"
EXPECTED_COUNT = 280
SHARD_COUNT = 8
SHARD_SIZE = 35
EXPECTED_MANIFEST_SHA256 = "fda189ab9ce7e2ae85c02d58987181b089a3cf55caf3dcf81d33d164bff0306a"
EXPECTED_LORA_SHA256 = "db46680166757682b2d33edc7102bb0d33d8e17228ab0e4e6877453c301a0cb1"
EXPECTED_TRAINING_CONTRACT_SHA256 = "4064edfc0436a0003b5f2fbbf4a5da1c9044b50ed56021190f12daa8cf6dc9f1"
EXPECTED_SCHEDULER_SHA256 = "247ecd6635dae7bf889a7ec69ba951c44d746cb4edef13b9c3b1b16bfeeedba5"
SHARD_SHA256 = {
    0: "5785d03146405e44661f6dac4191727ebeaf7154163a4503b0e445e6a84c78ca",
    1: "40a4b2356c9d8df35d1aea53fda14595d9cee40f76544b3febe64b09434bed42",
    2: "43c1035a6cc3eeb38d5173440a36bb8b689136ed5b7b3f4c443ea3c10316671a",
    3: "0c470924366b1d029b24b0f14a2033442accb18e499569ebe9f3430e97504922",
    4: "b7d6a51eeeb0bc2ff8123b81e47de2d1a56b808199e560808a98b40a6e99eee2",
    5: "220e7ffda0aaa331cbaa32079437a280184e563dacef91f37d7d756df8fe6584",
    6: "8309957c21ba7a7200a674904631c7c43d5bbd0db4c8d4107873de2d92cb61b3",
    7: "9525a046febce6adcbff70ded028362e4b9ed559d48d8fa9b72d663ef306cf1c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
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
    start = shard * SHARD_SIZE
    return [f"{index:08d}" for index in range(start, start + SHARD_SIZE)]


def expected_sample_seed(sample_id: str) -> int:
    payload = f"cogvideox_rr_inference_v1\0{42}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def audit_inference(run_root: Path) -> dict[str, Any]:
    require_equal(run_root.name, RUN_ID, "run id")
    inference_root = run_root / "inference"
    all_ids: list[str] = []
    shard_results: list[dict[str, Any]] = []
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
    exact_scheduler = {
        "class_name": "CogVideoXDPMScheduler",
        "config_sha256": EXPECTED_SCHEDULER_SHA256,
        "kind": "dpm",
        "num_train_timesteps": 1000,
        "prediction_type": "v_prediction",
        "timestep_spacing": "trailing",
    }

    for shard in range(SHARD_COUNT):
        expected_ids = expected_shard_ids(shard)
        shard_root = inference_root / f"shard{shard}"
        contract_path = shard_root / "inference_contract.json"
        manifest_path = shard_root / "inference_manifest.jsonl"
        log_path = run_root / "logs" / f"shard{shard}.log"
        for path in (contract_path, manifest_path, log_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        contract = load_json(contract_path)
        for key, expected in exact_contract.items():
            require_equal(contract.get(key), expected, f"shard{shard} contract.{key}")
        require_equal(contract.get("sample_ids"), expected_ids, f"shard{shard} sample ids")
        sample_ids_file = contract.get("sample_ids_file")
        if not isinstance(sample_ids_file, dict):
            raise ValueError(f"shard{shard} sample_ids_file is not an object")
        require_equal(sample_ids_file.get("sha256"), SHARD_SHA256[shard], f"shard{shard} sample-id hash")
        scheduler = contract.get("scheduler")
        if not isinstance(scheduler, dict):
            raise ValueError(f"shard{shard} scheduler is not an object")
        for key, expected in exact_scheduler.items():
            require_equal(scheduler.get(key), expected, f"shard{shard} scheduler.{key}")

        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        require_equal(len(rows), SHARD_SIZE, f"shard{shard} manifest count")
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
                "sample_seed": expected_sample_seed(sample_id),
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
        if '"status": "completed"' not in log_text or f'"count": {SHARD_SIZE}' not in log_text:
            raise ValueError(f"shard{shard} log lacks terminal completion record")
        all_ids.extend(expected_ids)
        shard_results.append(
            {
                "shard": shard,
                "count": SHARD_SIZE,
                "contract_sha256": sha256_file(contract_path),
                "manifest_sha256": sha256_file(manifest_path),
                "log_sha256": sha256_file(log_path),
            }
        )

    require_equal(all_ids, [f"{index:08d}" for index in range(EXPECTED_COUNT)], "full sample ids")
    prediction_paths = list(inference_root.glob("shard*/*/prediction.png"))
    require_equal(len(prediction_paths), EXPECTED_COUNT, "full prediction count")
    require_equal(len({path.parent.name for path in prediction_paths}), EXPECTED_COUNT, "unique prediction ids")
    return {"count": EXPECTED_COUNT, "shards": shard_results}


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.run_root.is_dir():
        raise FileNotFoundError(args.run_root)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "run_id": RUN_ID,
        "stage": "inference",
    }
    result.update(audit_inference(args.run_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
