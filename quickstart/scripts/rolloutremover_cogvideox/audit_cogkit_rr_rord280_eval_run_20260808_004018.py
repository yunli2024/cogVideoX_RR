#!/usr/bin/env python3
"""Run-scoped prepared/evaluation audit for the frozen 8-shard CogKit run."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_AUDIT = Path(__file__).with_name("audit_cogkit_rr_rord280_full_20260807.py")
RUN_ID = "run-20260808-004018-149f154b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepared", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_base_audit() -> Any:
    spec = importlib.util.spec_from_file_location("cogkit_rr_rord280_base_audit", BASE_AUDIT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base audit: {BASE_AUDIT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_ID = RUN_ID
    return module


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.run_root.is_dir():
        raise FileNotFoundError(args.run_root)
    if args.run_root.name != RUN_ID:
        raise ValueError(f"run id mismatch: {args.run_root.name!r} != {RUN_ID!r}")

    module = load_base_audit()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "run_id": RUN_ID,
        "stage": args.stage,
    }
    if args.stage == "prepared":
        result.update(module.audit_prepared(args.run_root))
    else:
        result.update(module.audit_evaluation(args.run_root))
        summary = module.load_json(
            args.run_root / "eval_outputs" / "all8_full_frame" / "summary.json"
        )
        metadata = summary.get("metadata")
        pairs = summary.get("pairs")
        if not isinstance(metadata, dict) or not isinstance(pairs, list):
            raise ValueError("evaluation summary metadata/pairs structure is invalid")
        config = metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError("evaluation summary config is invalid")
        module.require_equal(config.get("remove_crop"), True, "ReMOVE crop enabled")
        module.require_equal(
            config.get("remove_crop_strategy"),
            "official_bbox16_v1",
            "ReMOVE crop strategy",
        )
        module.require_equal(
            [pair.get("ReMOVE_crop_strategy") for pair in pairs],
            ["official_bbox16_v1"] * module.EXPECTED_COUNT,
            "pairwise ReMOVE crop strategies",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
