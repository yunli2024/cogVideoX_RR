#!/usr/bin/env python3
"""Export CogKit LoRA training state from a DCP checkpoint.

The exporter is intentionally independent of training and inference. It uses
PyTorch's public DCP conversion utility, retains only ``app.model`` (the PEFT
state saved by CogKit's AppState), and writes the safetensors filename consumed
by CogKit's LoRA loader.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


ADAPTER_WEIGHT_NAME = "adapter_model.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a complete CogKit DCP LoRA checkpoint into a CogKit-loadable adapter."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--expected-tensor-count", type=int)
    parser.add_argument("--expected-numel", type=int)
    parser.add_argument(
        "--provenance-file",
        action="append",
        default=[],
        type=Path,
        help="File to hash into export_manifest.json; may be repeated.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(args: list[str]) -> str:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkpoint_step(checkpoint_dir: Path) -> int | None:
    prefix = "checkpoint-"
    if not checkpoint_dir.name.startswith(prefix):
        return None
    try:
        return int(checkpoint_dir.name[len(prefix) :])
    except ValueError:
        return None


def inspect_source(checkpoint_dir: Path, expected_shards: int | None) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    metadata_path = checkpoint_dir / ".metadata"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty DCP metadata: {metadata_path}")

    shards = sorted(checkpoint_dir.glob("*.distcp"))
    if not shards:
        raise RuntimeError(f"No DCP shards found in: {checkpoint_dir}")
    if expected_shards is not None and len(shards) != expected_shards:
        raise RuntimeError(
            f"Expected {expected_shards} DCP shards, found {len(shards)} in {checkpoint_dir}"
        )
    empty = [str(path) for path in shards if path.stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Empty DCP shards: {empty}")

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": checkpoint_step(checkpoint_dir),
        "metadata": {
            "name": metadata_path.name,
            "size": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
        },
        "shards": [
            {"name": path.name, "size": path.stat().st_size} for path in shards
        ],
    }


def cogkit_lora_config(code_root: Path) -> dict[str, Any]:
    source_root = code_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"CogKit source root does not exist: {source_root}")
    sys.path.insert(0, str(source_root))
    from cogkit.utils.lora import _get_lora_config  # noqa: PLC0415

    config = _get_lora_config()
    targets = config.target_modules
    if isinstance(targets, set):
        targets = sorted(targets)
    elif isinstance(targets, tuple):
        targets = list(targets)
    return {
        "r": int(config.r),
        "lora_alpha": int(config.lora_alpha),
        "target_modules": targets,
        "adapter_weight_name": ADAPTER_WEIGHT_NAME,
    }


def extract_lora_state(converted_state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(converted_state, dict):
        raise RuntimeError(f"Converted DCP root must be a dict, got {type(converted_state)!r}")
    app_state = converted_state.get("app")
    if not isinstance(app_state, dict):
        raise RuntimeError(
            f"Converted DCP must contain dict key 'app'; keys={sorted(converted_state)}"
        )
    model_state = app_state.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise RuntimeError(
            f"Converted DCP app state must contain non-empty dict key 'model'; "
            f"keys={sorted(app_state)}"
        )

    non_tensors = [key for key, value in model_state.items() if not torch.is_tensor(value)]
    if non_tensors:
        raise RuntimeError(f"Non-tensor entries found in app.model: {non_tensors[:8]}")
    non_lora = [
        key
        for key in model_state
        if ".lora_A." not in key and ".lora_B." not in key
    ]
    if non_lora:
        raise RuntimeError(f"Non-LoRA entries found in app.model: {non_lora[:8]}")

    return {
        key: value.detach().cpu().contiguous()
        for key, value in sorted(model_state.items())
    }


def tensor_summary(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    nonfinite = [
        key
        for key, value in state.items()
        if (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite:
        raise RuntimeError(f"Non-finite LoRA tensors: {nonfinite[:8]}")

    key_signature = hashlib.sha256()
    dtype_counts: Counter[str] = Counter()
    total_numel = 0
    for key, value in state.items():
        dtype_counts[str(value.dtype)] += 1
        total_numel += value.numel()
        key_signature.update(key.encode("utf-8"))
        key_signature.update(str(tuple(value.shape)).encode("ascii"))
        key_signature.update(str(value.dtype).encode("ascii"))

    return {
        "tensor_count": len(state),
        "total_numel": total_numel,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "key_shape_dtype_sha256": key_signature.hexdigest(),
        "all_finite": True,
    }


def require_expected(
    summary: dict[str, Any], expected_tensor_count: int | None, expected_numel: int | None
) -> None:
    if expected_tensor_count is not None and summary["tensor_count"] != expected_tensor_count:
        raise RuntimeError(
            f"Expected {expected_tensor_count} LoRA tensors, got {summary['tensor_count']}"
        )
    if expected_numel is not None and summary["total_numel"] != expected_numel:
        raise RuntimeError(
            f"Expected {expected_numel} LoRA elements, got {summary['total_numel']}"
        )


def git_provenance(code_root: Path) -> dict[str, Any]:
    return {
        "head": command_output(["git", "-C", str(code_root), "rev-parse", "HEAD"]),
        "status_short": command_output(["git", "-C", str(code_root), "status", "--short"]).splitlines(),
    }


def file_provenance(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Provenance file does not exist: {resolved}")
        result.append(
            {
                "path": str(resolved),
                "size": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    return result


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    code_root = args.code_root.resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = output_dir.parent / f".{output_dir.name}.partial-{os.getpid()}"
    if partial_dir.exists():
        raise FileExistsError(f"Refusing to reuse partial output: {partial_dir}")
    partial_dir.mkdir()

    temporary_state = partial_dir / "converted_training_state.pt"
    adapter_path = partial_dir / ADAPTER_WEIGHT_NAME
    manifest_path = partial_dir / "export_manifest.json"

    try:
        source = inspect_source(checkpoint_dir, args.expected_shards)
        lora_config = cogkit_lora_config(code_root)
        provenance = file_provenance([Path(__file__), *args.provenance_file])

        dcp_to_torch_save(checkpoint_dir, temporary_state)
        converted_state = torch.load(
            temporary_state,
            map_location="cpu",
            weights_only=False,
        )
        lora_state = extract_lora_state(converted_state)

        app_state = converted_state["app"]
        if "optim" not in app_state:
            raise RuntimeError("Converted DCP app state is missing expected optimizer state")
        del app_state["optim"]
        del converted_state
        gc.collect()

        before_save = tensor_summary(lora_state)
        require_expected(before_save, args.expected_tensor_count, args.expected_numel)

        save_file(lora_state, adapter_path, metadata={"format": "pt"})
        reloaded = load_file(adapter_path, device="cpu")
        after_reload = tensor_summary(reloaded)
        require_expected(after_reload, args.expected_tensor_count, args.expected_numel)
        if before_save != after_reload:
            raise RuntimeError(
                "Safetensors reload changed key/shape/dtype/count summary: "
                f"before={before_save}, after={after_reload}"
            )

        manifest = {
            "schema_version": 1,
            "status": "PASSED",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "output": {
                "directory": str(output_dir),
                "adapter_weight_name": ADAPTER_WEIGHT_NAME,
                "adapter_size": adapter_path.stat().st_size,
                "adapter_sha256": sha256_file(adapter_path),
            },
            "lora_config": lora_config,
            "tensor_validation": after_reload,
            "expected": {
                "shard_count": args.expected_shards,
                "tensor_count": args.expected_tensor_count,
                "total_numel": args.expected_numel,
            },
            "code": {
                "root": str(code_root),
                "git": git_provenance(code_root),
                "files": provenance,
            },
            "conversion": {
                "api": "torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
                "torch_version": torch.__version__,
                "device": "cpu",
                "temporary_file_removed": True,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        temporary_state.unlink()
        partial_dir.rename(output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception:
        if temporary_state.is_file():
            temporary_state.unlink()
        print(f"Export failed; partial artifacts, if any, remain at: {partial_dir}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
