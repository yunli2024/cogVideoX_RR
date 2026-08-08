#!/usr/bin/env python3
"""One-time load gate for adapters produced by export_dcp_lora.py."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import CogVideoXTransformer3DModel
from peft import get_peft_model_state_dict
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-tensor-count", type=int)
    parser.add_argument("--expected-numel", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    adapter_dir = args.adapter_dir.resolve()
    code_root = args.code_root.resolve()
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    sys.path.insert(0, str(code_root / "src"))
    from cogkit.utils.lora import inject_lora  # noqa: PLC0415

    started = time.time()
    expected = load_file(adapter_path, device="cpu")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    inject_lora(transformer, adapter_dir)
    actual = get_peft_model_state_dict(transformer)

    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys != actual_keys:
        raise RuntimeError(
            f"LoRA key mismatch: missing={sorted(expected_keys - actual_keys)[:8]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
        )

    total_numel = sum(tensor.numel() for tensor in actual.values())
    if args.expected_tensor_count is not None and len(actual) != args.expected_tensor_count:
        raise RuntimeError(
            f"Expected {args.expected_tensor_count} tensors, loaded {len(actual)}"
        )
    if args.expected_numel is not None and total_numel != args.expected_numel:
        raise RuntimeError(f"Expected {args.expected_numel} elements, loaded {total_numel}")

    mismatched = []
    for key in sorted(expected):
        expected_tensor = expected[key]
        actual_tensor = actual[key].detach().cpu()
        if expected_tensor.shape != actual_tensor.shape or not torch.equal(
            expected_tensor, actual_tensor.to(expected_tensor.dtype)
        ):
            mismatched.append(key)
            if len(mismatched) == 8:
                break
    if mismatched:
        raise RuntimeError(f"Loaded LoRA tensor values differ: {mismatched}")

    result = {
        "status": "PASSED",
        "model_path": str(model_path),
        "adapter_dir": str(adapter_dir),
        "tensor_count": len(actual),
        "total_numel": total_numel,
        "adapter_dtypes_after_load": sorted({str(tensor.dtype) for tensor in actual.values()}),
        "exact_values_after_dtype_roundtrip": True,
        "elapsed_seconds": round(time.time() - started, 3),
        "device": "cpu",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
