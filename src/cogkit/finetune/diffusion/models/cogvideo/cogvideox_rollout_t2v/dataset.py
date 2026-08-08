"""Fail-closed loader for precomputed CogVideoX RolloutRemover keyframes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


PREPROCESSING_CONVENTION = "uint8_0_255_to_float_0_1_to_vae_minus1_plus1_v1"
LATENT_CONVENTION = "cogvideox_posterior_times_scaling_factor_v1"
TEMPORAL_VAE_POLICY = "independent_single_frame_encode_decode_v1"
ENCODING_ROLES = {"source", "gt", "mask_sam", "mask_check"}
MANIFEST_ASSET_FIELDS = ("source", "gt", "mask_sam", "mask_check")


@dataclass(frozen=True)
class RolloutCacheRecord:
    sample_id: str
    tensor_path: Path
    metadata_path: Path


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {context} {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} {path} must contain one JSON object")
    return payload


def _resolve_manifest_asset(
    dataset_root: Path, value: object, *, field: str, line_number: int
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest line {line_number}: {field} must be a non-empty path")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (dataset_root / candidate).resolve()
    try:
        resolved.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(
            f"manifest line {line_number}: {field} escapes dataset root: {value!r}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest line {line_number}: missing {field}: {resolved}")
    return resolved


class CogVideoXRolloutCacheDataset(Dataset):
    REQUIRED_TENSORS = {
        "source", "gt", "mask_sam", "mask_check", "mask_sam_binary", "mask_check_binary"
    }

    def __init__(
        self,
        dataset_root: Path,
        manifest: Path,
        cache_dir: Path,
        expected_model_id: str,
        expected_scaling_factor: float,
        expected_posterior_mode: str,
        expected_pixel_height: int,
        expected_pixel_width: int,
        expected_latent_shape: tuple[int, int, int],
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.manifest = Path(manifest).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(f"dataset root not found: {self.dataset_root}")
        if not self.manifest.is_file():
            raise FileNotFoundError(f"rollout manifest not found: {self.manifest}")
        if not self.cache_dir.is_dir():
            raise NotADirectoryError(f"rollout cache directory not found: {self.cache_dir}")

        if len(expected_latent_shape) != 3 or any(int(value) <= 0 for value in expected_latent_shape):
            raise ValueError("expected_latent_shape must be a positive [C,h,w] tuple")

        records: list[RolloutCacheRecord] = []
        seen_ids: set[str] = set()
        with self.manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"manifest line {line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"manifest line {line_number} must be a JSON object")
                sample_id = payload.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"manifest line {line_number} has no sample_id")
                if (
                    Path(sample_id).name != sample_id
                    or sample_id in {".", ".."}
                    or "/" in sample_id
                    or "\\" in sample_id
                ):
                    raise ValueError(
                        f"manifest line {line_number}: unsafe sample_id {sample_id!r}"
                    )
                if sample_id in seen_ids:
                    raise ValueError(f"manifest line {line_number}: duplicate sample_id {sample_id!r}")
                seen_ids.add(sample_id)
                for field in MANIFEST_ASSET_FIELDS:
                    if field not in payload:
                        raise ValueError(f"manifest line {line_number}: missing field {field}")
                    _resolve_manifest_asset(
                        self.dataset_root,
                        payload[field],
                        field=field,
                        line_number=line_number,
                    )
                tensor_path = self.cache_dir / f"{sample_id}.pt"
                metadata_path = self.cache_dir / f"{sample_id}.json"
                if not tensor_path.is_file() or not metadata_path.is_file():
                    raise FileNotFoundError(f"missing cache pair for sample {sample_id}")
                metadata = _load_json_object(metadata_path, context="cache metadata")
                metadata_required = {
                    "sample_id",
                    "model_id",
                    "scaling_factor",
                    "posterior_mode",
                    "vae_seed_by_role",
                    "preprocessing_convention",
                    "latent_convention",
                    "temporal_vae_policy",
                    "source_shape",
                    "gt_shape",
                    "mask_sam_shape",
                    "mask_check_shape",
                    "pixel_height",
                    "pixel_width",
                }
                missing_metadata = metadata_required.difference(metadata)
                if missing_metadata:
                    raise ValueError(
                        f"cache metadata {sample_id} missing {sorted(missing_metadata)}"
                    )
                expected = {
                    "sample_id": sample_id,
                    "model_id": expected_model_id,
                    "posterior_mode": expected_posterior_mode,
                    "pixel_height": expected_pixel_height,
                    "pixel_width": expected_pixel_width,
                    "preprocessing_convention": PREPROCESSING_CONVENTION,
                    "latent_convention": LATENT_CONVENTION,
                    "temporal_vae_policy": TEMPORAL_VAE_POLICY,
                }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ValueError(
                            f"cache metadata mismatch for {sample_id}/{key}: "
                            f"found={metadata.get(key)!r}, expected={value!r}"
                        )
                if abs(float(metadata["scaling_factor"]) - float(expected_scaling_factor)) > 1e-12:
                    raise ValueError(
                        f"cache metadata mismatch for {sample_id}/scaling_factor: "
                        f"found={metadata['scaling_factor']!r}, expected={expected_scaling_factor!r}"
                    )
                if set(metadata["vae_seed_by_role"]) != ENCODING_ROLES:
                    raise ValueError(f"cache {sample_id} must record one VAE seed per image role")
                for role in ENCODING_ROLES:
                    shape_key = f"{role}_shape"
                    if metadata[shape_key] != list(expected_latent_shape):
                        raise ValueError(
                            f"cache metadata mismatch for {sample_id}/{shape_key}: "
                            f"found={metadata[shape_key]!r}, expected={list(expected_latent_shape)!r}"
                        )
                records.append(RolloutCacheRecord(sample_id, tensor_path, metadata_path))
        if not records:
            raise ValueError("rollout manifest contains no samples")
        self.records = tuple(records)
        self.expected_latent_shape = expected_latent_shape

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensors = torch.load(record.tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(tensors, dict):
            raise ValueError(f"cache tensor {record.tensor_path} must contain a dictionary")
        missing = self.REQUIRED_TENSORS.difference(tensors)
        if missing:
            raise ValueError(f"cache entry {record.sample_id} missing {sorted(missing)}")
        for role in ("source", "gt", "mask_sam", "mask_check"):
            if not isinstance(tensors[role], torch.Tensor):
                raise TypeError(f"cache {record.sample_id}/{role} must be a torch.Tensor")
            if tuple(tensors[role].shape) != self.expected_latent_shape:
                raise ValueError(
                    f"unexpected {record.sample_id}/{role} shape {tuple(tensors[role].shape)}"
                )
            if not tensors[role].is_floating_point() or not torch.isfinite(tensors[role]).all().item():
                raise ValueError(f"invalid floating latent in {record.sample_id}/{role}")
        mask_shape = (1, self.expected_latent_shape[-2], self.expected_latent_shape[-1])
        for role in ("mask_sam_binary", "mask_check_binary"):
            if not isinstance(tensors[role], torch.Tensor):
                raise TypeError(f"cache {record.sample_id}/{role} must be a torch.Tensor")
            if tuple(tensors[role].shape) != mask_shape:
                raise ValueError(
                    f"unexpected {record.sample_id}/{role} shape {tuple(tensors[role].shape)}"
                )
            if not torch.all((tensors[role] == 0) | (tensors[role] == 1)).item():
                raise ValueError(f"cache {record.sample_id}/{role} must contain only 0 and 1")
        return {"sample_id": record.sample_id, "tensors": tensors}
