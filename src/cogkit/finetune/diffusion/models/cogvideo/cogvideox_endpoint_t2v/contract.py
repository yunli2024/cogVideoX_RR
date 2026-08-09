"""Direct-target endpoint contract for CogVideoX object removal."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class EndpointContract:
    """Three-frame mask/source-conditioned direct-target topology."""

    name: str = "mask_source_direct_target_3f_v1"
    mask_sam_probability: float = 0.5
    loss_normalization: str = "active_weight_mean"
    loss_outlier_abs_threshold: float | None = 50.0

    def __post_init__(self) -> None:
        if self.name != "mask_source_direct_target_3f_v1":
            raise ValueError(f"unknown endpoint contract: {self.name}")
        if not 0 <= self.mask_sam_probability <= 1:
            raise ValueError("mask_sam_probability must lie in [0, 1]")
        if self.loss_normalization not in {"full_tensor_mean", "active_weight_mean"}:
            raise ValueError("unsupported loss normalization")
        if self.loss_outlier_abs_threshold is not None and self.loss_outlier_abs_threshold <= 0:
            raise ValueError("loss_outlier_abs_threshold must be positive or None")

    @property
    def total_frames(self) -> int:
        return 3

    @property
    def clean_condition_indices(self) -> tuple[int, int]:
        return (0, 1)

    @property
    def supervised_frame_indices(self) -> tuple[int]:
        return (2,)

    @property
    def frame_roles(self) -> tuple[str, str, str]:
        return ("mask_condition", "source_condition", "final_target")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "total_frames": self.total_frames,
                "frame_roles": list(self.frame_roles),
                "clean_condition_indices": list(self.clean_condition_indices),
                "supervised_frame_indices": list(self.supervised_frame_indices),
                "constructs_intermediate_rollout": False,
            }
        )
        return payload


@dataclass(frozen=True)
class EndpointSampleProvenance:
    sample_id: str
    condition_mask_kind: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _stable_u63(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def select_condition_mask_kind(
    contract: EndpointContract, *, endpoint_seed: int, sample_id: str, epoch: int
) -> str:
    # Keep the exact RR condition-selection namespace so matched samples receive
    # the same mask kind in the same epoch.
    draw = _stable_u63(
        "condition_mask_v1", int(endpoint_seed), sample_id, int(epoch)
    ) / float(1 << 63)
    return "mask_sam" if draw < contract.mask_sam_probability else "mask_check"


def construct_cached_endpoint(
    entries: Sequence[dict[str, torch.Tensor]],
    sample_ids: Sequence[str],
    contract: EndpointContract,
    *,
    endpoint_seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[EndpointSampleProvenance, ...]]:
    if len(entries) != len(sample_ids):
        raise ValueError("entries and sample_ids must have identical length")
    if not entries:
        raise ValueError("cannot construct an empty endpoint batch")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    frames_per_sample = []
    provenance = []
    for entry, sample_id in zip(entries, sample_ids):
        if not isinstance(entry, dict):
            raise TypeError(f"cache entry {sample_id!r} must be a tensor dictionary")
        missing = {"source", "gt", "mask_sam", "mask_check"}.difference(entry)
        if missing:
            raise ValueError(f"cache entry {sample_id} missing {sorted(missing)}")
        condition_kind = select_condition_mask_kind(
            contract,
            endpoint_seed=endpoint_seed,
            sample_id=sample_id,
            epoch=epoch,
        )
        tensors = (entry[condition_kind], entry["source"], entry["gt"])
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise TypeError("cached endpoint latents must be floating torch.Tensor values")
            if tensor.ndim != 3:
                raise ValueError(f"cached endpoint latents must be [C,h,w], got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError("cached endpoint latents contain NaN or infinity")
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("mask, source and target latents must share one shape")
        frames_per_sample.append(torch.stack(tensors, dim=0))
        provenance.append(EndpointSampleProvenance(sample_id, condition_kind))

    endpoint = torch.stack(frames_per_sample, dim=0)
    frame_weights = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    return endpoint, frame_weights, tuple(provenance)


def clamp_endpoint_conditions(
    noisy: torch.Tensor, clean: torch.Tensor, contract: EndpointContract
) -> torch.Tensor:
    if noisy.shape != clean.shape or clean.ndim != 5:
        raise ValueError("noisy and clean must share [B,F,C,h,w] shape")
    if clean.shape[1] != contract.total_frames:
        raise ValueError("tensor frame count disagrees with endpoint contract")
    clamped = noisy.clone()
    clean_indices = list(contract.clean_condition_indices)
    clamped[:, clean_indices] = clean[:, clean_indices]
    return clamped


def endpoint_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestep_weights: torch.Tensor,
    frame_weights: torch.Tensor,
    contract: EndpointContract,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.ndim != 5:
        raise ValueError("prediction and target must share [B,F,C,h,w] shape")
    if target.shape[1] != contract.total_frames:
        raise ValueError("tensor frame count disagrees with endpoint contract")
    if tuple(timestep_weights.shape) != (target.shape[0],):
        raise ValueError("timestep_weights must have shape [B]")

    difference = prediction.float() - target.float()
    weights = timestep_weights.to(device=difference.device, dtype=difference.dtype)
    if not torch.isfinite(weights).all().item() or (weights < 0).any().item():
        raise ValueError("timestep_weights must be finite and non-negative")
    while weights.ndim < target.ndim:
        weights = weights.unsqueeze(-1)
    error = weights * difference.square()
    threshold = contract.loss_outlier_abs_threshold
    if threshold is not None:
        error = error * (difference.abs() <= threshold).to(error.dtype)

    frame_weights = frame_weights.to(device=error.device, dtype=error.dtype)
    if frame_weights.ndim == 1:
        frame_weights = frame_weights.unsqueeze(0).expand(target.shape[0], -1)
    if tuple(frame_weights.shape) != (target.shape[0], contract.total_frames):
        raise ValueError("frame_weights must be [F] or [B,F]")
    if (frame_weights[:, list(contract.clean_condition_indices)] != 0).any().item():
        raise ValueError("clean condition frames must have zero loss weight")
    if (frame_weights[:, contract.supervised_frame_indices[0]] <= 0).any().item():
        raise ValueError("final target frame must have positive loss weight")

    masked = error * frame_weights[:, :, None, None, None]
    if contract.loss_normalization == "full_tensor_mean":
        return masked.mean()
    active = frame_weights.sum() * target.shape[2] * target.shape[3] * target.shape[4]
    return masked.sum() / active.clamp(min=1)
