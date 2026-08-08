"""WAN14B-RR-aligned trajectory, clamp, and loss contract for CogKit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class RolloutPreset:
    name: str
    corruption_frames: int
    restoration_frames: int
    interpolation_gamma: float
    mask_sam_probability: float
    loss_normalization: str = "full_tensor_mean"
    loss_outlier_abs_threshold: float | None = 50.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("preset name must be non-empty")
        if self.corruption_frames < 0 or self.restoration_frames < 0:
            raise ValueError("trajectory frame counts must be non-negative")
        if self.interpolation_gamma <= 0:
            raise ValueError("interpolation_gamma must be positive")
        if not 0 <= self.mask_sam_probability <= 1:
            raise ValueError("mask_sam_probability must lie in [0, 1]")
        if self.loss_normalization not in {"full_tensor_mean", "active_weight_mean"}:
            raise ValueError("unsupported loss normalization")
        if (
            self.loss_outlier_abs_threshold is not None
            and self.loss_outlier_abs_threshold <= 0
        ):
            raise ValueError("loss_outlier_abs_threshold must be positive or None")

    @property
    def mask_condition_index(self) -> int:
        return 0

    @property
    def mask_prediction_index(self) -> int:
        return 1

    @property
    def source_condition_index(self) -> int:
        return 2

    @property
    def clean_condition_indices(self) -> tuple[int, int]:
        return (self.mask_condition_index, self.source_condition_index)

    @property
    def corruption_indices(self) -> tuple[int, ...]:
        return tuple(range(3, 3 + self.corruption_frames))

    @property
    def noisy_anchor_index(self) -> int:
        return 3 + self.corruption_frames

    @property
    def final_gt_index(self) -> int:
        return self.noisy_anchor_index + self.restoration_frames + 1

    @property
    def total_frames(self) -> int:
        return self.final_gt_index + 1

    @property
    def restoration_indices(self) -> tuple[int, ...]:
        start = self.noisy_anchor_index + 1
        return tuple(range(start, start + self.restoration_frames))

    @property
    def supervised_frame_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.total_frames) if i not in self.clean_condition_indices)

    @property
    def frame_roles(self) -> tuple[str, ...]:
        roles = ["mask_condition", "mask_check_prediction", "source_condition"]
        roles.extend(f"corruption_{index}" for index in range(1, self.corruption_frames + 1))
        roles.append("noisy_anchor")
        roles.extend(f"restoration_{index}" for index in range(1, self.restoration_frames + 1))
        roles.append("final_gt")
        return tuple(roles)

    @property
    def corruption_alphas(self) -> tuple[float, ...]:
        denominator = self.corruption_frames + 1
        return tuple((i / denominator) ** self.interpolation_gamma for i in range(1, denominator))

    @property
    def restoration_alphas(self) -> tuple[float, ...]:
        denominator = self.restoration_frames + 1
        return tuple(
            1.0 - (1.0 - i / denominator) ** self.interpolation_gamma
            for i in range(1, denominator)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "total_frames": self.total_frames,
                "frame_roles": list(self.frame_roles),
                "clean_condition_indices": list(self.clean_condition_indices),
                "supervised_frame_indices": list(self.supervised_frame_indices),
                "corruption_alphas": list(self.corruption_alphas),
                "restoration_alphas": list(self.restoration_alphas),
            }
        )
        return payload


@dataclass(frozen=True)
class TrajectorySampleProvenance:
    sample_id: str
    condition_mask_kind: str
    anchor_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BASE_PRESET = RolloutPreset(
    name="wan14b_maskpred_c4r5_g1p2_14f",
    corruption_frames=4,
    restoration_frames=5,
    interpolation_gamma=1.2,
    mask_sam_probability=0.5,
)


def resolve_rollout_preset(name: str, **overrides: Any) -> RolloutPreset:
    if name != _BASE_PRESET.name:
        raise ValueError(f"unknown rollout preset: {name}")
    values = {
        field: getattr(_BASE_PRESET, field) if value is None else value
        for field, value in overrides.items()
    }
    threshold = values.get("loss_outlier_abs_threshold", _BASE_PRESET.loss_outlier_abs_threshold)
    if threshold == 0:
        threshold = None
    return RolloutPreset(
        name=name if all(value is None for value in overrides.values()) else f"{name}__yaml_override",
        corruption_frames=int(values.get("corruption_frames", _BASE_PRESET.corruption_frames)),
        restoration_frames=int(values.get("restoration_frames", _BASE_PRESET.restoration_frames)),
        interpolation_gamma=float(values.get("interpolation_gamma", _BASE_PRESET.interpolation_gamma)),
        mask_sam_probability=float(values.get("mask_sam_probability", _BASE_PRESET.mask_sam_probability)),
        loss_normalization=str(values.get("loss_normalization", _BASE_PRESET.loss_normalization)),
        loss_outlier_abs_threshold=None if threshold is None else float(threshold),
    )


def _stable_u63(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def anchor_seed_for_sample(rollout_seed: int, sample_id: str) -> int:
    return _stable_u63("rollout_anchor_v1", int(rollout_seed), sample_id)


def select_condition_mask_kind(
    preset: RolloutPreset, *, rollout_seed: int, sample_id: str, epoch: int
) -> str:
    draw = _stable_u63(
        "condition_mask_v1", int(rollout_seed), sample_id, int(epoch)
    ) / float(1 << 63)
    return "mask_sam" if draw < preset.mask_sam_probability else "mask_check"


def construct_cached_rollout(
    entries: Sequence[dict[str, torch.Tensor]],
    sample_ids: Sequence[str],
    preset: RolloutPreset,
    rollout_seed: int,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[TrajectorySampleProvenance, ...]]:
    if len(entries) != len(sample_ids):
        raise ValueError("entries and sample_ids must have identical length")
    if not entries:
        raise ValueError("cannot construct an empty rollout batch")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    required = {
        "source",
        "gt",
        "mask_sam",
        "mask_check",
        "mask_sam_binary",
        "mask_check_binary",
    }
    sources, targets, mask_checks, conditions, binaries = [], [], [], [], []
    condition_kinds = []
    for entry, sample_id in zip(entries, sample_ids):
        if not isinstance(entry, dict):
            raise TypeError(f"cache entry {sample_id!r} must be a tensor dictionary")
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"cache entry {sample_id} missing {sorted(missing)}")
        kind = select_condition_mask_kind(
            preset,
            rollout_seed=rollout_seed,
            sample_id=sample_id,
            epoch=epoch,
        )
        sources.append(entry["source"])
        targets.append(entry["gt"])
        mask_checks.append(entry["mask_check"])
        conditions.append(entry[kind])
        binaries.append(entry["mask_check_binary"])
        condition_kinds.append(kind)

    for tensor in sources + targets + mask_checks + conditions:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError("cached keyframe latents must be floating torch.Tensor values")
        if tensor.ndim != 3:
            raise ValueError(f"cached keyframe latents must be [C,h,w], got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError("cached keyframe latents contain NaN or infinity")

    source = torch.stack(sources)
    gt = torch.stack(targets)
    mask_check = torch.stack(mask_checks)
    mask_condition = torch.stack(conditions)
    mask_binary = torch.stack(binaries)
    if len({tuple(x.shape) for x in (source, gt, mask_check, mask_condition)}) != 1:
        raise ValueError("all cached keyframe batches must share one shape")
    if mask_binary.shape != (source.shape[0], 1, source.shape[-2], source.shape[-1]):
        raise ValueError("mask_check_binary must be [B,1,h,w]")
    if not torch.all((mask_binary == 0) | (mask_binary == 1)).item():
        raise ValueError("mask_check_binary must contain only 0 and 1")

    anchors = []
    provenance = []
    generator_device = source.device if source.device.type != "mps" else torch.device("cpu")
    for index, (sample_id, condition_kind) in enumerate(zip(sample_ids, condition_kinds)):
        anchor_seed = anchor_seed_for_sample(rollout_seed, sample_id)
        generator = torch.Generator(device=generator_device).manual_seed(anchor_seed)
        noise = torch.randn(
            source[index : index + 1].shape,
            generator=generator,
            device=generator_device,
            dtype=source.dtype,
        )
        if noise.device != source.device:
            noise = noise.to(source.device)
        anchors.append(
            torch.where(mask_binary[index : index + 1] > 0.5, noise, source[index : index + 1])
        )
        provenance.append(
            TrajectorySampleProvenance(
                sample_id=sample_id,
                condition_mask_kind=condition_kind,
                anchor_seed=anchor_seed,
            )
        )
    anchor = torch.cat(anchors)

    frames = [mask_condition, mask_check, source]
    frames.extend((1 - alpha) * source + alpha * anchor for alpha in preset.corruption_alphas)
    frames.append(anchor)
    frames.extend((1 - alpha) * anchor + alpha * gt for alpha in preset.restoration_alphas)
    frames.append(gt)
    rollout = torch.stack(frames, dim=1)
    if rollout.shape[1] != preset.total_frames:
        raise AssertionError("trajectory and preset frame counts disagree")
    frame_weights = torch.ones(preset.total_frames, dtype=torch.float32)
    frame_weights[list(preset.clean_condition_indices)] = 0
    return rollout, frame_weights, tuple(provenance)


def clamp_clean_conditions(noisy: torch.Tensor, clean: torch.Tensor, preset: RolloutPreset):
    if noisy.shape != clean.shape or noisy.ndim != 5:
        raise ValueError("noisy and clean must share [B,F,C,h,w] shape")
    if noisy.shape[1] != preset.total_frames:
        raise ValueError("tensor frame count disagrees with rollout preset")
    clamped = noisy.clone()
    clean_indices = list(preset.clean_condition_indices)
    clamped[:, clean_indices] = clean[:, clean_indices]
    return clamped


def masked_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestep_weights: torch.Tensor,
    frame_weights: torch.Tensor,
    preset: RolloutPreset,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.ndim != 5:
        raise ValueError("prediction and target must share [B,F,C,h,w] shape")
    if target.shape[1] != preset.total_frames:
        raise ValueError("tensor frame count disagrees with rollout preset")
    if not isinstance(timestep_weights, torch.Tensor):
        raise TypeError("timestep_weights must be a torch.Tensor")
    if tuple(timestep_weights.shape) != (target.shape[0],):
        raise ValueError("timestep_weights must have shape [B]")

    # Match WAN14B-RR's custom_mse_loss: residuals and MSE are evaluated in
    # float32 even when the backbone forward runs in bf16.
    difference = prediction.float() - target.float()
    weights = timestep_weights.to(device=difference.device, dtype=difference.dtype)
    if not torch.isfinite(weights).all().item() or (weights < 0).any().item():
        raise ValueError("timestep_weights must be finite and non-negative")
    while weights.ndim < target.ndim:
        weights = weights.unsqueeze(-1)
    error = weights * difference.square()
    threshold = preset.loss_outlier_abs_threshold
    if threshold is not None:
        error = error * (difference.abs() <= threshold).to(error.dtype)
    if not isinstance(frame_weights, torch.Tensor):
        raise TypeError("frame_weights must be a torch.Tensor")
    frame_weights = frame_weights.to(device=error.device, dtype=error.dtype)
    if frame_weights.ndim == 1:
        frame_weights = frame_weights.unsqueeze(0).expand(target.shape[0], -1)
    if tuple(frame_weights.shape) != (target.shape[0], preset.total_frames):
        raise ValueError("frame_weights must be [F] or [B,F]")
    if not torch.isfinite(frame_weights).all().item() or (frame_weights < 0).any().item():
        raise ValueError("frame_weights must be finite and non-negative")
    if (frame_weights[:, list(preset.clean_condition_indices)] != 0).any().item():
        raise ValueError("clean condition frames must have zero loss weight")
    masked = error * frame_weights[:, :, None, None, None]
    if preset.loss_normalization == "full_tensor_mean":
        return masked.mean()
    active = frame_weights.sum() * target.shape[2] * target.shape[3] * target.shape[4]
    return masked.sum() / active.clamp(min=1)
