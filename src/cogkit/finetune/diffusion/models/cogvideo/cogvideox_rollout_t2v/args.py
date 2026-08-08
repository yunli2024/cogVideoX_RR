"""Isolated CogKit arguments for cached RolloutRemover training."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from cogkit.finetune.base import BaseArgs

from .contract import RolloutPreset, resolve_rollout_preset


class CogVideoXRolloutArgs(BaseArgs):
    """CogKit base arguments plus an explicit RolloutRemover contract."""

    model_config = {"frozen": True, "extra": "forbid"}

    train_resolution: list[int]
    enable_slicing: bool = True
    enable_tiling: bool = True
    enable_packing: bool = False
    gen_fps: int | None = None

    rollout_manifest: Path
    rollout_cache_dir: Path
    rollout_prompt: str = "Remove the masked object and its side effect"
    rollout_preset: str = "wan14b_maskpred_c4r5_g1p2_14f"
    rollout_seed: int = 42
    rollout_corruption_frames: int | None = None
    rollout_restoration_frames: int | None = None
    rollout_interpolation_gamma: float | None = None
    rollout_mask_sam_probability: float | None = None
    rollout_loss_normalization: Literal["full_tensor_mean", "active_weight_mean"] | None = None
    rollout_loss_outlier_abs_threshold: float | None = None
    rollout_expected_vae_scaling_factor: float = 0.7
    rollout_expected_posterior_mode: Literal["mode"] = "mode"

    @field_validator("rollout_seed")
    @classmethod
    def validate_rollout_seed(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("rollout_seed must be non-negative")
        return int(value)

    @field_validator("train_resolution")
    @classmethod
    def validate_train_resolution(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(int(item) <= 0 for item in value):
            raise ValueError("train_resolution must be [rollout_frames, height, width]")
        return [int(item) for item in value]

    def resolved_preset(self) -> RolloutPreset:
        return resolve_rollout_preset(
            self.rollout_preset,
            corruption_frames=self.rollout_corruption_frames,
            restoration_frames=self.rollout_restoration_frames,
            interpolation_gamma=self.rollout_interpolation_gamma,
            mask_sam_probability=self.rollout_mask_sam_probability,
            loss_normalization=self.rollout_loss_normalization,
            loss_outlier_abs_threshold=self.rollout_loss_outlier_abs_threshold,
        )

    @model_validator(mode="after")
    def validate_rollout_contract(self):
        preset = self.resolved_preset()
        frames, height, width = self.train_resolution
        if frames != preset.total_frames:
            raise ValueError(
                f"train_resolution declares {frames} states but preset requires {preset.total_frames}"
            )
        if height % 16 or width % 16:
            raise ValueError("height and width must be divisible by VAE scale 8 times patch size 2")
        if self.enable_packing:
            raise ValueError("packing is not implemented for RolloutRemover")
        if self.do_validation:
            raise ValueError("adapter v1 has no rollout-aware validation; set do_validation: false")
        if self.rollout_expected_vae_scaling_factor <= 0:
            raise ValueError("rollout_expected_vae_scaling_factor must be positive")
        if self.training_type != "lora":
            raise ValueError("RolloutRemover CogVideoX adapter currently supports LoRA only")
        return self
