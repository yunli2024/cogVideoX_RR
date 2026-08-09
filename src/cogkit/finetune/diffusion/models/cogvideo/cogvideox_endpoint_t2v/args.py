"""Arguments for direct-target CogVideoX endpoint training."""

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from cogkit.finetune.base import BaseArgs

from .contract import EndpointContract


class CogVideoXEndpointArgs(BaseArgs):
    model_config = {"frozen": True, "extra": "forbid"}

    train_resolution: list[int]
    enable_slicing: bool = True
    enable_tiling: bool = True
    enable_packing: bool = False
    gen_fps: int | None = None

    endpoint_manifest: Path
    endpoint_cache_dir: Path
    endpoint_prompt: str = "Remove the masked object and its side effect"
    endpoint_seed: int = 42
    endpoint_mask_sam_probability: float = 0.5
    endpoint_loss_normalization: Literal["full_tensor_mean", "active_weight_mean"] = "full_tensor_mean"
    endpoint_loss_outlier_abs_threshold: float | None = 50.0
    endpoint_expected_vae_scaling_factor: float = 0.7
    endpoint_expected_posterior_mode: Literal["mode"] = "mode"

    @field_validator("endpoint_seed")
    @classmethod
    def validate_endpoint_seed(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("endpoint_seed must be non-negative")
        return int(value)

    @field_validator("train_resolution")
    @classmethod
    def validate_train_resolution(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(int(item) <= 0 for item in value):
            raise ValueError("train_resolution must be [frames, height, width]")
        return [int(item) for item in value]

    def resolved_contract(self) -> EndpointContract:
        threshold = self.endpoint_loss_outlier_abs_threshold
        return EndpointContract(
            mask_sam_probability=self.endpoint_mask_sam_probability,
            loss_normalization=self.endpoint_loss_normalization,
            loss_outlier_abs_threshold=threshold,
        )

    @model_validator(mode="after")
    def validate_endpoint_contract(self):
        contract = self.resolved_contract()
        frames, height, width = self.train_resolution
        if frames != contract.total_frames:
            raise ValueError(
                f"train_resolution declares {frames} states but endpoint contract requires {contract.total_frames}"
            )
        if height % 16 or width % 16:
            raise ValueError("height and width must be divisible by VAE scale 8 times patch size 2")
        if self.enable_packing:
            raise ValueError("packing is not implemented for EndpointRemover")
        if self.do_validation:
            raise ValueError("adapter v1 has no endpoint-aware validation; set do_validation: false")
        if self.endpoint_expected_vae_scaling_factor <= 0:
            raise ValueError("endpoint_expected_vae_scaling_factor must be positive")
        if self.training_type != "lora":
            raise ValueError("EndpointRemover CogVideoX adapter currently supports LoRA only")
        return self
