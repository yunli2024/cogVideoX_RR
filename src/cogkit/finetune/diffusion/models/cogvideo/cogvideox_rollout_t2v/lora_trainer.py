"""Minimal WAN14B-RR method adapter over CogKit's native CogVideoX trainer."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing_extensions import override

from cogkit.finetune import register
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_t2v.lora_trainer import (
    CogVideoXT2VLoraTrainer,
)
from cogkit.finetune.utils import free_memory, is_main_process

from .args import CogVideoXRolloutArgs
from .contract import (
    clamp_clean_conditions,
    construct_cached_rollout,
    masked_velocity_loss,
)
from .dataset import CogVideoXRolloutCacheDataset


class CogVideoXRolloutT2VLoraTrainer(CogVideoXT2VLoraTrainer):
    """Override only dataset, RR conditions, and RR loss semantics."""

    @override
    def _init_args(self, uargs_fpath: Path) -> CogVideoXRolloutArgs:
        return CogVideoXRolloutArgs.parse_from_yaml(uargs_fpath)

    @override
    def prepare_dataset(self) -> None:
        self.rollout_preset = self.uargs.resolved_preset()
        self._rollout_epoch = 0
        _, pixel_height, pixel_width = self.uargs.train_resolution
        vae_scale = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        self._validate_model_contract(vae_scale)
        latent_shape = (
            int(self.state.transformer_config.in_channels),
            pixel_height // vae_scale,
            pixel_width // vae_scale,
        )
        self.train_dataset = CogVideoXRolloutCacheDataset(
            dataset_root=self.uargs.data_root,
            manifest=self.uargs.rollout_manifest,
            cache_dir=self.uargs.rollout_cache_dir,
            expected_model_id=str(self.uargs.model_path),
            expected_scaling_factor=self.uargs.rollout_expected_vae_scaling_factor,
            expected_posterior_mode=self.uargs.rollout_expected_posterior_mode,
            expected_pixel_height=pixel_height,
            expected_pixel_width=pixel_width,
            expected_latent_shape=latent_shape,
        )

        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.text_encoder.to(self.state.device, dtype=self.state.weight_dtype)
        self._rollout_prompt_embedding = self.encode_text(self.uargs.rollout_prompt).to("cpu")
        self.components.text_encoder.to("cpu")
        free_memory()

        self._write_resolved_training_contract(
            vae_scale=vae_scale,
            latent_shape=latent_shape,
        )

        self.train_data_loader = DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.uargs.batch_size,
            num_workers=self.uargs.num_workers,
            pin_memory=self.uargs.pin_memory,
            sampler=DistributedSampler(
                self.train_dataset,
                num_replicas=self.state.world_size,
                rank=self.state.global_rank,
                shuffle=True,
                seed=0 if self.uargs.seed is None else self.uargs.seed,
            ),
        )

    @override
    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        sample_ids = [str(sample["sample_id"]) for sample in samples]
        prompt_embedding = self._rollout_prompt_embedding.unsqueeze(0).expand(
            len(samples), -1, -1
        )
        return {
            "sample_ids": sample_ids,
            "cache_entries": [sample["tensors"] for sample in samples],
            "prompt_embedding": prompt_embedding,
        }

    @override
    def on_train_epoch_start(self, epoch: int) -> None:
        super().on_train_epoch_start(epoch)
        self._rollout_epoch = int(epoch)
        if self.uargs.seed is not None:
            rank_epoch_seed = (
                int(self.uargs.seed)
                + self._rollout_epoch * self.state.world_size
                + self.state.global_rank
            )
            torch.cuda.manual_seed_all(rank_epoch_seed)

    @override
    def _prepare_training_batch(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rollout, frame_weights, provenance = construct_cached_rollout(
            batch["cache_entries"],
            batch["sample_ids"],
            self.rollout_preset,
            rollout_seed=self.uargs.rollout_seed,
            epoch=self._rollout_epoch,
        )
        batch["encoded_videos"] = rollout.permute(0, 2, 1, 3, 4).contiguous()
        batch["frame_weights"] = frame_weights
        batch["trajectory_provenance"] = tuple(item.to_dict() for item in provenance)
        return super()._prepare_training_batch(batch)

    @override
    def _add_training_noise(
        self,
        *,
        clean_latent: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        noisy = super()._add_training_noise(
            clean_latent=clean_latent,
            noise=noise,
            timesteps=timesteps,
            batch=batch,
        )
        return clamp_clean_conditions(noisy, clean_latent, self.rollout_preset)

    @override
    def _compute_training_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestep_weights: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return masked_velocity_loss(
            prediction,
            target,
            timestep_weights,
            batch["frame_weights"],
            self.rollout_preset,
        )

    def _validate_model_contract(self, vae_scale: int) -> None:
        transformer_config = self.state.transformer_config
        actual_scaling_factor = float(self.components.vae.config.scaling_factor)
        expected_scaling_factor = float(self.uargs.rollout_expected_vae_scaling_factor)
        if abs(actual_scaling_factor - expected_scaling_factor) > 1e-12:
            raise ValueError(
                "loaded VAE scaling factor disagrees with rollout cache contract: "
                f"actual={actual_scaling_factor}, expected={expected_scaling_factor}"
            )
        if vae_scale != 8:
            raise ValueError(f"RolloutRemover CogVideoX-5B requires VAE spatial scale 8, got {vae_scale}")
        if int(transformer_config.in_channels) != 16:
            raise ValueError(
                f"RolloutRemover CogVideoX-5B requires 16 latent channels, got {transformer_config.in_channels}"
            )
        if int(transformer_config.patch_size) != 2:
            raise ValueError(
                f"RolloutRemover CogVideoX-5B requires spatial patch size 2, got {transformer_config.patch_size}"
            )
        if transformer_config.patch_size_t is not None:
            raise ValueError("RolloutRemover adapter requires CogVideoX with patch_size_t=None")
        if not transformer_config.use_rotary_positional_embeddings:
            raise ValueError("RolloutRemover CogVideoX-5B requires native 3D rotary embeddings")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_resolved_training_contract(
        self, *, vae_scale: int, latent_shape: tuple[int, int, int]
    ) -> None:
        if not is_main_process():
            return
        payload = {
            "schema_version": 1,
            "method": "RolloutRemover",
            "backbone": "CogVideoX-5B",
            "model_path": str(self.uargs.model_path),
            "data_root": str(self.uargs.data_root),
            "manifest": str(self.uargs.rollout_manifest),
            "manifest_sha256": self._sha256(Path(self.uargs.rollout_manifest)),
            "cache_dir": str(self.uargs.rollout_cache_dir),
            "prompt": self.uargs.rollout_prompt,
            "seed": self.uargs.seed,
            "diffusion_seed_policy": "seed_plus_epoch_times_world_size_plus_global_rank_v1",
            "rollout_seed": self.uargs.rollout_seed,
            "mask_selection": "deterministic_sample_epoch_hash_v1",
            "anchor": "mask_check_binary_gaussian_inside_source_outside_v1",
            "trajectory": self.rollout_preset.to_dict(),
            "model_admission": {
                "vae_scaling_factor": float(self.components.vae.config.scaling_factor),
                "vae_spatial_scale": vae_scale,
                "latent_shape": list(latent_shape),
                "patch_size": int(self.state.transformer_config.patch_size),
                "patch_size_t": self.state.transformer_config.patch_size_t,
                "use_rotary_positional_embeddings": bool(
                    self.state.transformer_config.use_rotary_positional_embeddings
                ),
            },
            "optimization": {
                "training_type": self.uargs.training_type,
                "strategy": self.uargs.strategy,
                "world_size": self.state.world_size,
                "batch_size_per_rank": self.uargs.batch_size,
                "gradient_accumulation_steps": self.uargs.gradient_accumulation_steps,
                "learning_rate": self.uargs.learning_rate,
                "optimizer": self.uargs.optimizer,
                "lr_scheduler": self.uargs.lr_scheduler,
            },
        }
        output_path = Path(self.uargs.output_dir) / "rollout_training_contract.json"
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if output_path.exists():
            if output_path.read_text(encoding="utf-8") != serialized:
                raise ValueError(
                    f"existing rollout contract disagrees with current run: {output_path}"
                )
            return
        output_path.write_text(serialized, encoding="utf-8")


register("cogvideox-rollout-t2v", "lora", CogVideoXRolloutT2VLoraTrainer)
