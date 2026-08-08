import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.contract import (
    RolloutPreset,
    anchor_seed_for_sample,
    clamp_clean_conditions,
    construct_cached_rollout,
    masked_velocity_loss,
    resolve_rollout_preset,
    select_condition_mask_kind,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.dataset import (
    CogVideoXRolloutCacheDataset,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_t2v.lora_trainer import (
    CogVideoXT2VLoraTrainer,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.lora_trainer import (
    CogVideoXRolloutT2VLoraTrainer,
)
from cogkit.finetune.base.base_trainer import BaseTrainer


def _entry(offset: float = 0.0) -> dict[str, torch.Tensor]:
    source = torch.full((2, 3, 4), offset, dtype=torch.float32)
    gt = torch.full_like(source, offset + 4.0)
    mask_sam = torch.full_like(source, offset + 1.0)
    mask_check = torch.full_like(source, offset + 2.0)
    mask_binary = torch.zeros((1, 3, 4), dtype=torch.float32)
    mask_binary[:, 1:, 2:] = 1
    return {
        "source": source,
        "gt": gt,
        "mask_sam": mask_sam,
        "mask_check": mask_check,
        "mask_sam_binary": mask_binary.clone(),
        "mask_check_binary": mask_binary,
    }


def test_wan14b_contract_layout_is_exact() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    assert preset.total_frames == 14
    assert preset.clean_condition_indices == (0, 2)
    assert preset.supervised_frame_indices == (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    assert preset.frame_roles[0:3] == (
        "mask_condition",
        "mask_check_prediction",
        "source_condition",
    )
    assert preset.noisy_anchor_index == 7
    assert preset.final_gt_index == 13


def test_cached_rollout_matches_live_wan_formula() -> None:
    preset = resolve_rollout_preset(
        "wan14b_maskpred_c4r5_g1p2_14f", mask_sam_probability=1.0
    )
    entry = _entry()
    rollout, frame_weights, provenance = construct_cached_rollout(
        [entry], ["sample-a"], preset, rollout_seed=42, epoch=3
    )
    generator = torch.Generator(device="cpu").manual_seed(
        anchor_seed_for_sample(42, "sample-a")
    )
    noise = torch.randn((1, 2, 3, 4), generator=generator)
    expected_anchor = torch.where(
        entry["mask_check_binary"].unsqueeze(0) > 0.5,
        noise,
        entry["source"].unsqueeze(0),
    )

    assert rollout.shape == (1, 14, 2, 3, 4)
    assert torch.equal(rollout[:, 0], entry["mask_sam"].unsqueeze(0))
    assert torch.equal(rollout[:, 1], entry["mask_check"].unsqueeze(0))
    assert torch.equal(rollout[:, 2], entry["source"].unsqueeze(0))
    assert torch.equal(rollout[:, 7], expected_anchor)
    assert torch.equal(rollout[:, 13], entry["gt"].unsqueeze(0))
    for frame_index, alpha in zip(preset.corruption_indices, preset.corruption_alphas):
        expected = (1.0 - alpha) * rollout[:, 2] + alpha * expected_anchor
        assert torch.allclose(rollout[:, frame_index], expected)
    for frame_index, alpha in zip(preset.restoration_indices, preset.restoration_alphas):
        expected = (1.0 - alpha) * expected_anchor + alpha * rollout[:, 13]
        assert torch.allclose(rollout[:, frame_index], expected)
    assert provenance[0].condition_mask_kind == "mask_sam"
    assert frame_weights.tolist() == [0.0, 1.0, 0.0] + [1.0] * 11


def test_mask_selection_is_sample_epoch_deterministic() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    first = [
        select_condition_mask_kind(preset, rollout_seed=42, sample_id="x", epoch=epoch)
        for epoch in range(32)
    ]
    second = [
        select_condition_mask_kind(preset, rollout_seed=42, sample_id="x", epoch=epoch)
        for epoch in range(32)
    ]
    assert first == second
    assert set(first) == {"mask_sam", "mask_check"}


def test_clean_clamp_and_wan_full_tensor_loss() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    clean = torch.randn((2, 14, 3, 2, 2))
    noisy = torch.randn_like(clean)
    clamped = clamp_clean_conditions(noisy, clean, preset)
    assert torch.equal(clamped[:, 0], clean[:, 0])
    assert torch.equal(clamped[:, 2], clean[:, 2])
    assert torch.equal(clamped[:, 1], noisy[:, 1])

    frame_weights = torch.ones(14)
    frame_weights[[0, 2]] = 0
    loss = masked_velocity_loss(
        torch.ones_like(clean),
        torch.zeros_like(clean),
        torch.ones(2),
        frame_weights,
        preset,
    )
    assert loss.item() == pytest.approx(12.0 / 14.0)


def test_clean_clamp_overwrites_nonfinite_scheduler_values() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    clean = torch.zeros((1, 14, 1, 1, 1))
    noisy = torch.ones_like(clean)
    noisy[:, 0] = torch.nan
    noisy[:, 2] = torch.inf
    clamped = clamp_clean_conditions(noisy, clean, preset)
    assert torch.equal(clamped[:, 0], clean[:, 0])
    assert torch.equal(clamped[:, 2], clean[:, 2])
    assert torch.equal(clamped[:, 1], noisy[:, 1])


def test_wan_loss_computes_bfloat16_residuals_in_float32() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    prediction = torch.full((1, 14, 1, 1, 1), 1.1, dtype=torch.bfloat16)
    target = torch.full_like(prediction, 0.1)
    frame_weights = torch.ones(14)
    frame_weights[[0, 2]] = 0
    loss = masked_velocity_loss(
        prediction,
        target,
        torch.ones(1),
        frame_weights,
        preset,
    )
    difference = prediction.float() - target.float()
    expected = (difference.square() * frame_weights[None, :, None, None, None]).mean()
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(expected.item())


def test_contract_rejects_nonbinary_mask_and_clean_loss_weight() -> None:
    preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    entry = _entry()
    entry["mask_check_binary"][0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="only 0 and 1"):
        construct_cached_rollout([entry], ["bad"], preset, rollout_seed=0, epoch=0)

    target = torch.zeros((1, 14, 1, 1, 1))
    with pytest.raises(ValueError, match="clean condition"):
        masked_velocity_loss(
            target,
            target,
            torch.ones(1),
            torch.ones(14),
            preset,
        )


def _write_cache_sample(root: Path, *, sample_id: str = "000001") -> tuple[Path, Path]:
    for name in ("shot", "bg", "mask-sam", "mask-check"):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{sample_id}.png").write_bytes(b"asset")
    cache_dir = root / "cache"
    cache_dir.mkdir()
    tensors = _entry()
    torch.save(tensors, cache_dir / f"{sample_id}.pt")
    metadata = {
        "sample_id": sample_id,
        "model_id": "/model/CogVideoX-5b",
        "scaling_factor": 0.7,
        "posterior_mode": "mode",
        "vae_seed_by_role": {name: index for index, name in enumerate(("source", "gt", "mask_sam", "mask_check"))},
        "preprocessing_convention": "uint8_0_255_to_float_0_1_to_vae_minus1_plus1_v1",
        "latent_convention": "cogvideox_posterior_times_scaling_factor_v1",
        "temporal_vae_policy": "independent_single_frame_encode_decode_v1",
        "source_shape": [2, 3, 4],
        "gt_shape": [2, 3, 4],
        "mask_sam_shape": [2, 3, 4],
        "mask_check_shape": [2, 3, 4],
        "pixel_height": 24,
        "pixel_width": 32,
    }
    (cache_dir / f"{sample_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "source": f"shot/{sample_id}.png",
                "gt": f"bg/{sample_id}.png",
                "mask_sam": f"mask-sam/{sample_id}.png",
                "mask_check": f"mask-check/{sample_id}.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, cache_dir


def test_cache_dataset_is_fail_closed(tmp_path: Path) -> None:
    manifest, cache_dir = _write_cache_sample(tmp_path)
    dataset = CogVideoXRolloutCacheDataset(
        dataset_root=tmp_path,
        manifest=manifest,
        cache_dir=cache_dir,
        expected_model_id="/model/CogVideoX-5b",
        expected_scaling_factor=0.7,
        expected_posterior_mode="mode",
        expected_pixel_height=24,
        expected_pixel_width=32,
        expected_latent_shape=(2, 3, 4),
    )
    assert len(dataset) == 1
    assert dataset[0]["sample_id"] == "000001"

    tensors = torch.load(cache_dir / "000001.pt", weights_only=True)
    tensors["mask_sam_binary"][0, 0, 0] = 0.25
    torch.save(tensors, cache_dir / "000001.pt")
    with pytest.raises(ValueError, match="only 0 and 1"):
        dataset[0]


def test_cache_dataset_rejects_unsafe_sample_id(tmp_path: Path) -> None:
    manifest, cache_dir = _write_cache_sample(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sample_id"] = "../escape"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe sample_id"):
        CogVideoXRolloutCacheDataset(
            dataset_root=tmp_path,
            manifest=manifest,
            cache_dir=cache_dir,
            expected_model_id="/model/CogVideoX-5b",
            expected_scaling_factor=0.7,
            expected_posterior_mode="mode",
            expected_pixel_height=24,
            expected_pixel_width=32,
            expected_latent_shape=(2, 3, 4),
        )


def test_native_cogvideox_loss_hook_preserves_original_reduction() -> None:
    trainer = object.__new__(CogVideoXT2VLoraTrainer)
    prediction = torch.tensor([[[[[1.0]]], [[[3.0]]]]])
    target = torch.zeros_like(prediction)
    weights = torch.tensor([2.0])
    loss = trainer._compute_training_loss(
        prediction=prediction,
        target=target,
        timestep_weights=weights,
        batch={},
    )
    assert loss.item() == pytest.approx(10.0)


def test_native_cogvideox_noise_hook_delegates_to_scheduler() -> None:
    class Scheduler:
        @staticmethod
        def add_noise(clean, noise, timesteps):
            return clean + 2 * noise + timesteps[:, None, None, None, None]

    trainer = object.__new__(CogVideoXT2VLoraTrainer)
    trainer.components = SimpleNamespace(scheduler=Scheduler())
    clean = torch.ones((1, 2, 1, 1, 1))
    noise = torch.full_like(clean, 3)
    result = trainer._add_training_noise(
        clean_latent=clean,
        noise=noise,
        timesteps=torch.tensor([4]),
        batch={},
    )
    assert torch.equal(result, torch.full_like(clean, 11))


def test_rollout_trainer_reuses_native_compute_loss() -> None:
    assert "compute_loss" not in CogVideoXRolloutT2VLoraTrainer.__dict__
    assert (
        CogVideoXRolloutT2VLoraTrainer.compute_loss
        is CogVideoXT2VLoraTrainer.compute_loss
    )


def test_rollout_adapter_runs_native_compute_loss_and_backward() -> None:
    class Scheduler:
        config = SimpleNamespace(num_train_timesteps=8)
        alphas_cumprod = torch.linspace(0.1, 0.8, 8)

        @staticmethod
        def add_noise(clean, noise, timesteps):
            del timesteps
            return 0.75 * clean + 0.25 * noise

        @staticmethod
        def get_velocity(prediction, noisy, timesteps):
            del noisy, timesteps
            return prediction

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25))
            self.last_hidden_shape = None

        def forward(self, *, hidden_states, **kwargs):
            del kwargs
            self.last_hidden_shape = tuple(hidden_states.shape)
            return (hidden_states * self.scale,)

    transformer = Transformer()
    trainer = object.__new__(CogVideoXRolloutT2VLoraTrainer)
    trainer.state = SimpleNamespace(
        device=torch.device("cpu"),
        transformer_config=SimpleNamespace(
            patch_size_t=None,
            use_rotary_positional_embeddings=True,
        ),
    )
    trainer.components = SimpleNamespace(
        scheduler=Scheduler(),
        transformer=transformer,
        vae=SimpleNamespace(config=SimpleNamespace(block_out_channels=(1, 1, 1, 1))),
    )
    trainer.uargs = SimpleNamespace(seed=123, rollout_seed=42)
    trainer.rollout_preset = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")
    trainer._rollout_epoch = 0
    trainer.prepare_rotary_positional_embeddings = lambda **kwargs: None
    batch = {
        "sample_ids": ["integration"],
        "cache_entries": [_entry()],
        "prompt_embedding": torch.zeros((1, 2, 3)),
    }

    torch.manual_seed(123)
    loss = trainer.compute_loss(batch)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss).item()
    assert transformer.last_hidden_shape == (1, 14, 2, 3, 4)
    assert batch["frame_weights"].tolist() == [0.0, 1.0, 0.0] + [1.0] * 11
    assert batch["trajectory_provenance"][0]["anchor_seed"] == anchor_seed_for_sample(
        42, "integration"
    )
    loss.backward()
    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad).item()
    assert transformer.scale.grad.abs().item() > 0


def test_base_epoch_hook_advances_dataloader_sampler() -> None:
    class Sampler:
        epoch = None

        def set_epoch(self, epoch):
            self.epoch = epoch

    sampler = Sampler()
    trainer = SimpleNamespace(train_data_loader=SimpleNamespace(sampler=sampler))
    BaseTrainer.on_train_epoch_start(trainer, 7)
    assert sampler.epoch == 7
