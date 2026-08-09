from types import SimpleNamespace

import pytest
import torch

from cogkit.finetune.diffusion.models.cogvideo.cogvideox_endpoint_t2v.contract import (
    EndpointContract,
    clamp_endpoint_conditions,
    construct_cached_endpoint,
    endpoint_velocity_loss,
    select_condition_mask_kind,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_endpoint_t2v.lora_trainer import (
    CogVideoXEndpointT2VLoraTrainer,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_t2v.lora_trainer import (
    CogVideoXT2VLoraTrainer,
)


def _entry(offset: float = 0.0) -> dict[str, torch.Tensor]:
    source = torch.full((2, 3, 4), offset, dtype=torch.float32)
    return {
        "source": source,
        "gt": torch.full_like(source, offset + 4.0),
        "mask_sam": torch.full_like(source, offset + 1.0),
        "mask_check": torch.full_like(source, offset + 2.0),
    }


def test_endpoint_layout_has_no_intermediate_states() -> None:
    contract = EndpointContract(mask_sam_probability=1.0)
    endpoint, weights, provenance = construct_cached_endpoint(
        [_entry()], ["sample-a"], contract, endpoint_seed=42, epoch=0
    )
    assert contract.frame_roles == ("mask_condition", "source_condition", "final_target")
    assert contract.clean_condition_indices == (0, 1)
    assert contract.supervised_frame_indices == (2,)
    assert endpoint.shape == (1, 3, 2, 3, 4)
    assert torch.equal(endpoint[:, 0], _entry()["mask_sam"].unsqueeze(0))
    assert torch.equal(endpoint[:, 1], _entry()["source"].unsqueeze(0))
    assert torch.equal(endpoint[:, 2], _entry()["gt"].unsqueeze(0))
    assert weights.tolist() == [0.0, 0.0, 1.0]
    assert provenance[0].condition_mask_kind == "mask_sam"
    assert contract.to_dict()["constructs_intermediate_rollout"] is False


def test_endpoint_mask_selection_matches_rr_hash_policy() -> None:
    contract = EndpointContract()
    first = [
        select_condition_mask_kind(contract, endpoint_seed=42, sample_id="x", epoch=epoch)
        for epoch in range(32)
    ]
    second = [
        select_condition_mask_kind(contract, endpoint_seed=42, sample_id="x", epoch=epoch)
        for epoch in range(32)
    ]
    assert first == second
    assert set(first) == {"mask_sam", "mask_check"}


def test_endpoint_clamps_only_conditions_and_supervises_only_target() -> None:
    contract = EndpointContract()
    clean = torch.randn((2, 3, 2, 2, 2))
    noisy = torch.randn_like(clean)
    clamped = clamp_endpoint_conditions(noisy, clean, contract)
    assert torch.equal(clamped[:, 0], clean[:, 0])
    assert torch.equal(clamped[:, 1], clean[:, 1])
    assert torch.equal(clamped[:, 2], noisy[:, 2])

    loss = endpoint_velocity_loss(
        torch.ones_like(clean),
        torch.zeros_like(clean),
        torch.ones(2),
        torch.tensor([0.0, 0.0, 1.0]),
        contract,
    )
    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(1.0)


def test_endpoint_adapter_runs_native_compute_loss_and_backward() -> None:
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
    trainer = object.__new__(CogVideoXEndpointT2VLoraTrainer)
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
    trainer.uargs = SimpleNamespace(seed=42, endpoint_seed=42)
    trainer.endpoint_contract = EndpointContract()
    trainer._endpoint_epoch = 0
    trainer.prepare_rotary_positional_embeddings = lambda **kwargs: None
    batch = {
        "sample_ids": ["integration"],
        "cache_entries": [_entry()],
        "prompt_embedding": torch.zeros((1, 2, 3)),
    }

    torch.manual_seed(42)
    loss = trainer.compute_loss(batch)
    assert torch.isfinite(loss).item()
    assert transformer.last_hidden_shape == (1, 3, 2, 3, 4)
    assert batch["frame_weights"].tolist() == [0.0, 0.0, 1.0]
    loss.backward()
    assert transformer.scale.grad is not None
    assert torch.isfinite(transformer.scale.grad).item()


def test_endpoint_trainer_reuses_native_compute_loss() -> None:
    assert "compute_loss" not in CogVideoXEndpointT2VLoraTrainer.__dict__
    assert CogVideoXEndpointT2VLoraTrainer.compute_loss is CogVideoXT2VLoraTrainer.compute_loss
