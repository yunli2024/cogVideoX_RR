from types import SimpleNamespace

import torch
from diffusers import CogVideoXDPMScheduler

from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.contract import (
    resolve_rollout_preset,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.inference import (
    build_initial_rollout_latents,
    decode_rollout_states_individually,
    restore_clean_conditions,
    sample_rollout_latents,
    sample_seed,
    zero_condition_prediction,
)


PRESET = resolve_rollout_preset("wan14b_maskpred_c4r5_g1p2_14f")


def _conditions():
    mask = torch.full((1, 2, 3, 4), -0.25)
    source = torch.full((1, 2, 3, 4), 0.75)
    return mask, source


def test_initial_latents_use_only_two_clean_conditions():
    mask, source = _conditions()
    latents = build_initial_rollout_latents(
        mask_condition=mask,
        source_condition=source,
        preset=PRESET,
        generator=torch.Generator().manual_seed(7),
        init_noise_sigma=2.0,
    )
    assert latents.shape == (1, 14, 2, 3, 4)
    torch.testing.assert_close(latents[:, 0], mask)
    torch.testing.assert_close(latents[:, 2], source)
    assert not torch.equal(latents[:, 1], mask)
    assert not torch.equal(latents[:, -1], source)


def test_restore_and_zero_are_non_mutating():
    mask, source = _conditions()
    values = torch.randn(1, 14, 2, 3, 4)
    original = values.clone()
    restored = restore_clean_conditions(
        values,
        mask_condition=mask,
        source_condition=source,
        preset=PRESET,
    )
    torch.testing.assert_close(values, original)
    torch.testing.assert_close(restored[:, 0], mask)
    torch.testing.assert_close(restored[:, 2], source)

    prediction = torch.ones_like(values)
    zeroed = zero_condition_prediction(prediction, PRESET)
    assert torch.count_nonzero(zeroed[:, 0]) == 0
    assert torch.count_nonzero(zeroed[:, 2]) == 0
    assert torch.count_nonzero(zeroed[:, 1]) > 0
    assert torch.count_nonzero(prediction[:, 0]) > 0


def test_sample_seed_is_stable_and_sample_specific():
    assert sample_seed(42, "080001") == sample_seed(42, "080001")
    assert sample_seed(42, "080001") != sample_seed(42, "080002")


class _FakeScheduler:
    init_noise_sigma = 1.0
    order = 1

    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        del kwargs
        self.timesteps = torch.arange(
            num_inference_steps, 0, -1, device=device, dtype=torch.long
        )

    def scale_model_input(self, sample, timestep):
        del timestep
        return sample

    def step(self, prediction, timestep, sample, return_dict=False, **kwargs):
        del timestep, return_dict, kwargs
        return (sample - 0.1 * prediction,)


class _FakeTransformer(torch.nn.Module):
    config = SimpleNamespace(use_rotary_positional_embeddings=True)

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def __call__(self, *, hidden_states, **kwargs):
        del kwargs
        return (torch.ones_like(hidden_states),)


class _FakePipeline:
    def __init__(self):
        self.scheduler = _FakeScheduler()
        self.transformer = _FakeTransformer()
        self._execution_device = torch.device("cpu")

    def _prepare_rotary_positional_embeddings(self, height, width, frames, device):
        return height, width, frames, device

    def prepare_extra_step_kwargs(self, generator, eta):
        del generator, eta
        return {}


class _FakeDPMPipeline(_FakePipeline):
    def __init__(self):
        super().__init__()
        self.scheduler = CogVideoXDPMScheduler()


def test_sampler_keeps_clean_conditions_through_all_steps():
    mask, source = _conditions()
    pipeline = _FakePipeline()
    result = sample_rollout_latents(
        pipeline,
        preset=PRESET,
        mask_condition=mask,
        source_condition=source,
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=None,
        generator=torch.Generator().manual_seed(9),
        height=48,
        width=64,
        num_inference_steps=3,
        guidance_scale=1.0,
    )
    torch.testing.assert_close(result[:, 0], mask)
    torch.testing.assert_close(result[:, 2], source)
    assert not torch.equal(result[:, 1], torch.zeros_like(result[:, 1]))


def test_sampler_supports_native_cogvideox_dpm_scheduler():
    mask, source = _conditions()
    pipeline = _FakeDPMPipeline()
    result = sample_rollout_latents(
        pipeline,
        preset=PRESET,
        mask_condition=mask,
        source_condition=source,
        prompt_embeds=torch.zeros(1, 2, 3),
        negative_prompt_embeds=None,
        generator=torch.Generator().manual_seed(11),
        height=48,
        width=64,
        num_inference_steps=3,
        guidance_scale=1.0,
    )
    torch.testing.assert_close(result[:, 0], mask)
    torch.testing.assert_close(result[:, 2], source)
    assert torch.isfinite(result).all()


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.5)
        self.calls = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def decode(self, value):
        self.calls.append(tuple(value.shape))
        return SimpleNamespace(sample=value[:, :3])


def test_decode_is_one_vae_call_per_state_with_temporal_length_one():
    vae = _FakeVAE()
    latents = torch.randn(1, 4, 3, 2, 2)
    decoded = decode_rollout_states_individually(vae, latents)
    assert decoded.shape == (1, 4, 3, 2, 2)
    assert vae.calls == [(1, 3, 1, 2, 2)] * 4


if __name__ == "__main__":
    tests = [
        test_initial_latents_use_only_two_clean_conditions,
        test_restore_and_zero_are_non_mutating,
        test_sample_seed_is_stable_and_sample_specific,
        test_sampler_keeps_clean_conditions_through_all_steps,
        test_sampler_supports_native_cogvideox_dpm_scheduler,
        test_decode_is_one_vae_call_per_state_with_temporal_length_one,
    ]
    for test in tests:
        test()
    print(f"PASSED: {len(tests)} focused CogVideoX-RR inference tests")
