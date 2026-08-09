"""CogKit-native sampling primitives for direct-target EndpointRemover.

Inference mirrors the training topology exactly: two clean condition states
followed by one Gaussian-initialized target state.  No intermediate rollout is
constructed or presented to the Transformer.
"""

from __future__ import annotations

import hashlib
import math

import torch
from diffusers import CogVideoXDPMScheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps

from .contract import EndpointContract


INFERENCE_POLICY = "cogkit_native_condition_clamped_endpoint_v1"
SEED_POLICY = "sha256_global_seed_plus_sample_id_endpoint_v1"
TEMPORAL_VAE_POLICY = "independent_single_frame_encode_decode_v1"


def sample_seed(global_seed: int, sample_id: str) -> int:
    """Return a rank-independent deterministic seed for one sample."""

    payload = f"cogvideox_endpoint_inference_v1\0{int(global_seed)}\0{sample_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _validate_condition_pair(
    mask_condition: torch.Tensor,
    source_condition: torch.Tensor,
) -> None:
    if mask_condition.shape != source_condition.shape or mask_condition.ndim != 4:
        raise ValueError("condition latents must share [B,C,h,w] shape")
    if not mask_condition.is_floating_point() or not source_condition.is_floating_point():
        raise TypeError("condition latents must be floating tensors")
    if not torch.isfinite(mask_condition).all().item():
        raise ValueError("mask condition latent contains NaN or infinity")
    if not torch.isfinite(source_condition).all().item():
        raise ValueError("source condition latent contains NaN or infinity")


def restore_endpoint_conditions(
    values: torch.Tensor,
    *,
    mask_condition: torch.Tensor,
    source_condition: torch.Tensor,
    contract: EndpointContract,
) -> torch.Tensor:
    """Restore the two clean endpoint conditions without mutating the input."""

    _validate_condition_pair(mask_condition, source_condition)
    if values.ndim != 5:
        raise ValueError("endpoint tensor must have [B,F,C,h,w] shape")
    if values.shape[0] != mask_condition.shape[0] or values.shape[2:] != mask_condition.shape[1:]:
        raise ValueError("endpoint tensor and condition latents disagree")
    if values.shape[1] != contract.total_frames:
        raise ValueError("endpoint tensor frame count disagrees with contract")

    restored = values.clone()
    restored[:, 0] = mask_condition.to(device=values.device, dtype=values.dtype)
    restored[:, 1] = source_condition.to(device=values.device, dtype=values.dtype)
    return restored


def zero_condition_prediction(
    prediction: torch.Tensor,
    contract: EndpointContract,
) -> torch.Tensor:
    """Suppress predicted updates to the two clean condition states."""

    if prediction.ndim != 5 or prediction.shape[1] != contract.total_frames:
        raise ValueError("prediction must have contract-aligned [B,F,C,h,w] shape")
    result = prediction.clone()
    result[:, list(contract.clean_condition_indices)] = 0
    return result


def build_initial_endpoint_latents(
    *,
    mask_condition: torch.Tensor,
    source_condition: torch.Tensor,
    contract: EndpointContract,
    generator: torch.Generator,
    init_noise_sigma: float,
) -> torch.Tensor:
    """Create `[clean mask, clean source, Gaussian target]`."""

    _validate_condition_pair(mask_condition, source_condition)
    sigma = float(init_noise_sigma)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("scheduler init_noise_sigma must be finite and positive")
    batch, channels, height, width = source_condition.shape
    target_noise = torch.randn(
        (batch, channels, height, width),
        generator=generator,
        device=source_condition.device,
        dtype=source_condition.dtype,
    ) * sigma
    return torch.stack((mask_condition, source_condition, target_noise), dim=1)


@torch.no_grad()
def encode_endpoint_prompt(
    pipeline,
    *,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the native CogVideoX prompt encoder unchanged."""

    return pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=float(guidance_scale) > 1.0,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def sample_endpoint_latents(
    pipeline,
    *,
    contract: EndpointContract,
    mask_condition: torch.Tensor,
    source_condition: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    generator: torch.Generator,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    use_dynamic_cfg: bool = False,
    eta: float = 0.0,
    attention_kwargs: dict | None = None,
) -> torch.Tensor:
    """Denoise only the target state while clamping mask and source."""

    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if guidance_scale < 1.0:
        raise ValueError("guidance_scale must be at least 1")
    _validate_condition_pair(mask_condition, source_condition)

    try:
        device = next(pipeline.transformer.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("cannot resolve CogVideoX Transformer execution device") from exc
    if device.type == "meta":
        raise RuntimeError("CogVideoX Transformer is still on the meta device")
    weight_dtype = prompt_embeds.dtype
    mask_condition = mask_condition.to(device=device, dtype=weight_dtype)
    source_condition = source_condition.to(device=device, dtype=weight_dtype)
    if prompt_embeds.shape[0] != source_condition.shape[0]:
        raise ValueError("prompt and condition batch sizes disagree")

    do_cfg = guidance_scale > 1.0
    if do_cfg:
        if negative_prompt_embeds is None:
            raise ValueError("negative_prompt_embeds are required when CFG is enabled")
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError("positive and negative prompt embeddings must share shape")
        model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0).to(
            device=device, dtype=weight_dtype
        )
    else:
        model_prompt_embeds = prompt_embeds.to(device=device, dtype=weight_dtype)

    latents = build_initial_endpoint_latents(
        mask_condition=mask_condition,
        source_condition=source_condition,
        contract=contract,
        generator=generator,
        init_noise_sigma=float(pipeline.scheduler.init_noise_sigma),
    )
    timesteps, effective_steps = retrieve_timesteps(
        pipeline.scheduler, num_inference_steps, device=device
    )
    image_rotary_emb = (
        pipeline._prepare_rotary_positional_embeddings(
            height, width, contract.total_frames, device
        )
        if pipeline.transformer.config.use_rotary_positional_embeddings
        else None
    )
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    old_pred_original_sample = None

    for step_index, timestep in enumerate(timesteps):
        latents = restore_endpoint_conditions(
            latents,
            mask_condition=mask_condition,
            source_condition=source_condition,
            contract=contract,
        )
        model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
        model_input = pipeline.scheduler.scale_model_input(model_input, timestep)
        prediction = pipeline.transformer(
            hidden_states=model_input,
            encoder_hidden_states=model_prompt_embeds,
            timestep=timestep.expand(model_input.shape[0]),
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0].float()

        if do_cfg:
            unconditioned, conditioned = prediction.chunk(2)
            current_guidance = float(guidance_scale)
            if use_dynamic_cfg:
                progress = (effective_steps - float(timestep.item())) / effective_steps
                current_guidance = 1 + guidance_scale * (
                    1 - math.cos(math.pi * progress**5)
                ) / 2
            prediction = unconditioned + current_guidance * (conditioned - unconditioned)
        prediction = zero_condition_prediction(prediction, contract)

        if isinstance(pipeline.scheduler, CogVideoXDPMScheduler):
            previous_timestep = timesteps[step_index - 1] if step_index > 0 else None
            latents, old_pred_original_sample = pipeline.scheduler.step(
                prediction,
                old_pred_original_sample,
                timestep,
                previous_timestep,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )
            if old_pred_original_sample is not None:
                old_pred_original_sample = restore_endpoint_conditions(
                    old_pred_original_sample,
                    mask_condition=mask_condition,
                    source_condition=source_condition,
                    contract=contract,
                )
        else:
            latents = pipeline.scheduler.step(
                prediction,
                timestep,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )[0]

        latents = restore_endpoint_conditions(
            latents.to(dtype=weight_dtype),
            mask_condition=mask_condition,
            source_condition=source_condition,
            contract=contract,
        )

    return latents


@torch.no_grad()
def decode_endpoint_states_individually(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode `[B,F,C,h,w]` through F temporal-length-one VAE calls."""

    if latents.ndim != 5 or not latents.is_floating_point():
        raise ValueError("latents must be a floating [B,F,C,h,w] tensor")
    if not torch.isfinite(latents).all().item():
        raise ValueError("latents contain NaN or infinity")
    scaling_factor = float(vae.config.scaling_factor)
    if not math.isfinite(scaling_factor) or scaling_factor <= 0:
        raise ValueError("VAE scaling_factor must be finite and positive")
    try:
        vae_device = next(vae.parameters()).device
    except (AttributeError, StopIteration):
        vae_device = latents.device
    vae_dtype = getattr(vae, "dtype", latents.dtype)

    decoded_states = []
    for frame_index in range(latents.shape[1]):
        one_state = (
            latents[:, frame_index]
            .unsqueeze(2)
            .to(device=vae_device, dtype=vae_dtype)
            / scaling_factor
        )
        decoded = vae.decode(one_state).sample
        if decoded.ndim != 5 or decoded.shape[2] != 1:
            raise ValueError(
                "single-state CogVideoX decoding must return [B,3,1,H,W]; "
                f"got {tuple(decoded.shape)}"
            )
        decoded_states.append(decoded.squeeze(2))
    return torch.stack(decoded_states, dim=1)
