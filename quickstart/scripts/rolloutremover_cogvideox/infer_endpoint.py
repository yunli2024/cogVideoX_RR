"""Cached-latent inference CLI for CogVideoX5B EndpointRemover."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from cogkit.finetune.diffusion.models.cogvideo.cogvideox_endpoint_t2v.contract import (
    EndpointContract,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_rollout_t2v.dataset import (
    LATENT_CONVENTION,
    PREPROCESSING_CONVENTION,
    TEMPORAL_VAE_POLICY,
)
from cogkit.finetune.diffusion.models.cogvideo.cogvideox_endpoint_t2v.inference import (
    INFERENCE_POLICY,
    SEED_POLICY,
    decode_endpoint_states_individually,
    encode_endpoint_prompt,
    sample_endpoint_latents,
    sample_seed,
)
from cogkit.utils import load_pipeline


SCHEMA_VERSION = 1
DEFAULT_PROMPT = "Remove the masked object and its side effect"
LORA_WEIGHT_NAME = "adapter_model.safetensors"
REQUIRED_TENSORS = {
    "source",
    "gt",
    "mask_sam",
    "mask_check",
    "mask_sam_binary",
    "mask_check_binary",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CogKit-native cached CogVideoX5B EndpointRemover inference"
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--training_contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_ids", nargs="+", required=True)
    parser.add_argument(
        "--mask_condition_kind",
        choices=("mask_check", "mask_sam"),
        default="mask_check",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--use_dynamic_cfg", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--save_states", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dry_run_contract", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _validate_paths(args: argparse.Namespace) -> Path:
    for path, kind in (
        (args.model_path, "dir"),
        (args.lora_path, "dir"),
        (args.cache_dir, "dir"),
        (args.training_contract, "file"),
        (args.manifest, "file"),
    ):
        valid = path.is_dir() if kind == "dir" else path.is_file()
        if not valid:
            raise FileNotFoundError(f"missing {kind}: {path}")
    lora_weight = args.lora_path / LORA_WEIGHT_NAME
    if not lora_weight.is_file():
        raise FileNotFoundError(f"missing CogKit LoRA weight: {lora_weight}")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if len(set(args.sample_ids)) != len(args.sample_ids):
        raise ValueError("--sample_ids must not contain duplicates")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("height and width must be positive and divisible by 16")
    if args.num_inference_steps <= 0 or args.guidance_scale < 1:
        raise ValueError("invalid inference step count or guidance scale")
    return lora_weight


def _load_manifest_records(
    manifest: Path, sample_ids: list[str]
) -> dict[str, dict[str, Any]]:
    requested = set(sample_ids)
    selected: dict[str, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("sample_id"), str):
                raise ValueError(f"invalid manifest record at line {line_number}")
            sample_id = payload["sample_id"]
            if sample_id in requested:
                if sample_id in selected:
                    raise ValueError(f"duplicate requested sample in manifest: {sample_id}")
                selected[sample_id] = payload
    missing = requested.difference(selected)
    if missing:
        raise ValueError(f"requested sample IDs missing from manifest: {sorted(missing)}")
    return selected


def _validate_training_contract(
    args: argparse.Namespace,
    training: dict[str, Any],
    contract: EndpointContract,
) -> None:
    if training.get("method") != "EndpointRemover" or training.get("backbone") != "CogVideoX-5B":
        raise ValueError("training contract is not CogVideoX5B EndpointRemover")
    if training.get("model_path") != str(args.model_path):
        raise ValueError("model path disagrees with training contract")
    if training.get("prompt") != args.prompt:
        raise ValueError("prompt disagrees with training contract")
    topology = training.get("topology")
    if not isinstance(topology, dict):
        raise ValueError("training contract has no endpoint topology")
    expected_topology = contract.to_dict()
    for key in (
        "total_frames",
        "frame_roles",
        "clean_condition_indices",
        "supervised_frame_indices",
        "constructs_intermediate_rollout",
    ):
        if topology.get(key) != expected_topology[key]:
            raise ValueError(f"training endpoint topology disagrees on {key}")
    if training.get("intermediate_rollout", "missing") is not None:
        raise ValueError("training contract unexpectedly declares an intermediate rollout")
    admission = training.get("model_admission")
    if not isinstance(admission, dict):
        raise ValueError("training contract has no model admission")
    expected_shape = [16, args.height // 8, args.width // 8]
    if admission.get("latent_shape") != expected_shape:
        raise ValueError("inference geometry disagrees with training latent shape")
    if abs(float(admission.get("vae_scaling_factor")) - 0.7) > 1e-12:
        raise ValueError("training VAE scaling factor is not 0.7")


def _load_cached_sample(
    *,
    cache_dir: Path,
    sample_id: str,
    model_path: Path,
    height: int,
    width: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"unsafe sample ID: {sample_id!r}")
    tensor_path = cache_dir / f"{sample_id}.pt"
    metadata_path = cache_dir / f"{sample_id}.json"
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing cache pair for {sample_id}")
    metadata = _load_json(metadata_path)
    expected_metadata = {
        "sample_id": sample_id,
        "model_id": str(model_path),
        "pixel_height": height,
        "pixel_width": width,
        "posterior_mode": "mode",
        "preprocessing_convention": PREPROCESSING_CONVENTION,
        "latent_convention": LATENT_CONVENTION,
        "temporal_vae_policy": TEMPORAL_VAE_POLICY,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"cache metadata mismatch for {sample_id}/{key}: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if abs(float(metadata.get("scaling_factor")) - 0.7) > 1e-12:
        raise ValueError(f"cache VAE scaling factor mismatch for {sample_id}")

    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict):
        raise ValueError(f"cache tensor is not a dictionary: {tensor_path}")
    missing = REQUIRED_TENSORS.difference(tensors)
    if missing:
        raise ValueError(f"cache {sample_id} missing {sorted(missing)}")
    expected_shape = (16, height // 8, width // 8)
    for role in ("source", "gt", "mask_sam", "mask_check"):
        value = tensors[role]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            raise ValueError(f"cache {sample_id}/{role} has wrong shape")
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"cache {sample_id}/{role} is not a finite latent")
    return tensors, metadata


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("decoded frame must be [3,H,W]")
    pixels = ((frame.detach().cpu().float().clamp(-1, 1) + 1) / 2).clamp(0, 1)
    array = (pixels.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    lora_weight = _validate_paths(args)
    contract = EndpointContract()
    training = _load_json(args.training_contract)
    _validate_training_contract(args, training, contract)
    manifest_records = _load_manifest_records(args.manifest, args.sample_ids)
    cached = {
        sample_id: _load_cached_sample(
            cache_dir=args.cache_dir,
            sample_id=sample_id,
            model_path=args.model_path,
            height=args.height,
            width=args.width,
        )
        for sample_id in args.sample_ids
    }

    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "task": "CogVideoX5B-EndpointRemover-cached-inference",
        "inference_policy": INFERENCE_POLICY,
        "seed_policy": SEED_POLICY,
        "temporal_vae_policy": TEMPORAL_VAE_POLICY,
        "model_path": str(args.model_path),
        "lora_path": str(args.lora_path),
        "lora_sha256": sha256_file(lora_weight),
        "training_contract": str(args.training_contract),
        "training_contract_sha256": sha256_file(args.training_contract),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "cache_dir": str(args.cache_dir),
        "sample_ids": args.sample_ids,
        "mask_condition_kind": args.mask_condition_kind,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "use_dynamic_cfg": bool(args.use_dynamic_cfg),
        "seed": args.seed,
        "dtype": args.dtype,
        "topology": contract.to_dict(),
        "pure_output": True,
        "paste_back": False,
        "output_blending": False,
        "gt_used_as_model_input": False,
        "save_states": bool(args.save_states),
    }
    if args.dry_run_contract:
        print(json.dumps(run_contract, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the requested device")

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    weight_dtype = _dtype(args.dtype)
    pipeline = load_pipeline(
        str(args.model_path),
        lora_model_id_or_path=str(args.lora_path),
        dtype=weight_dtype,
    ).to(device)
    pipeline.transformer.eval()
    pipeline.text_encoder.eval()
    pipeline.vae.eval()
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()
    if abs(float(pipeline.vae.config.scaling_factor) - 0.7) > 1e-12:
        raise ValueError("loaded VAE scaling factor is not 0.7")

    prompt_embeds, negative_prompt_embeds = encode_endpoint_prompt(
        pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        device=device,
        dtype=weight_dtype,
        max_sequence_length=int(pipeline.transformer.config.max_text_seq_length),
    )
    pipeline.text_encoder.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(args.output_dir / "inference_contract.json", run_contract)
    manifest_output = args.output_dir / "inference_manifest.jsonl"

    for sample_id in args.sample_ids:
        tensors, cache_metadata = cached[sample_id]
        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(exist_ok=False)
        seed_value = sample_seed(args.seed, sample_id)
        generator = torch.Generator(device=device).manual_seed(seed_value)
        mask_condition = tensors[args.mask_condition_kind].unsqueeze(0)
        source_condition = tensors["source"].unsqueeze(0)
        latents = sample_endpoint_latents(
            pipeline,
            contract=contract,
            mask_condition=mask_condition,
            source_condition=source_condition,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            use_dynamic_cfg=args.use_dynamic_cfg,
        )

        generated_indices = list(range(contract.total_frames)) if args.save_states else [2]
        generated_decoded = decode_endpoint_states_individually(
            pipeline.vae,
            latents[:, generated_indices],
        )[0]
        generated_frames = {
            index: _to_pil(generated_decoded[position])
            for position, index in enumerate(generated_indices)
        }
        reference_latents = torch.stack(
            [
                tensors[args.mask_condition_kind],
                tensors["source"],
                tensors["gt"],
            ],
            dim=0,
        ).unsqueeze(0)
        reference_decoded = decode_endpoint_states_individually(
            pipeline.vae,
            reference_latents,
        )[0]
        reference_frames = [_to_pil(frame) for frame in reference_decoded]

        mask_path = sample_dir / "mask_condition.png"
        source_path = sample_dir / "source.png"
        gt_path = sample_dir / "gt.png"
        prediction_path = sample_dir / "prediction.png"
        reference_frames[0].save(mask_path)
        reference_frames[1].save(source_path)
        reference_frames[2].save(gt_path)
        generated_frames[2].save(prediction_path)

        state_paths: list[str] = []
        if args.save_states:
            states_dir = sample_dir / "states"
            states_dir.mkdir()
            for index, role in enumerate(contract.frame_roles):
                frame_path = states_dir / f"{index:02d}_{role}.png"
                generated_frames[index].save(frame_path)
                state_paths.append(str(frame_path.relative_to(args.output_dir)))

        sample_payload = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "sample_seed": seed_value,
            "manifest_record": manifest_records[sample_id],
            "cache_metadata": cache_metadata,
            "mask_condition_kind": args.mask_condition_kind,
            "prediction": str(prediction_path.relative_to(args.output_dir)),
            "source_reference": str(source_path.relative_to(args.output_dir)),
            "mask_reference": str(mask_path.relative_to(args.output_dir)),
            "gt_reference": str(gt_path.relative_to(args.output_dir)),
            "generated_states": state_paths,
            "pure_output": True,
            "paste_back": False,
            "output_blending": False,
            "gt_used_as_model_input": False,
        }
        _write_json_exclusive(sample_dir / "metadata.json", sample_payload)
        with manifest_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample_payload, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "completed",
                "count": len(args.sample_ids),
                "output_dir": str(args.output_dir),
                "manifest": str(manifest_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
