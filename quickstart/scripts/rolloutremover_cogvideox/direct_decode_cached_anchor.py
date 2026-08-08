"""Directly decode the deterministic CogVideoX RolloutRemover F7 anchor.

This diagnostic bypasses the Transformer, LoRA, scheduler, and inference loop.
It reconstructs the exact training-time semantic anchor from cached keyframe
latents and the production deterministic seed contract, then decodes it with
the base CogVideoX VAE under controlled dtype/tiling settings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLCogVideoX
from PIL import Image, ImageDraw


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--cache_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u63(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().cpu().float()
    return {
        "shape": list(x.shape),
        "dtype": str(value.dtype),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def masked_stats(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    x = value.detach().cpu().float()
    m = mask.detach().cpu().bool().expand_as(x)
    inside = x[m]
    outside = x[~m]
    return {
        "inside_mean": float(inside.mean()),
        "inside_std": float(inside.std()),
        "outside_mean": float(outside.mean()),
        "outside_std": float(outside.std()),
    }


def masked_neighbor_correlation(
    value: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    x = value.detach().cpu().float()
    m = mask.detach().cpu().bool()

    def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.flatten()
        b = b.flatten()
        a = a - a.mean()
        b = b - b.mean()
        denominator = torch.sqrt((a.square().sum()) * (b.square().sum()))
        return float((a * b).sum() / denominator.clamp_min(1e-12))

    valid_x = (m[:, :, 1:] & m[:, :, :-1]).expand(x.shape[0], -1, -1)
    valid_y = (m[:, 1:, :] & m[:, :-1, :]).expand(x.shape[0], -1, -1)
    return {
        "horizontal": correlation(x[:, :, 1:][valid_x], x[:, :, :-1][valid_x]),
        "vertical": correlation(x[:, 1:, :][valid_y], x[:, :-1, :][valid_y]),
    }


def to_uint8(frame: torch.Tensor) -> np.ndarray:
    pixels = ((frame.detach().cpu().float().clamp(-1, 1) + 1) / 2).clamp(0, 1)
    return (
        pixels.permute(1, 2, 0).numpy().clip(0, 1) * 255.0
    ).round().astype(np.uint8)


def grid_score(image: np.ndarray, mask: np.ndarray, period: int = 8) -> dict[str, float]:
    pixels = image.astype(np.float32)
    mask = mask.astype(bool)
    gx = np.abs(pixels[:, 1:] - pixels[:, :-1]).mean(axis=2)
    gy = np.abs(pixels[1:] - pixels[:-1]).mean(axis=2)
    valid_x = mask[:, 1:] & mask[:, :-1]
    valid_y = mask[1:] & mask[:-1]
    boundary_x = np.zeros_like(valid_x)
    boundary_y = np.zeros_like(valid_y)
    boundary_x[:, period - 1 :: period] = True
    boundary_y[period - 1 :: period, :] = True

    def ratio(gradient: np.ndarray, valid: np.ndarray, boundary: np.ndarray) -> float:
        on = gradient[valid & boundary]
        off = gradient[valid & ~boundary]
        if on.size == 0 or off.size == 0:
            return float("nan")
        return float(on.mean() / max(float(off.mean()), 1e-12))

    horizontal = ratio(gx, valid_x, boundary_x)
    vertical = ratio(gy, valid_y, boundary_y)
    return {
        "period_pixels": period,
        "horizontal_boundary_over_interior": horizontal,
        "vertical_boundary_over_interior": vertical,
        "mean_boundary_over_interior": float(np.nanmean([horizontal, vertical])),
    }


@torch.no_grad()
def decode_one(
    vae: AutoencoderKLCogVideoX,
    latent: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if latent.ndim != 3:
        raise ValueError(f"expected [C,H,W], got {tuple(latent.shape)}")
    scaling_factor = float(vae.config.scaling_factor)
    one_state = (
        latent.unsqueeze(0).unsqueeze(2).to(device=device, dtype=vae.dtype)
        / scaling_factor
    )
    decoded = vae.decode(one_state).sample
    if tuple(decoded.shape[:3]) != (1, 3, 1):
        raise ValueError(f"unexpected decoded shape: {tuple(decoded.shape)}")
    return decoded[0, :, 0].detach().cpu().float()


def load_vae(
    model_path: Path, dtype: torch.dtype, device: torch.device
) -> AutoencoderKLCogVideoX:
    vae = AutoencoderKLCogVideoX.from_pretrained(
        str(model_path),
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    vae.eval()
    vae.enable_slicing()
    if not math.isclose(float(vae.config.scaling_factor), 0.7, abs_tol=1e-12):
        raise ValueError("CogVideoX VAE scaling factor is not 0.7")
    return vae


def save_image(path: Path, frame: torch.Tensor) -> np.ndarray:
    array = to_uint8(frame)
    Image.fromarray(array).save(path)
    return array


def make_montage(output_dir: Path, names: list[str]) -> None:
    images = [Image.open(output_dir / name).convert("RGB") for name in names]
    label_height = 28
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width * 2, (height + label_height) * 3), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(zip(names, images)):
        x = (index % 2) * width
        y = (index // 2) * (height + label_height)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 7), name, fill="black")
    canvas.save(output_dir / "comparison.png")


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if not args.cache_path.is_file() or not args.metadata_path.is_file():
        raise FileNotFoundError("cache tensor/metadata pair is incomplete")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is unavailable")

    metadata = json.loads(args.metadata_path.read_text(encoding="utf-8"))
    if metadata.get("sample_id") != args.sample_id:
        raise ValueError("metadata sample_id mismatch")
    if metadata.get("model_id") != str(args.model_path):
        raise ValueError("metadata model_id mismatch")
    if metadata.get("posterior_mode") != "mode":
        raise ValueError("unexpected cache posterior convention")

    tensors = torch.load(args.cache_path, map_location="cpu", weights_only=True)
    required = {"source", "gt", "mask_check", "mask_check_binary"}
    missing = required.difference(tensors)
    if missing:
        raise ValueError(f"cache missing tensors: {sorted(missing)}")
    source = tensors["source"].contiguous()
    gt = tensors["gt"].contiguous()
    mask_condition = tensors["mask_check"].contiguous()
    mask_binary = tensors["mask_check_binary"].contiguous()
    if tuple(source.shape) != (16, 60, 90):
        raise ValueError(f"unexpected source shape: {tuple(source.shape)}")
    if tuple(mask_binary.shape) != (1, 60, 90):
        raise ValueError(f"unexpected mask shape: {tuple(mask_binary.shape)}")
    if not torch.all((mask_binary == 0) | (mask_binary == 1)).item():
        raise ValueError("mask is not binary")

    anchor_seed = stable_u63("rollout_anchor_v1", args.global_seed, args.sample_id)
    generator = torch.Generator(device="cpu").manual_seed(anchor_seed)
    iid_noise = torch.randn(source.shape, generator=generator, dtype=source.dtype)
    exact_anchor = torch.where(mask_binary > 0.5, iid_noise, source)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "source": source,
            "gt": gt,
            "mask_condition": mask_condition,
            "mask_binary": mask_binary,
            "iid_noise": iid_noise,
            "exact_training_anchor_f7": exact_anchor,
        },
        args.output_dir / "diagnostic_latents.pt",
    )
    pixel_mask = (
        F.interpolate(mask_binary.unsqueeze(0).float(), size=(480, 720), mode="nearest")
        .squeeze()
        .numpy()
        .astype(bool)
    )
    Image.fromarray((pixel_mask.astype(np.uint8) * 255)).save(
        args.output_dir / "mask_binary_nearest.png"
    )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    frames: dict[str, torch.Tensor] = {}
    vae_bf16 = load_vae(args.model_path, torch.bfloat16, device)
    vae_bf16.disable_tiling()
    frames["source_bf16_nontiled"] = decode_one(vae_bf16, source, device)
    frames["mask_condition_bf16_nontiled"] = decode_one(
        vae_bf16, mask_condition, device
    )
    frames["anchor_exact_bf16_nontiled"] = decode_one(
        vae_bf16, exact_anchor, device
    )
    frames["full_iid_bf16_nontiled"] = decode_one(vae_bf16, iid_noise, device)
    vae_bf16.enable_tiling()
    frames["anchor_exact_bf16_tiled"] = decode_one(vae_bf16, exact_anchor, device)
    del vae_bf16
    gc.collect()
    torch.cuda.empty_cache()

    vae_fp32 = load_vae(args.model_path, torch.float32, device)
    vae_fp32.disable_tiling()
    frames["anchor_exact_fp32_nontiled"] = decode_one(
        vae_fp32, exact_anchor, device
    )
    del vae_fp32
    gc.collect()
    torch.cuda.empty_cache()

    arrays = {
        name: save_image(args.output_dir / f"{name}.png", frame)
        for name, frame in frames.items()
    }
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": args.sample_id,
        "anchor_seed": anchor_seed,
        "latent": {
            "source": tensor_stats(source),
            "iid_noise": tensor_stats(iid_noise),
            "exact_anchor": tensor_stats(exact_anchor),
            "iid_noise_masked": masked_stats(iid_noise, mask_binary),
            "exact_anchor_masked": masked_stats(exact_anchor, mask_binary),
            "iid_noise_mask_neighbor_correlation": masked_neighbor_correlation(
                iid_noise, mask_binary
            ),
            "exact_anchor_mask_neighbor_correlation": masked_neighbor_correlation(
                exact_anchor, mask_binary
            ),
        },
        "decoded_grid_scores": {
            name: grid_score(array, pixel_mask, period=8)
            for name, array in arrays.items()
        },
        "decoded_ab": {
            "bf16_tiled_vs_nontiled_mae_minus1_plus1": float(
                (
                    frames["anchor_exact_bf16_tiled"]
                    - frames["anchor_exact_bf16_nontiled"]
                )
                .abs()
                .mean()
            ),
            "fp32_vs_bf16_nontiled_mae_minus1_plus1": float(
                (
                    frames["anchor_exact_fp32_nontiled"]
                    - frames["anchor_exact_bf16_nontiled"]
                )
                .abs()
                .mean()
            ),
        },
    }
    contract = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "cogvideox_cached_anchor_direct_decode_v1",
        "sample_id": args.sample_id,
        "global_seed": args.global_seed,
        "anchor_seed": anchor_seed,
        "model_path": str(args.model_path),
        "cache_path": str(args.cache_path),
        "cache_sha256": sha256_file(args.cache_path),
        "metadata_path": str(args.metadata_path),
        "metadata_sha256": sha256_file(args.metadata_path),
        "script_sha256": sha256_file(Path(__file__)),
        "latent_contract": "torch.where(mask_check_binary, iid_N01, source_latent)",
        "temporal_decode": "independent_T1",
        "transformer_used": False,
        "scheduler_used": False,
        "lora_used": False,
        "pixel_noise_used": False,
        "dtype_ab": ["bf16", "fp32"],
        "tiling_ab": [False, True],
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_montage(
        args.output_dir,
        [
            "source_bf16_nontiled.png",
            "mask_condition_bf16_nontiled.png",
            "anchor_exact_bf16_nontiled.png",
            "anchor_exact_bf16_tiled.png",
            "anchor_exact_fp32_nontiled.png",
            "full_iid_bf16_nontiled.png",
        ],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
