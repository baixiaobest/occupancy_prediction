from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.VAE_prediction import VAEPredictionDecoder, build_prediction_vae_models


@dataclass(frozen=True)
class VAEDecoderTapBundle:
    decoder: VAEPredictionDecoder
    decoder_context_frames: int
    latent_channels: int
    latent_height: int
    latent_width: int
    num_upsample_blocks: int


def _to_stride_list(raw: Sequence[int | Sequence[int]]) -> list[tuple[int, int]]:
    stride_list: list[tuple[int, int]] = []
    for value in raw:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 2:
                raise ValueError("stride entries must have length 2")
            stride_list.append((int(value[0]), int(value[1])))
        else:
            s = int(value)
            stride_list.append((s, s))
    return stride_list


def _normalize_dynamic_context_tensor(
    x: torch.Tensor,
    *,
    expected_channels: int | None = None,
    expected_time: int | None = None,
    expected_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(
            "dynamic_context must have shape (B, C, T, H, W), "
            f"got {tuple(x.shape)}"
        )
    if expected_channels is not None and int(x.shape[1]) != int(expected_channels):
        raise ValueError(
            f"dynamic_context channel dimension must be {int(expected_channels)}, "
            f"got {int(x.shape[1])}"
        )
    if expected_time is not None and int(x.shape[2]) != int(expected_time):
        raise ValueError(
            f"dynamic_context time dimension must be {int(expected_time)}, "
            f"got {int(x.shape[2])}"
        )
    if expected_hw is not None and tuple(int(v) for v in x.shape[-2:]) != tuple(int(v) for v in expected_hw):
        raise ValueError(
            "dynamic_context HxW mismatch: expected "
            f"{tuple(int(v) for v in expected_hw)}, got {tuple(int(v) for v in x.shape[-2:])}"
        )
    return x.float()


def _normalize_static_context_tensor(
    x: torch.Tensor,
    *,
    expected_channels: int | None = None,
    expected_hw: tuple[int, int] | None = None,
    target_t: int | None = None,
) -> torch.Tensor:
    if x.ndim == 4:
        x = x.unsqueeze(2)
    elif x.ndim != 5:
        raise ValueError(
            "static_map must have shape (B, C, H, W) or (B, C, T, H, W), "
            f"got {tuple(x.shape)}"
        )

    if expected_channels is not None and int(x.shape[1]) != int(expected_channels):
        raise ValueError(
            f"static_map channel dimension must be {int(expected_channels)}, "
            f"got {int(x.shape[1])}"
        )
    if expected_hw is not None and tuple(int(v) for v in x.shape[-2:]) != tuple(int(v) for v in expected_hw):
        raise ValueError(
            "static_map HxW mismatch: expected "
            f"{tuple(int(v) for v in expected_hw)}, got {tuple(int(v) for v in x.shape[-2:])}"
        )

    if target_t is not None:
        static_t = int(x.shape[2])
        target_t = int(target_t)
        if static_t == 1 and target_t > 1:
            x = x.expand(-1, -1, target_t, -1, -1)
        elif static_t != target_t:
            raise ValueError(
                f"static_map time dimension must be 1 or {target_t}, got {static_t}"
            )

    return x.float()


def load_vae_checkpoint_dict(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("VAE checkpoint must be a dict")
    return payload


def infer_dynamic_occupancy_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError(
            "dynamic occupancy shape must be (C, T, H, W), "
            f"got shape={shape}"
        )
    c, t, h, w = shape
    if int(c) <= 0 or int(t) <= 0 or int(h) <= 0 or int(w) <= 0:
        raise ValueError("dynamic occupancy dimensions must be positive")
    return int(c), int(t), int(h), int(w)


def infer_static_occupancy_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) == 3:
        c, h, w = shape
        if int(c) <= 0 or int(h) <= 0 or int(w) <= 0:
            raise ValueError("static occupancy dimensions must be positive")
        return int(c), 1, int(h), int(w)
    if len(shape) == 4:
        c, t, h, w = shape
        if int(c) <= 0 or int(t) <= 0 or int(h) <= 0 or int(w) <= 0:
            raise ValueError("static occupancy dimensions must be positive")
        return int(c), int(t), int(h), int(w)
    raise ValueError(
        "static occupancy shape must be (C, H, W) or (C, T, H, W), "
        f"got shape={shape}"
    )


def state_to_xy_vec(x: torch.Tensor, *, key_name: str) -> torch.Tensor:
    x = x.float().flatten(start_dim=1)
    if int(x.shape[1]) < 2:
        raise ValueError(f"{key_name} must provide at least 2 values per batch")
    return x[:, :2]


def load_vae_decoder_tap_bundle(checkpoint_path: Path) -> VAEDecoderTapBundle:
    ckpt = load_vae_checkpoint_dict(checkpoint_path)
    model_cfg = ckpt.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError("Checkpoint must contain dict key 'model_config'")
    if "decoder" not in ckpt:
        raise ValueError("Checkpoint must contain key 'decoder'")

    latent_channel = int(model_cfg["latent_channel"])
    channels = [int(c) for c in model_cfg["channels"]]
    decoder_downsample_channels = [int(c) for c in model_cfg["decoder_downsample_channels"]]
    decoder_context_latent_channel = int(model_cfg.get("decoder_context_latent_channel", latent_channel))
    static_stem_channels = int(model_cfg["static_stem_channels"])
    velocity_mlp_dim = int(model_cfg.get("velocity_mlp_dim", 16))
    encoder_velocity_condition_channels = int(model_cfg.get("encoder_velocity_condition_channels", 0))
    decoder_velocity_condition_channels = int(model_cfg.get("decoder_velocity_condition_channels", 0))
    decoder_position_mlp_dim = int(model_cfg.get("decoder_position_mlp_dim", 16))
    decoder_position_condition_channels = int(model_cfg.get("decoder_position_condition_channels", 0))

    input_shape = tuple(int(v) for v in model_cfg["input_shape"])
    output_shape = tuple(int(v) for v in model_cfg["output_shape"])
    downsample_strides = _to_stride_list(model_cfg["downsample_strides"])
    upsample_strides = _to_stride_list(model_cfg["upsample_strides"])
    upsample_channels = [int(c) for c in model_cfg["upsample_channels"]]

    decoder_context_frames = int(model_cfg.get("decoder_context_len", input_shape[1]))

    _, decoder = build_prediction_vae_models(
        input_shape=input_shape,
        output_shape=output_shape,
        latent_channel=latent_channel,
        channels=channels,
        decoder_downsample_channels=decoder_downsample_channels,
        decoder_context_latent_channel=decoder_context_latent_channel,
        static_stem_channels=static_stem_channels,
        velocity_mlp_dim=velocity_mlp_dim,
        encoder_velocity_condition_channels=encoder_velocity_condition_channels,
        decoder_velocity_condition_channels=decoder_velocity_condition_channels,
        decoder_position_mlp_dim=decoder_position_mlp_dim,
        decoder_position_condition_channels=decoder_position_condition_channels,
        decoder_context_frames=decoder_context_frames,
        downsample_strides=downsample_strides,
        decoder_context_downsample_strides=downsample_strides,
        upsample_strides=upsample_strides,
        upsample_channels=upsample_channels,
        device="cpu",
    )
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    latent_h = int(input_shape[2])
    latent_w = int(input_shape[3])
    for stride_h, stride_w in downsample_strides:
        latent_h = (latent_h + int(stride_h) - 1) // int(stride_h)
        latent_w = (latent_w + int(stride_w) - 1) // int(stride_w)

    return VAEDecoderTapBundle(
        decoder=decoder,
        decoder_context_frames=int(decoder_context_frames),
        latent_channels=int(latent_channel),
        latent_height=int(latent_h),
        latent_width=int(latent_w),
        num_upsample_blocks=len(upsample_strides),
    )


__all__ = [
    "_to_stride_list",
    "_normalize_dynamic_context_tensor",
    "_normalize_static_context_tensor",
    "VAEDecoderTapBundle",
    "infer_dynamic_occupancy_shape",
    "infer_static_occupancy_shape",
    "load_vae_checkpoint_dict",
    "load_vae_decoder_tap_bundle",
    "state_to_xy_vec",
]
