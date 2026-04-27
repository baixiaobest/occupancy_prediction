from __future__ import annotations

from pathlib import Path

import torch


def load_decoder_context_len_from_checkpoint(checkpoint_path: Path | str) -> int:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError("VAE checkpoint must be a dict")

    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError("VAE checkpoint must contain dict key 'model_config'")

    if "decoder_context_len" not in model_cfg:
        raise ValueError(
            "VAE checkpoint model_config must contain key 'decoder_context_len'"
        )

    context_len = int(model_cfg["decoder_context_len"])
    if context_len <= 0:
        raise ValueError("decoder_context_len must be > 0 in VAE checkpoint")

    return context_len


__all__ = ["load_decoder_context_len_from_checkpoint"]