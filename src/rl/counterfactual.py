from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CounterfactualRolloutBatch:
    """Batched counterfactual rollout outputs.

    Shapes:
    - candidate_velocity_plans: (B, K, H, 2)
    - candidate_position_offsets: (B, K, H, 2)
    - predicted_logits: (B, K, H, 1, H_occ, W_occ)
    - tapped_features: optional (B, K, H, C_tap, H_tap, W_tap)
    """

    candidate_velocity_plans: torch.Tensor
    candidate_position_offsets: torch.Tensor
    predicted_logits: torch.Tensor
    tapped_features: torch.Tensor | None = None

    def flatten_candidates_for_q(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return tensors flattened from (B, K, ...) to (B*K, ...)."""
        batch_size, num_candidates = self.candidate_velocity_plans.shape[:2]
        planned_velocities = self.candidate_velocity_plans.reshape(batch_size * num_candidates, *self.candidate_velocity_plans.shape[2:])
        if self.tapped_features is None:
            return planned_velocities, None
        tapped_features = self.tapped_features.reshape(batch_size * num_candidates, *self.tapped_features.shape[2:])
        return planned_velocities, tapped_features


@torch.no_grad()
def sample_random_velocity_plans(
    current_velocity: torch.Tensor,
    *,
    num_candidates: int,
    horizon: int,
    max_speed: float,
    delta_std: float = 0.25,
    dt: float = 0.1,
    include_current_velocity_candidate: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample smooth random velocity plans around the current velocity.

    Uses a velocity random walk in action space and clamps each step to `max_speed`.
    """
    velocity = torch.as_tensor(current_velocity, dtype=torch.float32)
    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError("current_velocity must have shape (B, 2)")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be > 0")
    if delta_std < 0.0:
        raise ValueError("delta_std must be >= 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    batch_size = int(velocity.shape[0])
    device = velocity.device
    delta_v = torch.randn(
        (batch_size, num_candidates, horizon, 2),
        generator=generator,
        device=device,
        dtype=velocity.dtype,
    ) * float(delta_std) * float(dt)

    plans = velocity[:, None, None, :] + torch.cumsum(delta_v, dim=2)
    if include_current_velocity_candidate:
        plans[:, 0, :, :] = velocity[:, None, :]

    speeds = torch.linalg.vector_norm(plans, dim=-1, keepdim=True)
    scale = torch.clamp(float(max_speed) / speeds.clamp_min(1e-6), max=1.0)
    return plans * scale


@torch.no_grad()
def integrate_velocity_plans(
    velocity_plans: torch.Tensor,
    *,
    dt: float,
    initial_position_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate velocity plans into future position offsets."""
    plans = torch.as_tensor(velocity_plans, dtype=torch.float32)
    if plans.ndim != 4 or plans.shape[-1] != 2:
        raise ValueError("velocity_plans must have shape (B, K, H, 2)")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    offsets = torch.cumsum(plans * float(dt), dim=2)
    if initial_position_offset is None:
        return offsets

    initial = torch.as_tensor(initial_position_offset, dtype=plans.dtype, device=plans.device)
    if initial.ndim == 2 and initial.shape == (plans.shape[0], 2):
        initial = initial[:, None, None, :]
    elif initial.ndim == 3 and initial.shape == (plans.shape[0], plans.shape[1], 2):
        initial = initial[:, :, None, :]
    else:
        raise ValueError(
            "initial_position_offset must have shape (B, 2) or (B, K, 2)"
        )
    return offsets + initial


@torch.no_grad()
def rollout_counterfactual_futures(
    *,
    decoder: nn.Module,
    dynamic_context: torch.Tensor,
    static_map: torch.Tensor,
    candidate_velocity_plans: torch.Tensor,
    latent_channels: int,
    latent_shape: tuple[int, int, int],
    dt: float,
    current_position_offset: torch.Tensor | None = None,
    tap_layer: int | None = None,
    binary_feedback: bool = False,
    threshold: float = 0.5,
    latent_samples: torch.Tensor | None = None,
) -> CounterfactualRolloutBatch:
    """Roll out imagined futures for a batch of candidate action sequences.

    This uses the trained decoder autoregressively, taking the first predicted frame
    at each step and feeding it back into the decoder context.
    """
    context = torch.as_tensor(dynamic_context, dtype=torch.float32)
    plans = torch.as_tensor(candidate_velocity_plans, dtype=torch.float32, device=context.device)

    if context.ndim != 5 or context.shape[1] != 1:
        raise ValueError("dynamic_context must have shape (B, 1, T_ctx, H, W)")
    if plans.ndim != 4 or plans.shape[-1] != 2:
        raise ValueError("candidate_velocity_plans must have shape (B, K, H, 2)")
    if latent_channels <= 0:
        raise ValueError("latent_channels must be > 0")
    if len(latent_shape) != 3 or any(int(v) <= 0 for v in latent_shape):
        raise ValueError("latent_shape must be a 3-tuple of positive ints")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1]")

    batch_size, _, context_frames, _, _ = context.shape
    _, num_candidates, horizon, _ = plans.shape
    device = context.device

    static_x = torch.as_tensor(static_map, dtype=torch.float32, device=device)
    if static_x.ndim not in (4, 5):
        raise ValueError("static_map must have shape (B, 1, H, W) or (B, 1, T, H, W)")
    if static_x.shape[0] != batch_size:
        raise ValueError("static_map batch size must match dynamic_context")

    position_offsets = integrate_velocity_plans(
        plans,
        dt=dt,
        initial_position_offset=current_position_offset,
    )

    flat_batch = batch_size * num_candidates
    rollout_context = (
        context[:, None, ...]
        .expand(-1, num_candidates, -1, -1, -1, -1)
        .reshape(flat_batch, 1, context_frames, context.shape[-2], context.shape[-1])
        .clone()
    )

    if static_x.ndim == 4:
        static_x = static_x[:, None, ...].expand(-1, num_candidates, -1, -1, -1).reshape(flat_batch, *static_x.shape[1:])
    else:
        static_x = static_x[:, None, ...].expand(-1, num_candidates, -1, -1, -1, -1).reshape(flat_batch, *static_x.shape[1:])

    flat_plans = plans.reshape(flat_batch, horizon, 2)
    flat_offsets = position_offsets.reshape(flat_batch, horizon, 2)

    if latent_samples is None:
        z = torch.randn(
            (flat_batch, latent_channels, latent_shape[0], latent_shape[1], latent_shape[2]),
            device=device,
            dtype=context.dtype,
        )
    else:
        z = torch.as_tensor(latent_samples, dtype=torch.float32, device=device)
        expected_shape = (batch_size, num_candidates, latent_channels, latent_shape[0], latent_shape[1], latent_shape[2])
        if tuple(z.shape) != expected_shape:
            raise ValueError(
                f"latent_samples must have shape {expected_shape}, got {tuple(z.shape)}"
            )
        z = z.reshape(flat_batch, latent_channels, latent_shape[0], latent_shape[1], latent_shape[2])

    predicted_logits: torch.Tensor | None = None
    tapped_features: torch.Tensor | None = None

    for step in range(horizon):
        step_velocity = flat_plans[:, step, :]
        step_position_offset = flat_offsets[:, step, :]

        if tap_layer is None:
            logits_full = decoder(
                z,
                rollout_context,
                static_x,
                step_velocity,
                step_position_offset,
            )
            tap_step = None
        else:
            logits_full, tap_step = decoder(
                z,
                rollout_context,
                static_x,
                step_velocity,
                step_position_offset,
                tap_layer=tap_layer,
            )

        logits_step = logits_full[:, :, 0]
        if predicted_logits is None:
            predicted_logits = torch.empty(
                (flat_batch, horizon, *logits_step.shape[1:]),
                dtype=logits_step.dtype,
                device=device,
            )
        predicted_logits[:, step] = logits_step

        if tap_step is not None:
            if tapped_features is None:
                tapped_features = torch.empty(
                    (flat_batch, horizon, *tap_step.shape[1:]),
                    dtype=tap_step.dtype,
                    device=device,
                )
            tapped_features[:, step] = tap_step

        feedback = torch.sigmoid(logits_step)
        if binary_feedback:
            feedback = (feedback >= threshold).to(dtype=feedback.dtype)
        rollout_context = torch.cat([rollout_context[:, :, 1:], feedback.unsqueeze(2)], dim=2)

    if predicted_logits is None:
        raise RuntimeError("counterfactual rollout did not produce any predictions")

    predicted_logits = predicted_logits.reshape(batch_size, num_candidates, horizon, *predicted_logits.shape[2:])
    if tapped_features is not None:
        tapped_features = tapped_features.reshape(batch_size, num_candidates, horizon, *tapped_features.shape[2:])

    return CounterfactualRolloutBatch(
        candidate_velocity_plans=plans,
        candidate_position_offsets=position_offsets,
        predicted_logits=predicted_logits,
        tapped_features=tapped_features,
    )


__all__ = [
    "CounterfactualRolloutBatch",
    "integrate_velocity_plans",
    "rollout_counterfactual_futures",
    "sample_random_velocity_plans",
]
