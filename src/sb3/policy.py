from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.VAE_prediction import VAEPredictionDecoder, build_prediction_vae_models


def _build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation_fn: type[torch.nn.Module],
) -> tuple[torch.nn.Module, int]:
    layers: list[torch.nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        dim = int(hidden_dim)
        if dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        layers.append(torch.nn.Linear(last_dim, dim))
        layers.append(activation_fn())
        last_dim = dim

    if not layers:
        return torch.nn.Identity(), int(input_dim)
    return torch.nn.Sequential(*layers), int(last_dim)


def _product(shape: Sequence[int]) -> int:
    out = 1
    for v in shape:
        out *= int(v)
    return int(out)


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


class OccupancyFusionFeaturesExtractor(BaseFeaturesExtractor):
    """Fuse map observations and vector observations into one latent tensor.

    Expected dict observation keys by default:
    - dynamic_context: dynamic occupancy context maps.
    - static_map: static occupancy map.
    - all remaining keys are flattened and treated as vector features.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        *,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        conv_channels: Sequence[int] = (16, 32, 64),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: int = 1,
        activation_fn: type[torch.nn.Module] = torch.nn.ReLU,
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("OccupancyFusionFeaturesExtractor requires a Dict observation space")
        if dynamic_key not in observation_space.spaces:
            raise ValueError(f"Missing dynamic occupancy key '{dynamic_key}' in observation space")
        if static_key not in observation_space.spaces:
            raise ValueError(f"Missing static occupancy key '{static_key}' in observation space")

        self.dynamic_key = str(dynamic_key)
        self.static_key = str(static_key)
        self.vector_keys = tuple(
            key for key in observation_space.spaces.keys() if key not in {self.dynamic_key, self.static_key}
        )

        dynamic_shape = tuple(int(v) for v in observation_space.spaces[self.dynamic_key].shape)
        static_shape = tuple(int(v) for v in observation_space.spaces[self.static_key].shape)

        dynamic_channels, dynamic_t, dynamic_h, dynamic_w = self._infer_dynamic_shape(dynamic_shape)
        static_channels, static_t, static_h, static_w = self._infer_static_shape(static_shape)
        if dynamic_h != static_h or dynamic_w != static_w:
            raise ValueError(
                "dynamic and static occupancy maps must have matching HxW, "
                f"got dynamic=({dynamic_h}, {dynamic_w}) and static=({static_h}, {static_w})"
            )
        if static_t not in (1, dynamic_t):
            raise ValueError(
                "static occupancy time dimension must be 1 or match dynamic horizon, "
                f"got static_t={static_t}, dynamic_t={dynamic_t}"
            )

        self.dynamic_channels = int(dynamic_channels)
        self.static_channels = int(static_channels)
        self.dynamic_time_steps = int(dynamic_t)

        # Pack (C, T) into channels for Conv2D, mirroring VAE time-to-channel conditioning.
        static_effective_t = int(dynamic_t)
        map_in_channels = int((self.dynamic_channels * self.dynamic_time_steps) + (self.static_channels * static_effective_t))
        vector_dim = sum(
            _product(tuple(int(v) for v in observation_space.spaces[key].shape)) for key in self.vector_keys
        )

        super().__init__(observation_space=observation_space, features_dim=1)

        conv_layers: list[torch.nn.Module] = []
        in_ch = map_in_channels
        for out_ch in conv_channels:
            out_dim = int(out_ch)
            if out_dim <= 0:
                raise ValueError("conv channels must be positive")
            conv_layers.append(
                torch.nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_dim,
                    kernel_size=int(conv_kernel_size),
                    stride=int(conv_stride),
                    padding=int(conv_padding),
                )
            )
            conv_layers.append(activation_fn())
            in_ch = out_dim
        self.map_encoder = torch.nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, map_in_channels, dynamic_h, dynamic_w)
            encoded = self.map_encoder(dummy)
            map_latent_dim = int(encoded.flatten(start_dim=1).shape[1])

        self.map_height = int(dynamic_h)
        self.map_width = int(dynamic_w)
        self._features_dim = int(vector_dim + map_latent_dim)

    @staticmethod
    def _infer_dynamic_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        # Dynamic occupancy is expected as (C, T, H, W) in observation space.
        if len(shape) != 4:
            raise ValueError(
                "dynamic occupancy shape must be (C, T, H, W), "
                f"got shape={shape}"
            )
        c, t, h, w = shape
        if int(c) <= 0 or int(t) <= 0 or int(h) <= 0 or int(w) <= 0:
            raise ValueError("dynamic occupancy dimensions must be positive")
        return int(c), int(t), int(h), int(w)

    @staticmethod
    def _infer_static_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        # Static occupancy can be (C, H, W) or (C, T, H, W).
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

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        dynamic = _normalize_dynamic_context_tensor(
            observations[self.dynamic_key],
            expected_channels=self.dynamic_channels,
            expected_time=self.dynamic_time_steps,
            expected_hw=(self.map_height, self.map_width),
        )
        static = _normalize_static_context_tensor(
            observations[self.static_key],
            expected_channels=self.static_channels,
            expected_hw=(self.map_height, self.map_width),
            target_t=int(dynamic.shape[2]),
        )

        b, c_dyn, t_ctx, h, w = dynamic.shape
        _, c_static, _, _, _ = static.shape
        dynamic_2d = dynamic.reshape(b, c_dyn * t_ctx, h, w)
        static_2d = static.reshape(b, c_static * t_ctx, h, w)
        map_input = torch.cat([static_2d, dynamic_2d], dim=1)
        map_latent = self.map_encoder(map_input).flatten(start_dim=1)

        if self.vector_keys:
            vector_features = torch.cat(
                [observations[key].float().flatten(start_dim=1) for key in self.vector_keys],
                dim=1,
            )
            return torch.cat([vector_features, map_latent], dim=1)
        return map_latent


class VAEDecoderTapFeaturesExtractor(BaseFeaturesExtractor):
    """Use a frozen VAE decoder tap feature as occupancy map latent.

    The extractor loads a decoder checkpoint once, keeps decoder weights frozen,
    and concatenates tapped map features with flattened vector observations.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        *,
        vae_checkpoint: str,
        tap_layer: int | None = None,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        velocity_key: str = "current_velocity",
        position_key: str = "goal_position",
    ) -> None:
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("VAEDecoderTapFeaturesExtractor requires a Dict observation space")

        self.dynamic_key = str(dynamic_key)
        self.static_key = str(static_key)
        self.velocity_key = str(velocity_key)
        self.position_key = str(position_key)

        required_keys = [self.dynamic_key, self.static_key, self.velocity_key, self.position_key]
        for key in required_keys:
            if key not in observation_space.spaces:
                raise ValueError(f"Missing key '{key}' in observation space")

        self.vector_keys = tuple(
            key for key in observation_space.spaces.keys() if key not in {self.dynamic_key, self.static_key}
        )
        vector_dim = sum(
            _product(tuple(int(v) for v in observation_space.spaces[key].shape)) for key in self.vector_keys
        )

        super().__init__(observation_space=observation_space, features_dim=1)

        (
            self.decoder,
            self.decoder_context_frames,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            num_upsample_blocks,
        ) = self._load_decoder_from_checkpoint(Path(vae_checkpoint))

        self.register_buffer(
            "fixed_z",
            torch.randn(
                (1, self.latent_channels, 1, self.latent_height, self.latent_width),
                dtype=torch.float32,
            ),
            persistent=True,
        )

        if tap_layer is None:
            self.tap_layer = int(num_upsample_blocks)
        else:
            self.tap_layer = int(tap_layer)

        for param in self.decoder.parameters():
            param.requires_grad = False
        self.decoder.eval()

        with torch.no_grad():
            dynamic_shape = tuple(int(v) for v in observation_space.spaces[self.dynamic_key].shape)
            static_shape = tuple(int(v) for v in observation_space.spaces[self.static_key].shape)
            dynamic_dummy = torch.zeros((1,) + dynamic_shape, dtype=torch.float32)
            static_dummy = torch.zeros((1,) + static_shape, dtype=torch.float32)
            dynamic_dummy = _normalize_dynamic_context_tensor(dynamic_dummy, expected_channels=1)
            static_dummy = _normalize_static_context_tensor(static_dummy, expected_channels=1)
            velocity_dummy = self._state_to_vec(
                torch.zeros((1,) + tuple(int(v) for v in observation_space.spaces[self.velocity_key].shape), dtype=torch.float32),
                key_name=self.velocity_key,
            )
            position_dummy = self._state_to_vec(
                torch.zeros((1,) + tuple(int(v) for v in observation_space.spaces[self.position_key].shape), dtype=torch.float32),
                key_name=self.position_key,
            )
            tapped = self.decoder(
                self.fixed_z,
                dynamic_dummy,
                static_dummy,
                velocity_dummy,
                position_dummy,
                tap_layer=self.tap_layer,
                tap_only=True,
            )
            map_latent_dim = int(tapped.flatten(start_dim=1).shape[1])

        self._features_dim = int(vector_dim + map_latent_dim)

    @staticmethod
    def _load_checkpoint(path: Path) -> dict:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("VAE checkpoint must be a dict")
        return payload

    def _load_decoder_from_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[VAEPredictionDecoder, int, int, int, int, int]:
        ckpt = self._load_checkpoint(checkpoint_path)
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

        return (
            decoder,
            int(decoder_context_frames),
            int(latent_channel),
            int(latent_h),
            int(latent_w),
            len(upsample_strides),
        )

    @staticmethod
    def _state_to_vec(x: torch.Tensor, *, key_name: str) -> torch.Tensor:
        x = x.float().flatten(start_dim=1)
        if int(x.shape[1]) < 2:
            raise ValueError(f"{key_name} must provide at least 2 values per batch")
        return x[:, :2]

    def train(self, mode: bool = True) -> VAEDecoderTapFeaturesExtractor:
        super().train(mode)
        # Keep decoder in eval mode even when policy switches to train mode.
        self.decoder.eval()
        return self

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        dynamic = _normalize_dynamic_context_tensor(observations[self.dynamic_key], expected_channels=1)
        static = _normalize_static_context_tensor(observations[self.static_key], expected_channels=1)
        velocity_vec = self._state_to_vec(observations[self.velocity_key], key_name=self.velocity_key)
        position_vec = self._state_to_vec(observations[self.position_key], key_name=self.position_key)

        z = self.fixed_z.expand(dynamic.shape[0], -1, -1, -1, -1).to(
            device=dynamic.device,
            dtype=dynamic.dtype,
        )

        with torch.no_grad():
            tapped = self.decoder(
                z,
                dynamic,
                static,
                velocity_vec,
                position_vec,
                tap_layer=self.tap_layer,
                tap_only=True,
            )
            map_latent = tapped.flatten(start_dim=1)

        if self.vector_keys:
            vector_features = torch.cat(
                [observations[key].float().flatten(start_dim=1) for key in self.vector_keys],
                dim=1,
            )
            return torch.cat([vector_features, map_latent], dim=1)
        return map_latent


class OccupancyActorNetwork(torch.nn.Module):
    """Actor MLP over fused [vector features, encoded occupancy map]."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.net, self.output_dim = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OccupancyCriticNetwork(torch.nn.Module):
    """Critic MLP over fused [vector features, encoded occupancy map]."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: Sequence[int],
        activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.net, self.output_dim = _build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation_fn=activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OccupancyExtractor(torch.nn.Module):
    """Actor-critic extractor composed of explicit actor and critic networks."""

    def __init__(
        self,
        *,
        feature_dim: int,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        actor_activation_fn: type[torch.nn.Module],
        critic_activation_fn: type[torch.nn.Module],
    ) -> None:
        super().__init__()
        self.policy_net = OccupancyActorNetwork(
            input_dim=feature_dim,
            hidden_dims=actor_hidden_dims,
            activation_fn=actor_activation_fn,
        )
        self.value_net = OccupancyCriticNetwork(
            input_dim=feature_dim,
            hidden_dims=critic_hidden_dims,
            activation_fn=critic_activation_fn,
        )
        self.latent_dim_pi = self.policy_net.output_dim
        self.latent_dim_vf = self.value_net.output_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class OccupancyActorCriticPolicy(ActorCriticPolicy):
    """PPO policy with occupancy-map fusion + explicit actor/critic MLP heads."""

    def __init__(
        self,
        *args,
        actor_hidden_dims: Sequence[int] = (64, 64),
        critic_hidden_dims: Sequence[int] = (64, 64),
        actor_activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        critic_activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        dynamic_key: str = "dynamic_context",
        static_key: str = "static_map",
        velocity_key: str = "current_velocity",
        position_key: str = "goal_position",
        map_extractor_type: str = "conv",
        vae_checkpoint: str | None = None,
        vae_tap_layer: int | None = None,
        map_conv_channels: Sequence[int] = (16, 32, 64),
        map_conv_kernel_size: int = 3,
        map_conv_stride: int = 2,
        map_conv_padding: int = 1,
        map_activation_fn: type[torch.nn.Module] = torch.nn.ReLU,
        **kwargs,
    ) -> None:
        self.actor_hidden_dims = tuple(int(dim) for dim in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(dim) for dim in critic_hidden_dims)
        self.actor_activation_fn = actor_activation_fn
        self.critic_activation_fn = critic_activation_fn

        kwargs.setdefault("ortho_init", True)
        kwargs.setdefault("share_features_extractor", True)

        features_extractor_kwargs = dict(kwargs.pop("features_extractor_kwargs", {}))

        extractor_type = str(map_extractor_type).lower()
        if "features_extractor_class" not in kwargs:
            if extractor_type == "conv":
                kwargs["features_extractor_class"] = OccupancyFusionFeaturesExtractor
                features_extractor_kwargs.setdefault("dynamic_key", str(dynamic_key))
                features_extractor_kwargs.setdefault("static_key", str(static_key))
                features_extractor_kwargs.setdefault("conv_channels", tuple(int(v) for v in map_conv_channels))
                features_extractor_kwargs.setdefault("conv_kernel_size", int(map_conv_kernel_size))
                features_extractor_kwargs.setdefault("conv_stride", int(map_conv_stride))
                features_extractor_kwargs.setdefault("conv_padding", int(map_conv_padding))
                features_extractor_kwargs.setdefault("activation_fn", map_activation_fn)
            elif extractor_type == "vae_tap":
                if vae_checkpoint is None:
                    raise ValueError("vae_checkpoint is required when map_extractor_type='vae_tap'")
                kwargs["features_extractor_class"] = VAEDecoderTapFeaturesExtractor
                features_extractor_kwargs.setdefault("vae_checkpoint", str(vae_checkpoint))
                features_extractor_kwargs.setdefault("tap_layer", vae_tap_layer)
                features_extractor_kwargs.setdefault("dynamic_key", str(dynamic_key))
                features_extractor_kwargs.setdefault("static_key", str(static_key))
                features_extractor_kwargs.setdefault("velocity_key", str(velocity_key))
                features_extractor_kwargs.setdefault("position_key", str(position_key))
            else:
                raise ValueError(f"Unsupported map_extractor_type: {map_extractor_type}")

        kwargs["features_extractor_kwargs"] = features_extractor_kwargs

        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = OccupancyExtractor(
            feature_dim=self.features_dim,
            actor_hidden_dims=self.actor_hidden_dims,
            critic_hidden_dims=self.critic_hidden_dims,
            actor_activation_fn=self.actor_activation_fn,
            critic_activation_fn=self.critic_activation_fn,
        )


__all__ = [
    "OccupancyActorCriticPolicy",
    "OccupancyActorNetwork",
    "OccupancyCriticNetwork",
    "OccupancyExtractor",
    "OccupancyFusionFeaturesExtractor",
    "VAEDecoderTapFeaturesExtractor",
]
