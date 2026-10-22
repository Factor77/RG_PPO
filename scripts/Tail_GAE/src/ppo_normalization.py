"""Leakage-safe SB3 normalization helpers for rolling PPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any
try:
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    VecNormalize = object  # type: ignore


class FrozenNormalizationEnv:
    """Expose a FinRL environment with frozen VecNormalize statistics."""

    def __init__(self, env: Any, statistics_path: str | Path) -> None:
        self._env = env
        self._statistics_path = Path(statistics_path)

    def get_sb_env(self) -> tuple[Any, Any]:
        try:
            from stable_baselines3.common.vec_env import VecNormalize
        except ImportError as exc:
            raise ImportError("stable_baselines3 VecNormalize is required") from exc
        vec_env, obs = self._env.get_sb_env()
        normalized = VecNormalize.load(str(self._statistics_path), vec_env)
        normalized.training = False
        normalized.norm_reward = False
        return normalized, normalized.normalize_obs(obs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def make_training_normalizer(
    vec_env: Any,
    *,
    normalize_observations: bool,
    normalize_rewards: bool,
    warmup_count: int = 100,
) -> Any:
    """Wrap an environment whose running statistics may update in training.

    During the first *warmup_count* samples, raw observations pass through
    unchanged while RunningMeanStd accumulates statistics.  Once the count
    threshold is reached normalisation engages automatically.

    This avoids the zero-vector observation produced when the very first
    ``obs_rms.update`` sets ``mean≈obs`` and ``var≈0`` (Bug 5).
    """
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise ImportError("stable_baselines3 VecNormalize is required") from exc
    return _WarmedUpVecNormalize(
        vec_env,
        training=True,
        norm_obs=normalize_observations,
        norm_reward=normalize_rewards,
        warmup_count=warmup_count,
    )

class _WarmedUpVecNormalize(VecNormalize):
    """VecNormalize that skips observation normalization during warm-up."""

    def __init__(self, *args: Any, warmup_count: int = 100, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._warmup_count = warmup_count

    def normalize_obs(self, obs: Any) -> Any:
        if self.obs_rms.count < self._warmup_count:
               return obs
        return super().normalize_obs(obs)


def load_frozen_normalizer(vec_env: Any, statistics_path: str | Path) -> Any:
    """Attach one train-fitted normalization state without updating it."""
    try:
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError as exc:
        raise ImportError("stable_baselines3 VecNormalize is required") from exc
    normalized = VecNormalize.load(str(statistics_path), vec_env)
    normalized.training = False
    normalized.norm_reward = False
    return normalized
