"""VecNormalize wrapper that z-scores only the first base_dim observation
coordinates and passes the tail-risk block through unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize


class SliceVecNormalize(VecNormalize):
    """Normalize obs[:, :base_dim] only; obs[:, base_dim:] passes through bit-identically."""

    def __init__(self, venv: Any, *, base_dim: int, warmup_count: int = 100, **kwargs: Any) -> None:
        super().__init__(venv, **kwargs)
        assert 1 <= base_dim <= self.observation_space.shape[0]
        # Re-create stats so they track ONLY the base block
        self.obs_rms = RunningMeanStd(shape=(base_dim,))
        self.base_dim = base_dim
        self.warmup_count = warmup_count

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        assert isinstance(obs, np.ndarray)  # Box obs only
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            # Update stats with the base block only (parent updates the full obs)
            self.obs_rms.update(obs[:, : self.base_dim])
        return self.normalize_obs(obs)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        # Raw pass-through during warm-up (avoids the zero-vector bug of the
        # first obs_rms.update, mirroring ppo_normalization._WarmedUpVecNormalize)
        if self.obs_rms.count < self.warmup_count:
            return obs
        # Support 1-D observations (e.g. terminal_observation in infos)
        if isinstance(obs, np.ndarray) and obs.ndim == 1:
            return self.normalize_obs(obs[None, :])[0]
        assert isinstance(obs, np.ndarray) and obs.ndim == 2 and obs.shape[1] >= self.base_dim
        obs_ = obs.copy()
        obs_[:, : self.base_dim] = self._normalize_obs(obs_[:, : self.base_dim], self.obs_rms)
        return obs_.astype(np.float32)

    def step_wait(self) -> tuple[Any, Any, Any, Any]:
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, np.ndarray)  # Box obs only
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            # Update stats with the base block only
            self.obs_rms.update(obs[:, : self.base_dim])

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations (DummyVecEnv stores the env's raw
        # step return there, which is a plain list for FinRL-style envs)
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                to = infos[idx]["terminal_observation"]
                if not isinstance(to, np.ndarray):
                    to = np.asarray(to)
                infos[idx]["terminal_observation"] = self.normalize_obs(to)

        self.returns[dones] = 0
        return obs, rewards, dones, infos


def load_slice_normalizer(path: str | Path, venv: Any, *, base_dim: int | None = None) -> SliceVecNormalize:
    """Restore a saved SliceVecNormalize with venv attached and updates disabled.

    SB3's VecNormalize.load unpickles the original class instance, so base_dim is
    preserved automatically; it does NOT touch training/norm_reward flags.
    """
    normalizer = SliceVecNormalize.load(str(path), venv)
    normalizer.training = False
    normalizer.norm_reward = False
    if base_dim is not None and base_dim != normalizer.base_dim:
        normalizer.base_dim = base_dim
        if normalizer.obs_rms.mean.shape != (base_dim,):
            normalizer.obs_rms = RunningMeanStd(shape=(base_dim,))
    return normalizer


class FrozenSliceNormalizationEnv:
    """Expose an env with frozen SliceVecNormalize statistics."""

    def __init__(self, env: Any, statistics_path: str | Path, *, base_dim: int) -> None:
        self._env = env
        self._statistics_path = Path(statistics_path)
        self._base_dim = base_dim

    def get_sb_env(self) -> tuple[Any, Any]:
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv([lambda: self._env])
        normalized = load_slice_normalizer(self._statistics_path, vec_env, base_dim=self._base_dim)
        return normalized, normalized.normalize_obs(vec_env.reset())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
