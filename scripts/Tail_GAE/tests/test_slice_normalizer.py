"""Tests for slice_normalizer.SliceVecNormalize and friends."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from slice_normalizer import FrozenSliceNormalizationEnv, SliceVecNormalize, load_slice_normalizer

BASE_DIM = 61
N_RISK = 7
DIM = BASE_DIM + N_RISK


class _FakeEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._obs = np.zeros(DIM, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self._obs, {}

    def step(self, action):
        return self._obs, 0.0, False, False, {}


def _make_env() -> DummyVecEnv:
    return DummyVecEnv([lambda: _FakeEnv()])


def _random_obs(n: int = 1) -> np.ndarray:
    base = (np.random.randn(n, BASE_DIM) * 2.0).astype(np.float32)
    risk = (np.random.randn(n, N_RISK) * 7.0 + 3.0).astype(np.float32)
    return np.concatenate([base, risk], axis=1)


def test_risk_block_pass_through():
    normalizer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    for _ in range(5):
        normalizer.step(np.zeros((1, 1)))
    obs = _random_obs(3)
    out = normalizer.normalize_obs(obs)
    assert np.array_equal(out[:, BASE_DIM:], obs[:, BASE_DIM:])


def test_base_block_z_score():
    np.random.seed(0)
    normalizer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    normalizer.obs_rms.update((np.random.randn(200, BASE_DIM) * 2.0).astype(np.float32))
    obs = _random_obs(200)
    out = normalizer.normalize_obs(obs)
    z = out[:, :BASE_DIM]
    assert abs(z.mean()) < 0.3
    assert abs(z.std() - 1.0) < 0.3
    expected = np.clip(
        (obs[:, :BASE_DIM] - normalizer.obs_rms.mean) / np.sqrt(normalizer.obs_rms.var + normalizer.epsilon),
        -normalizer.clip_obs,
        normalizer.clip_obs,
    )
    assert np.allclose(z, expected, atol=1e-5)


def test_stats_isolation():
    normalizer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    for _ in range(10):
        normalizer.step(np.zeros((1, 1)))
    assert normalizer.obs_rms.mean.shape == (BASE_DIM,)
    assert normalizer.obs_rms.var.shape == (BASE_DIM,)


def test_eval_consistency(tmp_path):
    np.random.seed(0)
    trainer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    for _ in range(50):
        trainer.obs_rms.update((np.random.randn(32, BASE_DIM) * 2.0).astype(np.float32))
    path = tmp_path / "train_stats.pkl"
    trainer.save(str(path))
    obs = _random_obs(4)
    train_out = trainer.normalize_obs(obs)
    frozen = load_slice_normalizer(str(path), _make_env(), base_dim=BASE_DIM)
    assert not frozen.training
    assert not frozen.norm_reward
    frozen_out = frozen.normalize_obs(obs)
    assert np.array_equal(frozen_out[:, BASE_DIM:], train_out[:, BASE_DIM:])
    assert np.array_equal(frozen_out[:, :BASE_DIM], train_out[:, :BASE_DIM])


def test_warmup():
    np.random.seed(0)
    normalizer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM)
    obs = _random_obs(2)
    assert np.array_equal(normalizer.normalize_obs(obs), obs)
    normalizer.obs_rms.update((np.random.randn(200, BASE_DIM) * 2.0).astype(np.float32))
    out = normalizer.normalize_obs(obs)
    assert not np.array_equal(out, obs)


def test_wiring_smoke(tmp_path):
    np.random.seed(0)
    trainer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    trainer.obs_rms.update((np.random.randn(200, BASE_DIM) * 2.0).astype(np.float32))
    path = tmp_path / "stats.pkl"
    trainer.save(str(path))
    wrapped = FrozenSliceNormalizationEnv(_FakeEnv(), str(path), base_dim=BASE_DIM)
    vec_env, obs = wrapped.get_sb_env()
    assert obs.shape == (1, DIM)
    assert wrapped.observation_space.shape == (DIM,)


def test_reset_updates_stats_with_base_block_only():
    # The parent's reset() updates obs_rms with the FULL obs; the slice
    # override must update with the base block only (risk block excluded).
    np.random.seed(3)
    normalizer = SliceVecNormalize(_make_env(), base_dim=BASE_DIM, warmup_count=0)
    # Force the fake env to return non-trivial obs
    normalizer.venv.envs[0]._obs = np.ones(DIM, dtype=np.float32) * 5.0
    normalizer.reset()
    assert normalizer.obs_rms.mean.shape == (BASE_DIM,)
    assert normalizer.obs_rms.var.shape == (BASE_DIM,)
    assert np.allclose(normalizer.obs_rms.mean, 5.0, atol=1e-3)
    # Risk block still passes through unchanged after reset
    obs = _random_obs(2)
    out = normalizer.normalize_obs(obs)
    assert np.array_equal(out[:, BASE_DIM:], obs[:, BASE_DIM:])
