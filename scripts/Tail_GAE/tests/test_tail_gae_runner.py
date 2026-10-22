"""Unit tests for TailGAEConfig, the frozen hyperparameter layer."""

from __future__ import annotations

from dataclasses import asdict, FrozenInstanceError

import pytest

from tail_gae_runner import TailGAEConfig, run_tail_gae_config

FULL_KWARGS = dict(
    alpha=1.0,
    p_tail=0.10,
    train_start="2013-04-01",
    train_end="2024-12-31",
    val_test_start="2025-01-01",
    val_test_end="2026-06-01",
    run_id="1207_Tail_GAE_V2_7Risk_seed_0",
    ppo_timesteps=40_000,
    n_steps=1024,
    batch_size=256,
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.15,
    target_kl=0.015,
    max_grad_norm=0.5,
    net_arch={"pi": [256, 256], "vf": [256, 256]},
)


def test_full_config_values():
    cfg = TailGAEConfig(**FULL_KWARGS)
    for key, value in FULL_KWARGS.items():
        assert getattr(cfg, key) == value


def test_frozen():
    cfg = TailGAEConfig(**FULL_KWARGS)
    with pytest.raises(FrozenInstanceError):
        cfg.alpha = 2.0


def test_defaults_match_1200_baseline():
    cfg = TailGAEConfig(
        alpha=0.0,
        p_tail=0.025,
        train_start="2013-04-01",
        train_end="2024-12-31",
        val_test_start="2025-01-01",
        val_test_end="2026-06-01",
        run_id="x",
    )
    assert cfg.gamma == 0.99
    assert cfg.gae_lambda == 0.95
    assert cfg.learning_rate == 1e-4
    assert cfg.n_steps == 1024
    assert cfg.batch_size == 256
    assert cfg.clip_range == 0.15
    assert cfg.target_kl == 0.015
    assert cfg.max_grad_norm == 0.5
    assert cfg.ppo_timesteps == 40_000
    assert cfg.net_arch == {"pi": [256, 256], "vf": [256, 256]}


def test_asdict_roundtrip():
    cfg = TailGAEConfig(**FULL_KWARGS)
    assert TailGAEConfig(**asdict(cfg)) == cfg


def test_run_tail_gae_config_placeholder():
    assert isinstance(run_tail_gae_config(), dict)
