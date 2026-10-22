"""Frozen hyperparameter layer for Tail_GAE V2.

Data-only module: centralizes every experiment hyperparameter in one frozen
dataclass so values cannot drift between runs. No computation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class TailGAEConfig:
    alpha: float                      # Tail-GAE strength; 0.0 = standard GAE (1200), 0.5 = 1205, 1.0 = 1207/1210
    p_tail: float                     # tail threshold quantile; 0.025 (V1 legacy), 0.10 (V2 two-layer)
    train_start: str                  # "2013-04-01"
    train_end: str                    # "2024-12-31"
    val_test_start: str               # "2025-01-01"
    val_test_end: str                 # "2026-06-01"
    run_id: str                       # e.g. "1207_Tail_GAE_V2_7Risk_seed_0"
    ppo_timesteps: int = 40_000
    n_steps: int = 1024
    batch_size: int = 256
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.15
    target_kl: float = 0.015
    max_grad_norm: float = 0.5
    net_arch: dict = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


def run_tail_gae_config() -> Dict[str, Any]:
    """Placeholder for the V2 training entry point.

    Actual training is driven by notebook_runner (built by a separate agent);
    this function will return the full run config dict once that lands.
    """
    return {}
