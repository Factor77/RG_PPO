"""Unit tests for tail_gae_env.DownsideBreadthStockTradingEnv."""

import numpy as np
import pandas as pd
import pytest

from tail_gae_env import (
    DOWNSIDE_BREADTH_PILOT,
    RETURN_PPO,
    RISK_OBSERVATION_FIELDS,
    SUPPORTED_EXPERIMENTS,
    DownsideBreadthStockTradingEnv,
    get_risk_observation_dim,
)

STOCK_DIM = 3
N_DAYS = 15
STATE_SPACE = 1 + 2 * STOCK_DIM


def make_panel(seed=0, n_stocks=STOCK_DIM, n_days=N_DAYS):
    np.random.seed(seed)
    dates = [f"2020-01-{d + 1:02d}" for d in range(n_days)]
    rows = []
    for day in dates:
        for tic in range(n_stocks):
            rows.append(
                {
                    "date": day,
                    "tic": f"T{tic}",
                    "close": float(np.random.uniform(50.0, 200.0)),
                    "volume": float(np.random.uniform(1e5, 1e6)),
                }
            )
    df = pd.DataFrame(rows).sort_values(["date", "tic"], ignore_index=True)
    # Mirror FinRL data_split: index is the factorized day code, repeated per
    # tic, so env.df.loc[day] selects the whole day block.
    df.index = df["date"].factorize()[0]
    return df


def env_kwargs():
    return dict(
        stock_dim=STOCK_DIM,
        hmax=100,
        initial_amount=1_000_000,
        buy_cost_pct=[0.001] * STOCK_DIM,
        sell_cost_pct=[0.001] * STOCK_DIM,
        reward_scaling=1e-4,
        state_space=STATE_SPACE,
        action_space=STOCK_DIM,
        tech_indicator_list=[],
        turbulence_threshold=None,
        mode="train",
        iteration=0,
        model_name="ppo",
        print_verbosity=0,
    )


def test_construct_pins_return_reward():
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    assert env.reward_mode == "return"


def test_legacy_dsb_params_ignored():
    signal_table = pd.DataFrame(
        {
            "date": [f"2020-01-{d + 1:02d}" for d in range(N_DAYS)],
            "risk_score": np.random.default_rng(1).uniform(0.0, 1.0, N_DAYS),
        }
    )
    env = DownsideBreadthStockTradingEnv(
        make_panel(),
        signal_table=signal_table,
        lambda_breadth=0.3,
        use_breadth_observation=True,
        **env_kwargs(),
    )
    assert env.reward_mode == "return"


def test_reward_mode_pinned_to_return():
    with pytest.raises(ValueError):
        DownsideBreadthStockTradingEnv(make_panel(), reward_mode="mda", **env_kwargs())
    with pytest.raises(ValueError):
        DownsideBreadthStockTradingEnv(make_panel(), reward_mode="aust_original", **env_kwargs())


def test_risk_observation_asserted_off():
    with pytest.raises(AssertionError):
        DownsideBreadthStockTradingEnv(make_panel(), use_risk_observation=True, **env_kwargs())


def test_round_trip_no_v1_risk_fields():
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    obs, info = env.reset()
    assert len(obs) == STATE_SPACE
    assert env.observation_space.shape == (STATE_SPACE,)
    for _ in range(3):
        result = env.step(np.zeros(STOCK_DIM))
        assert isinstance(result, tuple) and len(result) == 5
        obs, reward, terminal, truncated, info = result
        assert len(obs) == STATE_SPACE
        assert np.isfinite(reward)


def test_module_constants():
    assert RETURN_PPO == "return"
    assert SUPPORTED_EXPERIMENTS == ("1200", "1205", "1207", "1210", "1215", "1216", "1217", "1218")
    assert isinstance(DOWNSIDE_BREADTH_PILOT, str)
    assert len(RISK_OBSERVATION_FIELDS) == 5
    assert get_risk_observation_dim() == 5
