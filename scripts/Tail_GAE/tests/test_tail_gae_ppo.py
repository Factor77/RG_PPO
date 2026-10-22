"""Unit tests for tail_gae_ppo.TailGAEPPO (spec §5.8)."""

import gymnasium
import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from risk_state_env import RiskStateEnvWrapper
from tail_gae_buffer import TailGAERolloutBuffer
from tail_gae_env import DownsideBreadthStockTradingEnv
from tail_gae_ppo import TailGAEPPO

STOCK_DIM = 3
N_DAYS = 200
# finrl.config.INDICATORS: the FinRL sell/buy path indexes the tech block.
INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
STATE_SPACE = 1 + 2 * STOCK_DIM + len(INDICATORS) * STOCK_DIM
N_STEPS = 32


def make_dates(n_days=N_DAYS):
    return [f"2020-01-{d + 1:02d}" if d < 31 else f"2020-02-{d - 30:02d}" for d in range(n_days)]


def make_panel(seed=0, n_stocks=STOCK_DIM, n_days=N_DAYS):
    np.random.seed(seed)
    rows = []
    for day in make_dates(n_days):
        for tic in range(n_stocks):
            row = {
                "date": day,
                "tic": f"T{tic}",
                "close": float(np.random.uniform(50.0, 200.0)),
                "volume": float(np.random.uniform(1e5, 1e6)),
            }
            for ind in INDICATORS:
                row[ind] = float(np.random.uniform(-1.0, 1.0))
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["date", "tic"], ignore_index=True)
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
        tech_indicator_list=INDICATORS,
        turbulence_threshold=None,
        mode="train",
        iteration=0,
        model_name="ppo",
        print_verbosity=0,
    )


class _VecEnvFacade(RiskStateEnvWrapper, gymnasium.Env):
    """DummyVecEnv needs a gymnasium.Env; the frozen wrapper is a plain class."""

    def close(self):
        return None


def make_vec(names=("current_drawdown", "worst20")):
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    wrapper = _VecEnvFacade(
        env,
        list(names),
        i_by_date={d: 0 for d in make_dates()},
        worst20_by_date={d: 0.09 for d in make_dates()},
    )
    wrapper.action_space = wrapper.env.action_space
    return wrapper, DummyVecEnv([lambda: wrapper])


def ppo_kwargs(**overrides):
    kw = dict(n_steps=N_STEPS, batch_size=N_STEPS, n_epochs=1, seed=0, verbose=0, device="cpu")
    kw.update(overrides)
    return kw


def test_main_path_uses_tail_gae_buffer():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO(
        "MlpPolicy", vec, tail_alpha=1.0,
        risk_wrapper=wrapper, **ppo_kwargs(),
    )
    assert model.rollout_buffer.__class__ is TailGAERolloutBuffer


def test_main_path_forwards_tail_params_to_buffer():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO(
        "MlpPolicy", vec, tail_alpha=0.5,
        risk_wrapper=wrapper, tilt_shift=True, **ppo_kwargs(),
    )
    buf = model.rollout_buffer
    assert buf.tail_alpha == 0.5
    assert buf.risk_wrapper is wrapper
    assert buf.tilt_shift is True


def test_fallback_path_keeps_external_buffer():
    np.random.seed(0)
    wrapper, vec = make_vec()
    external = RolloutBuffer(
        N_STEPS, vec.observation_space, vec.action_space, "cpu", 0.95, 0.99, n_envs=1,
    )
    model = TailGAEPPO("MlpPolicy", vec, rollout_buffer=external, **ppo_kwargs())
    assert model.rollout_buffer is external


def test_default_alpha_zero_constructs_without_tail_args():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO("MlpPolicy", vec, **ppo_kwargs())
    assert model.rollout_buffer.tail_alpha == 0.0
    assert model.rollout_buffer.risk_wrapper is None


def test_tail_alpha_without_risk_wrapper_fails_fast():
    np.random.seed(0)
    wrapper, vec = make_vec()
    with pytest.raises(AssertionError, match="risk_wrapper"):
        TailGAEPPO("MlpPolicy", vec, tail_alpha=1.0, **ppo_kwargs())


def test_learn_smoke_with_live_wrapper_protocol():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO(
        "MlpPolicy", vec, tail_alpha=1.0,
        risk_wrapper=wrapper, **ppo_kwargs(),
    )
    model.learn(total_timesteps=2 * N_STEPS)
    assert model.num_timesteps == 2 * N_STEPS
    computed = model.rollout_buffer._computed_I
    assert computed is not None
    assert len(computed) == N_STEPS
