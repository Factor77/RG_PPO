"""V4.13 P0（spec Part 7.1）: 方向标签生命周期修复的回归测试。

历史缺陷：RiskStateEnvWrapper.step() 在连续触发日只覆盖当日超限股票的
last_direct_grad，未先清零——触发日→触发日、股票由超限→未超限时旧 d_i 残留。
修复：每步先清零，再仅对当步 over 股票写入 ±1。
"""

import numpy as np
import pytest
import torch as th
from types import SimpleNamespace

import pandas as pd

from risk_state_env import RiskStateEnvWrapper

STOCK_DIM = 3
REF = 0.85


class _FakeBaseEnv:
    """Minimal base env for the wrapper's d_i lifecycle tests（同 U1 夹具口径）。"""

    stock_dim = STOCK_DIM

    def __init__(self, states, dates):
        self._states = list(states)
        self._dates = list(dates)
        self.asset_memory = [1_000_000.0, 1_010_000.0]
        self.date_memory = [self._dates[0]]
        self.reward_adapter = SimpleNamespace(
            accounting_state=SimpleNamespace(recent_portfolio_returns=[])
        )
        self.df = pd.DataFrame(
            [{"date": d, "tic": f"T{i}", "close": 100.0} for d in dates for i in range(2)]
        )
        obs_dim = 1 + 2 * STOCK_DIM + 8 * STOCK_DIM
        self.observation_space = SimpleNamespace(
            shape=(obs_dim,),
            low=np.zeros(obs_dim),
            high=np.ones(obs_dim),
            dtype=np.float32,
        )
        self.action_space = SimpleNamespace(shape=(STOCK_DIM,), dtype=np.float32)

    def step(self, actions):
        self.date_memory.pop()
        self.date_memory.append(self._dates.pop(0))
        return self._states.pop(0), 0.0, False, False, {}


def make_state(weights, prices=(100.0,) * STOCK_DIM, cash=500_000.0):
    asset = 1_010_000.0
    state = [cash]
    state.extend(prices)
    state.extend([w * asset / p for p, w in zip(prices, weights)])
    state.extend([0.0] * (STOCK_DIM * 8))
    return state


def make_wrapper(states, dates, i_by_date):
    wrapper = RiskStateEnvWrapper(
        _FakeBaseEnv(states, dates),
        ["current_drawdown"],
        i_by_date=i_by_date,
        worst20_by_date={d: 0.09 for d in dates},
        direct_penalty_ref=REF,
    )
    wrapper._assemble = lambda state: np.asarray(state, dtype=float)
    wrapper.n_risk_dims = 0
    return wrapper


# ① 连续两天均触发，股票从超限→未超限：第二天 d_i 必须为 0（历史缺陷回归）
def test_consecutive_trigger_days_stale_label_is_cleared():
    wrapper = make_wrapper(
        [make_state(weights=(0.5, 0.0, 0.0)),   # d0: 股 1 超限 → d=[1,0,0]
         make_state(weights=(0.02, 0.0, 0.0))],  # d1: 股 1 已回到阈值内
        ["d0", "d1"],
        {"d0": 1, "d1": 1},
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [1, 0, 0]
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [0, 0, 0], \
        "P0 回归失败：连续触发日由超限→未超限后 d_i 未清零"


# ② 触发→不触发：全向量归零
def test_trigger_then_non_trigger_zeroes_all():
    wrapper = make_wrapper(
        [make_state(weights=(0.5, -0.6, 0.0))] * 2,
        ["d0", "d1"],
        {"d0": 1, "d1": 0},
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [1, -1, 0]
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [0, 0, 0]


# ③ 同一步只有 over=True 的股票可取非零标签
def test_only_over_stocks_can_have_nonzero_label():
    wrapper = make_wrapper(
        [make_state(weights=(0.5, 0.02, -0.6))], ["d0"], {"d0": 1}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [1, 0, -1]
    # 非零位置与超限掩码一一对应
    w_ref = REF / STOCK_DIM
    weights = (0.5, 0.02, -0.6)
    for i, w in enumerate(weights):
        over = abs(w) > w_ref
        nonzero = wrapper.last_direct_grad[i] != 0
        assert nonzero == over, f"stock {i}: label={wrapper.last_direct_grad[i]}, over={over}"


# ④ β=0 时固定 seed 更新轨迹与无 4C 实现等价（learn smoke, 逐位一致）
def test_beta_zero_trajectory_equals_no_4c_bitwise():
    from test_tail_gae_ppo import (
        _VecEnvFacade,
        env_kwargs,
        make_dates,
        make_panel,
        make_vec,
        ppo_kwargs,
        N_STEPS,
    )
    from tail_gae_env import DownsideBreadthStockTradingEnv
    from tail_gae_ppo import TailGAEPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def build(ref):
        np.random.seed(0)
        env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
        wrapper = _VecEnvFacade(
            env,
            ["current_drawdown", "worst20"],
            i_by_date={d: 1 for d in make_dates()},
            worst20_by_date={d: 0.09 for d in make_dates()},
            direct_penalty_ref=ref,
        )
        wrapper.action_space = wrapper.env.action_space
        vec = DummyVecEnv([lambda: wrapper])
        model = TailGAEPPO(
            "MlpPolicy", vec, tail_alpha=1.0, risk_wrapper=wrapper,
            direct_penalty_beta=0.0, use_mu_ref=True, **ppo_kwargs(),
        )
        model.learn(total_timesteps=2 * N_STEPS)
        return model

    model_with_ref = build(0.85)   # β=0, 4C 配置在但权重为零
    model_no_4c = build(None)      # 4C 完全关闭
    for (n1, p1), (n2, p2) in zip(model_with_ref.policy.state_dict().items(),
                                  model_no_4c.policy.state_dict().items()):
        assert n1 == n2
        assert th.equal(p1, p2), f"param {n1} diverged: β=0 与无 4C 应逐位等价"
