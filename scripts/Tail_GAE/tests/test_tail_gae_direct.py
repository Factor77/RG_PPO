"""Unit tests for V4.5/1230 direct-gradient penalty (spec §6.1).

U1 d_i 计算（env 侧）; U2/U3 直通 loss 梯度方向（hinge）; U4 buffer 列对齐;
U5 β=0 逐位等价; U6 bridge 等价.
"""

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from types import SimpleNamespace
from gymnasium import spaces

from risk_state_env import RiskStateEnvWrapper
from tail_gae_buffer import TailGAERolloutBuffer, TailRolloutBufferSamples
from tail_gae_ppo import TailGAEPPO, direct_penalty_loss

STOCK_DIM = 3
REF = 0.85


class _FakeBaseEnv:
    """Minimal base env for RiskStateEnvWrapper.step() 的 d_i 逻辑测试 (U1)."""

    stock_dim = STOCK_DIM

    def __init__(self, states, dates):
        self._states = list(states)
        self._dates = list(dates)
        self.asset_memory = [1_000_000.0, 1_010_000.0]
        self.date_memory = [self._dates[0]]  # 当前日期
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
        self.date_memory.pop()  # 移除旧"当前日期"
        self.date_memory.append(self._dates.pop(0))  # 推进到下一日期
        return self._states.pop(0), 0.0, False, False, {}


def make_state(weights, prices=(100.0,) * STOCK_DIM, cash=500_000.0):
    """构造与 w = p*s/asset 一致的 state（asset = asset_memory[-1]）."""
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


# ---------------------------------------------------------------------------
# U1: d_i 计算（env 侧）
# ---------------------------------------------------------------------------

def test_u1_di_sign_positive_over_limit():
    wrapper = make_wrapper(
        [make_state(weights=(0.5, 0.1, -0.4))], ["d0"], {"d0": 1}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [1, 0, -1]


def test_u1_di_sign_negative_over_limit():
    wrapper = make_wrapper(
        [make_state(weights=(-0.6, 0.05, 0.02))], ["d0"], {"d0": 1}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [-1, 0, 0]


def test_u1_di_zero_when_not_over_limit():
    wrapper = make_wrapper(
        [make_state(weights=(0.05, -0.05, 0.02))], ["d0"], {"d0": 1}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [0, 0, 0]


def test_u1_di_zero_when_not_trigger_day():
    wrapper = make_wrapper(
        [make_state(weights=(0.5, -0.6, 0.7))], ["d0"], {"d0": 0}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [0, 0, 0]


def test_u1_di_resets_to_zero_on_next_step():
    # 触发日超限 → 下一日非触发日归零（last_direct_grad 每步重置）
    wrapper = make_wrapper(
        [make_state(weights=(0.5, 0, 0))] * 2, ["d0", "d1"], {"d0": 1, "d1": 0}
    )
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [1, 0, 0]
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_direct_grad.tolist() == [0, 0, 0]


# ---------------------------------------------------------------------------
# U2/U3: 直通 loss 梯度方向（hinge）
# ---------------------------------------------------------------------------

def test_u2_gradient_pushes_mu_down_for_d_plus_one():
    # ∇_μ L = β·I·d（spec §6.1 U2）: d=+1 → 梯度 +1, 梯度下降把 μ 推向下（减仓）
    mean_actions = th.tensor([[0.5, 0.3]], requires_grad=True)
    d_i = th.tensor([[1.0, 0.0]])
    i_t = th.tensor([[1.0]])
    loss = direct_penalty_loss(mean_actions, d_i, i_t)
    loss.backward()
    g = mean_actions.grad
    assert g[0, 0] > 0  # ∇=+1: 更新方向（-∇）把 μ1 推向下
    assert g[0, 1] == 0.0  # μ2 零梯度


def test_u2_gradient_pushes_mu_up_for_d_minus_one():
    # d=-1 → ∇=-1, 更新方向（-∇）把 μ 推向上（负仓降 |w|）
    mean_actions = th.tensor([[-0.4]], requires_grad=True)
    d_i = th.tensor([[-1.0]])
    i_t = th.tensor([[1.0]])
    loss = direct_penalty_loss(mean_actions, d_i, i_t)
    loss.backward()
    assert mean_actions.grad[0, 0] < 0


def test_u2_hinge_zero_when_mu_crossed_zero():
    mean_actions = th.tensor([[-0.2]], requires_grad=True)
    d_i = th.tensor([[1.0]])
    i_t = th.tensor([[1.0]])
    loss = direct_penalty_loss(mean_actions, d_i, i_t)
    loss.backward()
    assert loss.item() == 0.0
    assert mean_actions.grad[0, 0] == 0.0


def test_u3_zero_contribution_for_non_trigger():
    mean_actions = th.tensor([[0.5], [0.5]], requires_grad=True)
    d_i = th.tensor([[1.0], [1.0]])
    i_t = th.tensor([[0.0], [1.0]])
    loss = direct_penalty_loss(mean_actions, d_i, i_t)
    assert abs(loss.item() - 0.5) < 1e-6  # 只有第二个样本贡献: 0.5/1
    loss.backward()
    assert mean_actions.grad[0, 0] == 0.0  # i_t=0 样本梯度为零
    assert mean_actions.grad[1, 0] > 0.0  # i_t=1 样本 ∇=+1（更新推 μ 向下）


def test_u3_all_non_trigger_loss_zero():
    mean_actions = th.tensor([[0.5], [0.4]], requires_grad=True)
    d_i = th.tensor([[1.0], [1.0]])
    i_t = th.tensor([[0.0], [0.0]])
    loss = direct_penalty_loss(mean_actions, d_i, i_t)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# U4: buffer 列对齐
# ---------------------------------------------------------------------------

def test_u4_buffer_columns_aligned_with_samples():
    n = 8
    dim = 4
    buf = TailGAERolloutBuffer(
        n, spaces.Box(-1, 1, (dim,)), spaces.Box(-1, 1, (2,)),
        device="cpu", gae_lambda=0.95, gamma=0.99, n_envs=1,
        tail_alpha=0.0,
        risk_wrapper=SimpleNamespace(last_I_t=1, last_direct_grad=np.array([1, -1])),
    )
    rng = np.random.default_rng(0)
    for _ in range(n):
        obs = rng.normal(size=(1, dim))
        act = rng.normal(size=(1, 2))
        buf.add(obs, act, 0.0, th.zeros(1), th.zeros(1), -th.ones(1))
    assert buf.i_t[:, 0, 0].tolist() == [1.0] * n
    assert np.all(buf.d_i[:, 0] == [1, -1])
    buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))
    for samples in buf.get(batch_size=4):
        assert isinstance(samples, TailRolloutBufferSamples)
        assert samples.observations.shape[0] == samples.i_t.shape[0] == 4
        assert samples.i_t.shape == (4, 1)
        assert samples.d_i.shape == (4, 2)
        assert np.allclose(samples.d_i.numpy(), np.tile([1, -1], (4, 1)))


def test_u4_buffer_columns_varied_per_step():
    n = 6
    dim = 4
    d_seq = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1])] * 2
    i_seq = [1, 0, 1, 0, 1, 0]
    wrapper = SimpleNamespace()
    buf = TailGAERolloutBuffer(
        n, spaces.Box(-1, 1, (dim,)), spaces.Box(-1, 1, (2,)),
        device="cpu", gae_lambda=0.95, gamma=0.99, n_envs=1,
        tail_alpha=0.0, risk_wrapper=wrapper,
    )
    for t in range(n):
        wrapper.last_I_t = i_seq[t]
        wrapper.last_direct_grad = d_seq[t]
        buf.add(np.zeros((1, dim)), np.zeros((1, 2)), 0.0, th.zeros(1), th.zeros(1), -th.ones(1))
    buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))
    for samples in buf.get(batch_size=None):
        got_i = samples.i_t[:, 0].numpy()
        got_d = samples.d_i.numpy()
        assert samples.observations.shape[0] == n
        assert sorted(got_i.tolist()) == sorted(i_seq)
        rows = got_d.tolist()
        assert all(list(d) in rows for d in d_seq)  # 每行 d 都在样本中出现


# ---------------------------------------------------------------------------
# U5: β=0 时与 SB3 PPO 逐位等价
# ---------------------------------------------------------------------------

def test_u5_zero_beta_bitwise_equal_to_sb3():
    """同一模型（同 seed 构造）+ 同一份 rollout 数据 → train() 后参数逐位相同.

    手工构造 buffer（与 learn 采样链无关, 消除隐式 RNG 依赖）; 两个 train()
    前分别重置 np RNG（get() 的 permutation 相同）.
    """
    from test_tail_gae_ppo import make_vec, ppo_kwargs

    np.random.seed(0)
    wrapper, vec = make_vec()
    m_direct = TailGAEPPO("MlpPolicy", vec, direct_penalty_beta=0.0, **ppo_kwargs())
    np.random.seed(0)
    wrapper2, vec2 = make_vec()
    m_sb3 = PPO("MlpPolicy", vec2, **ppo_kwargs())

    n_steps = ppo_kwargs()["n_steps"]
    obs_dim = vec.observation_space.shape[0]
    for m in (m_direct, m_sb3):
        m._logger = configure(None, [])  # 未跑 learn 时无 logger, train() 需要
        buf = m.rollout_buffer
        rng = np.random.default_rng(7)  # 每模型独立 rng → 数据完全相同
        for _ in range(n_steps):
            obs = rng.normal(size=(1, obs_dim))
            act = rng.normal(size=(1, STOCK_DIM))
            buf.add(obs, act, float(rng.normal()), float(rng.integers(0, 2)),
                    th.zeros(1), th.zeros(1))
        buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))
    np.random.seed(0)
    m_direct.train()
    np.random.seed(0)
    m_sb3.train()
    for (n1, p1), (n2, p2) in zip(m_direct.policy.named_parameters(), m_sb3.policy.named_parameters()):
        assert n1 == n2
        assert th.equal(p1.detach().cpu(), p2.detach().cpu()), f"{n1} 参数不一致"


# ---------------------------------------------------------------------------
# U6: bridge/1200 等价（无 risk wrapper）
# ---------------------------------------------------------------------------

def test_u6_bridge_columns_zero_and_trains():
    from tail_gae_env import DownsideBreadthStockTradingEnv
    from test_tail_gae_ppo import make_panel, env_kwargs, ppo_kwargs

    np.random.seed(0)
    base = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    vec = DummyVecEnv([lambda: base])
    m = TailGAEPPO("MlpPolicy", vec, direct_penalty_beta=0.0, **ppo_kwargs())
    m.learn(total_timesteps=2 * ppo_kwargs()["n_steps"], progress_bar=False)
    # buffer 列恒零（无 wrapper 时 add() 不写, 初始零）
    assert m.rollout_buffer.i_t_flat is not None
    assert float(m.rollout_buffer.i_t_flat.sum()) == 0.0
    assert float(m.rollout_buffer.d_i_flat.sum()) == 0.0


def test_u6_wrapper_with_zero_beta_columns_zero():
    from test_tail_gae_ppo import make_vec, ppo_kwargs

    np.random.seed(0)
    wrapper, vec = make_vec()
    m = TailGAEPPO("MlpPolicy", vec, risk_wrapper=wrapper, direct_penalty_beta=0.0, **ppo_kwargs())
    m.learn(total_timesteps=2 * ppo_kwargs()["n_steps"], progress_bar=False)
    # wrapper 未配直通 ref（runner 传 None）→ d 列恒零
    assert float(m.rollout_buffer.d_i_flat.sum()) == 0.0
