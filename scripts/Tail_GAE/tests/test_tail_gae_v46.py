"""V4.6 Step 3C/Step 4 单测（spec §7.1 U2 扩展 / U7 / U9）。

U2 扩展: hinge 达标形式边界——μ ≤ μ_ref 时梯度为零;
U7: adaptive direct-penalty coefficient（自适应直通惩罚系数）λ 更新——
    Ĉ > ε 增、Ĉ < ε 减、clip 到 [0, λ_max]、每 train() 只更新一次、
    无触发 rollout → Ĉ=0 → 按 −η·ε 衰减、constraint_cost 列与 i_t/d_i 同索引对齐;
U9: 状态依赖 μ_ref 解析——100 股/目标 80 股/hmax=100 → μ_ref = −0.2（负号断言）、
    pre-action 对齐（env.step 前捕获）、欠仓可买为正、d_i^exec 成交瞬间口径.
"""

import numpy as np
import pandas as pd
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.logger import configure
from types import SimpleNamespace

from risk_state_env import RiskStateEnvWrapper
from tail_gae_buffer import TailGAERolloutBuffer
from tail_gae_ppo import TailGAEPPO, direct_penalty_loss
from test_tail_gae_ppo import make_vec, ppo_kwargs

STOCK_DIM = 3
REF = 0.85
N_STEPS = 32


# ---------------------------------------------------------------------------
# U2 扩展: 达标形式边界（spec §7.1 U2 / §5.1）
# ---------------------------------------------------------------------------

def test_u2_target_form_zero_gradient_below_mu_ref():
    # μ < μ_ref → max(0, d·(μ−μ_ref)) = 0 → loss = 0 且梯度为零（达标即停）.
    # 恰在边界（μ == μ_ref）: loss = 0; 梯度是 hinge 在折点处的次梯度
    # （PyTorch 取 1.0, 与 ReLU(0) 同约定——测度零, 不影响机制）, 只断言 loss.
    mu_ref = th.tensor([[0.2]])
    for mu_val in (0.1, 0.0, -0.5):
        mean_actions = th.tensor([[mu_val]], requires_grad=True)
        loss = direct_penalty_loss(mean_actions, th.tensor([[1.0]]), th.tensor([[1.0]]), mu_ref)
        assert loss.item() == 0.0
        loss.backward()
        assert mean_actions.grad[0, 0] == 0.0
    mean_actions = th.tensor([[0.2]], requires_grad=True)
    loss = direct_penalty_loss(mean_actions, th.tensor([[1.0]]), th.tensor([[1.0]]), mu_ref)
    assert loss.item() == 0.0


def test_u2_target_form_gradient_pushes_toward_mu_ref():
    # μ > μ_ref → ∇ = +d: 更新方向（−∇）把 μ 推向 μ_ref 即停, 不再推到 0.
    mean_actions = th.tensor([[0.5]], requires_grad=True)
    mu_ref = th.tensor([[0.2]])
    loss = direct_penalty_loss(mean_actions, th.tensor([[1.0]]), th.tensor([[1.0]]), mu_ref)
    assert np.isclose(loss.item(), 0.3)  # 0.5 − 0.2
    loss.backward()
    assert mean_actions.grad[0, 0] == pytest.approx(1.0)
    # d=−1（负仓）: max(0, −(μ−μ_ref)), μ 低于 μ_ref → ∇=−1 推 μ 向上（回达标）
    mean_actions2 = th.tensor([[-0.7]], requires_grad=True)
    loss2 = direct_penalty_loss(mean_actions2, th.tensor([[-1.0]]), th.tensor([[1.0]]), mu_ref)
    assert np.isclose(loss2.item(), 0.9)  # −(−0.7 − 0.2)
    loss2.backward()
    assert mean_actions2.grad[0, 0] == pytest.approx(-1.0)


def test_u2_mu_ref_none_keeps_legacy_form():
    # mu_ref=None → 与 v4.5 原式 max(0, d·μ) 完全一致（默认关闭, 回归等价）
    mean_actions = th.tensor([[0.5, -0.3]], requires_grad=True)
    d_i = th.tensor([[1.0, -1.0]])
    i_t = th.tensor([[1.0]])
    legacy = direct_penalty_loss(mean_actions, d_i, i_t)
    target = direct_penalty_loss(mean_actions, d_i, i_t, None)
    assert legacy.item() == target.item() == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# U7: λ 更新（spec §7.1 U7 / §4.3）
# ---------------------------------------------------------------------------

class _LambdaStub:
    """U7: 每步可控的 wrapper 桩——i_t/d_i/constraint_cost/mu_ref 同一索引写入."""

    def __init__(self, n, i_seq, cc_seq, d_seq=None, mu_ref_seq=None):
        self.names = ["current_drawdown"]
        self._risk_block_history = [np.zeros(1, dtype=np.float32)] * n
        self.i_seq = list(i_seq)
        self.cc_seq = list(cc_seq)
        self.d_seq = d_seq if d_seq is not None else [np.zeros(STOCK_DIM, dtype=int)] * n
        self.mu_ref_seq = mu_ref_seq if mu_ref_seq is not None else [np.zeros(STOCK_DIM)] * n
        self._t = -1
        self.last_I_t = None
        self.last_direct_grad = np.zeros(STOCK_DIM, dtype=int)
        self.last_constraint_cost = 0.0
        self.last_mu_ref = np.zeros(STOCK_DIM)

    def step(self):
        self._t += 1
        self.last_I_t = self.i_seq[self._t]
        self.last_direct_grad = np.asarray(self.d_seq[self._t], dtype=int)
        self.last_constraint_cost = float(self.cc_seq[self._t])
        self.last_mu_ref = np.asarray(self.mu_ref_seq[self._t], dtype=float)


def _lambda_model(stub, **overrides):
    """构造 λ 配置模型（手工填 buffer, 不走 env 采样链, 消除隐式 RNG 依赖）."""
    np.random.seed(0)
    wrapper, vec = make_vec()
    kw = dict(
        risk_wrapper=stub,
        direct_penalty_beta=0.0,
        direct_penalty_ref=REF,
    )
    kw.update(ppo_kwargs())
    kw.update(overrides)
    m = TailGAEPPO("MlpPolicy", vec, **kw)
    m._logger = configure(None, [])
    return m


def _fill_lambda_buffer(model, stub, seed=7):
    buf = model.rollout_buffer
    obs_dim = model.observation_space.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(N_STEPS):
        stub.step()
        buf.add(
            rng.normal(size=(1, obs_dim)),
            rng.normal(size=(1, STOCK_DIM)),
            float(rng.normal()),
            0.0,
            th.zeros(1),
            th.zeros(1),
        )
    buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))


def test_u7_lambda_rises_when_cost_above_epsilon():
    # Ĉ=0.10 > ε=0.02, η=0.5 → λ ← 0 + 0.5·0.08 = 0.04
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.0, lambda_eta=0.5,
                      lambda_max=1.0, constraint_epsilon=0.02)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == pytest.approx(0.5 * (0.10 - 0.02), abs=1e-9)
    assert np.isclose(m.logger.name_to_value["train/constraint_cost"], 0.10)
    assert np.isclose(m.logger.name_to_value["train/lambda"], m._lambda)


def test_u7_lambda_decays_when_cost_below_epsilon():
    # Ĉ=0（无触发）< ε=0.02 → λ ← 0.10 − 0.1·0.02 = 0.098（软预算释放力度）
    stub = _LambdaStub(N_STEPS, [0] * N_STEPS, [0.0] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.10, lambda_eta=0.1,
                      lambda_max=1.0, constraint_epsilon=0.02)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == pytest.approx(0.10 - 0.1 * 0.02, abs=1e-9)


def test_u7_lambda_no_trigger_rollout_decays_by_eta_eps():
    # 明确断言"无触发 rollout → Ĉ=0 → λ 按 −η·ε 衰减"（spec §4.3 更新契约）
    stub = _LambdaStub(N_STEPS, [0] * N_STEPS, [0.0] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.30, lambda_eta=0.4,
                      lambda_max=1.0, constraint_epsilon=0.05)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == pytest.approx(0.30 - 0.4 * 0.05, abs=1e-9)


def test_u7_lambda_clipped_to_lambda_max():
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.20] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.9, lambda_eta=1.0,
                      lambda_max=1.0, constraint_epsilon=0.0)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == pytest.approx(1.0, abs=1e-12)  # 0.9+0.2 → clip 1.0


def test_u7_lambda_clipped_at_zero():
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.01, lambda_eta=1.0,
                      lambda_max=1.0, constraint_epsilon=0.5)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == 0.0  # 0.01 + (0.10−0.5) < 0 → clip 0


def test_u7_lambda_updated_once_per_train():
    # n_epochs=2、batch=16 < n_steps=32: 若按 minibatch 更新, 同一批 rollout 的
    # 违反会被重复累计（4 minibatch × 2 epoch = 8 次 → λ=0.8）; 按 spec 每
    # train() 一次 → 只更新一次 → λ = 0.1
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.0, lambda_eta=1.0,
                      lambda_max=10.0, constraint_epsilon=0.0,
                      n_epochs=2, batch_size=16)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == pytest.approx(0.10, abs=1e-6)  # float32 存储容差


def test_u7_lambda_disabled_by_default_no_new_logs():
    # 默认（全部 None/0）= 关闭: λ 不动、不新增日志、行为等价现有（U5 守门）
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS)
    m = _lambda_model(stub)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert m._lambda == 0.0
    assert "train/constraint_cost" not in m.logger.name_to_value
    assert "train/lambda" not in m.logger.name_to_value


def test_u7_lambda_coefficient_used_when_beta_zero():
    # direct_penalty_beta=0（β 通道关闭）但 λ 激活（lambda_eta=0.0 冻结,
    # λ_init=0.5）→ 直通分支必须进入并记录 train/direct_loss → 系数来自 λ 而非 β
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS,
                       d_seq=[np.ones(STOCK_DIM, dtype=int)] * N_STEPS)
    m = _lambda_model(stub, lambda_init=0.5, lambda_eta=0.0,
                      lambda_max=1.0, constraint_epsilon=0.0)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert "train/direct_loss" in m.logger.name_to_value
    assert m._lambda == pytest.approx(0.5, abs=1e-12)  # η=0 → 冻结


def test_u4_mu_ref_flows_into_train_with_use_mu_ref():
    # use_mu_ref=True 时 train() 从 buffer 新列读 μ_ref（plumbing smoke）;
    # d_i=0 → 直通 loss = 0, 训练不抛错; use_mu_ref=False（默认）不访问该列
    stub = _LambdaStub(N_STEPS, [1] * N_STEPS, [0.10] * N_STEPS,
                       mu_ref_seq=[np.ones(STOCK_DIM)] * N_STEPS)
    m = _lambda_model(stub, use_mu_ref=True, lambda_init=0.5, lambda_eta=0.0,
                      lambda_max=1.0, constraint_epsilon=0.0)
    _fill_lambda_buffer(m, stub)
    m.train()
    assert "train/direct_loss" in m.logger.name_to_value
    assert m.logger.name_to_value["train/direct_loss"] == pytest.approx(0.0, abs=1e-12)


def test_u7_constraint_cost_column_aligned_with_i_t():
    # constraint_cost 列与 i_t/d_i 同索引对齐（spec §7.1 U7）: 每个 pos 的
    # c_t 与同一 step 的 I_t/d_i 一起写入、一起被 get() 读出. get() 对样本
    # 做置换, 故按"行三元组 (i_t, d_i, c_t) 与同一 add() 步完全一致"断言——
    # 每步 d_i 互异, 三元组唯一标识来源步.
    n = 6
    # 每步三元组 (i_t, d_i, c_t) 互异 → 样本行唯一对应来源步, 可证三列同源
    cc_seq = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    i_seq = [1, 0, 1, 0, 1, 0]
    d_seq = [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [0, 1]]
    rows = {t: (i_seq[t], tuple(d_seq[t]), cc_seq[t]) for t in range(n)}
    wrapper = SimpleNamespace()
    wrapper.last_mu_ref = np.zeros(2)
    buf = TailGAERolloutBuffer(
        n, spaces.Box(-1, 1, (4,)), spaces.Box(-1, 1, (2,)),
        device="cpu", gae_lambda=0.95, gamma=0.99, n_envs=1,
        tail_alpha=0.0, risk_wrapper=wrapper,
    )
    for t in range(n):
        wrapper.last_I_t = i_seq[t]
        wrapper.last_direct_grad = np.array(d_seq[t])
        wrapper.last_constraint_cost = cc_seq[t]
        buf.add(np.zeros((1, 4)), np.zeros((1, 2)), 0.0, th.zeros(1), th.zeros(1), -th.ones(1))
    buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))
    for samples in buf.get(batch_size=None):
        assert samples.constraint_cost.shape == (n, 1)
        assert samples.mu_ref.shape == (n, 2)
        assert np.allclose(samples.mu_ref.numpy(), 0.0)  # 未配置 → 恒零
        got_i = samples.i_t[:, 0].numpy().astype(int)
        got_d = samples.d_i.numpy().astype(int)
        got_cc = samples.constraint_cost[:, 0].numpy()
        matched = set()
        for t in range(n):
            src = [s for s in range(n)
                   if rows[s][0] == got_i[t]
                   and rows[s][1] == tuple(got_d[t].tolist())
                   and np.isclose(rows[s][2], float(got_cc[t]), atol=1e-6)]
            assert len(src) == 1, f"row {t}: (i,d,cc) 不与任何同一 add() 步对齐"
            matched.add(src[0])
        assert matched == set(range(n))  # 每个来源步恰好出现一次（无丢失/重复）


# ---------------------------------------------------------------------------
# U9: μ_ref 解析（spec §7.1 U9 / §5.1）
# ---------------------------------------------------------------------------

class _StateFakeBaseEnv:
    """带 state 属性与 hmax 的基座 env（U9: pre-action 捕获口径）.

    state 布局与基座 env 一致: [cash, p_1..p_30, s_1..s_30, ...]; env.step
    推进日期、把 state 更新为 post-action 状态、追加资产价值.
    """

    def __init__(self, stock_dim, pre_state, post_state, dates):
        self.stock_dim = stock_dim
        self.hmax = 100
        self._post = list(post_state)
        self._dates = list(dates)
        self.state = list(pre_state)  # 当前（pre-action）状态
        self.asset_memory = [self._asset_of(pre_state)]
        self.date_memory = [self._dates[0]]
        self.reward_adapter = SimpleNamespace(
            accounting_state=SimpleNamespace(recent_portfolio_returns=[])
        )
        self.df = pd.DataFrame(
            [{"date": d, "tic": f"T{i}", "close": 100.0}
             for d in dates for i in range(2)]
        )
        obs_dim = 1 + 2 * stock_dim + 8 * stock_dim
        self.observation_space = SimpleNamespace(
            shape=(obs_dim,),
            low=np.zeros(obs_dim),
            high=np.ones(obs_dim),
            dtype=np.float32,
        )
        self.action_space = SimpleNamespace(shape=(stock_dim,), dtype=np.float32)

    def _asset_of(self, st):
        st = np.asarray(st, dtype=float)
        return float(st[0] + np.sum(st[1:1 + self.stock_dim] * st[1 + self.stock_dim:1 + 2 * self.stock_dim]))

    def step(self, actions):
        self.date_memory.pop()
        self.date_memory.append(self._dates[0])
        self.state = list(self._post)
        self.asset_memory.append(self._asset_of(self._post))
        return list(self._post), 0.0, False, False, {}


def _make_exec_env(shares_pre, shares_post, *, stock_dim=30, post_prices=None):
    """V_pre 恒 283_000 的执行 env（target_0 = floor(0.85/30·283000/100) = 80）."""
    prices = [100.0] * stock_dim
    cash = 283_000.0 - 100.0 * shares_pre
    pre = [cash] + prices + [float(shares_pre)] + [0.0] * (stock_dim - 1)
    pp = list(post_prices) if post_prices is not None else list(prices)
    post = [cash] + pp + [float(shares_post)] + [0.0] * (stock_dim - 1)
    return _StateFakeBaseEnv(stock_dim, pre, post, ["d0"])


def _u9_wrapper(env, i_by_date, *, exec_aligned=False, ref=REF):
    wrapper = RiskStateEnvWrapper(
        env,
        ["current_drawdown"],
        i_by_date=i_by_date,
        worst20_by_date={"d0": 0.09},
        direct_penalty_ref=ref,
        direct_ref_exec_aligned=exec_aligned,
    )
    wrapper._assemble = lambda state: np.asarray(state, dtype=float)
    wrapper.n_risk_dims = 0
    return wrapper


def test_u9_mu_ref_negative_when_over_target():
    # 100 股 / 目标 80 股 / hmax=100 → μ_ref = −0.2（负号断言: 达标动作 = 卖 20 股）
    env = _make_exec_env(shares_pre=100, shares_post=78)
    wrapper = _u9_wrapper(env, {"d0": 1})
    wrapper.step(np.zeros(30))
    assert wrapper.last_mu_ref[0] == pytest.approx(-0.2, abs=1e-9)
    # 其余空仓股: 每股目标 = ref/30·V/p = 80 股（整股化）, 欠仓可买 → μ_ref = +0.8
    assert np.all(wrapper.last_mu_ref[1:] == pytest.approx(0.8, abs=1e-9))


def test_u9_mu_ref_pre_action_aligned():
    # pre-action 对齐（spec §7.1 U9）: μ_ref 用 env.step 之前的 shares 计算——
    # post 口径（78 股 → +0.02）与正确值（−0.2）不同, 断言排除 post 口径
    env = _make_exec_env(shares_pre=100, shares_post=78)
    wrapper = _u9_wrapper(env, {"d0": 1})
    wrapper.step(np.zeros(30))
    assert wrapper.last_mu_ref[0] == pytest.approx(-0.2, abs=1e-9)
    assert wrapper.last_mu_ref[0] != pytest.approx((80 - 78) / 100.0)


def test_u9_mu_ref_positive_when_under_target():
    # 欠仓可买（spec §5.1 P0-1）: 60 股 → 目标 80 股 → μ_ref = +0.2
    # （允许买到目标, 旧公式 μ_ref=0 会把正常加仓也压向 0）
    env = _make_exec_env(shares_pre=60, shares_post=60)
    wrapper = _u9_wrapper(env, {"d0": 1})
    wrapper.step(np.zeros(30))
    assert wrapper.last_mu_ref[0] == pytest.approx(0.2, abs=1e-9)


def test_u9_mu_ref_zero_when_at_target():
    env = _make_exec_env(shares_pre=80, shares_post=80)
    wrapper = _u9_wrapper(env, {"d0": 1})
    wrapper.step(np.zeros(30))
    assert wrapper.last_mu_ref[0] == pytest.approx(0.0, abs=1e-9)


def test_u9_mu_ref_zero_without_direct_config():
    # 直通配置未激活（direct_penalty_ref=None, 默认）→ μ_ref/constraint_cost/d_i 恒零
    env = _make_exec_env(shares_pre=100, shares_post=78)
    wrapper = RiskStateEnvWrapper(
        env,
        ["current_drawdown"],
        i_by_date={"d0": 1},
        worst20_by_date={"d0": 0.09},
    )
    wrapper._assemble = lambda state: np.asarray(state, dtype=float)
    wrapper.n_risk_dims = 0
    wrapper.step(np.zeros(30))
    assert np.all(wrapper.last_mu_ref == 0.0)
    assert wrapper.last_constraint_cost == 0.0
    assert np.all(wrapper.last_direct_grad == 0)


def test_u9_constraint_cost_handcalc():
    # c_t = I_t · Σ max(0, w_i − w_ref); w_ref = 0.85/30 = 0.028333…
    # post 90 股@100 → V = 273000 + 9000 = 282000 → w_0 = 0.0319149 > w_ref
    env = _make_exec_env(shares_pre=100, shares_post=90)
    wrapper = _u9_wrapper(env, {"d0": 1})
    wrapper.step(np.zeros(30))
    w_0 = 90.0 * 100.0 / 282_000.0
    expected = max(0.0, w_0 - 0.85 / 30.0)
    assert expected > 0.0
    assert wrapper.last_constraint_cost == pytest.approx(expected, abs=1e-12)


def test_u9_constraint_cost_zero_on_non_trigger():
    env = _make_exec_env(shares_pre=100, shares_post=90)
    wrapper = _u9_wrapper(env, {"d0": 0})
    wrapper.step(np.zeros(30))
    assert wrapper.last_constraint_cost == 0.0


def test_u9_exec_aligned_di_not_flagged_on_price_rise():
    # Step 4 d_i^exec（spec §5.1）: 正确卖出 22 股后价格跳涨 10%——post-action
    # 新价格口径把达标动作标成超限（w = 0.0305 > w_ref）; 成交瞬间口径（原成交
    # 价格估值）w^exec = 0.0278 < w_ref → d = 0（不罚）. 历史口径默认保持.
    env = _make_exec_env(shares_pre=100, shares_post=78, post_prices=[110.0] * 30)
    wrapper = _u9_wrapper(env, {"d0": 1}, exec_aligned=True)
    wrapper.step(np.zeros(30))
    assert wrapper.last_direct_grad[0] == 0
    env2 = _make_exec_env(shares_pre=100, shares_post=78, post_prices=[110.0] * 30)
    wrapper2 = _u9_wrapper(env2, {"d0": 1}, exec_aligned=False)
    wrapper2.step(np.zeros(30))
    assert wrapper2.last_direct_grad[0] == 1
