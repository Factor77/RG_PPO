"""Unit tests for the V4.12 1270 non-symmetric PCGrad contract (spec Part 3).

Covers the five pre-registered implementation acceptance items:
① pcgrad off / all-i_t=0 → bitwise identical to the closed path (learn smoke);
② synthetic gradient projection math;
③ vf/log_std gradient zero leakage (projection only touches actor-mean params);
④ all-i_t=0 minibatch passthrough (covered by ①);
⑤ determinism (pure function, no RNG).
"""

import numpy as np
import pytest
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from test_tail_gae_ppo import (
    _VecEnvFacade,
    env_kwargs,
    make_dates,
    make_panel,
    make_vec,
    ppo_kwargs,
    N_STEPS,
    STATE_SPACE,
)
from risk_state_env import RiskStateEnvWrapper
from tail_gae_env import DownsideBreadthStockTradingEnv
from tail_gae_ppo import TailGAEPPO, _pcgrad_grads, pcgrad_project, direct_penalty_loss


# ---------------------------------------------------------------------------
# ② 合成梯度投影（纯函数）
# ---------------------------------------------------------------------------

def _t(x):
    return th.tensor(x, dtype=th.float64)


def test_project_conflicting_direction():
    g_p = [_t([1.0, 0.0])]
    g_r = [_t([-1.0, 1.0])]
    g_r_proj, cos, skipped = pcgrad_project(g_p, g_r)
    assert not skipped
    assert abs(cos - (-1.0 / (2.0 ** 0.5))) < 1e-12
    assert th.allclose(g_r_proj[0], _t([0.0, 1.0]), atol=1e-12)
    # 投影后与 g_p 正交
    assert abs(float((g_r_proj[0] * g_p[0]).sum())) < 1e-12


def test_no_projection_when_positive_cos():
    g_p = [_t([1.0, 0.0])]
    g_r = [_t([1.0, 1.0])]
    g_r_proj, cos, skipped = pcgrad_project(g_p, g_r)
    assert not skipped
    assert cos > 0
    assert th.allclose(g_r_proj[0], g_r[0], atol=1e-12)


def test_skip_when_gp_below_eps():
    g_p = [_t([1e-9, 0.0])]
    g_r = [_t([-1.0, 1.0])]
    g_r_proj, cos, skipped = pcgrad_project(g_p, g_r)
    assert skipped
    assert cos == 0.0
    assert th.allclose(g_r_proj[0], g_r[0], atol=1e-12)


def test_determinism_no_rng():
    g_p = [_t([0.7, -0.2, 0.4])]
    g_r = [_t([-0.5, 0.9, 0.1])]
    out1 = pcgrad_project(g_p, g_r)
    out2 = pcgrad_project(g_p, g_r)
    for a, b in zip(out1[0], out2[0]):
        assert th.equal(a, b)
    assert out1[1] == out2[1]
    assert out1[2] == out2[2]


def test_multiple_param_groups_project_jointly():
    g_p = [_t([1.0, 0.0]), _t([2.0, 0.0])]
    g_r = [_t([-1.0, 1.0]), _t([0.0, 2.0])]
    g_r_proj, cos, skipped = pcgrad_project(g_p, g_r)
    assert not skipped
    # dot = -1 + 0 = -1; |g_p|^2 = 1 + 4 = 5 → alpha = -0.2
    assert th.allclose(g_r_proj[0], _t([-1.0 + 0.2, 1.0]), atol=1e-12)
    assert th.allclose(g_r_proj[1], _t([0.0 + 0.4, 2.0]), atol=1e-12)
    assert abs(float((g_r_proj[0] * g_p[0]).sum() + (g_r_proj[1] * g_p[1]).sum())) < 1e-12


# ---------------------------------------------------------------------------
# ③ 梯度分域：投影只触及 actor-mean 参数（vf/log_std 零泄漏）
# ---------------------------------------------------------------------------

def test_pcgrad_grads_touch_only_actor_mean_params():
    import gymnasium

    policy = ActorCriticPolicy(
        observation_space=gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_SPACE,)),
        action_space=gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3,)),
        lr_schedule=lambda _: 1e-4,
        net_arch={"pi": [16, 16], "vf": [16, 16]},
    )
    obs = th.randn(8, STATE_SPACE)
    dist = policy.get_distribution(obs)
    mean = dist.distribution.mean
    policy_loss = (mean ** 2).sum()
    i_t = th.ones(8, 1)
    d_i = th.ones(8, 3)
    direct_loss = direct_penalty_loss(mean, d_i, i_t, mu_ref=None)
    pairs = _pcgrad_grads(policy, policy_loss, 0.03 * direct_loss)
    assert pairs, "expected non-empty actor-mean gradient pairs"
    names = [n for n, _ in policy.named_parameters()]
    touched = set()
    for gp, gr, p in pairs:
        assert gp is not None and gr is not None
        name = [n for n, q in policy.named_parameters() if q is p][0]
        touched.add(name)
        assert name.startswith(("mlp_extractor.policy_net", "action_net")), name
    assert all(n == "log_std" or n.startswith(("mlp_extractor.value_net", "value_net"))
               for n in names if n not in touched
               if not n.startswith(("mlp_extractor.policy_net", "action_net")))


# ---------------------------------------------------------------------------
# ①/④ 全 i_t=0（或 pcgrad 关闭）→ 与关闭路径逐位一致（learn smoke）
# ---------------------------------------------------------------------------

def test_all_it_zero_pcgrad_on_equals_pcgrad_off_bitwise():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model_off = TailGAEPPO(
        "MlpPolicy", vec, tail_alpha=1.0, risk_wrapper=wrapper,
        direct_penalty_beta=0.03, **ppo_kwargs(),
    )
    model_off.learn(total_timesteps=2 * N_STEPS)

    np.random.seed(0)
    wrapper2, vec2 = make_vec()
    model_on = TailGAEPPO(
        "MlpPolicy", vec2, tail_alpha=1.0, risk_wrapper=wrapper2,
        direct_penalty_beta=0.03, pcgrad=True, **ppo_kwargs(),
    )
    model_on.learn(total_timesteps=2 * N_STEPS)

    for (n1, p1), (n2, p2) in zip(model_off.policy.state_dict().items(),
                                  model_on.policy.state_dict().items()):
        assert n1 == n2
        assert th.equal(p1, p2), f"param {n1} diverged with pcgrad=True on all-i_t=0 data"


def test_pcgrad_on_learn_smoke_with_triggers_records_metrics():
    np.random.seed(0)
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    wrapper = _VecEnvFacade(
        env,
        ["current_drawdown", "worst20"],
        i_by_date={d: 1 for d in make_dates()},
        worst20_by_date={d: 0.09 for d in make_dates()},
    )
    wrapper.action_space = wrapper.env.action_space
    vec = DummyVecEnv([lambda: wrapper])
    model = TailGAEPPO(
        "MlpPolicy", vec, tail_alpha=1.0, risk_wrapper=wrapper,
        direct_penalty_beta=0.03, pcgrad=True, **ppo_kwargs(),
    )
    model.learn(total_timesteps=2 * N_STEPS)
    assert model.num_timesteps == 2 * N_STEPS
    assert model.logger.name_to_value.get("train/pcgrad_cos") is not None
    assert model.logger.name_to_value.get("train/pcgrad_cos_cum") is not None
