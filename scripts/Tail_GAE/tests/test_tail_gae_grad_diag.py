"""V4.6 B0 梯度诊断单测（spec §3.3 / §3.5）。

U-B0a 熵项对 log_std 梯度解析断言：∂[ent_coef·(−H)]/∂log_std_i = −ent_coef
      （与 σ 无关），30 维 L2 = √30 × ent_coef ≈ 0.0548；
U-B0b grad_diag=True 的 train() 在 logger 记录三个梯度标量；
U-B0c grad_diag=False 不记录（默认零开销）。
"""

import numpy as np
import torch as th
from stable_baselines3.common.logger import configure

from tail_gae_ppo import TailGAEPPO
from test_tail_gae_ppo import make_vec, ppo_kwargs

N_STEPS = 32
STOCK_DIM = 3


def _entropy_grad(model, obs):
    """复刻 train() 内 B0 块：熵项对 log_std 的梯度（backward 前 autograd.grad）。"""
    dist = model.policy.get_distribution(obs)
    entropy_loss = -th.mean(dist.entropy())
    log_std = [p for n, p in model.policy.named_parameters() if n == "log_std"][0]
    g = th.autograd.grad(model.ent_coef * entropy_loss, log_std, retain_graph=True)[0]
    return g


def _fill_buffer(model, vec, rng_seed=7):
    buf = model.rollout_buffer
    obs_dim = vec.observation_space.shape[0]
    rng = np.random.default_rng(rng_seed)
    for _ in range(N_STEPS):
        buf.add(
            rng.normal(size=(1, obs_dim)),
            rng.normal(size=(1, STOCK_DIM)),
            float(rng.normal()),
            float(rng.integers(0, 2)),
            th.zeros(1),
            th.zeros(1),
        )
    buf.compute_returns_and_advantage(th.zeros(1), np.zeros((1,)))


def test_b0a_entropy_gradient_matches_analytic():
    """熵项对 log_std 每维梯度 = −ent_coef（与 σ 无关），L2 = √dims × ent_coef。"""
    wrapper, vec = make_vec()
    model = TailGAEPPO("MlpPolicy", vec, **ppo_kwargs())
    obs = th.zeros(1, *vec.observation_space.shape)
    g = _entropy_grad(model, obs)
    assert np.allclose(g.detach().cpu().numpy(), -model.ent_coef, atol=1e-6)
    expected = float(np.sqrt(g.numel()) * model.ent_coef)
    assert np.isclose(g.detach().norm().item(), expected, rtol=1e-5)
    # spec 理论基准（30 维动作空间口径）：√30 × 0.01 ≈ 0.0548
    assert np.isclose(float(np.sqrt(30.0) * 0.01), 0.054772, rtol=1e-3)


def test_b0b_train_records_three_grad_scalars():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO("MlpPolicy", vec, grad_diag=True, **ppo_kwargs())
    model._logger = configure(None, [])
    _fill_buffer(model, vec)
    model.train()
    for key in ("train/grad_logstd_ent", "train/grad_logstd_pg", "train/grad_mu_pg"):
        assert key in model.logger.name_to_value, f"missing {key}"
        assert np.isfinite(model.logger.name_to_value[key])


def test_b0c_disabled_no_extra_logs():
    np.random.seed(0)
    wrapper, vec = make_vec()
    model = TailGAEPPO("MlpPolicy", vec, grad_diag=False, **ppo_kwargs())
    model._logger = configure(None, [])
    _fill_buffer(model, vec)
    model.train()
    for key in ("train/grad_logstd_ent", "train/grad_logstd_pg", "train/grad_mu_pg"):
        assert key not in model.logger.name_to_value
