"""Unit tests for tail_gae_buffer.TailGAERolloutBuffer (spec §5.7)."""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer

from tail_gae_buffer import TailGAERolloutBuffer

N_STEPS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


class StubRiskWrapper:
    """Duck-typed stand-in for RiskStateEnvWrapper (frozen protocol)."""

    def __init__(self, r_seq, c_seq, i_seq, dates):
        self.R, self.C, self.I, self.dates = list(r_seq), list(c_seq), list(i_seq), list(dates)
        self.r_history, self.c_history = [], []
        self.i_history, self.date_history = [], []
        # G3 diagnostics read the wrapper's declared names + per-step risk
        # block; a constant dd block makes the quantile split degenerate and
        # the bucket fall back to row index (deterministic for tests).
        self.names = ["current_drawdown"]
        self._risk_block_history = [np.zeros(1, dtype=np.float32)] * len(i_seq)
        self._t = -1
        self.last_R_t = self.last_C_t = self.last_I_t = self.last_date = None
        # V4.5/1230 契约（spec §3.1）: wrapper 恒暴露 last_direct_grad（30 维）.
        self.last_direct_grad = np.zeros(2, dtype=int)  # 与 ACT_SPACE 对齐

    def step(self):
        self._t += 1
        self.last_R_t = self.R[self._t]
        self.last_C_t = self.C[self._t]
        self.last_I_t = self.I[self._t]
        self.last_date = self.dates[self._t]
        self.r_history.append(self.last_R_t)
        self.c_history.append(self.last_C_t)
        self.i_history.append(self.last_I_t)
        self.date_history.append(self.last_date)


def make_standard(seed=42):
    return RolloutBuffer(N_STEPS, OBS_SPACE, ACT_SPACE, "cpu", GAE_LAMBDA, GAMMA, n_envs=1)


def make_tail(tail_alpha, seed=42, **kwargs):
    return TailGAERolloutBuffer(
        N_STEPS, OBS_SPACE, ACT_SPACE, "cpu", GAE_LAMBDA, GAMMA,
        tail_alpha=tail_alpha, **kwargs,
    )


def fill_and_compute(buf, rewards, values, last_values, episode_starts, dones, seed=42):
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(N_STEPS, 4)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(N_STEPS, 2)).astype(np.float32)
    log_probs = rng.uniform(-1.0, 1.0, size=(N_STEPS,)).astype(np.float32)
    wrapper = getattr(buf, "risk_wrapper", None)
    for t in range(N_STEPS):
        if wrapper is not None:
            wrapper.step()
        buf.add(
            obs[t], actions[t], rewards[t], float(episode_starts[t]),
            th.tensor(values[t], dtype=th.float32), th.tensor(log_probs[t], dtype=th.float32),
        )
    buf.compute_returns_and_advantage(th.tensor(last_values, dtype=th.float32), np.asarray(dones))
    return buf


# delta_t = r_t + gamma * v_{t+1} * (1 - e_{t+1}) - v_t for all steps below.
REWARDS = np.array([0.1, 0.0, 0.2, -0.1], dtype=np.float32)
VALUES = np.array([1.0, 0.5, 0.3, 0.2], dtype=np.float32)
LAST_VALUE = np.array([0.1], dtype=np.float32)


def test_alpha_zero_equals_standard_including_episode_boundaries():
    # Episode boundary after step 1 and a terminal last step exercise both
    # recursion branches (episode_starts vs dones).
    episode_starts = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    dones = np.array([1.0], dtype=np.float32)
    std = fill_and_compute(make_standard(), REWARDS, VALUES, LAST_VALUE, episode_starts, dones)
    tail = fill_and_compute(make_tail(0.0), REWARDS, VALUES, LAST_VALUE, episode_starts, dones)
    assert np.allclose(tail.advantages, std.advantages, rtol=0, atol=0)
    assert np.allclose(tail.returns, std.returns, rtol=0, atol=0)


def test_single_tail_day_propagates_backward():
    # I = [0, 1, 0, 0]; step 1 delta = 0.99 * 0.3 - 0.5 = -0.203 < 0.
    delta_1 = REWARDS[1] + GAMMA * VALUES[2] - VALUES[1]
    assert delta_1 < 0
    rewards = np.zeros(4, dtype=np.float32)
    i_seq = [0, 1, 0, 0]
    stub = StubRiskWrapper([0.0, -0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], i_seq, list(range(4)))
    episode_starts = np.zeros(4, dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    std = fill_and_compute(make_standard(), rewards, VALUES, LAST_VALUE, episode_starts, dones)
    tail = fill_and_compute(
        make_tail(1.0, risk_wrapper=stub),
        rewards, VALUES, LAST_VALUE, episode_starts, dones,
    )
    # Non-tail steps (2 and 3) untouched; later recursion is identical.
    assert np.allclose(tail.advantages[2:], std.advantages[2:], rtol=0, atol=0)
    # tilde_delta_1 = delta_1 + min(delta_1, 0) = 2 * delta_1, more negative.
    assert tail.advantages[1] < std.advantages[1]
    assert np.allclose(tail.advantages[1], std.advantages[1] + delta_1, rtol=0, atol=0)
    # Credit backtracking: step 0's advantage drops by gamma * lambda * delta_1.
    assert tail.advantages[0] < std.advantages[0]
    assert np.allclose(tail.advantages[0], std.advantages[0] + GAMMA * GAE_LAMBDA * delta_1, rtol=0, atol=0)


def test_positive_residual_untouched_when_i_is_one():
    # I = [0, 1, 0, 0]; step 1 delta = 1.0 + 0.99 * 0.5 - 0.1 = 1.395 > 0.
    values = np.array([1.0, 0.1, 0.5, 0.2], dtype=np.float32)
    rewards = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    delta_1 = rewards[1] + GAMMA * values[2] - values[1]
    assert delta_1 > 0
    i_seq = [0, 1, 0, 0]
    stub = StubRiskWrapper([0.0, -0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], i_seq, list(range(4)))
    episode_starts = np.zeros(4, dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    std = fill_and_compute(make_standard(), rewards, values, LAST_VALUE, episode_starts, dones)
    tail = fill_and_compute(
        make_tail(1.0, risk_wrapper=stub),
        rewards, values, LAST_VALUE, episode_starts, dones,
    )
    assert tail._computed_I.tolist() == i_seq  # I_t == 1 at the positive-delta step
    assert np.allclose(tail.advantages, std.advantages, rtol=0, atol=0)
    assert np.allclose(tail.returns, std.returns, rtol=0, atol=0)


def test_i_t_from_recorded_risk_passes_through():
    # V4.2: the wrapper's per-date latch lookup is authoritative — the buffer
    # copies the recorded flags straight through, no recompute cross-check.
    i_seq = [0, 1, 1, 0]
    stub = StubRiskWrapper([0.0, -0.02, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], i_seq, list(range(4)))
    episode_starts = np.zeros(4, dtype=np.float32)
    dones = np.array([0.0], dtype=np.float32)
    tail = fill_and_compute(
        make_tail(1.0, risk_wrapper=stub),
        REWARDS, VALUES, LAST_VALUE, episode_starts, dones,
    )
    assert tail._i_hist == i_seq  # recorded wrapper flags
    assert np.array_equal(tail._computed_I, i_seq)
    # Returns still follow the standard delta recursion.
    std = fill_and_compute(make_standard(), REWARDS, VALUES, LAST_VALUE, episode_starts, dones)
    assert np.allclose(tail.returns, std.returns, rtol=0, atol=0)


def test_alpha_zero_without_wrapper_runs_untouched():
    tail = fill_and_compute(make_tail(0.0), REWARDS, VALUES, LAST_VALUE,
                            np.zeros(4, dtype=np.float32), np.array([0.0]))
    assert tail._i_hist == [] and tail._computed_I is None
    std = fill_and_compute(make_standard(), REWARDS, VALUES, LAST_VALUE,
                           np.zeros(4, dtype=np.float32), np.array([0.0]))
    assert np.allclose(tail.advantages, std.advantages, rtol=0, atol=0)


def test_n_envs_assert_fires_with_risk_wrapper():
    with pytest.raises(AssertionError):
        TailGAERolloutBuffer(
            N_STEPS, OBS_SPACE, ACT_SPACE, "cpu", GAE_LAMBDA, GAMMA, n_envs=2,
            tail_alpha=1.0,
            risk_wrapper=StubRiskWrapper([0.0], [0.0], [0], [0]),
        )


def _fill_partial(buf, n, seed=7):
    """Fill only the first *n* slots (pos = n, full = False)."""
    rng = np.random.default_rng(seed)
    wrapper = getattr(buf, "risk_wrapper", None)
    for t in range(n):
        if wrapper is not None:
            wrapper.step()
        buf.add(
            rng.uniform(-1.0, 1.0, size=(4,)).astype(np.float32),
            rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32),
            REWARDS[t], 0.0,
            th.tensor(VALUES[t], dtype=th.float32),
            th.tensor(float(t), dtype=th.float32),
        )
    return buf


def test_partial_rollout_alpha_zero_matches_standard():
    # total_timesteps % n_steps != 0 leaves pos < buffer_size; the α=0 path
    # must stay bitwise-identical to the parent on the same partial fill.
    std = _fill_partial(make_standard(), 3)
    tail = _fill_partial(make_tail(0.0), 3)
    std.compute_returns_and_advantage(th.tensor(LAST_VALUE, dtype=th.float32), np.asarray([0.0]))
    tail.compute_returns_and_advantage(th.tensor(LAST_VALUE, dtype=th.float32), np.asarray([0.0]))
    assert std.full is False and std.pos == 3
    assert np.allclose(tail.advantages, std.advantages, rtol=0, atol=0)
    assert np.allclose(tail.returns, std.returns, rtol=0, atol=0)


def test_partial_rollout_tail_path_handles_unwritten_slots():
    # I: d0=1, d1=1, d2=0 (wrapper latch lookup flags)
    wrapper = StubRiskWrapper(
        [-0.03, -0.02, 0.01], [-0.03, -0.05, -0.02], [1, 1, 0], ["d0", "d1", "d2"],
    )
    buf = _fill_partial(make_tail(1.0, risk_wrapper=wrapper), 3)
    buf.compute_returns_and_advantage(th.tensor(LAST_VALUE, dtype=th.float32), np.asarray([0.0]))
    assert buf.full is False and buf.pos == 3
    assert buf._computed_I.tolist()[:3] == [1, 1, 0]
    assert buf._computed_I.tolist()[3:] == [0]
    # Hand-computed tilde_GAE over the 3 valid steps; the unwritten slot 3
    # keeps the standard recursion (I=0, values[3]=0, bootstrap = last_values):
    # d0 = 0.1 + 0.99*0.5 - 1.0 = -0.405   -> tilde -0.810
    # d1 = 0.0 + 0.99*0.3 - 0.5 = -0.203   -> tilde -0.406
    # d2 = 0.2 + 0.99*0.0 - 0.3 = -0.100   -> tilde -0.100 (I=0, values[3]=0)
    # d3 = 0.0 + 0.99*0.1 - 0.0 =  0.099   (unwritten slot, standard)
    gae = 0.99 * 0.95
    a3 = 0.099
    a2 = -0.100 + gae * a3
    a1 = -0.406 + gae * a2
    a0 = -0.810 + gae * a1
    assert np.allclose(buf.advantages[0], a0, rtol=0, atol=1e-4)
    assert np.allclose(buf.advantages[1], a1, rtol=0, atol=1e-4)
    assert np.allclose(buf.advantages[2], a2, rtol=0, atol=1e-4)
    assert np.allclose(buf.advantages[3], a3, rtol=0, atol=1e-4)
