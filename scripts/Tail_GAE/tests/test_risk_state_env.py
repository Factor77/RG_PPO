"""Unit tests for risk_state_env.RiskStateEnvWrapper (spec 5.6)."""

import numpy as np
import pandas as pd
import pytest

from risk_state_env import RiskStateEnvWrapper
from tail_gae_env import DownsideBreadthStockTradingEnv

STOCK_DIM = 3
N_DAYS = 15
STATE_SPACE = 1 + 2 * STOCK_DIM

NAMES_7 = [
    "current_drawdown",
    "running_max_drawdown",
    "drawdown_delta",
    "gross_exposure",
    "downside_volatility_20",
    "streak20max",
    "worst20",
]


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


def make_dates(n_days=N_DAYS):
    return [f"2020-01-{d + 1:02d}" for d in range(n_days)]


def make_worst20(n_days=N_DAYS, value=0.09):
    return {d: value for d in make_dates(n_days)}


def make_i_by_date(n_days=N_DAYS, value=0):
    return {d: value for d in make_dates(n_days)}


def make_wrapper(
    names=NAMES_7,
    i_by_date=None,
    worst20_by_date=None,
    ew_by_date=None,
    ge_penalty_beta=0.0,
    ge_penalty_ref=0.95,
    ge_penalty_mode="total",
    policy_names=None,
):
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    if worst20_by_date is None:
        worst20_by_date = make_worst20()
    if i_by_date is None:
        i_by_date = make_i_by_date()
    return RiskStateEnvWrapper(
        env,
        names,
        i_by_date=i_by_date,
        worst20_by_date=worst20_by_date,
        ew_by_date=ew_by_date,
        ge_penalty_beta=ge_penalty_beta,
        ge_penalty_ref=ge_penalty_ref,
        ge_penalty_mode=ge_penalty_mode,
        policy_names=policy_names,
    )


def fake_step(wrapper, ret=0.0, drawdown=None):
    """Stub env.step: controlled portfolio return, no accounting mutation.

    The real env.step() advances accounting via advance_mdd_accounting, which
    would clobber hand-injected accounting values, and its portfolio return is
    not controllable with zero actions.
    """
    env = wrapper.env

    def _step(actions):
        env.asset_memory.append(env.asset_memory[-1] * (1.0 + ret))
        env.date_memory.append(env.date_memory[-1])
        if drawdown is not None:
            env.reward_adapter.accounting_state.current_drawdown = drawdown
        return list(env.state), 0.0, False, False, {}

    return _step


def test_dimension_contract():
    wrapper = make_wrapper(names=NAMES_7)
    base_dim = wrapper.env.observation_space.shape[0]
    assert wrapper.n_risk_dims == 7
    assert wrapper.observation_space.shape[0] == base_dim + 7
    assert wrapper.observation_space.dtype == wrapper.env.observation_space.dtype
    obs, _ = wrapper.reset()
    assert len(obs) == base_dim + 7
    obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert len(obs) == base_dim + 7


def test_internal_only_wrapper_keeps_policy_observation_at_base_dim():
    wrapper = make_wrapper(names=NAMES_7, policy_names=[])
    base_dim = wrapper.env.observation_space.shape[0]
    assert wrapper.n_internal_risk_dims == 7
    assert wrapper.n_risk_dims == 0
    assert wrapper.observation_space.shape == (base_dim,)
    assert wrapper.get_observation_metadata() == {
        "use_risk_observation": False,
        "base_observation_dim": base_dim,
        "risk_observation_dim": 0,
        "risk_observation_fields": [],
        "policy_observation_dim": base_dim,
        "phase9_observation_dim": 0,
        "phase9_observation_fields": [],
    }
    obs, _ = wrapper.reset()
    assert len(obs) == base_dim
    obs, reward, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert len(obs) == base_dim
    assert len(wrapper._risk_block_history[-1]) == 7
    assert reward == pytest.approx(wrapper.env.reward)


def test_policy_fields_must_be_internal_fields():
    with pytest.raises(ValueError, match="subset"):
        make_wrapper(names=["gross_exposure"], policy_names=["current_drawdown"])


def test_unknown_risk_name_rejected():
    with pytest.raises(AssertionError):
        make_wrapper(names=["streak20maxx"])


def test_risk_block_order_matches_names():
    names = ["gross_exposure", "worst20", "current_drawdown"]
    wrapper = make_wrapper(names=names)
    obs, _ = wrapper.reset()
    block = obs[wrapper.base_dim:]
    assert block[0] == pytest.approx(-1.0)  # no holdings -> exposure 0 -> 0*2-1
    assert block[1] == pytest.approx(0.09 / 3.0)  # worst20 table -> 0.09*(1/3)
    assert block[2] == pytest.approx(-1.0)  # fresh accounting drawdown 0 -> 0*2-1


def test_transform_affine_applied(monkeypatch):
    wrapper = make_wrapper(names=["current_drawdown"])
    wrapper.reset()
    wrapper.env.reward_adapter.accounting_state.current_drawdown = 0.25
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim] == pytest.approx(0.25 * 2.0 - 1.0)


def test_drawdown_delta_tracking(monkeypatch):
    wrapper = make_wrapper(names=["drawdown_delta"])
    wrapper.reset()
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper, drawdown=0.0))
    obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim] == 0.0  # first step of episode: delta 0
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper, drawdown=0.2))
    obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim] == pytest.approx((0.2 - 0.0) * 10.0)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper, drawdown=0.05))
    obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim] == pytest.approx((0.05 - 0.2) * 10.0)


def _set_high_exposure_state(wrapper):
    """Mutate env.state so the post-step GE exceeds the 0.95 penalty ref.

    state layout (stock_dim=3): [cash, p1..p3, s1..s3].  GE = Σ|p·s|/value;
    value = cash + Σp·s.  cash=50_000, p=100, s=5_000*3 gives gross 1.5e6,
    value 1.55e6 -> GE 0.9677 (> 0.95).
    """
    st = wrapper.env.state
    st[0] = 50_000.0
    st[1:4] = [100.0] * STOCK_DIM
    st[4:7] = [5_000.0] * STOCK_DIM
    wrapper.env.asset_memory[-1] = 50_000.0 + 1.5e6


def test_ge_penalty_trigger_day_over_ref(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=1),
        ge_penalty_beta=0.05,
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    ge = 0.9677
    assert reward == pytest.approx(-0.05 * (ge - 0.95), abs=1e-4)
    assert wrapper.last_ge_penalty == pytest.approx(0.05 * (ge - 0.95), abs=1e-4)
    assert wrapper.ge_penalty_history == [wrapper.last_ge_penalty]


def test_ge_penalty_trigger_day_below_ref_not_penalized(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=1),
        ge_penalty_beta=0.05,
    )
    wrapper.reset()
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert reward == 0.0  # no holdings -> GE 0 < ref -> no penalty
    assert wrapper.last_ge_penalty == 0.0


def test_ge_penalty_non_trigger_day_not_penalized(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=0),
        ge_penalty_beta=0.05,
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert reward == 0.0  # I_t = 0 -> penalty term zeroed
    assert wrapper.last_ge_penalty == 0.0


def test_ge_penalty_beta_zero_no_op(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=1),
        ge_penalty_beta=0.0,
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert reward == 0.0  # beta=0 -> legacy reward unchanged
    assert wrapper.last_ge_penalty == 0.0


def test_ge_penalty_per_stock_trigger_day_handcalc(monkeypatch):
    # 1219 per-stock tax: penalty = beta * sum_i max(0, w_i - ref/stock_dim),
    # w_i = p_i*s_i/V.  Hand-set state: cash=50k, 3 stocks of 100*5000 ->
    # w_i = 500000/1550000 = 0.3225806, ref/3 = 0.2833333, excess 0.0392473.
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=1),
        ge_penalty_beta=0.10, ge_penalty_ref=0.85, ge_penalty_mode="per_stock",
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    st = wrapper.env.state
    V = st[0] + sum(st[i] * st[1 + STOCK_DIM + (i - 1)] for i in range(1, 1 + STOCK_DIM))
    w = [st[i] * st[1 + STOCK_DIM + (i - 1)] / V for i in range(1, 1 + STOCK_DIM)]
    expected = 0.10 * sum(max(0.0, wi - 0.85 / STOCK_DIM) for wi in w)
    assert expected > 0.0
    assert reward == pytest.approx(-expected, abs=1e-9)
    assert wrapper.last_ge_penalty == pytest.approx(expected, abs=1e-9)
    assert len(wrapper.ge_penalty_history) == 1


def test_ge_penalty_per_stock_non_trigger_day_zero(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=0),
        ge_penalty_beta=0.10, ge_penalty_ref=0.85, ge_penalty_mode="per_stock",
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert reward == 0.0  # I_t = 0 -> per-stock tax zeroed
    assert wrapper.last_ge_penalty == 0.0


def test_ge_penalty_per_stock_beta_zero_no_op(monkeypatch):
    wrapper = make_wrapper(
        names=["gross_exposure"], i_by_date=make_i_by_date(value=1),
        ge_penalty_beta=0.0, ge_penalty_ref=0.85, ge_penalty_mode="per_stock",
    )
    wrapper.reset()
    _set_high_exposure_state(wrapper)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert reward == 0.0  # bridge: beta=0 -> no penalty in either mode
    assert wrapper.last_ge_penalty == 0.0


def test_last_I_t_tail_lookup_day(monkeypatch):
    # V4.2: I_t comes from the per-date latch lookup (market EW trigger), not
    # from the portfolio R_t/C_t thresholds; R_t/C_t stay portfolio diagnostics.
    wrapper = make_wrapper(i_by_date=make_i_by_date(value=1))
    wrapper.reset()
    wrapper._port_rets.extend([0.0] * 9)
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper, ret=-0.015))
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_R_t == pytest.approx(-0.015)
    assert wrapper.last_C_t == pytest.approx(-0.015)
    assert wrapper.last_I_t == 1  # lookup says tail regardless of R_t/C_t


def test_last_I_t_non_tail_lookup_day(monkeypatch):
    # A sharp portfolio drop must NOT trigger when the market latch says 0.
    wrapper = make_wrapper(i_by_date=make_i_by_date(value=0))
    wrapper.reset()
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper, ret=-0.02))
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_C_t == pytest.approx(-0.02)
    assert wrapper.last_I_t == 0


def test_last_I_t_per_date_lookup(monkeypatch):
    # The lookup is per date: flipping a later date's entry flips last_I_t.
    i_by_date = make_i_by_date(value=0)
    i_by_date["2020-01-03"] = 1
    wrapper = make_wrapper(i_by_date=i_by_date)
    wrapper.reset()
    monkeypatch.setattr(wrapper.env, "step", fake_step(wrapper))
    wrapper.step(np.zeros(STOCK_DIM))
    assert wrapper.last_I_t == 0


def test_i_by_date_missing_date_fails_fast():
    wrapper = make_wrapper(names=["worst20"], i_by_date={"2020-01-01": 1})
    wrapper.reset()  # first day present in the table
    with pytest.raises(KeyError):
        wrapper.step(np.zeros(STOCK_DIM))  # next date absent


def test_warmup_ar1_zero_and_worst20_present():
    wrapper = make_wrapper(names=["ar1_20d", "worst20"])
    obs, _ = wrapper.reset()
    assert obs[wrapper.base_dim] == 0.0  # <4 points: ar1_20d exactly 0
    assert obs[wrapper.base_dim + 1] == pytest.approx(0.09 / 3.0)
    for _ in range(2):
        obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim] == 0.0  # still <4 points
    assert obs[wrapper.base_dim + 1] == pytest.approx(0.09 / 3.0)


def test_worst20_missing_date_fails_fast():
    wrapper = make_wrapper(
        names=["worst20"], worst20_by_date={"2020-01-01": 0.09}
    )
    wrapper.reset()  # episode day 1 present in table
    with pytest.raises(KeyError):
        wrapper.step(np.zeros(STOCK_DIM))  # next date absent


def test_mkt_ew_features_in_obs():
    dates = make_dates()
    wrapper = make_wrapper(
        names=["gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        ew_by_date={
            "mkt_ew_10": {d: -0.03 for d in dates},
            "mkt_ew_5": {d: 0.01 for d in dates},
        },
    )
    obs, _ = wrapper.reset()
    block = obs[wrapper.base_dim:]
    assert block[0] == pytest.approx(-1.0)  # no holdings -> exposure 0 -> 0*2-1
    assert block[1] == pytest.approx(-0.03)  # pass-through transform (1, 0)
    assert block[2] == pytest.approx(0.01)
    obs, *_ = wrapper.step(np.zeros(STOCK_DIM))
    assert obs[wrapper.base_dim + 1] == pytest.approx(-0.03)


def test_mkt_ew_missing_date_fails_fast():
    wrapper = make_wrapper(names=["mkt_ew_10"], ew_by_date={"mkt_ew_10": {}})
    with pytest.raises(KeyError):
        wrapper.reset()  # lookup has no entry for the episode's first date


def test_history_protocol():
    wrapper = make_wrapper(names=["worst20"])
    wrapper.reset()
    for _ in range(5):
        wrapper.step(np.zeros(STOCK_DIM))
    n = 5
    assert len(wrapper.r_history) == n
    assert len(wrapper.c_history) == n
    assert len(wrapper.i_history) == n
    assert len(wrapper.date_history) == n
    assert wrapper.r_history is wrapper._port_rets
    assert wrapper.last_R_t == wrapper.r_history[-1]
    assert wrapper.last_C_t == wrapper.c_history[-1]
    assert wrapper.last_I_t == wrapper.i_history[-1]
    assert wrapper.last_date == wrapper.date_history[-1]
    assert wrapper.last_date == wrapper.env.date_memory[-1]


def test_histories_persist_across_reset():
    wrapper = make_wrapper(names=["worst20"])
    wrapper.reset()
    for _ in range(15):  # 15-day panel: terminal on the 15th step() call
        obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert done
    n_before = len(wrapper.r_history)
    obs, info = wrapper.reset()
    assert len(wrapper.r_history) == n_before  # reset keeps the live buffer
    wrapper.step(np.zeros(STOCK_DIM))
    assert len(wrapper.r_history) == n_before + 1
    assert len(wrapper.c_history) == len(wrapper.r_history)
    assert len(wrapper.i_history) == len(wrapper.r_history)
    assert len(wrapper.date_history) == len(wrapper.r_history)


def test_step_reset_return_shapes():
    wrapper = make_wrapper()
    obs, info = wrapper.reset()
    assert isinstance(info, dict)
    result = wrapper.step(np.zeros(STOCK_DIM))
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, done, truncated, info = result
    assert done is False
    assert truncated is False
    assert isinstance(info, dict)
    assert np.isfinite(reward)


def test_step_no_longer_triggers_diagnostics(capsys):
    # spec 6: step() hook removed (spec 3.2-2); diagnostics are callback-driven.
    wrapper = make_wrapper(names=["current_drawdown", "worst20"])
    obs, _ = wrapper.reset()
    done = False
    for _ in range(4100):
        if done:
            obs, _ = wrapper.reset()
        obs, reward, done, truncated, info = wrapper.step(np.zeros(STOCK_DIM))
    assert capsys.readouterr().out == ""


def test_print_diagnostics_direct(capsys):
    wrapper = make_wrapper(names=["current_drawdown", "worst20"])
    wrapper.reset()
    for _ in range(3):
        wrapper.step(np.zeros(STOCK_DIM))
    wrapper.print_diagnostics(iteration=5)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln]
    # single atomic print(block, flush=True): exactly the two feature lines,
    # nothing else (spec 3.2-1 / 3.3)
    assert len(lines) == 2
    for name in ["current_drawdown", "worst20"]:
        assert any(f"[risk_state_env] {name}:" in ln for ln in lines)
    assert all("(iteration=5)" in ln for ln in lines)
    assert all(
        "mean=" in ln and "std=" in ln and "min=" in ln and "max=" in ln
        for ln in lines
    )
    assert "(n=" not in out


def test_print_diagnostics_empty_history_no_output(capsys):
    wrapper = make_wrapper(names=["current_drawdown"])
    wrapper.print_diagnostics(iteration=5)
    assert capsys.readouterr().out == ""


def test_getattr_delegates_to_env():
    wrapper = make_wrapper(names=["worst20"])
    env = wrapper.env
    # all five env_method names delegate (spec 3.2-3); save_action_memory is
    # the one FinRL DRL_prediction additionally calls (spec P1-3)
    for name in [
        "save_asset_memory",
        "save_action_memory",
        "get_base_state",
        "get_reward_accounting_state",
        "get_account_continuity_state",
    ]:
        assert callable(getattr(wrapper, name)), name
        assert getattr(wrapper, name).__func__ is getattr(env, name).__func__, name
    # attribute passthrough rides the same delegation (trade-segment df check)
    assert wrapper.df is env.df
    # unknown attributes fail cleanly instead of recursing
    with pytest.raises(AttributeError):
        wrapper._no_such_attribute


def test_get_observation_metadata():
    wrapper = make_wrapper(names=NAMES_7)
    md = wrapper.get_observation_metadata()
    assert md["use_risk_observation"] is True
    # derived from self.base_dim, never a hardcoded 301 (spec P2-1)
    assert md["base_observation_dim"] == wrapper.base_dim
    assert md["base_observation_dim"] == wrapper.env.observation_space.shape[0]
    assert md["base_observation_dim"] != 301
    assert md["risk_observation_dim"] == 7
    assert md["risk_observation_fields"] == NAMES_7
    assert md["policy_observation_dim"] == wrapper.base_dim + 7


def test_port_rets_primed_from_accounting_state():
    # spec 3.2-5 / P1-7: constructor prefills _port_rets from
    # reward_adapter.accounting_state.recent_portfolio_returns so
    # portfolio_window features stay continuous across trade segments
    env = DownsideBreadthStockTradingEnv(make_panel(), **env_kwargs())
    env.reward_adapter.accounting_state.recent_portfolio_returns = (
        -0.0100,
        0.0200,
        0.0300,
    )
    wrapper = RiskStateEnvWrapper(
        env,
        NAMES_7,
        i_by_date=make_i_by_date(),
        worst20_by_date=make_worst20(),
    )
    assert wrapper._port_rets == [-0.0100, 0.0200, 0.0300]
    # r_history must keep returning the same live object (step() appends to it)
    assert wrapper.r_history is wrapper._port_rets


def test_port_rets_empty_when_accounting_state_has_none():
    wrapper = make_wrapper(names=["worst20"])
    assert wrapper._port_rets == []
    assert wrapper.r_history is wrapper._port_rets
