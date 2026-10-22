"""Unit tests for the V3 rolling runner and its audit injection channel.

Covers spec §6 RollingRunner dry-run + §4.1 runner_kwargs injection:

- PU-6 hard constraint (rebalance/validation window must be 63)
- window derivation = 4 (iterations 126/189/252/315) and the dry-run plan
- 301-dim observation contract: V2 branch (base_dim + declared names) and
  the 1200 bridge (parent schema, use_risk_observation=False)
- runner_kwargs injection through audit_mda_ppo_run (P0-3); None keeps the
  1200 default construction untouched
- bridge behavior: no RiskStateEnvWrapper, no diagnostics callback, no
  [risk_state_env] output; reward_mode="return" reaches the env (P1-12)
- _DiagnosticsCallback: exactly 8 blocks at iterations 5..40 per window and
  the off-by-one backfill rule (P2-5)
- V2 wiring against the real tiny env: wrapper stashed, dims, callback list
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.Tail_GAE.src import tail_gae_rolling_runner as tgr
from scripts.Tail_GAE.src.mda_ppo_audit import MDAAuditConfig, audit_mda_ppo_run
from scripts.Tail_GAE.src.single_ppo_env import RISK_OBSERVATION_FIELDS
from scripts.Tail_GAE.src.tail_gae_env import DownsideBreadthStockTradingEnv

NAMES_5 = [
    "current_drawdown",
    "running_max_drawdown",
    "drawdown_delta",
    "downside_volatility_20",
    "gross_exposure",
]


# ---------------------------------------------------------------------------
# Fakes (mirror tests/test_single_ppo_runner_training.py patterns)
# ---------------------------------------------------------------------------

class FakeVecEnv:
    def __init__(self):
        self.observation_space = None

    def step(self, action):
        return "next_obs", [0.0], [False], [{}]


class FakeEnv:
    instances: list["FakeEnv"] = []

    def __init__(self, *, df, reward_mode, lambda_value, clip_c, **kwargs):
        self.df = df
        self.reward_mode = reward_mode
        self.lambda_value = lambda_value
        self.clip_c = clip_c
        self.mode = kwargs.get("mode")
        self.kwargs = kwargs
        self.instances.append(self)

    def get_sb_env(self):
        return FakeVecEnv(), None

    def get_observation_metadata(self):
        # Mirrors the real base env (single_ppo_env.py:443): the V2 runner
        # forces use_risk_observation=False on the bare env, which reports the
        # fixed single-PPO schema (P0-2 scenario).
        base_dim = int(self.kwargs["state_space"])
        return {
            "use_risk_observation": False,
            "base_observation_dim": base_dim,
            "risk_observation_dim": len(RISK_OBSERVATION_FIELDS),
            "risk_observation_fields": list(RISK_OBSERVATION_FIELDS),
            "policy_observation_dim": base_dim,
        }


def fake_data_split(data, start, end):
    return pd.DataFrame(
        {"date": [start, end], "tic": ["A", "A"], "close": [1.0, 1.0]}
    )


def test_n2_training_normalizer_fits_first_window_then_loads_frozen(tmp_path, monkeypatch):
    runner = object.__new__(tgr.TailGAERollingRunner)
    runner.normalization_config = {
        "enabled": True,
        "fit_once_on_meta_train": True,
    }
    runner._shared_normalization_paths = {}
    runner._risk_base_dim = lambda: 301
    vec_env = object()
    created = object()
    loaded = object()
    calls = []

    def fake_create(venv, **kwargs):
        calls.append(("create", venv, kwargs))
        return created

    def fake_load(path, venv, *, base_dim):
        calls.append(("load", path, venv, base_dim))
        return loaded

    monkeypatch.setattr(tgr, "SliceVecNormalize", fake_create)
    monkeypatch.setattr(tgr, "load_slice_normalizer", fake_load)

    assert runner._wrap_training_env(vec_env=vec_env, seed=4) is created
    assert calls == [
        (
            "create",
            vec_env,
            {
                "base_dim": 301,
                "norm_obs": True,
                "norm_reward": False,
                "training": True,
                "warmup_count": 100,
            },
        )
    ]

    first_window_path = tmp_path / "ppo_rolling_seed_4.pkl"
    first_window_path.write_bytes(b"saved")
    runner._shared_normalization_paths[4] = first_window_path
    assert runner._wrap_training_env(vec_env=vec_env, seed=4) is loaded
    assert calls[-1] == ("load", first_window_path, vec_env, 301)


def test_legacy_tail_training_normalizer_still_refits_each_window(monkeypatch):
    runner = object.__new__(tgr.TailGAERollingRunner)
    runner.normalization_config = {
        "enabled": True,
        "fit_once_on_meta_train": False,
    }
    runner._shared_normalization_paths = {0: Path("ignored.pkl")}
    runner._risk_base_dim = lambda: 301
    created = []

    def fake_create(venv, **kwargs):
        created.append((venv, kwargs))
        return len(created)

    monkeypatch.setattr(tgr, "SliceVecNormalize", fake_create)
    assert runner._wrap_training_env(vec_env="window-1", seed=0) == 1
    assert runner._wrap_training_env(vec_env="window-2", seed=0) == 2
    assert [call[0] for call in created] == ["window-1", "window-2"]


def test_tail_training_override_saves_both_normalizer_states(tmp_path, monkeypatch):
    """N1/N3 must not lose the parent's reward-component save hook."""
    runner = object.__new__(tgr.TailGAERollingRunner)
    runner.ppo_model_kwargs = {"n_steps": 8, "policy_kwargs": {"net_arch": [16]}}
    runner.tail_alpha = 0.0
    runner._current_risk_wrapper = None
    runner.tilt_shift = False
    runner.direct_penalty_beta = 0.03
    runner.direct_penalty_ref = 0.85
    runner.output_root = tmp_path
    runner.run_id = "n3"
    runner.ppo_timesteps = 16
    runner.training_callback_factory = None
    runner.training_event_callback = None
    runner._training_quality = {}
    calls = []

    class FakeModel:
        num_timesteps = 0

        def learn(self, **kwargs):
            self.num_timesteps = kwargs["total_timesteps"]

    monkeypatch.setattr(tgr, "TailGAEPPO", lambda *args, **kwargs: FakeModel())
    runner._save_training_normalizer = lambda **kwargs: calls.append("observation")
    runner._save_reward_component_normalizer = lambda **kwargs: calls.append("reward_component")

    window = types.SimpleNamespace(index=0, iteration=126, label="iter_126")
    model = runner._train_ppo_for_window(window=window, train_env=object(), seed=3)

    assert model.num_timesteps == 16
    assert calls == ["observation", "reward_component"]


def make_tiny_panel(n_stocks=3, n_days=15, seed=0):
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


def tiny_env_kwargs():
    return dict(
        stock_dim=3,
        hmax=100,
        initial_amount=1_000_000,
        buy_cost_pct=[0.001] * 3,
        sell_cost_pct=[0.001] * 3,
        reward_scaling=1e-4,
        state_space=7,  # 1 + 2*3 (tech_indicator_list=[])
        action_space=3,
        tech_indicator_list=[],
        turbulence_threshold=None,
        mode="train",
        iteration=0,
        model_name="ppo",
        print_verbosity=0,
    )


def fake_tiny_data_split(data, start, end):
    return make_tiny_panel()


class NoopCallback:
    def flush(self):
        pass


def noop_callback_factory(**kwargs):
    return NoopCallback()


def base_kwargs(tmp_path: Path, **overrides) -> dict:
    kwargs = dict(
        processed_data={"date": ["2020-01-01", "2020-04-01", "2020-07-01"]},
        env_kwargs={
            "stock_dim": 2,
            "state_space": 301,
            "action_space": 2,
            "hmax": 100,
            "initial_amount": 1_000_000,
            "buy_cost_pct": [0.001] * 2,
            "sell_cost_pct": [0.001] * 2,
            "reward_scaling": 1e-4,
            "tech_indicator_list": [],
        },
        train_period=("2010-01-01", "2020-01-01"),
        val_test_period=("2020-01-01", "2021-01-01"),
        rebalance_window=63,
        validation_window=63,
        ppo_model_kwargs={"n_steps": 8},
        ppo_timesteps=16,
        reward_mode="return",
        lambda_value=None,
        lambda_grid=[0.0],
        clip_c=0.02,
        run_id="tg_test",
        seed=0,
        output_root=tmp_path / "scripts" / "MDA_PPO" / "runs",
        experiment_name="tail_gae_test",
        rolling_window_count=4,
        data_split_fn=fake_data_split,
        env_cls=FakeEnv,
    )
    kwargs.update(overrides)
    return kwargs


def make_runner(tmp_path: Path, **overrides) -> tgr.TailGAERollingRunner:
    return tgr.TailGAERollingRunner(**base_kwargs(tmp_path, **overrides))


# ---------------------------------------------------------------------------
# PU-6 hard constraint + constructor injection
# ---------------------------------------------------------------------------

def test_pu6_rejects_non_63_rebalance_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PU-6"):
        make_runner(tmp_path, rebalance_window=62)
    with pytest.raises(ValueError, match="PU-6"):
        make_runner(tmp_path, rebalance_window=126)


def test_pu6_rejects_non_63_validation_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="PU-6"):
        make_runner(tmp_path, validation_window=62)
    with pytest.raises(ValueError, match="PU-6"):
        make_runner(tmp_path, validation_window=64)


def test_pu6_accepts_63_63(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    assert runner.rebalance_window == 63
    assert runner.validation_window == 63


def test_constructor_injects_v2_parameters(tmp_path: Path) -> None:
    worst20 = {"2020-01-01": 0.05}
    i_by_date = {"2020-01-01": 0}
    runner = make_runner(
        tmp_path,
        tail_alpha=0.5,
        i_by_date=i_by_date,
        risk_state_names=NAMES_5,
        worst20_by_date=worst20,
        expected_risk_observation_fields=NAMES_5,
    )
    assert runner.tail_alpha == 0.5
    assert runner.i_by_date == i_by_date
    assert runner.risk_state_names == NAMES_5
    assert runner.worst20_by_date == worst20
    assert runner.expected_risk_observation_fields == NAMES_5
    assert runner._current_risk_wrapper is None


def test_w0_decouples_internal_and_policy_risk_fields(tmp_path: Path) -> None:
    runner = make_runner(
        tmp_path,
        i_by_date={f"2020-01-{day:02d}": 0 for day in range(1, 16)},
        internal_risk_fields=NAMES_5,
        policy_risk_fields=[],
        expected_risk_observation_fields=[],
        compute_direct_diagnostics=True,
    )
    assert runner.risk_state_names == NAMES_5
    assert runner.internal_risk_fields == NAMES_5
    assert runner.policy_risk_fields == []
    window = runner.build_rolling_windows()[0]
    env = runner._build_env(window=window, mode="train", seed=0)
    assert runner._observation_metadata == {
        "use_risk_observation": False,
        "base_observation_dim": 301,
        "risk_observation_dim": 0,
        "risk_observation_fields": [],
        "policy_observation_dim": 301,
        "phase9_observation_dim": 0,
        "phase9_observation_fields": [],
    }


def test_constructor_validates_alpha_and_i_by_date(tmp_path: Path) -> None:
    # tail_alpha > 0 needs the I_t data source (risk_state_names)
    with pytest.raises(ValueError, match="tail_alpha"):
        make_runner(tmp_path, tail_alpha=0.5)
    # risk_state_names requires the precomputed latch series
    with pytest.raises(ValueError, match="i_by_date"):
        make_runner(
            tmp_path,
            tail_alpha=0.5,
            risk_state_names=NAMES_5,
        )
    # 1200 bridge: alpha=0 and no names is the only legal no-risk combo
    runner = make_runner(tmp_path, tail_alpha=0.0)
    assert runner.risk_state_names == []


# ---------------------------------------------------------------------------
# Window derivation (iterations 126/189/252/315) + dry-run plan
# ---------------------------------------------------------------------------

def test_window_derivation_four_windows(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-01", "2026-06-01").strftime("%Y-%m-%d").tolist()
    runner = make_runner(
        tmp_path,
        processed_data={"date": dates},
        val_test_period=(dates[0], dates[-1]),
        rolling_window_count=None,
    )
    windows = runner.build_rolling_windows()
    assert [window.iteration for window in windows] == [126, 189, 252, 315]
    assert [window.label for window in windows] == [
        "iter_126",
        "iter_189",
        "iter_252",
        "iter_315",
    ]
    assert [window.index for window in windows] == [0, 1, 2, 3]


def test_dry_run_plan_walks_through_four_windows(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-01", "2026-06-01").strftime("%Y-%m-%d").tolist()
    runner = make_runner(
        tmp_path,
        processed_data={"date": dates},
        val_test_period=(dates[0], dates[-1]),
        rolling_window_count=None,
    )
    plan = runner.run(dry_run=True)
    assert [window["iteration"] for window in plan["rolling_windows"]] == [
        126,
        189,
        252,
        315,
    ]
    assert plan["rebalance_window"] == 63
    assert plan["validation_window"] == 63
    assert plan["reward_mode"] == "return"
    assert plan["planned_models"] == ["ppo"]
    assert plan["training_budget"]["rolling_window_count"] == 4
    assert plan["training_budget"]["estimated_total_ppo_train_count"] == 4
    assert plan["training_budget"]["estimated_total_ppo_timesteps"] == 4 * 16


def test_max_smoke_windows_limits_derivation(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-01-01", "2026-06-01").strftime("%Y-%m-%d").tolist()
    runner = make_runner(
        tmp_path,
        processed_data={"date": dates},
        val_test_period=(dates[0], dates[-1]),
        rolling_window_count=None,
        max_smoke_windows=1,
    )
    windows = runner.build_rolling_windows()
    assert len(windows) == 1
    assert windows[0].iteration == 126


# ---------------------------------------------------------------------------
# Observation contract: V2 branch (301 base + declared names) and bridge
# ---------------------------------------------------------------------------

def test_v2_observation_contract_base_301_plus_names(tmp_path: Path) -> None:
    runner = make_runner(
        tmp_path,
        use_risk_observation=True,
        risk_state_names=NAMES_5,
        expected_risk_observation_fields=NAMES_5,
        tail_alpha=0.5,
        i_by_date={"2020-01-01": 0},
    )
    window = runner.build_rolling_windows()[0]
    env = runner._build_env(window=window, mode="train", seed=0)
    # P0-1: the bare base env is always constructed with use_risk_observation
    # forced False, even when the run-level switch is True
    assert env.kwargs["use_risk_observation"] is False
    assert env.reward_mode == "return"
    assert runner._observation_metadata == {
        "use_risk_observation": True,
        "base_observation_dim": 301,
        "risk_observation_dim": 5,
        "risk_observation_fields": list(NAMES_5),
        "policy_observation_dim": 301 + 5,
        "phase9_observation_dim": 0,
        "phase9_observation_fields": [],
    }
    assert runner._risk_base_dim() == 301


def test_bridge_observation_contract_base_301(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, use_risk_observation=False)
    window = runner.build_rolling_windows()[0]
    env = runner._build_env(window=window, mode="train", seed=0)
    assert env.kwargs["use_risk_observation"] is False
    assert env.reward_mode == "return"
    # parent branch: fixed schema, no risk dims added
    assert runner._observation_metadata == {
        "use_risk_observation": False,
        "base_observation_dim": 301,
        "risk_observation_dim": len(RISK_OBSERVATION_FIELDS),
        "risk_observation_fields": list(RISK_OBSERVATION_FIELDS),
        "policy_observation_dim": 301,
    }


def test_risk_base_dim_unavailable_before_env_build(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    with pytest.raises(RuntimeError, match="base observation dim"):
        runner._risk_base_dim()


# ---------------------------------------------------------------------------
# 1200 bridge: no wrapper, no diagnostics, reward_mode="return" (P1-12)
# ---------------------------------------------------------------------------

def test_bridge_build_train_env_skips_wrapper(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    window = runner.build_rolling_windows()[0]
    train_env = runner._build_train_env(window, seed=0)
    assert runner._current_risk_wrapper is None
    # parent path: get_sb_env() vec env handed to the training normalizer,
    # which is disabled here -> the vec env comes back unchanged
    assert isinstance(train_env, FakeVecEnv)


def test_bridge_training_mounts_no_diagnostics_callback(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class StubPPO:
        def __init__(self, policy, env, **kwargs):
            captured["kwargs"] = kwargs
            self.num_timesteps = 0

        def learn(self, **kwargs):
            captured["learn_kwargs"] = kwargs
            self.num_timesteps = int(kwargs["total_timesteps"])

    monkeypatch.setattr(tgr, "TailGAEPPO", StubPPO)
    events: list[dict[str, object]] = []
    runner = make_runner(
        tmp_path,
        tail_alpha=0.0,
        training_event_callback=events.append,
        training_callback_factory=noop_callback_factory,
    )
    window = runner.build_rolling_windows()[0]
    train_env = runner._build_train_env(window, seed=0)
    runner._train_ppo_for_window(window=window, train_env=train_env, seed=0)

    assert captured["kwargs"]["tail_alpha"] == 0.0
    assert captured["kwargs"]["risk_wrapper"] is None
    # no diagnostics callback for the bridge (spec §3.3): only the audit
    # training callback is mounted, so nothing [risk_state_env] ever prints
    callbacks = captured["learn_kwargs"]["callback"]
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], NoopCallback)
    assert not any(
        isinstance(callback, tgr._DiagnosticsCallback) for callback in callbacks
    )
    assert events[0]["status"] == "completed"


def test_v2_training_mounts_diagnostics_callback(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class StubPPO:
        def __init__(self, policy, env, **kwargs):
            captured["kwargs"] = kwargs
            self.num_timesteps = 0
            # the V2 wiring calls open_rl_log/close_rl_log on the buffer
            self.rollout_buffer = types.SimpleNamespace(
                open_rl_log=lambda path: None,
                close_rl_log=lambda: None,
            )

        def learn(self, **kwargs):
            captured["learn_kwargs"] = kwargs
            self.num_timesteps = int(kwargs["total_timesteps"])

    monkeypatch.setattr(tgr, "TailGAEPPO", StubPPO)
    events: list[dict[str, object]] = []
    i_by_date = {"2020-01-01": 0}
    ew_by_date = {
        "mkt_ew_10": {"2020-01-01": -0.02},
        "mkt_ew_5": {"2020-01-01": 0.0},
    }
    runner = make_runner(
        tmp_path,
        env_cls=DownsideBreadthStockTradingEnv,
        data_split_fn=fake_tiny_data_split,
        env_kwargs=tiny_env_kwargs(),
        risk_state_names=NAMES_5,
        expected_risk_observation_fields=NAMES_5,
        tail_alpha=0.5,
        i_by_date=i_by_date,
        worst20_by_date={},
        ew_by_date=ew_by_date,
        tilt_shift=True,
        ge_penalty_beta=0.05,
        ge_penalty_ref=0.95,
        training_event_callback=events.append,
        training_callback_factory=noop_callback_factory,
        # _tail_mechanism_dir asserts the run root is a timestamp dir
        # (output_root.parents[2]); mirror the real layout for the diag wiring.
        output_root=tmp_path / "20260814_120000" / "scripts" / "MDA_PPO" / "runs",
    )
    window = runner.build_rolling_windows()[0]
    train_env = runner._build_train_env(window, seed=0)

    wrapper = runner._current_risk_wrapper
    assert wrapper is not None
    assert wrapper.n_risk_dims == 5
    assert wrapper.names == NAMES_5
    # wrapper's observation = base (7) + risk block (5)
    assert train_env.observation_space.shape[0] == 7 + 5

    runner._train_ppo_for_window(window=window, train_env=train_env, seed=0)
    assert captured["kwargs"]["tail_alpha"] == 0.5
    assert captured["kwargs"]["risk_wrapper"] is wrapper
    assert captured["kwargs"]["tilt_shift"] is True
    # i_by_date reaches the mechanism through the wrapper (latch lookup)
    assert wrapper.i_by_date == i_by_date
    # 1216: the market-EW tables reach the wrapper for the obs features
    assert wrapper.ew_by_date == ew_by_date
    # 1218: the ge-penalty params reach the wrapper
    assert wrapper.ge_penalty_beta == 0.05
    assert wrapper.ge_penalty_ref == 0.95
    callbacks = captured["learn_kwargs"]["callback"]
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], NoopCallback)
    assert isinstance(callbacks[1], tgr._DiagnosticsCallback)
    assert events[0]["status"] == "completed"


def test_w0_real_train_env_computes_internal_block_but_policy_stays_base_dim(
    tmp_path: Path,
) -> None:
    runner = make_runner(
        tmp_path,
        env_cls=DownsideBreadthStockTradingEnv,
        data_split_fn=fake_tiny_data_split,
        env_kwargs=tiny_env_kwargs(),
        internal_risk_fields=NAMES_5,
        policy_risk_fields=[],
        expected_risk_observation_fields=[],
        compute_direct_diagnostics=True,
        i_by_date={f"2020-01-{day:02d}": 0 for day in range(1, 16)},
        worst20_by_date={},
    )
    window = runner.build_rolling_windows()[0]
    train_env = runner._build_train_env(window, seed=0)
    wrapper = runner._current_risk_wrapper
    assert wrapper is not None
    assert wrapper.names == NAMES_5
    assert wrapper.policy_names == []
    assert wrapper.n_internal_risk_dims == 5
    assert wrapper.n_risk_dims == 0
    assert wrapper.direct_penalty_ref == pytest.approx(0.85)
    assert train_env.observation_space.shape == (7,)
    obs = train_env.reset()
    assert obs.shape == (1, 7)
    train_env.step(np.zeros((1, 3), dtype=float))
    assert len(wrapper._risk_block_history[-1]) == 5


# ---------------------------------------------------------------------------
# _DiagnosticsCallback: 8 blocks at 5..40, off-by-one backfill (P2-5)
# ---------------------------------------------------------------------------

class FakeDiagnosticWrapper:
    def __init__(self):
        self.iterations: list[int] = []

    def print_diagnostics(self, iteration: int) -> None:
        self.iterations.append(iteration)

    def flush_per_step_log(self) -> None:
        pass


def make_callback(wrapper, n_steps=1024, num_timesteps=40000):
    callback = tgr._DiagnosticsCallback(
        wrapper=wrapper, n_steps=n_steps, num_timesteps=num_timesteps
    )
    # _on_rollout_end reads model.rollout_buffer._tail_diag (None -> no rows)
    callback.model = types.SimpleNamespace(
        rollout_buffer=types.SimpleNamespace(_tail_diag=None)
    )
    return callback


def _drive_rollouts(callback, n):
    """SB3 loop order: collect_rollouts -> on_rollout_end -> dump_logs ->
    next rollout's first _on_step (which flushes the pending block)."""
    for _ in range(n):
        callback._on_rollout_end()
        callback._on_step()


def test_diagnostics_callback_emits_exactly_8_blocks_per_window() -> None:
    wrapper = FakeDiagnosticWrapper()
    callback = make_callback(wrapper)
    _drive_rollouts(callback, 40)  # 40000/1024 = 39.06 -> 40 rollout iterations
    assert wrapper.iterations == [5, 10, 15, 20, 25, 30, 35, 40]
    # P2-5: count % 5 == 0 -> no backfill, otherwise a 9th block would appear
    callback._on_training_end()
    assert wrapper.iterations == [5, 10, 15, 20, 25, 30, 35, 40]


def test_diagnostics_callback_backfills_final_block_when_count_not_multiple() -> None:
    wrapper = FakeDiagnosticWrapper()
    callback = make_callback(wrapper)
    # 3 rollouts: no complete 5-block yet
    _drive_rollouts(callback, 3)
    assert wrapper.iterations == []
    callback._on_training_end()
    assert wrapper.iterations == [40]  # ceil(40000 / 1024)
    # 38 rollouts: blocks 5..35 fired, the trailing 3 iterations are backfilled
    wrapper = FakeDiagnosticWrapper()
    callback = make_callback(wrapper)
    _drive_rollouts(callback, 38)
    assert wrapper.iterations == [5, 10, 15, 20, 25, 30, 35]
    callback._on_training_end()
    assert wrapper.iterations == [5, 10, 15, 20, 25, 30, 35, 40]


def test_diagnostics_callback_backfill_iteration_ceiling() -> None:
    wrapper = FakeDiagnosticWrapper()
    callback = make_callback(wrapper, num_timesteps=4096)
    _drive_rollouts(callback, 3)
    callback._on_training_end()
    assert wrapper.iterations == [4]  # ceil(4096 / 1024)


# ---------------------------------------------------------------------------
# audit_mda_ppo_run runner_kwargs injection (P0-3) and V2 contract closure
# ---------------------------------------------------------------------------

class StubAuditRunner:
    last_kwargs: dict = {}
    observation_contract: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)

    def run(self, dry_run=False):
        output_root = Path(type(self).last_kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        summary = output_root / "summary.csv"
        pd.DataFrame(
            {"date": ["2025-01-02"], "account_value": [1_000_000.0]}
        ).to_csv(summary, index=False)
        trade = output_root / "trade.csv"
        pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03"],
                "account_value": [1_000_000.0, 1_001_000.0],
            }
        ).to_csv(trade, index=False)
        return {
            "runner_summary_path": str(summary),
            "windows": [{"trade_account_value_path": str(trade)}],
            "observation": dict(type(self).observation_contract),
            "quality_gate_failed": False,
            "quality_gate_results": [],
        }


def make_audit_config(tmp_path: Path, **overrides) -> MDAAuditConfig:
    cfg = dict(
        seed=0,
        lambda_value=None,
        output_root=tmp_path,
        data_path=tmp_path / "data.csv",
        notebook_path=tmp_path / "missing.ipynb",
        dji_path=None,
        experiment_name="tail_gae_test",
        artifact_id="1205 tail_gae_test",
        reward_mode="return",
        run_directory_layout="artifact_seed_timestamp",
        deterministic=True,
        allow_ineligible_backtest=True,
        normalization_config={},
        quality_gate_config={},
        use_risk_observation=True,
        risk_observation_fields=list(NAMES_5),
    )
    cfg.update(overrides)
    return MDAAuditConfig(**cfg)


def run_audit_with_stub(
    tmp_path: Path,
    config: MDAAuditConfig,
    contract: dict,
    runner_kwargs: dict | None,
):
    StubAuditRunner.last_kwargs = {}
    StubAuditRunner.observation_contract = contract
    panel = pd.DataFrame(
        {"date": ["2025-01-02"], "tic": ["A"], "close": [1.0]}
    )
    return audit_mda_ppo_run(
        config=config,
        processed=panel,
        raw_data=panel,
        env_kwargs={"stock_dim": 1, "state_space": 301},
        train_period=["2025-01-01", "2025-01-02"],
        val_test_period=["2025-01-02", "2025-01-03"],
        rebalance_window=63,
        validation_window=63,
        ppo_model_kwargs={"n_steps": 8},
        ppo_timesteps=40000,
        runner_cls=StubAuditRunner,
        runner_kwargs=runner_kwargs,
    )


def test_audit_config_validates_risk_observation_fields(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="sequence of strings"):
        make_audit_config(tmp_path, risk_observation_fields="abc")
    with pytest.raises(ValueError, match="not be empty"):
        make_audit_config(tmp_path, risk_observation_fields=[])
    with pytest.raises(TypeError, match="non-empty strings"):
        make_audit_config(tmp_path, risk_observation_fields=["a", ""])
    # None default (1200) is untouched
    config = make_audit_config(
        tmp_path, use_risk_observation=False, risk_observation_fields=None
    )
    assert config.risk_observation_fields is None
    w0_config = make_audit_config(
        tmp_path, use_risk_observation=False, risk_observation_fields=[]
    )
    assert w0_config.risk_observation_fields == []


def test_audit_accepts_w0_internal_wrapper_zero_policy_fields(tmp_path: Path) -> None:
    config = make_audit_config(
        tmp_path,
        use_risk_observation=False,
        risk_observation_fields=[],
    )
    contract = {
        "use_risk_observation": False,
        "base_observation_dim": 301,
        "risk_observation_dim": 0,
        "risk_observation_fields": [],
        "policy_observation_dim": 301,
    }
    result = run_audit_with_stub(
        tmp_path,
        config,
        contract=contract,
        runner_kwargs={
            "internal_risk_fields": list(NAMES_5),
            "policy_risk_fields": [],
            "expected_risk_observation_fields": [],
        },
    )
    saved = json.loads(
        (result.run_dir / "audit" / "observation_contract.json").read_text()
    )
    assert saved == contract
    experiment_config = json.loads(
        (result.run_dir / "audit" / "experiment_config.json").read_text()
    )
    assert experiment_config["extra"]["risk_observation_dim"] == 0
    assert experiment_config["extra"]["risk_observation_fields"] == []


def test_audit_injects_runner_kwargs(tmp_path: Path) -> None:
    config = make_audit_config(tmp_path)
    contract = {
        "use_risk_observation": True,
        "base_observation_dim": 301,
        "risk_observation_dim": 5,
        "risk_observation_fields": list(NAMES_5),
        "policy_observation_dim": 306,
    }
    result = run_audit_with_stub(
        tmp_path,
        config,
        contract=contract,
        runner_kwargs={
            "tail_alpha": 0.5,
            "i_by_date": {"2025-01-02": 0},
            "risk_state_names": list(NAMES_5),
            "worst20_by_date": {"2025-01-02": 0.05},
            "expected_risk_observation_fields": list(NAMES_5),
        },
    )
    kwargs = StubAuditRunner.last_kwargs
    assert kwargs["tail_alpha"] == 0.5
    assert kwargs["i_by_date"] == {"2025-01-02": 0}
    assert kwargs["risk_state_names"] == list(NAMES_5)
    assert kwargs["expected_risk_observation_fields"] == list(NAMES_5)
    assert kwargs["worst20_by_date"] == {"2025-01-02": 0.05}
    # fixed audit-side kwargs are unchanged
    assert kwargs["rebalance_window"] == 63
    assert kwargs["validation_window"] == 63
    assert kwargs["reward_mode"] == "return"
    assert kwargs["seed"] == 0
    assert kwargs["run_id"] == "rolling"
    assert kwargs["ppo_timesteps"] == 40000
    assert kwargs["max_smoke_windows"] is None
    # P1-11: the config snapshot records the declared names, not the fixed schema
    exp_cfg = json.loads(
        (result.run_dir / "audit" / "experiment_config.json").read_text()
    )
    assert exp_cfg["extra"]["risk_observation_fields"] == list(NAMES_5)
    # observation contract persisted through the audit output
    saved = json.loads(
        (result.run_dir / "audit" / "observation_contract.json").read_text()
    )
    assert saved["policy_observation_dim"] == 306
    assert saved["base_observation_dim"] == 301


def test_audit_default_runner_kwargs_none_keeps_1200_construction(
    tmp_path: Path,
) -> None:
    config = make_audit_config(
        tmp_path, use_risk_observation=False, risk_observation_fields=None
    )
    contract = {
        "use_risk_observation": False,
        "base_observation_dim": None,
        "risk_observation_dim": len(RISK_OBSERVATION_FIELDS),
        "risk_observation_fields": list(RISK_OBSERVATION_FIELDS),
        "policy_observation_dim": None,
    }
    run_audit_with_stub(tmp_path, config, contract=contract, runner_kwargs=None)
    kwargs = StubAuditRunner.last_kwargs
    assert "tail_alpha" not in kwargs
    assert "i_by_date" not in kwargs
    assert "risk_state_names" not in kwargs
    assert "expected_risk_observation_fields" not in kwargs


def test_audit_v2_observation_contract_mismatch_raises(tmp_path: Path) -> None:
    config = make_audit_config(tmp_path)
    contract = {
        "use_risk_observation": True,
        "base_observation_dim": 301,
        "risk_observation_dim": 5,
        "risk_observation_fields": list(NAMES_5),
        "policy_observation_dim": 305,  # inconsistent: 301 + 5
    }
    with pytest.raises(RuntimeError, match="policy observation dimension"):
        run_audit_with_stub(
            tmp_path,
            config,
            contract=contract,
            runner_kwargs={"risk_state_names": list(NAMES_5)},
        )


def test_audit_v2_declared_field_length_mismatch_raises(tmp_path: Path) -> None:
    config = make_audit_config(tmp_path)
    contract = {
        "use_risk_observation": True,
        "base_observation_dim": 301,
        "risk_observation_dim": 7,
        "risk_observation_fields": [
            "current_drawdown",
            "running_max_drawdown",
            "drawdown_delta",
            "gross_exposure",
            "downside_volatility_20",
            "streak20max",
            "worst20",
        ],
        "policy_observation_dim": 308,
    }
    with pytest.raises(RuntimeError, match="do not match the declared"):
        run_audit_with_stub(
            tmp_path,
            config,
            contract=contract,
            runner_kwargs={"risk_state_names": list(NAMES_5)},
        )
