"""Tail-GAE rolling runner (spec §3.1) and per-window diagnostics callback (§3.3).

``TailGAERollingRunner`` subclasses ``SinglePPORollingRunner`` and inherits the
1200 (V1) rolling protocol unchanged — PU-6 63/63 assertions, window derivation
(iterations 126/189/252/315), cumulative-expanding training data, the
1-day-overlap validation/trade segments, and the trade continuity state chain —
while swapping in the V2 mechanism layer: ``RiskStateEnvWrapper`` around the
base env, ``SliceVecNormalize``/``FrozenSliceNormalizationEnv`` normalization,
and ``TailGAEPPO`` with its Tail-GAE rollout buffer.

The 1200 bridge (``risk_state_names == []``, alpha=0) runs the same rolling
protocol with the wrapper skipped and the parent's single-PPO environment path:
only the environment class and the model class differ from the baseline.
"""

from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from .risk_state_env import RiskStateEnvWrapper
    from .single_ppo_runner import RollingWindow, SinglePPORollingRunner
    from .slice_normalizer import (
        FrozenSliceNormalizationEnv,
        SliceVecNormalize,
        load_slice_normalizer,
    )
    from .tail_gae_ppo import TailGAEPPO
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from risk_state_env import RiskStateEnvWrapper
    from single_ppo_runner import RollingWindow, SinglePPORollingRunner
    from slice_normalizer import (
        FrozenSliceNormalizationEnv,
        SliceVecNormalize,
        load_slice_normalizer,
    )
    from tail_gae_ppo import TailGAEPPO


def _gross_exposure_from_base_state(base_state: Any, *, stock_dim: int) -> float:
    """bridge trade 落盘用 GE：w = Σ|p·s|/V，与 risk_oracle._extract_gross_exposure 同式。

    state 是 env 原生状态（base_state 布局: cash, price_1..n, share_1..n,
    <技术指标...>），与 wrapper 无关（spec §6.4 N1）。
    """
    prices = np.asarray(base_state[1 : 1 + stock_dim], dtype=float)
    shares = np.asarray(base_state[1 + stock_dim : 1 + 2 * stock_dim], dtype=float)
    value = float(base_state[0] + np.sum(prices * shares))
    if value <= 1e-8:
        return 0.0
    return float(min(np.sum(np.abs(prices * shares)) / value, 1.0))


class TailGAERollingRunner(SinglePPORollingRunner):
    """1200 rolling protocol with the V2 risk block + Tail-GAE mechanism.

    Only construction-related methods are overridden; the rolling machinery
    (window derivation, segment splitting, continuity chain, summary output)
    is fully inherited.
    """

    def __init__(
        self,
        *,
        processed_data: Any,
        env_kwargs: dict[str, Any],
        train_period: tuple[str, str] | list[str],
        val_test_period: tuple[str, str] | list[str],
        rebalance_window: int,
        validation_window: int,
        ppo_model_kwargs: dict[str, Any],
        ppo_timesteps: int,
        reward_mode: str,
        lambda_value: float | None,
        lambda_grid: Iterable[float],
        clip_c: float,
        run_id: str,
        seed: int | Iterable[int],
        output_root: str | Path,
        use_risk_observation: bool = False,
        experiment_name: str | None = None,
        rolling_window_count: int | None = None,
        max_smoke_windows: int | None = None,
        drl_agent_cls: type[Any] | None = None,
        data_split_fn: Callable[..., Any] | None = None,
        env_cls: type[Any] | None = None,
        training_callback_factory: Callable[..., Any] | None = None,
        training_event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        diagnostics_buffer_factory: Callable[..., Any] | None = None,
        normalization_config: Mapping[str, Any] | None = None,
        quality_gate_config: Mapping[str, Any] | None = None,
        # --- V3 additions (injected via audit runner_kwargs, spec §4.1) -------
        tail_alpha: float = 0.0,
        i_by_date: Mapping[str, int] | None = None,
        risk_state_names: Sequence[str] | None = None,
        internal_risk_fields: Sequence[str] | None = None,
        policy_risk_fields: Sequence[str] | None = None,
        signal_table: Any = None,  # signature compatibility only (spec P2-3)
        worst20_by_date: Mapping[str, float] | None = None,
        ew_by_date: Mapping[str, Mapping[str, float]] | None = None,
        ge_penalty_beta: float = 0.0,
        ge_penalty_ref: float = 0.95,
        ge_penalty_mode: str = "total",
        tilt_shift: bool = False,
        expected_risk_observation_fields: Sequence[str] | None = None,
        # --- V4.5/1230 additions (direct-gradient penalty, spec §3.4) ------
        direct_penalty_beta: float = 0.0,
        direct_penalty_ref: float = 0.85,
        compute_direct_diagnostics: bool = False,
        # V4.6 Step 4（spec §5.1 P0-1）: d_i 成交瞬间口径开关（默认关闭, 历史口径不变）
        direct_ref_exec_aligned: bool = False,
        # Phase 6（Trajectory-aware Direct Penalty）: 连续监督权重模式与查表
        supervision_mode: str = "legacy",
        supervision_weight_by_date: Mapping[str, float] | None = None,
        supervision_weight_by_window: Mapping[int, Mapping[str, float]] | None = None,
        # Phase 9（contract §1.3/§1.4）: gated 日历按 window 取表
        # {window: {date: {w_down, w_up, up_days_remaining, episode_age,
        # since_episode_end}}}; wrapper 按决策日 transition_date 查表.
        phase9_by_window: Mapping[int, Mapping[str, Mapping[str, float]]] | None = None,
        # Phase 9 §2.1（D-observation）: window -> date -> {rf20_descriptor,
        # episode_age, since_episode_end}——恒为 treatment 真实状态机列, 与 loss
        # 表解耦（1324 loss 读 dose-control 列、obs 仍读 treatment 列）.
        phase9_observation_by_window: Mapping[int, Mapping[str, Mapping[str, float]]] | None = None,
        use_phase9_observation: bool = False,
        # Phase 9 §3.2/§3.3（B/C）: β_up（回补项系数, 0.0 = 关闭）与 actor-mean
        # 近端带 δ_μ（None = 关闭）.
        recovery_penalty_beta: float = 0.0,
        aux_mean_prox_clip: float | None = None,
        # Δ1321b（spec §1.1/§6.1）: 条件化暴露地板 L_CEF 的剂量与参数
        # （β=0.0 = 关闭 = 与现码逐位等价）; 策略快照开关供 §4.3 事后诊断.
        cef_penalty_beta: float = 0.0,
        cef_floor_G: float = 0.90,
        cef_band_delta: float = 0.02,
        cef_min_capacity: float = 1e-3,
        save_policy_snapshot: bool = False,
        # V4.12 合同二（spec Part 3）: 非对称 PCGrad 开关（默认关闭 = 与现状逐位一致）
        pcgrad: bool = False,
        # --- V4.6 Step 5 additions (execution-layer risk overlay, spec §6) -
        risk_overlay: str | None = None,
        overlay_i_by_date: Mapping[str, int] | None = None,
        trade_log: bool = False,
    ) -> None:
        super().__init__(
            processed_data=processed_data,
            env_kwargs=env_kwargs,
            train_period=train_period,
            val_test_period=val_test_period,
            rebalance_window=rebalance_window,
            validation_window=validation_window,
            ppo_model_kwargs=ppo_model_kwargs,
            ppo_timesteps=ppo_timesteps,
            reward_mode=reward_mode,
            lambda_value=lambda_value,
            lambda_grid=lambda_grid,
            clip_c=clip_c,
            run_id=run_id,
            seed=seed,
            output_root=output_root,
            use_risk_observation=use_risk_observation,
            experiment_name=experiment_name,
            rolling_window_count=rolling_window_count,
            max_smoke_windows=max_smoke_windows,
            drl_agent_cls=drl_agent_cls,
            data_split_fn=data_split_fn,
            env_cls=env_cls,
            training_callback_factory=training_callback_factory,
            training_event_callback=training_event_callback,
            diagnostics_buffer_factory=diagnostics_buffer_factory,
            normalization_config=normalization_config,
            quality_gate_config=quality_gate_config,
        )
        self.tail_alpha = float(tail_alpha)
        self.i_by_date = dict(i_by_date or {})
        legacy_risk_fields = list(risk_state_names or [])
        self.internal_risk_fields = (
            legacy_risk_fields if internal_risk_fields is None else list(internal_risk_fields)
        )
        self.policy_risk_fields = (
            legacy_risk_fields if policy_risk_fields is None else list(policy_risk_fields)
        )
        if any(name not in self.internal_risk_fields for name in self.policy_risk_fields):
            raise ValueError("policy_risk_fields must be a subset of internal_risk_fields")
        # Backward-compatible internal signal alias. Existing experiments pass
        # only risk_state_names and therefore retain identical behavior.
        self.risk_state_names = list(self.internal_risk_fields)
        self.signal_table = signal_table
        self.worst20_by_date = dict(worst20_by_date or {})
        self.ew_by_date = {
            name: dict(by_date)
            for name, by_date in (ew_by_date or {}).items()
        }
        self.ge_penalty_beta = float(ge_penalty_beta)
        self.ge_penalty_ref = float(ge_penalty_ref)
        self.ge_penalty_mode = str(ge_penalty_mode)
        self.tilt_shift = bool(tilt_shift)
        self.direct_penalty_beta = float(direct_penalty_beta)
        self.direct_penalty_ref = float(direct_penalty_ref)
        self.compute_direct_diagnostics = bool(compute_direct_diagnostics)
        self.direct_ref_exec_aligned = bool(direct_ref_exec_aligned)
        self.supervision_mode = str(supervision_mode)
        if self.supervision_mode not in ("legacy", "trajectory"):
            raise ValueError(
                f"invalid supervision_mode={supervision_mode!r}; "
                "expected 'legacy' or 'trajectory'"
            )
        self.supervision_weight_by_date = None
        self.supervision_weight_by_window = None
        if self.supervision_mode == "trajectory":
            flat_table = dict(supervision_weight_by_date or {})
            window_tables = {
                int(index): dict(table)
                for index, table in (supervision_weight_by_window or {}).items()
            }
            if bool(flat_table) == bool(window_tables):
                raise ValueError(
                    "supervision_mode='trajectory' requires exactly one of "
                    "supervision_weight_by_date or supervision_weight_by_window"
                )
            tables = {-1: flat_table} if flat_table else window_tables
            for index, table in tables.items():
                if not table:
                    raise ValueError(f"empty supervision table for window {index}")
                for d, w in table.items():
                    if not np.isfinite(w) or not (0.0 <= w <= 1.0):
                        raise ValueError(
                            f"supervision weight out of range for window={index}, "
                            f"date={d!r}: {w!r} (require finite value in [0, 1])"
                        )
            self.supervision_weight_by_date = flat_table or None
            self.supervision_weight_by_window = window_tables or None
        # Phase 9: window -> date -> row 查表（按 window 取表, 禁止压平）.
        self.phase9_by_window = None
        if phase9_by_window:
            normalized: dict[int, dict[str, dict[str, float]]] = {}
            required = ("w_down", "w_up", "up_days_remaining", "episode_age", "since_episode_end")
            for index, table in phase9_by_window.items():
                if not isinstance(index, int):
                    raise ValueError("phase9_by_window keys must be int window indices")
                rows = {str(d): dict(row) for d, row in table.items()}
                for d, row in rows.items():
                    missing = [k for k in required if k not in row]
                    if missing:
                        raise ValueError(f"phase9 row window={index} date={d} missing {missing}")
                normalized[int(index)] = rows
            self.phase9_by_window = normalized
        # Phase 9 §2.1: observation 表按 window 规范化（treatment 真实列）.
        self.phase9_observation_by_window = None
        if phase9_observation_by_window:
            obs_norm: dict[int, dict[str, dict[str, float]]] = {}
            obs_required = ("rf20_descriptor", "episode_age", "since_episode_end")
            for index, table in phase9_observation_by_window.items():
                if not isinstance(index, int):
                    raise ValueError(
                        "phase9_observation_by_window keys must be int window indices")
                rows = {str(d): dict(row) for d, row in table.items()}
                for d, row in rows.items():
                    missing = [k for k in obs_required if k not in row]
                    if missing:
                        raise ValueError(
                            f"phase9 obs row window={index} date={d} missing {missing}")
                obs_norm[int(index)] = rows
            self.phase9_observation_by_window = obs_norm
        self.use_phase9_observation = bool(use_phase9_observation)
        if self.use_phase9_observation and not self.phase9_observation_by_window:
            raise ValueError(
                "use_phase9_observation=True requires a non-empty "
                "phase9_observation_by_window (contract §2.1)"
            )
        if self.use_phase9_observation and self.phase9_by_window is not None and (
            set(self.phase9_by_window) != set(self.phase9_observation_by_window or {})
        ):
            raise ValueError("phase9 loss/observation tables must share window indices")
        self.recovery_penalty_beta = float(recovery_penalty_beta)
        self.aux_mean_prox_clip = None if aux_mean_prox_clip is None else float(aux_mean_prox_clip)
        self.cef_penalty_beta = float(cef_penalty_beta)
        self.cef_floor_G = float(cef_floor_G)
        self.cef_band_delta = float(cef_band_delta)
        self.cef_min_capacity = float(cef_min_capacity)
        self.save_policy_snapshot = bool(save_policy_snapshot)
        self.pcgrad = bool(pcgrad)
        # Step 5 覆盖层（spec §6）: None=关闭（1200 等价）；触发源 runner 侧
        # 从 tail_gae_data 读入传给执行 env（训练期 env 侧按 mode 跳过）。
        self.risk_overlay = risk_overlay
        self.overlay_i_by_date = dict(overlay_i_by_date or {})
        # §6.4 N1: trade 段暴露比落盘守卫与 wrapper 解耦——trade_log=True 时
        # bridge 模式（无 wrapper）同样落盘，权重从 env 原生 state 直接算。
        self.trade_log = bool(trade_log)
        self.expected_risk_observation_fields = (
            None
            if expected_risk_observation_fields is None
            else list(expected_risk_observation_fields)
        )
        # The live RiskStateEnvWrapper of the current training window, stashed
        # by _build_train_env and consumed by _train_ppo_for_window (TailGAEPPO
        # reads R_t/C_t/I_t/date through it).  None for the 1200 bridge.
        self._current_risk_wrapper: RiskStateEnvWrapper | None = None
        # The RiskStateEnvWrapper of the current validation window, stashed by
        # _build_validation_env and consumed by the validation per-day logger.
        self._current_validation_wrapper: RiskStateEnvWrapper | None = None
        # V4.5/1230: trade 窗口的 wrapper, stashed by _build_trade_env, 供 trade
        # 段逐日暴露比日志（_run_trade_for_window/_predict_trade_account_value_and_state
        # override）读取 GE.
        self._current_trade_wrapper: RiskStateEnvWrapper | None = None
        self._current_trade_log_fh = None
        self._current_trade_log_csv = None
        self._current_trade_ge = None

        if self.tail_alpha != 0.0 and not self.risk_state_names:
            raise ValueError(
                "tail_alpha > 0 requires a non-empty risk_state_names "
                "(I_t data source for the Tail-GAE advantage)"
            )
        if self.risk_state_names and not self.i_by_date:
            raise ValueError(
                "risk_state_names requires i_by_date (V4.2 market-EW latch "
                "series, precomputed by tail_gae_data.load_tail_data)"
            )

    # ------------------------------------------------------------------
    # Observation contract (P0-2)
    # ------------------------------------------------------------------

    def _print_run_start(self) -> None:
        """Print W0/S1-safe internal and policy risk contracts separately."""
        print("============Start Single PPO Strategy============")
        print("============================================")
        print(f"run_id: {self.run_id}")
        print(f"experiment_name: {self.experiment_name}")
        print(f"reward_mode: {self.reward_mode}")
        print(f"ppo_timesteps: {self.ppo_timesteps}")
        print(f"seed(s): {self.seeds}")
        print(
            "risk_wrapper: "
            f"enabled={bool(self.internal_risk_fields)}, "
            f"internal_dim={len(self.internal_risk_fields)}, "
            f"internal_fields={self.internal_risk_fields}"
        )
        print(
            "policy_risk_observation: "
            f"enabled={bool(self.policy_risk_fields)}, "
            f"dim={len(self.policy_risk_fields)}, "
            f"fields={self.policy_risk_fields}"
        )

    def _register_observation_metadata(self, env: Any) -> None:
        """Register the policy observation contract (V2 branch, spec P0-2).

        The parent runs this on the bare base env, which reports
        ``use_risk_observation=False`` plus the fixed single-PPO schema; every
        equality check would fail against the V2 declared names (5/7/10 are not
        a fixed schema).  The non-None branch therefore skips the
        use_risk_observation/fields/dim equality checks and builds the contract
        from the base dim + the declared fields, closing the loop with the
        audit-side ``_validate_observation_metadata``.  ``None`` keeps the
        parent logic untouched (1200 unaffected).
        """
        if self.expected_risk_observation_fields is None:
            super()._register_observation_metadata(env)
            return
        if not hasattr(env, "get_observation_metadata"):
            return
        metadata = dict(env.get_observation_metadata())
        base_dim = int(metadata["base_observation_dim"])
        fields = list(self.policy_risk_fields)
        # Phase 9 §2.1: 注册对象是裸基座 env（phase9 键缺失时 .get 兜底 0）;
        # D 模式的实际 +3 维在 wrapper._assemble 追加, 由本 runner 声明——
        # 与 wrapper.get_observation_metadata（base+risk+phase9）一致.
        n_phase9 = 3 if self.use_phase9_observation else int(metadata.get("phase9_observation_dim", 0))
        phase9_fields = (
            ["rf20_descriptor", "episode_age", "since_episode_end"]
            if self.use_phase9_observation
            else []
        )
        normalized = {
            "use_risk_observation": bool(fields),
            "base_observation_dim": base_dim,
            "risk_observation_dim": len(fields),
            "risk_observation_fields": fields,
            "phase9_observation_dim": n_phase9,
            "phase9_observation_fields": phase9_fields,
            "policy_observation_dim": base_dim + len(fields) + n_phase9,
        }
        current_base_dim = self._observation_metadata["base_observation_dim"]
        if current_base_dim is not None and self._observation_metadata != normalized:
            raise RuntimeError(
                "training, validation, and trade observation contracts differ"
            )
        self._observation_metadata = normalized

    # ------------------------------------------------------------------
    # Environment construction (P0-1 / P1-5)
    # ------------------------------------------------------------------

    def _build_env(
        self,
        *,
        window: RollingWindow,
        mode: str,
        seed: int,
        previous_state: Any | None = None,
        previous_reward_state: Any | None = None,
        previous_continuity_state: Any | None = None,
    ) -> Any:
        """Build the base env with ``use_risk_observation=False`` forced (P0-1).

        The parent hard-passes ``self.use_risk_observation`` (True for V2 runs)
        which trips ``DownsideBreadthStockTradingEnv``'s
        ``assert use_risk_observation is False`` (tail_gae_env.py:44): the risk
        block comes from the wrapper, never from the base env.  The observation
        contract is declared by ``_register_observation_metadata`` instead.
        Everything else mirrors the parent.
        """
        env_cls = self._resolve_env_cls()
        data = self._window_data(window=window, mode=mode)
        kwargs = self._env_kwargs_for_mode(
            mode=mode,
            seed=seed,
            previous_state=previous_state,
        )
        if mode == "trade":
            kwargs["reward_accounting_state"] = previous_reward_state
            kwargs["expected_continuity_state"] = previous_continuity_state
        diagnostics_buffer = None
        if self.diagnostics_buffer_factory is not None:
            diagnostics_buffer = self.diagnostics_buffer_factory(
                mode=mode,
                window=window,
                seed=seed,
            )
        # Step 5: overlay 参数透传给执行 env（训练/验证/trade 同一构造路径，
        # 训练期由 env 侧 mode 守卫跳过——训练产物与 1200 完全一致）
        if self.risk_overlay is not None:
            kwargs["risk_overlay"] = self.risk_overlay
            if self.overlay_i_by_date:
                kwargs["overlay_i_by_date"] = self.overlay_i_by_date
        env = env_cls(
            df=data,
            reward_mode=self.reward_mode,
            lambda_value=self._effective_lambda_value(),
            clip_c=self.clip_c,
            diagnostics_buffer=diagnostics_buffer,
            use_risk_observation=False,
            **kwargs,
        )
        self._register_observation_metadata(env)
        return env

    def _build_train_env(self, window: RollingWindow, *, seed: int) -> Any:
        """Build the training vec env, wrapping the base env in the V2 wrapper.

        Only when ``risk_state_names`` is non-empty is the base env outsourced
        to ``RiskStateEnvWrapper`` (stashed as ``self._current_risk_wrapper``);
        the wrapper is then vec-env'd and handed to the Tail-GAE training
        normalizer.  The bridge (empty names) follows the parent logic
        verbatim (spec P1-5).
        """
        raw_env = self._build_env(window=window, mode="train", seed=seed)
        if not self.risk_state_names:
            if not hasattr(raw_env, "get_sb_env"):
                return raw_env
            sb_env = raw_env.get_sb_env()
            if isinstance(sb_env, tuple):
                vec_env = sb_env[0]
            else:
                vec_env = sb_env
            return self._wrap_training_env(vec_env=vec_env, seed=seed)

        wrapper = self._wrap_risk_state(raw_env, window=window)
        self._current_risk_wrapper = wrapper
        vec_env = DummyVecEnv([lambda: wrapper])
        return self._wrap_training_env(vec_env=vec_env, seed=seed)

    def _build_validation_env(self, window: RollingWindow, *, seed: int) -> Any:
        """Build the validation env, wrapped the same way as training.

        The wrapped env is stashed as ``self._current_validation_wrapper`` so
        the validation per-day logger can read I_t/gross_exposure/date off it.
        """
        env = self._build_env(window=window, mode="validation", seed=seed)
        if self.risk_state_names:
            env = self._wrap_risk_state(env, window=window)
            self._current_validation_wrapper = env
        else:
            self._current_validation_wrapper = None
        return self._wrap_evaluation_env(env=env, window=window, seed=seed)

    def _build_trade_env(
        self,
        window: RollingWindow,
        *,
        seed: int,
        previous_state: Any | None,
        previous_reward_state: Any | None,
        previous_continuity_state: Any | None,
    ) -> Any:
        """Build the trade env (continuity chain + same V2 wrapping)."""
        env = self._build_env(
            window=window,
            mode="trade",
            seed=seed,
            previous_state=previous_state,
            previous_reward_state=previous_reward_state,
            previous_continuity_state=previous_continuity_state,
        )
        if self.risk_state_names:
            env = self._wrap_risk_state(env, window=window)
            self._current_trade_wrapper = env
        else:
            self._current_trade_wrapper = None
        return self._wrap_evaluation_env(env=env, window=window, seed=seed)

    # ------------------------------------------------------------------
    # V4.5/1230: trade 段逐日暴露比日志（tail_mechanism/trade/{window}.csv）
    # 父类 trade 循环不落盘 GE——override 两方法, 循环内逐日写
    # (date, gross_exposure)。gross_exposure 与 validation 同口径: 风险块
    # 仿射经 wrapper 注册表反变换（risk_oracle (2.0,-1.0) → 原始 [0,1]）。
    # ------------------------------------------------------------------

    def _run_trade_for_window(
        self,
        *,
        window: RollingWindow,
        model: Any,
        env: Any,
        seed: int,
    ) -> tuple[Path, Any, Any, Any]:
        # 打开 trade 暴露比日志（§6.4 N1: 守卫与 wrapper 解耦——wrapper 存在
        # 照旧写；trade_log=True 时 bridge 同样落盘, 权重从 env state 直接算）
        self._current_trade_log_fh = None
        self._current_trade_log_csv = None
        if self.trade_log or self._current_trade_wrapper is not None:
            path = self._tail_mechanism_dir() / "trade" / f"{window.label}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._current_trade_log_fh = open(path, "w", newline="")
            self._current_trade_log_csv = csv.writer(self._current_trade_log_fh)
            self._current_trade_log_csv.writerow(["date", "gross_exposure"])
        try:
            output_path = self._account_value_output_path(
                mode="trade",
                window=window,
                seed=seed,
            )
            account_value, last_state, reward_state, continuity_state = (
                self._predict_trade_account_value_and_state(model=model, env=env)
            )
            self._write_account_value(account_value, output_path)
            self._write_overlay_log(env=env, mode="trade", window=window)
            return output_path, last_state, reward_state, continuity_state
        finally:
            if self._current_trade_log_fh is not None:
                self._current_trade_log_fh.close()
                self._current_trade_log_fh = None
                self._current_trade_log_csv = None

    def _write_overlay_log(self, *, env: Any, mode: str, window: RollingWindow) -> None:
        """把 env hook 内记录的 overlay 投影行落盘（C3 验收口径，spec §6.2）。

        路径 tail_mechanism/overlay/{mode}_{window.label}.csv；env 未启用
        overlay（无 overlay_log）时 no-op。行长 4: date, GE_before, GE_after,
        w_max_after——GE_after 是投影瞬间的 GE，非 post-step 日志口径。
        """
        rows = getattr(env, "overlay_log", None)
        if not rows:
            return
        path = self._tail_mechanism_dir() / "overlay" / f"{mode}_{window.label}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "ge_before", "ge_after", "w_max_after"])
            for row in rows:
                writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}"])

    def _predict_trade_account_value_and_state(
        self,
        *,
        model: Any,
        env: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """父类复制 + 逐日写 trade 段暴露比（spec v4.5 补落盘）。

        仅新增暴露比日志行, 交易/状态捕获逻辑与父类（tail_gae_trainer.py
        SinglePPORollingRunner._predict_trade_account_value_and_state）逐行一致。
        """
        if not hasattr(env, "get_sb_env"):
            raise AttributeError("trade environment must provide get_sb_env")
        if not hasattr(env, "df"):
            raise AttributeError("trade environment must expose df for trade length")

        test_env, test_obs = env.get_sb_env()
        trade_day_count = len(env.df.index.unique())
        last_state = None
        last_reward_state = None
        last_continuity_state = None
        account_memory = None

        # wrapper 路径沿用风险块反变换；bridge 路径用 env 原生 state 直接算
        # （§6.4 N1: 权重 w = shares·prices/V，与 wrapper 无关）
        wrapper = self._current_trade_wrapper
        if wrapper is not None:
            ge_idx = wrapper.names.index("gross_exposure")
            ge_a, ge_b = wrapper._dims[ge_idx][2]
        else:
            ge_idx = ge_a = ge_b = None
        for i in range(trade_day_count):
            action, _states = model.predict(test_obs, deterministic=True)
            test_obs, rewards, dones, info = test_env.step(action)

            if self._current_trade_log_csv is not None:
                if wrapper is not None:
                    ge_block = float(wrapper._risk_block_history[-1][ge_idx])
                    date, ge = wrapper.last_date, (ge_block - ge_b) / ge_a
                else:
                    date = env.date_memory[-1]
                    ge = _gross_exposure_from_base_state(
                        env.get_base_state(),
                        stock_dim=int(env.stock_dim),
                    )
                self._current_trade_log_csv.writerow([date, f"{ge:.6f}"])

            if i == trade_day_count - 2:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                # ``render`` may include the schema-defined risk observation.
                # FinRL's next trade environment must receive only its original
                # portfolio state; risk continuity travels separately below.
                last_state = test_env.env_method(method_name="get_base_state")
                last_reward_state = test_env.env_method(method_name="get_reward_accounting_state")
                last_continuity_state = test_env.env_method(method_name="get_account_continuity_state")

            if dones[0]:
                break

        if account_memory is None or last_state is None:
            raise RuntimeError(
                "failed to capture trade terminal state at the expected step; "
                "trade loop may have ended early or the capture index is wrong"
            )

        return account_memory[0], last_state[0], None if last_reward_state is None else last_reward_state[0], None if last_continuity_state is None else last_continuity_state[0]

    def _wrap_risk_state(
        self, env: Any, *, window: RollingWindow | None = None
    ) -> RiskStateEnvWrapper:
        """Outsource the base env to the risk-state wrapper."""
        supervision_table = self._supervision_table_for_window(window)
        phase9_table = self._phase9_table_for_window(window)
        obs_table = (
            self._phase9_obs_table_for_window(window)
            if self.use_phase9_observation
            else None
        )
        return RiskStateEnvWrapper(
            env,
            self.risk_state_names,
            i_by_date=self.i_by_date,
            worst20_by_date=self.worst20_by_date,
            ew_by_date=self.ew_by_date,
            ge_penalty_beta=self.ge_penalty_beta,
            ge_penalty_ref=self.ge_penalty_ref,
            ge_penalty_mode=self.ge_penalty_mode,
            # 直通配置关闭时（1218/1219/bridge）不计算 d_i（spec §3.1 守卫）
            direct_penalty_ref=(
                self.direct_penalty_ref
                if self.direct_penalty_beta != 0.0 or self.compute_direct_diagnostics
                else None
            ),
            # V4.6 Step 4: d_i 成交瞬间口径（默认 False = 历史 post-action 口径）
            direct_ref_exec_aligned=self.direct_ref_exec_aligned,
            policy_names=self.policy_risk_fields,
            supervision_mode=self.supervision_mode,
            supervision_weight_by_date=supervision_table,
            phase9_by_date=phase9_table,
            use_phase9_observation=self.use_phase9_observation,
            phase9_obs_by_date=obs_table,
        )

    def _phase9_obs_table_for_window(
        self, window: RollingWindow | None
    ) -> Mapping[str, Mapping[str, float]] | None:
        """Resolve the Phase 9 §2.1 observation table in the rolling-window context.

        恒为 treatment 真实状态机列（1324 loss 表为 dose-control 时 obs 表不变）;
        D 模式关闭时恒 None.
        """
        if not self.use_phase9_observation:
            return None
        if self.phase9_observation_by_window is None:
            raise ValueError(
                "use_phase9_observation=True requires window-specific "
                "phase9_observation_by_window"
            )
        if window is None:
            raise ValueError(
                "window context is required for window-specific phase9 obs tables"
            )
        try:
            return self.phase9_observation_by_window[int(window.index)]
        except KeyError:
            raise KeyError(
                f"no phase9 observation table for rolling window index {window.index}"
            ) from None

    def _phase9_table_for_window(
        self, window: RollingWindow | None
    ) -> Mapping[str, Mapping[str, float]] | None:
        """Resolve the Phase 9 gated table in the current rolling-window context."""
        if self.phase9_by_window is None:
            return None
        if window is None:
            raise ValueError("window context is required for window-specific phase9 tables")
        try:
            return self.phase9_by_window[int(window.index)]
        except KeyError:
            raise KeyError(
                f"no phase9 table for rolling window index {window.index}"
            ) from None

    def _supervision_table_for_window(
        self, window: RollingWindow | None
    ) -> Mapping[str, float] | None:
        """Resolve the trajectory table in the current rolling-window context."""
        supervision_table = self.supervision_weight_by_date
        if self.supervision_mode == "trajectory" and self.supervision_weight_by_window is not None:
            if window is None:
                raise ValueError("window context is required for window-specific supervision")
            try:
                supervision_table = self.supervision_weight_by_window[int(window.index)]
            except KeyError:
                raise KeyError(
                    f"no supervision table for rolling window index {window.index}"
                ) from None
        return supervision_table

    # ------------------------------------------------------------------
    # Normalization (P0-4 / P2-2)
    # ------------------------------------------------------------------

    def _wrap_training_env(self, *, vec_env: Any, seed: int) -> Any:
        """Fit per-window stats or reuse the frozen first-window base stats.

        ``warmup_count`` is the SliceVecNormalize parameter name (not
        ``warmup``, which would leak into VecNormalize's kwargs and raise
        TypeError).  The base dim comes from the registered observation
        contract -- the wrapper extends the observation with the risk block, so
        only the base part is z-scored.  Historical Tail-GAE runs fit one
        normalizer per window; V4.7/N2 can instead freeze the first window via
        ``fit_once_on_meta_train`` without touching the seven risk features.
        """
        if not self.normalization_config["enabled"]:
            return vec_env
        shared_path = self._shared_normalization_paths.get(seed)
        if self.normalization_config["fit_once_on_meta_train"] and shared_path is not None:
            if not shared_path.exists():
                raise FileNotFoundError("missing first-window normalization statistics")
            return load_slice_normalizer(
                shared_path,
                vec_env,
                base_dim=self._risk_base_dim(),
            )
        return SliceVecNormalize(
            vec_env,
            base_dim=self._risk_base_dim(),
            norm_obs=True,
            norm_reward=False,
            training=True,
            warmup_count=100,
        )

    def _wrap_evaluation_env(self, *, env: Any, window: RollingWindow, seed: int) -> Any:
        """Freeze this window's training stats over the evaluation env.

        Preserves the parent's "missing statistics -> FileNotFoundError" guard
        (spec P2-2, single_ppo_runner.py:1030).
        """
        if not self.normalization_config["enabled"]:
            return env
        path = self._shared_normalization_paths.get(seed)
        if path is None or not path.exists():
            raise FileNotFoundError("missing training-fitted normalization statistics")
        return FrozenSliceNormalizationEnv(env, path, base_dim=self._risk_base_dim())

    def _risk_base_dim(self) -> int:
        """Base observation dim (before the risk block), never hardcoded (P2-1).

        The primary source is the observation contract registered on the bare
        base env by ``_build_env``; the live wrapper's own metadata is the
        fallback.  Both derivations agree (base_dim + declared names, P1-13).
        """
        base_dim = self._observation_metadata.get("base_observation_dim")
        if base_dim is None:
            wrapper = self._current_risk_wrapper
            if wrapper is not None:
                base_dim = wrapper.get_observation_metadata()["base_observation_dim"]
        if base_dim is None:
            raise RuntimeError(
                "base observation dim is unavailable; build the environment first"
            )
        return int(base_dim)

    # ------------------------------------------------------------------
    # Training (TailGAEPPO wiring + diagnostics callback)
    # ------------------------------------------------------------------

    def _tail_mechanism_dir(self) -> Path:
        """返回 run 根下 tail_mechanism 目录, 建好 4 个类型子目录（V4, spec §4）.

        ``self.output_root`` 形如 ``.../{timestamp}/scripts/MDA_PPO/runs``,
        上三级（parents[2]）即时间戳运行目录。不经过 OutputPathManager——
        其 ``_assert_under_mda_runs`` 强制路径在 scripts/MDA_PPO/runs 下,
        与 run 根布局冲突。
        """
        output_root = Path(self.output_root)
        # P0 guard: 相对路径下 parents[2] 退化到 "."（CWD）——静默写错位置
        assert output_root.is_absolute(), \
            f"output_root 必须是绝对路径（相对时 parents[2] = CWD）: {self.output_root}"
        run_root = output_root.parents[2]    # 上三级 = 时间戳运行目录
        assert re.fullmatch(r"\d{8}_\d{6}", run_root.name), \
            f"run 根不是时间戳目录（疑为相对路径或布局变更）: {run_root}"
        mech_dir = run_root / "tail_mechanism"
        for sub in ("per_step", "rl", "diag", "risk_state_diag", "validation", "trade", "overlay"):
            (mech_dir / sub).mkdir(parents=True, exist_ok=True)
        return mech_dir

    def _train_ppo_for_window(
        self,
        *,
        window: RollingWindow,
        train_env: Any,
        seed: int,
    ) -> Any:
        """Train a fresh TailGAEPPO from scratch for this window (40k steps).

        ``PPO_MODEL_KWARGS`` + seed + tail_alpha + risk_wrapper (the window's
        stashed wrapper) + tensorboard_log.  The
        audit training callback is kept unchanged; the risk diagnostics
        callback is mounted only when a risk wrapper exists (bridge prints
        nothing, spec §3.3).  Training-normalizer statistics are saved through
        the parent's ``_save_training_normalizer`` exactly as before.
        """
        model_kwargs = dict(self.ppo_model_kwargs)
        # FinRL passes policy_kwargs separately to SB3; keeping it inside
        # model_kwargs would pass the keyword twice and fail at construction.
        policy_kwargs = model_kwargs.pop("policy_kwargs", None) or {}
        tb_log_name = f"ppo_{self.run_id}_{window.label}_seed_{seed}"
        # Mirror FinRL DRLAgent.get_model (finrl/.../stablebaselines3/models.py:143):
        # print the popped kwargs, and pass verbose=1 so SB3 prints
        # "Using cuda device" and the target_kl early-stopping line —
        # same output sequence as the 1200 baseline.
        print(model_kwargs)
        model = TailGAEPPO(
            "MlpPolicy",
            train_env,
            **model_kwargs,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=1,
            tail_alpha=self.tail_alpha,
            risk_wrapper=self._current_risk_wrapper,
            tilt_shift=self.tilt_shift,
            direct_penalty_beta=self.direct_penalty_beta,
            direct_penalty_ref=self.direct_penalty_ref,
            # Phase 9 §3.2/§3.3（B/C）: β_up 与 actor-mean 近端带。hmax/费用率
            # 不显式传——TailGAEPPO 构造时经 env 链（vecenv→wrapper→基座 env）
            # 解析为只读 tensor（§3.4）.
            recovery_penalty_beta=getattr(self, "recovery_penalty_beta", 0.0),
            aux_mean_prox_clip=getattr(self, "aux_mean_prox_clip", None),
            # Δ1321b §1.1: L_CEF 剂量与参数（默认 β=0.0 = 关闭, 零扰动）
            cef_penalty_beta=getattr(self, "cef_penalty_beta", 0.0),
            cef_floor_G=getattr(self, "cef_floor_G", 0.90),
            cef_band_delta=getattr(self, "cef_band_delta", 0.02),
            cef_min_capacity=getattr(self, "cef_min_capacity", 1e-3),
            # getattr 兜底手工构造路径（测试用 object.__new__, 属性缺失 = 关闭）
            pcgrad=getattr(self, "pcgrad", False),
            tensorboard_log=str(self.output_root / "tensorboard"),
        )

        callbacks: list[Any] = []
        if self.training_callback_factory is not None:
            callbacks.append(
                self.training_callback_factory(
                    model_name="ppo",
                    window=window,
                    seed=seed,
                )
            )
        # V4 接线（spec §4）: diag 落新布局 {run_root}/tail_mechanism/diag/,
        # 文件名不含 seed（运行目录已按 seed 分）。此块在现有 try(428) 之前,
        # 因旧构造处 train_kwargs 尚未定义（定义于 try 内）。
        if self._current_risk_wrapper is not None:
            diag_dir = self._tail_mechanism_dir()   # 建 4 子目录, 见下

            callbacks.append(
                _DiagnosticsCallback(
                    wrapper=self._current_risk_wrapper,
                    n_steps=int(model_kwargs.get("n_steps", 1024)),
                    num_timesteps=self.ppo_timesteps,
                    diag_path=diag_dir / "diag" / f"{window.label}.csv",
                )
            )

        started = time.time()
        event = {
            "model_name": "ppo",
            "window_index": window.index,
            "iteration": window.iteration,
            "window_label": window.label,
            "seed": seed,
            "tb_log_name": tb_log_name,
            "total_timesteps": self.ppo_timesteps,
            "status": "running",
        }
        try:
            train_kwargs = {
                "tb_log_name": tb_log_name,
                "total_timesteps": self.ppo_timesteps,
            }
            # SB3 learn() takes ``callback`` (a list is wrapped in a
            # CallbackList by MaybeCallback); ``callbacks`` is not a learn
            # keyword and would raise TypeError on the first training window.
            if callbacks:
                train_kwargs["callback"] = callbacks
            # V4 接线（spec §4）: G1a/G1b/G2 打开——任一 open 失败或 learn
            # 中途异常时, finally 的 close/flush 幂等兜底
            if self._current_risk_wrapper is not None:
                self._current_risk_wrapper.open_per_step_log(
                    diag_dir / "per_step" / f"{window.label}.csv"
                )
                self._current_risk_wrapper.set_diag_csv_path(
                    diag_dir / "risk_state_diag" / f"{window.label}.csv"
                )
                # V4.13 P0（spec Part 7.1）: 方向标签逐 step 审计文件
                self._current_risk_wrapper.open_label_log(
                    diag_dir / "risk_state_diag" / f"{window.label}_labels.csv"
                )
                # G1b: 句柄挂在 rollout_buffer 上（model 已创建）
                model.rollout_buffer.open_rl_log(
                    diag_dir / "rl" / f"{window.label}.csv"
                )
            model.learn(**train_kwargs)
            self._save_training_normalizer(train_env=train_env, window=window, seed=seed)
            # The Tail-GAE training override must preserve the parent's MDA
            # component-normalizer lifecycle.  Without this call, N1/N3 lose
            # the first-window state and silently refit return statistics in
            # every rolling window.
            self._save_reward_component_normalizer(train_env=train_env, seed=seed)
            event["status"] = "completed"
            # Δ1321b §4.3: window 训练末策略快照（配置关闭时零动作）
            if getattr(self, "save_policy_snapshot", False):
                snapshot_dir = Path(self.output_root) / "policy_snapshot"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                # risk_wrapper（及回调）持有写模式的日志句柄, 不可 pickle;
                # 快照仅需策略/价值参数, 显式排除（env 侧可由配置重建）.
                model.save(snapshot_dir / f"{window.label}_seed{seed}.zip",
                           exclude=["risk_wrapper", "_callback",
                                    "rollout_buffer_kwargs", "rollout_buffer_class"])
            return model
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = repr(exc)
            raise
        finally:
            # V4 接线（spec §4）: 先 close/flush（幂等兜底）, 再记 event;
            # 已有 flush 循环（下方 getattr(callback, "flush")）不动
            if self._current_risk_wrapper is not None:
                self._current_risk_wrapper.close_per_step_log()
                self._current_risk_wrapper.close_label_log()
                model.rollout_buffer.close_rl_log()
            for callback in callbacks:
                flush = getattr(callback, "flush", None)
                if callable(flush):
                    flush()
            event["num_timesteps_after_training"] = int(
                getattr(model, "num_timesteps", 0)
            )
            quality_metrics = None
            for callback in callbacks:
                metrics = getattr(callback, "training_quality_metrics", None)
                if callable(metrics):
                    quality_metrics = metrics()
                    break
            if quality_metrics is not None:
                event["training_quality"] = quality_metrics
                self._training_quality[(seed, window.index)] = dict(quality_metrics)
            if self.training_event_callback is not None:
                self.training_event_callback(event)

    # ------------------------------------------------------------------
    # Validation (per-day log: date, I_t, gross_exposure, action_mean,
    # V_pre; spec §8 待办 1)
    # ------------------------------------------------------------------

    def _run_validation_for_window(
        self,
        *,
        window: RollingWindow,
        model: Any,
        env: Any,
        seed: int,
    ) -> Path:
        output_path = self._account_value_output_path(
            mode="validation",
            window=window,
            seed=seed,
        )
        if self._current_validation_wrapper is not None:
            account_value = self._predict_and_log_validation(
                window=window,
                model=model,
                env=env,
            )
        else:
            account_value = self._predict_account_value(model=model, env=env)
        self._write_account_value(account_value, output_path)
        # Step 5: validation 段同样执行覆盖层（spec §6.2 作用时机），落 overlay 日志
        self._write_overlay_log(env=env, mode="validation", window=window)
        return output_path

    def _predict_and_log_validation(
        self,
        *,
        window: RollingWindow,
        model: Any,
        env: Any,
    ) -> Any:
        """Validation loop byte-identical to FinRL DRL_prediction + per-day log.

        Replicates finrl/agents/stablebaselines3/models.py DRL_prediction
        exactly (get_sb_env -> test_env.reset() -> predict/step loop ->
        capture save_asset_memory at max_steps-1 -> dones[0] break) so the
        validation account-value CSV keeps the established semantics.  Per
        day it additionally writes (date, I_t, gross_exposure, action_mean,
        V_pre) to tail_mechanism/validation/{window.label}.csv; gross_exposure
        is the raw value (risk-block affine inverted through the wrapper's own
        registry transform, not hardcoded) and V_pre is the critic value of
        the normalized observation the day's action was chosen on.
        """
        wrapper = self._current_validation_wrapper
        test_env, test_obs = env.get_sb_env()
        account_memory = None
        test_env.reset()
        day_count = len(env.df.index.unique())
        max_steps = day_count - 1
        log_path = self._tail_mechanism_dir() / "validation" / f"{window.label}.csv"
        ge_idx = wrapper.names.index("gross_exposure")
        ge_a, ge_b = wrapper._dims[ge_idx][2]
        with open(log_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["date", "I_t", "gross_exposure", "action_mean", "V_pre"])
            for i in range(day_count):
                v_pre = float(
                    model.policy.predict_values(
                        th.as_tensor(
                            np.asarray(test_obs, dtype=np.float32),
                            device=model.device,
                        )
                    )[0]
                    .detach()
                    .cpu()
                    .item()
                )
                action, _states = model.predict(test_obs, deterministic=True)
                test_obs, rewards, dones, info = test_env.step(action)
                ge_block = float(wrapper._risk_block_history[-1][ge_idx])
                writer.writerow(
                    [
                        wrapper.last_date,
                        int(wrapper.last_I_t),
                        f"{(ge_block - ge_b) / ge_a:.6f}",
                        f"{float(np.mean(action)):.6f}",
                        f"{v_pre:.6f}",
                    ]
                )
                if i == max_steps - 1:
                    account_memory = test_env.env_method(
                        method_name="save_asset_memory"
                    )
                if dones[0]:
                    print("hit end!")
                    break
        if account_memory is None:
            raise RuntimeError(
                "validation loop ended before the account-memory capture step"
            )
        return account_memory[0]


class _DiagnosticsCallback(BaseCallback):
    """Emit the ``[risk_state_env]`` feature block every 5 rollout iterations.

    One atomic ``print`` per block (spec §3.3): the wrapper joins all feature
    rows into a single string so SB3 logger dumps or pipe buffering cannot
    split the block.

    SB3's learn() loop runs ``collect_rollouts (-> on_rollout_end)`` then
    ``dump_logs`` (the iteration table), so printing directly in
    ``_on_rollout_end`` would put the block BEFORE its table.  The block is
    instead stashed in ``_pending`` and printed on the first ``_on_step`` of
    the next rollout (i.e. after the iteration-N table) — 1200-style order:
    table(5) -> block(5) -> table(6).  ``_on_training_end`` flushes the
    final pending block after the last table: with 40000/1024 = 39.06 the
    last ``_on_rollout_end`` fires at iteration 40 (count == 40), and the
    backfill branch only covers counts that are not multiples of 5 (spec
    P2-5 off-by-one): each window emits exactly 8 blocks at iterations 5..40.
    """

    def __init__(
        self,
        *,
        wrapper: RiskStateEnvWrapper,
        n_steps: int,
        num_timesteps: int,
        diag_path: Path | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._wrapper = wrapper
        self._n_steps = int(n_steps)
        self._num_timesteps = int(num_timesteps)
        self._diag_path = Path(diag_path) if diag_path is not None else None
        # 重跑防护: _flush_pending 是 append 模式, 同路径重跑同一窗口会累积旧行
        # （40→80 行, 违验收 #4 的"每 rollout 一行 × 40"）——每窗口新建 callback,
        # 构造时删旧文件, 保证每次训练从 0 行开始（与 G2 set_diag_csv_path 语义一致）
        if self._diag_path is not None and self._diag_path.exists():
            self._diag_path.unlink()
        self._diag_rows: list[dict[str, Any]] = []
        self._diag_written = 0   # 已写行游标（flush 幂等, spec §4）
        self._count = 0
        self._pending: int | None = None

    def _on_step(self) -> bool:
        if self._pending is not None:
            self._wrapper.print_diagnostics(iteration=self._pending)
            self._pending = None
        return True

    def _on_rollout_end(self) -> None:
        self._count += 1

        # --- collect tail mechanism diagnostics from the rollout buffer ---
        buffer = getattr(self.model, "rollout_buffer", None)
        diag = getattr(buffer, "_tail_diag", None)
        if diag is not None:
            diag["iteration"] = self._count
            self._diag_rows.append(dict(diag))

        # --- G1a: 每 rollout 末 flush wrapper 的 per-step 缓冲（spec §3.1.2）---
        # callback 仅在 wrapper 存在时被构造（runner 侧守卫）, 与现有
        # _on_step/_on_training_end 对 wrapper 的直接调用保持一致
        self._wrapper.flush_per_step_log()

        # --- 增量写盘本轮新 diag 行（中途异常不丢已累积行）---
        self._flush_pending()

        if self._count % 5 == 0:
            self._pending = self._count

    def _on_training_end(self) -> None:
        if self._pending is not None:
            self._wrapper.print_diagnostics(iteration=self._pending)
            self._pending = None
        elif self._count % 5 != 0:
            iteration = int(math.ceil(self._num_timesteps / self._n_steps))
            self._wrapper.print_diagnostics(iteration=iteration)

        # --- persist tail mechanism diagnostics（幂等, 已写行不重复）---
        self.flush()

    def flush(self) -> None:
        """幂等写盘: 把未写的新 diag 行追加到 CSV（runner finally 兜底调用）."""
        self._flush_pending()

    def _flush_pending(self) -> None:
        """append 模式写盘; 仅当文件不存在时写表头, 只写游标之后的新行."""
        if self._diag_path is None or self._diag_written >= len(self._diag_rows):
            return
        self._diag_path.parent.mkdir(parents=True, exist_ok=True)
        # 表头 = ["iteration"] + 21 个 _tail_diag 字段（共 22 列, spec §3.2.5）;
        # 字段顺序取首行的插入顺序（buffer 的 21 键 + 末尾 iteration）
        fieldnames = ["iteration"] + [
            k for k in self._diag_rows[0] if k != "iteration"
        ]
        write_header = not self._diag_path.exists()
        with open(self._diag_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in self._diag_rows[self._diag_written:]:
                writer.writerow(row)
        self._diag_written = len(self._diag_rows)
