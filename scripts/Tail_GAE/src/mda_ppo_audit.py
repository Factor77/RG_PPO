"""Audit orchestration for reproducible MDA-PPO notebook experiments."""

from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

try:
    from .backtest_metrics import concatenate_continuous_account_values
except ImportError:
    from .backtest_metrics import concatenate_continuous_account_values


__all__ = [
    "MDAAuditConfig",
    "MDAAuditResult",
    "audit_mda_ppo_run",
    "set_reproducible_seed",
]

SUPPORTED_RUN_DIRECTORY_LAYOUTS = {
    "standard",
    "artifact_seed_timestamp",
    "seed_timestamp",
}

try:
    from .mda_diagnostics import DiagnosticsBuffer, required_reward_fields
    from .single_ppo_runner import (
        RollingWindow,
        SUPPORTED_REWARD_MODES,
        SinglePPORollingRunner,
    )
    from .single_ppo_env import (
        RISK_OBSERVATION_FIELDS,
        get_risk_observation_dim,
    )
except ImportError:
    from mda_diagnostics import DiagnosticsBuffer, required_reward_fields
    from single_ppo_runner import (
        RollingWindow,
        SUPPORTED_REWARD_MODES,
        SinglePPORollingRunner,
    )
    from single_ppo_env import (
        RISK_OBSERVATION_FIELDS,
        get_risk_observation_dim,
    )

try:
    from scripts.ensemble.ensemble_audit import (
        ensure_audit_dirs,
        make_sb3_training_callback,
        save_json,
        set_reproducible_seed,
        snapshot_dataframe,
        snapshot_experiment_config,
        snapshot_rng_state,
        snapshot_runtime,
        write_file_manifest,
        write_result_metrics,
    )
except ImportError:
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.ensemble.ensemble_audit import (
        ensure_audit_dirs,
        make_sb3_training_callback,
        save_json,
        set_reproducible_seed,
        snapshot_dataframe,
        snapshot_experiment_config,
        snapshot_rng_state,
        snapshot_runtime,
        write_file_manifest,
        write_result_metrics,
    )


@dataclass(frozen=True)
class MDAAuditConfig:
    seed: int
    lambda_value: float | None
    output_root: str | Path
    data_path: str | Path
    notebook_path: str | Path
    dji_path: str | Path | None = None
    experiment_name: str = "mda_ppo"
    artifact_id: str | None = None
    reward_mode: str = "mda"
    clip_c: float = 0.02
    lambda_grid: Sequence[float] = (0.0, 0.03, 0.1, 0.3, 1.0)
    deterministic: bool = True
    log_freq: int = 1000
    max_windows: int | None = None
    run_name: str | None = None
    run_directory_prefix: str | None = None
    run_directory_layout: str = "standard"
    extra: Mapping[str, Any] = field(default_factory=dict)
    normalization_config: Mapping[str, Any] = field(default_factory=dict)
    quality_gate_config: Mapping[str, Any] = field(default_factory=dict)
    allow_ineligible_backtest: bool = False
    use_risk_observation: bool = False
    # Must stay the last field (P1-2): 1200 notebooks construct this config
    # with all-keyword arguments, so positional order is safe.
    risk_observation_fields: Sequence[str] | None = None
    # Phase 9 §2.1 (D-observation): 声明的追加字段（1323/1324 传非 None 才核对
    # phase9 维数; 旧 notebook 默认 None）. 必须保持最后一个字段（P1-2）.
    phase9_observation_fields: Sequence[str] | None = None

    def __post_init__(self) -> None:
        # Phase 9 §2.1: None = D 模式未声明, 不核对; 非 None 须为非空字符串序列.
        if self.phase9_observation_fields is not None:
            if not isinstance(self.phase9_observation_fields, Sequence) or isinstance(
                self.phase9_observation_fields, (str, bytes)
            ):
                raise TypeError(
                    "phase9_observation_fields must be a sequence of strings when provided"
                )
            if not all(
                isinstance(field, str) and field
                for field in self.phase9_observation_fields
            ):
                raise TypeError(
                    "phase9_observation_fields must be a sequence of non-empty strings"
                )
        # P2-4: validate at construction time (not at audit run time); the
        # None path (1200) is untouched.
        if self.risk_observation_fields is None:
            return
        if not isinstance(self.risk_observation_fields, Sequence) or isinstance(
            self.risk_observation_fields, (str, bytes)
        ):
            raise TypeError(
                "risk_observation_fields must be a sequence of strings when provided"
            )
        if not self.risk_observation_fields and self.use_risk_observation:
            raise ValueError(
                "risk_observation_fields must not be empty when risk observation is enabled"
            )
        if not all(
            isinstance(field, str) and field
            for field in self.risk_observation_fields
        ):
            raise TypeError(
                "risk_observation_fields must be a sequence of non-empty strings"
            )


@dataclass(frozen=True)
class MDAAuditResult:
    run_dir: Path
    df_summary: pd.DataFrame
    account_value: pd.DataFrame
    baseline: pd.DataFrame | None
    comparison: pd.DataFrame | None
    mda_stats: pd.DataFrame | None
    baseline_stats: pd.DataFrame | None
    runner_result: Mapping[str, Any]


def audit_mda_ppo_run(
    *,
    config: MDAAuditConfig,
    processed: pd.DataFrame,
    raw_data: pd.DataFrame,
    env_kwargs: Mapping[str, Any],
    train_period: Sequence[str],
    val_test_period: Sequence[str],
    rebalance_window: int,
    validation_window: int,
    ppo_model_kwargs: Mapping[str, Any],
    ppo_timesteps: int,
    runner_cls: type[SinglePPORollingRunner] = SinglePPORollingRunner,
    runner_kwargs: Mapping[str, Any] | None = None,
) -> MDAAuditResult:
    """Run one MDA-PPO experiment and persist analysis-ready artifacts."""

    _validate_config(config)
    run_dir = _make_run_dir(config)
    artifact_label = _artifact_label(config)
    account_value_label = _account_value_label(config)
    dirs = ensure_audit_dirs(run_dir)
    diagnostics_dir = dirs["audit"] / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(
        config.notebook_path,
        dirs["audit"] / Path(config.notebook_path).name,
    )

    set_reproducible_seed(config.seed, deterministic=config.deterministic)
    snapshot_runtime(exp_dir=run_dir)
    snapshot_rng_state(exp_dir=run_dir, label="before_training")
    snapshot_experiment_config(
        exp_dir=run_dir,
        seed=config.seed,
        train_period=train_period,
        val_test_period=val_test_period,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
        env_kwargs=env_kwargs,
        model_kwargs={"ppo": ppo_model_kwargs},
        timesteps_dict={"ppo": ppo_timesteps},
        notebook_name=Path(config.notebook_path).name,
        extra={
            **dict(config.extra),
            "experiment_name": config.experiment_name,
            "artifact_id": config.artifact_id,
            "reward_mode": config.reward_mode,
            "lambda_value": config.lambda_value,
            "lambda_grid": list(config.lambda_grid),
            "clip_c": config.clip_c if config.reward_mode == "mda" else None,
            "data_path": str(config.data_path),
            "dji_path": str(config.dji_path) if config.dji_path else None,
            "max_windows": config.max_windows,
            "run_directory_layout": config.run_directory_layout,
            "normalization_config": dict(config.normalization_config),
            "quality_gate_config": dict(config.quality_gate_config),
            "use_risk_observation": config.use_risk_observation,
            "risk_observation_dim": (
                len(config.risk_observation_fields)
                if config.risk_observation_fields is not None
                else get_risk_observation_dim()
            ),
            "risk_observation_fields": (
                list(config.risk_observation_fields)
                if config.risk_observation_fields is not None
                else list(RISK_OBSERVATION_FIELDS)
            ),
            "phase9_observation_dim": (
                len(config.phase9_observation_fields)
                if config.phase9_observation_fields is not None
                else 0
            ),
            "phase9_observation_fields": (
                list(config.phase9_observation_fields)
                if config.phase9_observation_fields is not None
                else []
            ),
        },
    )
    snapshot_dataframe(
        raw_data,
        exp_dir=run_dir,
        name="raw_data",
        key_columns=["date", "tic"],
    )
    snapshot_dataframe(
        processed,
        exp_dir=run_dir,
        name="processed",
        key_columns=["date", "tic"],
    )

    status: dict[str, Any] = {"status": "running", "run_dir": str(run_dir)}
    save_json(status, dirs["audit"] / "run_status.json")
    audit_collector = _MDAAuditCollector(
        run_dir=run_dir,
        log_freq=config.log_freq,
        experiment_name=artifact_label,
        reward_mode=config.reward_mode,
    )
    started = time.time()

    try:
        runner_output_root = run_dir / "scripts" / "MDA_PPO" / "runs"
        runner = runner_cls(
            processed_data=processed,
            env_kwargs=dict(env_kwargs),
            train_period=tuple(train_period),
            val_test_period=tuple(val_test_period),
            rebalance_window=rebalance_window,
            validation_window=validation_window,
            ppo_model_kwargs=dict(ppo_model_kwargs),
            ppo_timesteps=ppo_timesteps,
            reward_mode=config.reward_mode,
            lambda_value=config.lambda_value,
            lambda_grid=config.lambda_grid,
            clip_c=config.clip_c,
            run_id="rolling",
            seed=config.seed,
            output_root=runner_output_root,
            experiment_name=config.experiment_name,
            max_smoke_windows=config.max_windows,
            training_callback_factory=audit_collector.make_training_callback,
            training_event_callback=audit_collector.record_training_event,
            diagnostics_buffer_factory=audit_collector.make_diagnostics_buffer,
            normalization_config=config.normalization_config,
            quality_gate_config=config.quality_gate_config,
            use_risk_observation=config.use_risk_observation,
            **(
                dict(runner_kwargs)
                if runner_kwargs is not None
                else {}
            ),
        )
        runner_result = runner.run(dry_run=False)
        audit_collector.flush()
        observation_metadata = dict(
            runner_result.get(
                "observation",
                {
                    "use_risk_observation": config.use_risk_observation,
                    "base_observation_dim": None,
                    "risk_observation_dim": (
                        len(config.risk_observation_fields)
                        if config.risk_observation_fields is not None
                        else get_risk_observation_dim()
                    ),
                    "risk_observation_fields": (
                        list(config.risk_observation_fields)
                        if config.risk_observation_fields is not None
                        else list(RISK_OBSERVATION_FIELDS)
                    ),
                    "policy_observation_dim": None,
                },
            )
        )
        _validate_observation_metadata(
            observation_metadata,
            use_risk_observation=config.use_risk_observation,
            risk_observation_fields=config.risk_observation_fields,
            phase9_observation_fields=config.phase9_observation_fields,
        )
        save_json(
            observation_metadata,
            dirs["audit"] / "observation_contract.json",
        )

        df_summary = pd.read_csv(runner_result["runner_summary_path"])
        df_summary.to_csv(dirs["results"] / "df_summary.csv", index=False)
        gate_failed = bool(runner_result.get("quality_gate_failed", False))
        if gate_failed:
            failure = dict(runner_result.get("quality_gate_failure") or {})
            save_json(
                {
                    "eligible_for_final_backtest": False,
                    "exploratory_backtest_allowed": bool(config.allow_ineligible_backtest),
                    "failure": failure,
                    "gate_results": runner_result.get("quality_gate_results", []),
                    "normalization_config": dict(config.normalization_config),
                    "quality_gate_config": dict(config.quality_gate_config),
                },
                dirs["audit"] / "final_backtest_eligibility.json",
            )
            if not config.allow_ineligible_backtest:
                empty_account_value = pd.DataFrame(columns=["date", "account_value"])
                empty_account_value.to_csv(
                    dirs["results"] / f"df_account_value_{account_value_label}.csv",
                    index=False,
                )
                snapshot_rng_state(exp_dir=run_dir, label="after_training")
                status.update(
                    {
                        "status": "ineligible",
                        "elapsed_seconds": time.time() - started,
                        "summary_rows": int(len(df_summary)),
                        "ineligibility_reason": failure.get("quality_gate_reason"),
                    }
                )
                save_json(status, dirs["audit"] / "run_status.json")
                write_file_manifest(exp_dir=run_dir)
                return MDAAuditResult(
                    run_dir=run_dir,
                    df_summary=df_summary,
                    account_value=empty_account_value,
                    baseline=None,
                    comparison=None,
                    mda_stats=None,
                    baseline_stats=None,
                    runner_result=runner_result,
                )
        account_value = _collect_account_value(runner_result)
        save_json(
            {
                "eligible_for_final_backtest": not gate_failed,
                "exploratory_backtest": gate_failed,
                "gate_results": runner_result.get("quality_gate_results", []),
                "normalization_config": dict(config.normalization_config),
                "quality_gate_config": dict(config.quality_gate_config),
            },
            dirs["audit"] / "final_backtest_eligibility.json",
        )
        account_value.to_csv(
            dirs["results"] / f"df_account_value_{account_value_label}.csv",
            index=False,
        )
        mda_stats = _backtest_stats(account_value, value_col_name="account_value")
        if mda_stats is not None:
            mda_stats.to_csv(dirs["results"] / f"{artifact_label}_stats.csv")
        write_result_metrics(
            account_value,
            exp_dir=run_dir,
            label=artifact_label,
            value_col="account_value",
        )

        baseline = _load_baseline(
            config.dji_path,
            account_value,
            initial_amount=env_kwargs.get("initial_amount"),
        )
        comparison = None
        baseline_stats = None
        if baseline is not None:
            baseline.to_csv(dirs["results"] / "df_dji.csv", index=False)
            baseline_stats = _backtest_stats(
                baseline[["date", "dji"]].rename(columns={"dji": "close"}),
                value_col_name="close",
            )
            if baseline_stats is not None:
                baseline_stats.to_csv(dirs["results"] / "baseline_stats.csv")
            comparison = pd.merge(
                account_value[["date", "account_value"]].rename(
                    columns={"account_value": artifact_label}
                ),
                baseline[["date", "dji"]],
                on="date",
                how="inner",
            ).set_index("date")
            comparison.to_csv(dirs["results"] / "result.csv")

        snapshot_rng_state(exp_dir=run_dir, label="after_training")
        status.update(
            {
                "status": "completed_exploratory" if gate_failed else "completed",
                "elapsed_seconds": time.time() - started,
                "summary_rows": int(len(df_summary)),
                "account_value_rows": int(len(account_value)),
            }
        )
        save_json(status, dirs["audit"] / "run_status.json")
        write_file_manifest(exp_dir=run_dir)
        return MDAAuditResult(
            run_dir=run_dir,
            df_summary=df_summary,
            account_value=account_value,
            baseline=baseline,
            comparison=comparison,
            mda_stats=mda_stats,
            baseline_stats=baseline_stats,
            runner_result=runner_result,
        )
    except Exception:
        audit_collector.flush()
        snapshot_rng_state(exp_dir=run_dir, label="after_failure")
        status.update(
            {
                "status": "failed",
                "elapsed_seconds": time.time() - started,
                "traceback": traceback.format_exc(),
            }
        )
        save_json(status, dirs["audit"] / "run_status.json")
        write_file_manifest(exp_dir=run_dir)
        raise


class _ContextDiagnosticsBuffer(DiagnosticsBuffer):
    def __init__(
        self,
        context: Mapping[str, Any],
        *,
        required_fields: Sequence[str],
    ) -> None:
        super().__init__(required_fields=required_fields)
        self.context = dict(context)

    def record_step(self, record: Mapping[str, object]) -> Mapping[str, object]:
        return super().record_step({**self.context, **dict(record)})


class _MDAAuditCollector:
    def __init__(
        self,
        *,
        run_dir: Path,
        log_freq: int,
        experiment_name: str,
        reward_mode: str,
    ) -> None:
        self.run_dir = run_dir
        self.log_freq = log_freq
        self.experiment_name = experiment_name
        self.required_diagnostic_fields = required_reward_fields(reward_mode)
        self.training_rows: list[dict[str, Any]] = []
        self.diagnostics: list[tuple[str, _ContextDiagnosticsBuffer]] = []

    def make_training_callback(
        self,
        *,
        model_name: str,
        window: RollingWindow,
        seed: int,
    ) -> Any:
        return make_sb3_training_callback(
            exp_dir=self.run_dir,
            model_name=model_name,
            window_index=window.iteration,
            log_freq=self.log_freq,
        )

    def record_training_event(self, event: Mapping[str, Any]) -> None:
        self.training_rows.append(dict(event))
        self._flush_training_runs()

    def make_diagnostics_buffer(
        self,
        *,
        mode: str,
        window: RollingWindow,
        seed: int,
    ) -> _ContextDiagnosticsBuffer:
        context = {
            "mode": mode,
            "window_index": window.index,
            "iteration": window.iteration,
            "window_label": window.label,
            "seed": seed,
        }
        buffer = _ContextDiagnosticsBuffer(
            context,
            required_fields=self.required_diagnostic_fields,
        )
        filename = (
            f"diagnostics_{mode}_{self.experiment_name}_{window.label}_seed_{seed}.csv"
        )
        self.diagnostics.append((filename, buffer))
        return buffer

    def flush(self) -> None:
        self._flush_training_runs()
        diagnostics_dir = self.run_dir / "audit" / "diagnostics"
        for filename, buffer in self.diagnostics:
            buffer.flush_csv(diagnostics_dir / filename)
        call_counts: dict[str, int] = {}
        for row in self.training_rows:
            model_name = str(row["model_name"])
            call_counts[model_name] = call_counts.get(model_name, 0) + 1
        save_json(
            {"model_train_call_counts": call_counts},
            self.run_dir / "audit" / "training_call_counts.json",
        )

    def _flush_training_runs(self) -> None:
        path = self.run_dir / "audit" / "training_runs.csv"
        if self.training_rows:
            pd.DataFrame(self.training_rows).to_csv(path, index=False)
        elif not path.exists():
            pd.DataFrame(
                columns=[
                    "model_name",
                    "window_index",
                    "iteration",
                    "window_label",
                    "seed",
                    "tb_log_name",
                    "total_timesteps",
                    "status",
                    "elapsed_seconds",
                    "num_timesteps_after_training",
                ]
            ).to_csv(path, index=False)


def _validate_config(config: MDAAuditConfig) -> None:
    if not isinstance(config.use_risk_observation, bool):
        raise TypeError("use_risk_observation must be a boolean")
    if config.reward_mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(
            f"unsupported reward_mode={config.reward_mode!r}; "
            f"expected one of {sorted(SUPPORTED_REWARD_MODES)}"
        )
    _validate_path_component(config.experiment_name, field_name="experiment_name")

    # No artifact_id means a historical experiment: preserve its old naming
    # contract and output layout exactly.
    if config.artifact_id is None:
        expected_experiment_name = _expected_experiment_name(config.reward_mode)
        if config.experiment_name != expected_experiment_name:
            raise ValueError(
                "experiment_name does not match reward_mode for a legacy run; "
                f"expected {expected_experiment_name!r} for "
                f"reward_mode={config.reward_mode!r}"
            )
    else:
        _validate_path_component(config.artifact_id, field_name="artifact_id")

    if config.reward_mode == "mda" and config.lambda_value is None:
        raise ValueError("lambda_value is required when reward_mode='mda'")
    if config.lambda_value is not None and config.lambda_value < 0:
        raise ValueError("lambda_value must be non-negative")
    if config.reward_mode == "mda" and not config.lambda_grid:
        raise ValueError("lambda_grid must not be empty when reward_mode='mda'")
    if config.log_freq <= 0:
        raise ValueError("log_freq must be positive")
    if config.max_windows is not None and config.max_windows <= 0:
        raise ValueError("max_windows must be positive when provided")
    if config.run_directory_layout not in SUPPORTED_RUN_DIRECTORY_LAYOUTS:
        raise ValueError(
            "unsupported run_directory_layout="
            f"{config.run_directory_layout!r}; expected one of "
            f"{sorted(SUPPORTED_RUN_DIRECTORY_LAYOUTS)}"
        )


def _validate_observation_metadata(
    metadata: Mapping[str, Any],
    *,
    use_risk_observation: bool,
    risk_observation_fields: Sequence[str] | None = None,
    phase9_observation_fields: Sequence[str] | None = None,
) -> None:
    if risk_observation_fields is not None:
        # Non-None branch (V2): the risk field names are declared by the caller
        # (5/7/10 names, not the fixed schema), so the fixed-schema equality
        # checks are skipped; only the declared contract length must match the
        # contract registered by the runner. None -> existing per-field checks.
        declared_fields = list(risk_observation_fields)
        registered_fields = metadata.get("risk_observation_fields")
        if not isinstance(registered_fields, (list, tuple)):
            raise RuntimeError("runner risk observation fields are missing")
        if len(registered_fields) != len(declared_fields):
            raise RuntimeError(
                "runner risk observation fields do not match the declared "
                "audit contract"
            )
        expected_fields = declared_fields
    else:
        expected_fields = list(RISK_OBSERVATION_FIELDS)
        if metadata.get("use_risk_observation") is not use_risk_observation:
            raise RuntimeError(
                "runner observation switch does not match the audit configuration"
            )
        if metadata.get("risk_observation_fields") != expected_fields:
            raise RuntimeError("runner risk observation fields do not match the schema")
        if metadata.get("risk_observation_dim") != len(expected_fields):
            raise RuntimeError("runner risk observation dimension does not match the schema")

    # Phase 9 §2.1: 声明非 None 时逐名核对（R3 修复 2026-09-11: 原实现只比
    # 数量, 同长度错名可过门; risk 固定 schema 分支 :645 一直逐值比对）; None
    # （旧 notebook）时以 runner 注册的 phase9 键为准（缺失/空 = 0 维, 旧运行不变）.
    registered9_fields = metadata.get("phase9_observation_fields")
    registered9_dim = metadata.get("phase9_observation_dim")
    if phase9_observation_fields is not None:
        declared9 = list(phase9_observation_fields)
        if not isinstance(registered9_fields, (list, tuple)):
            raise RuntimeError("runner phase9 observation fields are missing")
        if list(registered9_fields) != declared9:
            raise RuntimeError(
                "runner phase9 observation fields do not match the declared "
                "audit contract"
            )
        expected9 = len(declared9)
    elif registered9_fields is None:
        expected9 = 0
    else:
        expected9 = len(list(registered9_fields))
    if registered9_dim is not None and int(registered9_dim) != expected9:
        raise RuntimeError("runner phase9 observation dim/fields disagree")

    base_dim = metadata.get("base_observation_dim")
    policy_dim = metadata.get("policy_observation_dim")
    if (base_dim is None) != (policy_dim is None):
        raise RuntimeError("runner observation dimensions are incomplete")
    if base_dim is None:
        return
    expected_policy_dim = int(base_dim)
    if use_risk_observation:
        expected_policy_dim += len(expected_fields)
    expected_policy_dim += expected9
    if int(policy_dim) != expected_policy_dim:
        raise RuntimeError("runner policy observation dimension is inconsistent")


def _make_run_dir(config: MDAAuditConfig) -> Path:
    run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{config.run_directory_prefix} " if config.run_directory_prefix else ""
    if config.run_directory_layout == "artifact_seed_timestamp":
        base_dir = (
            Path(config.output_root)
            / f"{prefix}{_artifact_label(config)}"
            / f"seed_{config.seed}"
            / run_name
        )
    elif config.run_directory_layout == "seed_timestamp":
        base_dir = (
            Path(config.output_root)
            / f"seed_{config.seed}"
            / run_name
        )
    elif (
        config.artifact_id is None
        and config.reward_mode == "aust_original"
    ):
        base_dir = (
            Path(config.output_root)
            / "original_reward"
            / f"seed_{config.seed}"
            / run_name
        )
    else:
        lambda_dir = (
            f"lambda_{float(config.lambda_value):g}"
            if config.reward_mode == "mda"
            else "lambda_na"
        )
        base_dir = (
            Path(config.output_root)
            / _artifact_label(config)
            / lambda_dir
            / f"seed_{config.seed}"
            / run_name
        )
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir.with_name(f"{base_dir.name}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _collect_account_value(runner_result: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in runner_result.get("windows", []):
        path = Path(window["trade_account_value_path"])
        if not path.exists():
            raise FileNotFoundError(f"missing MDA-PPO trade output: {path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("runner returned no MDA-PPO trade account values")
    return concatenate_continuous_account_values(frames)


def _load_baseline(
    dji_path: str | Path | None,
    account_value: pd.DataFrame,
    *,
    initial_amount: Any,
) -> pd.DataFrame | None:
    if dji_path is None or not Path(dji_path).exists():
        return None
    baseline = pd.read_csv(dji_path)
    baseline.columns = [str(column).lower() for column in baseline.columns]
    if "date" not in baseline.columns and "datadate" in baseline.columns:
        baseline = baseline.rename(columns={"datadate": "date"})
    if "date" not in baseline.columns or "close" not in baseline.columns:
        return None
    baseline["date"] = pd.to_datetime(baseline["date"]).dt.strftime("%Y-%m-%d")
    account_dates = pd.to_datetime(account_value["date"]).dt.strftime("%Y-%m-%d")
    baseline = baseline[baseline["date"].isin(set(account_dates))].copy()
    baseline = baseline.sort_values("date").reset_index(drop=True)
    if baseline.empty:
        return None
    amount = float(initial_amount or account_value["account_value"].iloc[0])
    baseline["dji"] = baseline["close"] / float(baseline["close"].iloc[0]) * amount
    return baseline[["date", "dji", "close"]]


def _backtest_stats(
    account_value: pd.DataFrame,
    *,
    value_col_name: str,
) -> pd.DataFrame | None:
    try:
        from finrl.plot import get_daily_return
        from pyfolio import timeseries
    except Exception:
        return None

    try:
        daily_return = get_daily_return(
            account_value,
            value_col_name=value_col_name,
        ).dropna()

        if daily_return.empty:
            return None

        raw_stats = timeseries.perf_stats(
            returns=daily_return,
            positions=None,
            transactions=None,
            turnover_denom="AGB",
        )
    except Exception:
        return None

    return pd.DataFrame(raw_stats)


def _copy_if_exists(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    if source_path.exists():
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _expected_experiment_name(reward_mode: str) -> str:
    if reward_mode == "aust_original":
        return "aust_ppo"
    if reward_mode == "return":
        return "return_ppo"
    return "mda_ppo"


def _artifact_label(config: MDAAuditConfig) -> str:
    if config.artifact_id is not None:
        return config.artifact_id
    return _expected_experiment_name(config.reward_mode)


def _account_value_label(config: MDAAuditConfig) -> str:
    if config.artifact_id is not None:
        return config.artifact_id
    if config.reward_mode == "mda":
        return "mda"
    return _artifact_label(config)


def _validate_path_component(value: str, *, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    if value != value.strip():
        raise ValueError(
            f"{field_name} must not have leading or trailing whitespace"
        )
    if Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{field_name} must be a single path component")
