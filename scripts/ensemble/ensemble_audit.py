"""Audit helpers for FinRL ensemble experiments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import statistics
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class AuditConfig:
    seed: int
    experiment_name: str
    output_root: str | os.PathLike[str]
    data_path: str | os.PathLike[str]
    notebook_path: str | os.PathLike[str]
    dji_path: str | os.PathLike[str] | None = None
    deterministic: bool = True
    log_freq: int = 1000
    run_name: str | None = None
    use_seed_dir_as_run_dir: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditResult:
    run_dir: Path
    df_summary: pd.DataFrame
    account_value: pd.DataFrame
    baseline: pd.DataFrame | None
    comparison: pd.DataFrame | None
    ensemble_stats: pd.DataFrame | None
    baseline_stats: pd.DataFrame | None


def set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, Stable-Baselines3 and Torch when available."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def audit_ensemble_run(
    *,
    ensemble_agent: Any,
    config: AuditConfig,
    model_kwargs: Mapping[str, Mapping[str, Any] | None],
    timesteps_dict: Mapping[str, int],
    run_ensemble_strategy: Callable[[], pd.DataFrame] | None = None,
    processed: pd.DataFrame | None = None,
    raw_data: pd.DataFrame | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    train_period: Sequence[str] | None = None,
    val_test_period: Sequence[str] | None = None,
    rebalance_window: int | None = None,
    validation_window: int | None = None,
    result_label: str = "ensemble",
    trade_model_name: str = "ensemble",
) -> AuditResult:
    """Run a FinRL rolling strategy and persist reproducibility/debug artifacts.

    ``result_label`` names strategy-level artifacts; ``trade_model_name`` is
    the model-name segment emitted by FinRL's ``DRL_prediction`` trade files.
    Both default to ``ensemble`` for the original AUST ensemble contract.
    """
    _validate_artifact_label(result_label, field_name="result_label")
    _validate_artifact_label(trade_model_name, field_name="trade_model_name")
    run_dir = _make_run_dir(config)
    dirs = ensure_audit_dirs(run_dir)
    _copy_if_exists(config.notebook_path, dirs["audit"] / Path(config.notebook_path).name)

    set_reproducible_seed(config.seed, deterministic=config.deterministic)
    snapshot_runtime(exp_dir=run_dir)
    snapshot_rng_state(exp_dir=run_dir, label="before_training")
    snapshot_experiment_config(
        exp_dir=run_dir,
        seed=config.seed,
        train_period=train_period or getattr(ensemble_agent, "train_period", ()),
        val_test_period=val_test_period or getattr(ensemble_agent, "val_test_period", ()),
        rebalance_window=rebalance_window or getattr(ensemble_agent, "rebalance_window", None),
        validation_window=validation_window or getattr(ensemble_agent, "validation_window", None),
        env_kwargs=env_kwargs or {},
        model_kwargs=model_kwargs,
        timesteps_dict=timesteps_dict,
        notebook_name=Path(config.notebook_path).name,
        extra={
            **dict(config.extra),
            "experiment_name": config.experiment_name,
            "result_label": result_label,
            "trade_model_name": trade_model_name,
            "data_path": str(config.data_path),
            "dji_path": str(config.dji_path) if config.dji_path else None,
        },
    )
    if raw_data is not None:
        snapshot_dataframe(raw_data, exp_dir=run_dir, name="raw_data", key_columns=["date", "tic"])
    if processed is not None:
        snapshot_dataframe(processed, exp_dir=run_dir, name="processed", key_columns=["date", "tic"])

    result: dict[str, Any] = {"status": "running", "run_dir": str(run_dir)}
    save_json(result, dirs["audit"] / "run_status.json")
    start = time.time()

    try:
        with _finrl_output_dirs(run_dir), _instrument_ensemble_agent(ensemble_agent, config, run_dir):
            df_summary = (
                run_ensemble_strategy()
                if run_ensemble_strategy is not None
                else ensemble_agent.run_ensemble_strategy(
                    model_kwargs.get("a2c"),
                    model_kwargs.get("ppo"),
                    model_kwargs.get("ddpg"),
                    model_kwargs.get("sac"),
                    model_kwargs.get("td3"),
                    timesteps_dict,
                )
            )
        summary_path = dirs["results"] / "df_summary.csv"
        df_summary.to_csv(summary_path, index=False)

        account_value = collect_ensemble_account_value(
            results_dir=run_dir / "results",
            unique_trade_date=getattr(ensemble_agent, "unique_trade_date"),
            rebalance_window=getattr(ensemble_agent, "rebalance_window"),
            validation_window=getattr(ensemble_agent, "validation_window"),
            trade_model_name=trade_model_name,
        )
        account_value.to_csv(dirs["results"] / "df_account_value.csv", index=False)
        ensemble_stats = _backtest_stats(account_value, value_col_name="account_value")
        if ensemble_stats is not None:
            ensemble_stats.to_csv(dirs["results"] / f"{result_label}_stats.csv")
        write_result_metrics(
            account_value,
            exp_dir=run_dir,
            label=result_label,
            value_col="account_value",
        )

        baseline = _load_baseline(config.dji_path, account_value, getattr(ensemble_agent, "initial_amount", None))
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
                    columns={"account_value": result_label}
                ),
                baseline[["date", "dji"]],
                on="date",
                how="inner",
            ).set_index("date")
            comparison.to_csv(dirs["results"] / "result.csv")

        snapshot_rng_state(exp_dir=run_dir, label="after_training")
        write_file_manifest(exp_dir=run_dir)
        result.update(
            {
                "status": "completed",
                "elapsed_seconds": time.time() - start,
                "summary_rows": int(len(df_summary)),
                "account_value_rows": int(len(account_value)),
            }
        )
        save_json(result, dirs["audit"] / "run_status.json")
        return AuditResult(run_dir, df_summary, account_value, baseline, comparison, ensemble_stats, baseline_stats)
    except Exception:
        snapshot_rng_state(exp_dir=run_dir, label="after_failure")
        write_file_manifest(exp_dir=run_dir)
        result.update(
            {
                "status": "failed",
                "elapsed_seconds": time.time() - start,
                "traceback": traceback.format_exc(),
            }
        )
        save_json(result, dirs["audit"] / "run_status.json")
        raise


def ensure_audit_dirs(exp_dir: str | os.PathLike[str]) -> dict[str, Path]:
    root = Path(exp_dir)
    audit_dir = root / "audit"
    dirs = {
        "root": root,
        "audit": audit_dir,
        "training": audit_dir / "training",
        "data": audit_dir / "data",
        "results": audit_dir / "results",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def snapshot_experiment_config(
    *,
    exp_dir: str | os.PathLike[str],
    seed: int,
    train_period: Sequence[str],
    val_test_period: Sequence[str],
    rebalance_window: int | None,
    validation_window: int | None,
    env_kwargs: Mapping[str, Any],
    model_kwargs: Mapping[str, Mapping[str, Any] | None],
    timesteps_dict: Mapping[str, int],
    notebook_name: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    payload = {
        "created_at_utc": _utc_now(),
        "notebook_name": notebook_name,
        "seed": seed,
        "train_period": list(train_period),
        "val_test_period": list(val_test_period),
        "rebalance_window": rebalance_window,
        "validation_window": validation_window,
        "env_kwargs": dict(env_kwargs),
        "model_kwargs": {name: dict(kwargs or {}) for name, kwargs in model_kwargs.items()},
        "timesteps_dict": dict(timesteps_dict),
        "extra": dict(extra or {}),
    }
    return save_json(payload, dirs["audit"] / "experiment_config.json")


def snapshot_runtime(
    *,
    exp_dir: str | os.PathLike[str],
    modules: Optional[Sequence[str]] = None,
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    modules = modules or (
        "finrl",
        "stable_baselines3",
        "gym",
        "gymnasium",
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
    )
    payload = {
        "created_at_utc": _utc_now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key)
            for key in (
                "PYTHONHASHSEED",
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "packages": {name: _module_version(name) for name in modules},
    }
    return save_json(payload, dirs["audit"] / "runtime.json")


def snapshot_rng_state(
    *,
    exp_dir: str | os.PathLike[str],
    label: str,
    include_torch: bool = True,
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    payload: dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "label": label,
        "python_random_state_hash": _hash_bytes(repr(random.getstate()).encode("utf-8")),
        "numpy_random_state_hash": _hash_bytes(repr(np.random.get_state()).encode("utf-8")),
    }
    if include_torch:
        try:
            import torch

            payload["torch_initial_seed"] = int(torch.initial_seed())
            payload["torch_cpu_rng_state_hash"] = _hash_bytes(torch.get_rng_state().cpu().numpy().tobytes())
            payload["torch_cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                payload["torch_cuda_rng_state_hashes"] = [
                    _hash_bytes(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()
                ]
        except Exception as exc:
            payload["torch_rng_error"] = repr(exc)
    return save_json(payload, dirs["audit"] / f"rng_state_{label}.json")


def snapshot_dataframe(
    df: pd.DataFrame,
    *,
    exp_dir: str | os.PathLike[str],
    name: str,
    key_columns: Optional[Sequence[str]] = None,
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    key_columns = list(key_columns or [])
    payload = {
        "created_at_utc": _utc_now(),
        "name": name,
        "shape": list(df.shape),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "null_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "hash": dataframe_hash(df),
        "key_ranges": _key_ranges(df, key_columns),
    }
    return save_json(payload, dirs["data"] / f"{name}_snapshot.json")


def dataframe_hash(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    meta = repr((list(df.columns), [str(dtype) for dtype in df.dtypes])).encode("utf-8")
    return _hash_bytes(meta + hashed)


def write_result_metrics(
    account_value: pd.DataFrame,
    *,
    exp_dir: str | os.PathLike[str],
    label: str,
    value_col: str = "account_value",
    date_col: Optional[str] = "date",
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    series = pd.to_numeric(account_value[value_col], errors="coerce").dropna()
    returns = series.pct_change().dropna()
    downside = returns[returns < 0]
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    payload = {
        "created_at_utc": _utc_now(),
        "label": label,
        "rows": int(len(series)),
        "start_date": _first_value(account_value, date_col),
        "end_date": _last_value(account_value, date_col),
        "start_value": _float_or_none(series.iloc[0] if len(series) else None),
        "end_value": _float_or_none(series.iloc[-1] if len(series) else None),
        "total_return": _float_or_none(series.iloc[-1] / series.iloc[0] - 1.0 if len(series) > 1 else None),
        "daily_return_mean": _float_or_none(returns.mean()),
        "daily_return_std": _float_or_none(returns.std()),
        "annualized_volatility": _float_or_none(returns.std() * np.sqrt(252)),
        "sharpe": _float_or_none(np.sqrt(252) * returns.mean() / returns.std() if returns.std() else None),
        "sortino": _float_or_none(np.sqrt(252) * returns.mean() / downside.std() if downside.std() else None),
        "max_drawdown": _float_or_none(drawdown.min()),
        "account_value_hash": dataframe_hash(account_value),
    }
    return save_json(payload, dirs["results"] / f"{label}_metrics.json")


def write_file_manifest(
    *,
    exp_dir: str | os.PathLike[str],
    patterns: Sequence[str] = ("*.csv", "*.json", "*.zip", "*.pkl", "*.txt", "events.out.tfevents.*"),
) -> Path:
    dirs = ensure_audit_dirs(exp_dir)
    root = Path(exp_dir)
    rows = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                rows.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _file_sha256(path),
                        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    }
                )
    manifest = pd.DataFrame(rows).sort_values("path") if rows else pd.DataFrame(columns=["path", "size_bytes", "sha256", "modified_utc"])
    out = dirs["audit"] / "file_manifest.csv"
    manifest.to_csv(out, index=False)
    return out


def make_sb3_training_callback(
    *,
    exp_dir: str | os.PathLike[str],
    model_name: str,
    window_index: Optional[int] = None,
    log_freq: int = 1000,
):
    from stable_baselines3.common.callbacks import BaseCallback

    dirs = ensure_audit_dirs(exp_dir)
    safe_model = str(model_name).lower()
    suffix = "" if window_index is None else f"_window_{window_index}"
    out = dirs["training"] / f"{safe_model}{suffix}_training_log.csv"

    class AuditTrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows: list[dict[str, Any]] = []
            self._ppo_updates: list[dict[str, float]] = []
            self._seen_ppo_updates: set[int] = set()

        def _on_step(self) -> bool:
            if self.num_timesteps == 1 or self.num_timesteps % log_freq == 0:
                self._record("step")
            return True

        def _on_rollout_end(self) -> None:
            self._record("rollout_end")

        def _on_training_end(self) -> None:
            self._record("training_end")
            self.flush()

        def _record(self, event: str) -> None:
            row = {
                "created_at_utc": _utc_now(),
                "event": event,
                "model_name": model_name,
                "window_index": window_index,
                "num_timesteps": int(self.num_timesteps),
                "n_calls": int(self.n_calls),
            }
            for key in ("rewards", "dones", "infos", "actions"):
                if key in self.locals:
                    row[f"{key}_summary"] = _summarize_arraylike(self.locals[key])
            logger = getattr(self.model, "logger", None)
            logger_values = getattr(logger, "name_to_value", None)
            if logger_values:
                row["logger_values"] = _to_jsonable(dict(logger_values))
                update = _ppo_update_metrics(logger_values)
                update_index = update.get("n_updates")
                if update_index is not None and int(update_index) not in self._seen_ppo_updates:
                    self._ppo_updates.append(update)
                    self._seen_ppo_updates.add(int(update_index))
            self.rows.append(row)

        def training_quality_metrics(self) -> dict[str, float]:
            """Summarize the final fifth of PPO updates for the trade gate."""
            if not self._ppo_updates:
                return {}
            late = self._ppo_updates[max(0, int(len(self._ppo_updates) * 0.8)):]
            result: dict[str, float] = {"update_count": float(len(self._ppo_updates))}
            for source, target in (
                ("explained_variance", "late_explained_variance_median"),
                ("approx_kl", "late_approx_kl_median"),
                ("clip_fraction", "late_clip_fraction_median"),
            ):
                values = [row[source] for row in late if source in row]
                if values:
                    result[target] = float(statistics.median(values))
            ev_values = [row["explained_variance"] for row in late if "explained_variance" in row]
            if ev_values:
                result["late_explained_variance_negative_share"] = sum(value < 0 for value in ev_values) / len(ev_values)
            return result

        def flush(self) -> None:
            if self.rows:
                pd.DataFrame(self.rows).to_csv(out, index=False)

    return AuditTrainingCallback()


def _ppo_update_metrics(logger_values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, key in (
        ("train/explained_variance", "explained_variance"),
        ("train/approx_kl", "approx_kl"),
        ("train/clip_fraction", "clip_fraction"),
        ("train/n_updates", "n_updates"),
    ):
        value = logger_values.get(name)
        if value is not None:
            try:
                result[key] = float(value)
            except (TypeError, ValueError):
                pass
    return result


def collect_ensemble_account_value(
    *,
    results_dir: str | os.PathLike[str],
    unique_trade_date: Sequence[Any],
    rebalance_window: int,
    validation_window: int,
    trade_model_name: str = "ensemble",
) -> pd.DataFrame:
    """Collect AUST rolling trade files for one emitted model name."""
    _validate_artifact_label(trade_model_name, field_name="trade_model_name")
    results_dir = Path(results_dir)
    df_account_value = pd.DataFrame()
    for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
        path = results_dir / f"account_value_trade_{trade_model_name}_{i}.csv"
        if path.exists():
            temp = pd.read_csv(path)
            df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
    if df_account_value.empty:
        raise FileNotFoundError(
            f"No {trade_model_name} account value CSVs found under {results_dir}"
        )

    trade_dates = pd.DataFrame({"date": unique_trade_date})[validation_window:].reset_index(drop=True)
    if "date" not in df_account_value.columns:
        df_account_value = df_account_value.join(trade_dates.iloc[: len(df_account_value)])
    return df_account_value


def _make_run_dir(config: AuditConfig) -> Path:
    if config.use_seed_dir_as_run_dir:
        if config.run_name is not None:
            raise ValueError(
                "run_name must be omitted when use_seed_dir_as_run_dir=True"
            )
        run_dir = Path(config.output_root) / f"seed_{config.seed}"
        if run_dir.exists():
            raise FileExistsError(
                f"refusing to mix outputs from separate runs under {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    run_name = config.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(config.output_root) / f"seed_{config.seed}" / run_name
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir.with_name(f"{base_dir.name}_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _validate_artifact_label(value: str, *, field_name: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty path-safe label")


@contextlib.contextmanager
def _finrl_output_dirs(run_dir: Path):
    from finrl import config

    old_cwd = Path.cwd()
    old_values = {
        "RESULTS_DIR": config.RESULTS_DIR,
        "TRAINED_MODEL_DIR": config.TRAINED_MODEL_DIR,
        "TENSORBOARD_LOG_DIR": config.TENSORBOARD_LOG_DIR,
    }
    new_values = {
        "RESULTS_DIR": str(run_dir / "results"),
        "TRAINED_MODEL_DIR": str(run_dir / "trained_models"),
        "TENSORBOARD_LOG_DIR": str(run_dir / "tensorboard_log"),
    }
    for value in new_values.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    try:
        for key, value in new_values.items():
            setattr(config, key, value)
        os.chdir(run_dir)
        yield
    finally:
        os.chdir(old_cwd)
        for key, value in old_values.items():
            setattr(config, key, value)


@contextlib.contextmanager
def _instrument_ensemble_agent(ensemble_agent: Any, config: AuditConfig, run_dir: Path):
    original_get_model = ensemble_agent.get_model
    original_train_model = ensemble_agent.train_model
    training_rows: list[dict[str, Any]] = []
    call_counts: dict[str, int] = {}

    def audited_get_model(model_name, env, policy="MlpPolicy", policy_kwargs=None, model_kwargs=None, seed=None, verbose=1):
        effective_seed = config.seed if seed is None else seed
        model = original_get_model(
            model_name,
            env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            model_kwargs=model_kwargs,
            seed=effective_seed,
            verbose=verbose,
        )
        setattr(model, "_audit_model_name", model_name)
        setattr(model, "_audit_seed", effective_seed)
        return model

    def audited_train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000, callbacks=None):
        callback = make_sb3_training_callback(
            exp_dir=run_dir,
            model_name=model_name,
            window_index=iter_num,
            log_freq=config.log_freq,
        )
        callback_list = list(callbacks or [])
        callback_list.append(callback)
        started = time.time()
        row = {
            "model_name": model_name,
            "iteration": iter_num,
            "tb_log_name": tb_log_name,
            "total_timesteps": total_timesteps,
            "seed": getattr(model, "_audit_seed", None),
            "status": "running",
        }
        training_rows.append(row)
        try:
            trained = original_train_model(
                model,
                model_name,
                tb_log_name=tb_log_name,
                iter_num=iter_num,
                total_timesteps=total_timesteps,
                callbacks=callback_list,
            )
            row["status"] = "completed"
            return trained
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            raise
        finally:
            flush = getattr(callback, "flush", None)
            if callable(flush):
                flush()
            row["elapsed_seconds"] = time.time() - started
            row["num_timesteps_after_training"] = int(getattr(model, "num_timesteps", 0))
            call_counts[model_name] = call_counts.get(model_name, 0) + 1
            pd.DataFrame(training_rows).to_csv(run_dir / "audit" / "training_runs.csv", index=False)

    try:
        ensemble_agent.get_model = audited_get_model
        ensemble_agent.train_model = audited_train_model
        yield
    finally:
        ensemble_agent.get_model = original_get_model
        ensemble_agent.train_model = original_train_model
        save_json({"model_train_call_counts": call_counts}, run_dir / "audit" / "training_call_counts.json")


def _load_baseline(
    dji_path: str | os.PathLike[str] | None,
    account_value: pd.DataFrame,
    initial_amount: float | None,
) -> pd.DataFrame | None:
    if dji_path is None or not Path(dji_path).exists():
        return None
    baseline = pd.read_csv(dji_path)
    baseline.columns = [str(c).lower() for c in baseline.columns]
    if "date" not in baseline.columns and "datadate" in baseline.columns:
        baseline = baseline.rename(columns={"datadate": "date"})
    if "date" not in baseline.columns or "close" not in baseline.columns:
        return None
    baseline["date"] = pd.to_datetime(baseline["date"]).dt.strftime("%Y-%m-%d")
    dates = pd.to_datetime(account_value["date"]).dt.strftime("%Y-%m-%d")
    baseline = baseline[baseline["date"].isin(set(dates))].copy()
    baseline = baseline.sort_values("date").reset_index(drop=True)
    if baseline.empty:
        return None
    start_value = float(baseline["close"].iloc[0])
    amount = float(initial_amount or account_value["account_value"].iloc[0])
    baseline["dji"] = baseline["close"] / start_value * amount
    return baseline[["date", "dji", "close"]]


def _backtest_stats(account_value: pd.DataFrame, *, value_col_name: str) -> pd.DataFrame | None:
    try:
        from finrl.plot import backtest_stats

        return pd.DataFrame(backtest_stats(account_value=account_value, value_col_name=value_col_name))
    except TypeError:
        try:
            from finrl.plot import backtest_stats

            return pd.DataFrame(backtest_stats(account_value=account_value))
        except Exception:
            return None
    except Exception:
        return None


def _copy_if_exists(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    src_path = Path(src)
    if src_path.exists():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)


def _summarize_arraylike(value: Any) -> JSONValue:
    try:
        arr = np.asarray(value, dtype=float)
        return {
            "shape": list(arr.shape),
            "mean": _float_or_none(np.nanmean(arr)),
            "std": _float_or_none(np.nanstd(arr)),
            "min": _float_or_none(np.nanmin(arr)),
            "max": _float_or_none(np.nanmax(arr)),
        }
    except Exception:
        return repr(value)


def _to_jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return repr(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _key_ranges(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, Any]]:
    ranges: dict[str, dict[str, Any]] = {}
    for col in columns:
        if col in df.columns and len(df[col]):
            ranges[col] = {
                "min": _to_jsonable(df[col].min()),
                "max": _to_jsonable(df[col].max()),
                "nunique": int(df[col].nunique(dropna=True)),
            }
    return ranges


def _first_value(df: pd.DataFrame, col: Optional[str]) -> Optional[Any]:
    if col and col in df.columns and len(df[col]):
        return _to_jsonable(df[col].iloc[0])
    return None


def _last_value(df: pd.DataFrame, col: Optional[str]) -> Optional[Any]:
    if col and col in df.columns and len(df[col]):
        return _to_jsonable(df[col].iloc[-1])
    return None


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
