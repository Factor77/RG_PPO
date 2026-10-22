"""Main training entry point callable from Jupyter notebooks (Tail_GAE V3).

V3 spec §5: the entry is audit-driven.  ``run_tail_gae_experiment`` builds the
pinned ``MDAAuditConfig`` (reward_mode="return", lambda_value=None, no quality
gate, V2 normalization record) and delegates to ``audit_mda_ppo_run`` with
``runner_cls=TailGAERollingRunner``: the 1200 rolling protocol (63/63 PU-6,
4 x 40k windows at iterations 126/189/252/315) with the V2 mechanism layer
(risk-state block via ``RiskStateEnvWrapper``, Tail-GAE advantage, and
SliceVecNormalize) injected through ``runner_kwargs`` (spec §4.1).

The V2 single-segment training path (``_make_env`` / ``_make_tail_gae_ppo`` /
``_train_model`` / ``_rollout_trade``, 61-dim branch, direct
SliceVecNormalize wiring) was removed per spec §5 point 6; the diagnostics
callback is mounted by the runner per training window, only when a risk
wrapper exists (spec §3.3), so the entry no longer prints diagnostics.

1200 bridge semantics: ``experiment_id="1200"`` resolves ``risk_state_names``
to ``[]`` and ``alpha=0``; ``risk_observation_fields=None`` and
``expected_risk_observation_fields=None`` then send the runner down the parent
(1200) protocol unchanged, while ``reward_mode="return"`` stays pinned (the
DownsideBreadthStockTradingEnv only accepts "return").
"""

from __future__ import annotations

import json
import os
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Peer-module imports (dual-mode: package-style and flat-import universe)
# ---------------------------------------------------------------------------

try:
    from .mda_ppo_audit import MDAAuditConfig, audit_mda_ppo_run
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from mda_ppo_audit import MDAAuditConfig, audit_mda_ppo_run

try:
    from .notebook_config import (
        DATA_FILE,
        DJI_FILE,
        PPO_MODEL_KWARGS,
        PPO_TIMESTEPS,
        get_experiment_config,
    )
except ImportError:
    from notebook_config import (
        DATA_FILE,
        DJI_FILE,
        PPO_MODEL_KWARGS,
        PPO_TIMESTEPS,
        get_experiment_config,
    )

try:
    from .tail_gae_data import load_tail_data
except ImportError:
    from tail_gae_data import load_tail_data

try:
    from .tail_gae_rolling_runner import TailGAERollingRunner
except ImportError:
    from tail_gae_rolling_runner import TailGAERollingRunner

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    """Apply the same Python / NumPy / SB3 / Torch seed contract as 0900."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed)
    except Exception:
        pass
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Notebook path resolution
# ---------------------------------------------------------------------------

def _resolve_notebook_path(
    experiment_id: str, seed: int, n_risk_dims: int
) -> Path:
    """Best-effort path of the executing notebook (audit copies it into the run).

    Per-experiment notebook directories live under
    ``ultimate_experiment/<experiment_id>*/`` (the 1205 directory is currently
    ``1205 00 Tail_GAE``) and the file is named
    ``Tail_GAE_risk_<n_risk_dims>_seed_<seed>.ipynb``.  A missing file is
    tolerated: ``audit_mda_ppo_run`` skips the copy when the path does not
    exist.
    """
    project_root = Path(__file__).resolve().parents[3]
    filename = f"Tail_GAE_risk_{n_risk_dims}_seed_{seed}.ipynb"
    candidates: list[Path] = [
        project_root / "ultimate_experiment" / f"{experiment_id} Tail_GAE" / filename,
    ]
    try:
        candidates.extend(
            sorted(
                project_root.glob(
                    f"ultimate_experiment/{experiment_id}*/{filename}"
                )
            )
        )
    except Exception:
        pass
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return candidates[0]


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

def run_tail_gae_experiment(experiment_id: str, seed: int) -> dict:
    """Run one Tail-GAE configuration through the V3 audit-driven pipeline.

    Parameters
    ----------
    experiment_id : str
        One of ``"1200"``, ``"1205"``, ``"1207"``, ``"1210"`` -- maps to a
        date-range + risk-state configuration in ``notebook_config.py``.
    seed : int
        Random seed (0--9).

    Returns
    -------
    dict
        Keys: ``run_dir`` (Path), ``account_value`` (DataFrame [date,
        account_value]), ``df_summary``, ``mda_stats``, ``baseline``,
        ``baseline_stats``, ``comparison``, ``runner_result``,
        ``observation_contract``, ``diagnostics``, ``config``, ``q_by_date``,
        ``dji_return_by_date``, ``window_dates``.
    """

    # ------------------------------------------------------------------
    # 1. Load experiment configuration
    # ------------------------------------------------------------------
    exp_cfg = get_experiment_config(experiment_id, seed)
    experiment_name: str = exp_cfg["experiment_name"]
    alpha: float = float(exp_cfg["alpha"])
    p_tail: float = float(exp_cfg["p_tail"])
    ge_penalty_beta: float = float(exp_cfg.get("ge_penalty_beta", 0.0))
    ge_penalty_ref: float = float(exp_cfg.get("ge_penalty_ref", 0.95))
    ge_penalty_mode: str = str(exp_cfg.get("ge_penalty_mode", "total"))
    risk_state_names: list[str] = list(exp_cfg["risk_state_names"])
    n_risk_dims: int = len(risk_state_names)
    use_risk_observation: bool = bool(exp_cfg["use_risk_observation"])
    output_root: Path = Path(exp_cfg["output_root"])
    train_period: tuple[str, str] = (exp_cfg["train_start"], exp_cfg["train_end"])
    val_test_period: tuple[str, str] = (
        exp_cfg["val_test_start"],
        exp_cfg["val_test_end"],
    )

    print("=" * 60)
    print(f"Tail_GAE Experiment: {experiment_id}  seed={seed}")
    print(f"  notebook: {exp_cfg['notebook_name']}")
    print(f"  train:    {train_period[0]} -> {train_period[1]}")
    print(f"  val/test: {val_test_period[0]} -> {val_test_period[1]}")
    print(f"  alpha:    {alpha}")
    print(f"  p_tail:   {p_tail}")
    print(f"  ge_penalty_beta: {ge_penalty_beta}  ref: {ge_penalty_ref}")
    print(f"  risk dims: {n_risk_dims}  ({', '.join(risk_state_names) if risk_state_names else 'none'})")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Load data (301-dim base state, spec §2)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading tail data ...")
    tail_data = load_tail_data(p_tail=p_tail)
    processed: pd.DataFrame = tail_data["processed"]
    signal_table: pd.DataFrame = tail_data["signal_table"]
    env_kwargs: dict[str, Any] = dict(tail_data["env_kwargs"])
    i_by_date: dict[str, int] = dict(tail_data["i_by_date"])
    worst20_by_date: dict[str, float] = dict(tail_data["worst20_by_date"])
    q_by_date: dict[str, float] = dict(tail_data["q_by_date"])
    dji_return_by_date: dict[str, float] = dict(tail_data["dji_return_by_date"])
    window_dates: list[str] = list(tail_data["window_dates"])
    raw_data: pd.DataFrame = pd.read_csv(DATA_FILE)  # original CSV (audit snapshot)

    # ------------------------------------------------------------------
    # 3. Build the pinned MDAAuditConfig (spec §5 point 3)
    # ------------------------------------------------------------------
    print("\n[2/4] Building MDAAuditConfig ...")
    audit_config = MDAAuditConfig(
        seed=seed,
        lambda_value=None,  # pinned: no MDA reward; return mode only
        reward_mode="return",  # DownsideBreadthStockTradingEnv accepts only "return"
        deterministic=True,  # deterministic trade/validation backtest
        output_root=output_root,
        data_path=DATA_FILE,
        notebook_path=_resolve_notebook_path(experiment_id, seed, n_risk_dims),
        dji_path=DJI_FILE,
        experiment_name=experiment_name.lower(),  # OutputPathManager: lowercase only
        artifact_id=f"{experiment_id} {experiment_name.lower()}",
        run_directory_layout="seed_timestamp",
        log_freq=1024,
        # V2 does not adopt the quality gate / eligibility marking (spec §1.1).
        allow_ineligible_backtest=True,
        use_risk_observation=use_risk_observation,
        # Record-only; the actual normalization is driven by SliceVecNormalize
        # inside the runner (training-updated, eval-frozen, no fit-once).
        normalization_config={
            "enabled": True,
            "normalize_observations": True,
            "normalize_rewards": False,
            "fit_once_on_meta_train": False,
            "normalize_mda_components": False,
        },
        # None = 1200 bridge (parent schema); V2 runs declare their names.
        risk_observation_fields=risk_state_names or None,
    )

    # ------------------------------------------------------------------
    # 4. Run the audit-driven rolling training
    # ------------------------------------------------------------------
    print("\n[3/4] Running audit-driven rolling training (63/63, 4 x 40k) ...")
    _set_reproducible_seed(seed, deterministic=True)

    mda_result = audit_mda_ppo_run(
        config=audit_config,
        processed=processed,
        raw_data=raw_data,
        env_kwargs=env_kwargs,
        train_period=train_period,
        val_test_period=val_test_period,
        rebalance_window=63,  # PU-6 hard constraint
        validation_window=63,  # PU-6 hard constraint
        ppo_model_kwargs=PPO_MODEL_KWARGS,
        ppo_timesteps=PPO_TIMESTEPS,
        runner_cls=TailGAERollingRunner,
        runner_kwargs={
            "tail_alpha": alpha,
            "i_by_date": i_by_date,
            "risk_state_names": risk_state_names,
            "signal_table": signal_table,
            "worst20_by_date": worst20_by_date,
            "ew_by_date": tail_data["ew_by_date"],
            "ge_penalty_beta": ge_penalty_beta,
            "ge_penalty_ref": ge_penalty_ref,
            "ge_penalty_mode": ge_penalty_mode,
            "tilt_shift": bool(exp_cfg.get("tilt_shift", False)),
            "expected_risk_observation_fields": risk_state_names or None,
            "direct_penalty_beta": float(exp_cfg.get("direct_penalty_beta", 0.0)),
            "direct_penalty_ref": float(exp_cfg.get("direct_penalty_ref", 0.85)),
        },
    )

    run_dir = Path(mda_result.run_dir)
    account_value: pd.DataFrame = mda_result.account_value
    runner_result: dict[str, Any] = dict(mda_result.runner_result)
    observation_contract: dict[str, Any] = dict(
        runner_result.get("observation", {})
    )

    # ------------------------------------------------------------------
    # 5. Save results (legacy layout contract: run_dir-root copies)
    # ------------------------------------------------------------------
    print("\n[4/4] Saving results ...")

    if len(account_value) > 0:
        account_value.to_csv(run_dir / "account_value.csv", index=False)
    with (run_dir / "observation_contract.json").open("w", encoding="utf-8") as fh:
        json.dump(observation_contract, fh, indent=2, sort_keys=True)

    config_snapshot = {
        "experiment_id": experiment_id,
        "seed": seed,
        "experiment_name": experiment_name,
        "alpha": alpha,
        "p_tail": p_tail,
        "ge_penalty_beta": ge_penalty_beta,
        "ge_penalty_ref": ge_penalty_ref,
        "ge_penalty_mode": ge_penalty_mode,
        "risk_state_names": risk_state_names,
        "n_risk_dims": n_risk_dims,
        "use_risk_observation": use_risk_observation,
        "reward_mode": "return",
        "lambda_value": None,
        "deterministic": True,
        "tail_trigger": "latch_market_EW_k10_rel5>0",
        "normalization_config": dict(audit_config.normalization_config),
        "quality_gate_config": dict(audit_config.quality_gate_config),
        "rebalance_window": 63,
        "validation_window": 63,
        "ppo_timesteps": PPO_TIMESTEPS,
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(config_snapshot, fh, indent=2, sort_keys=True, default=str)

    manifest = {
        "experiment_id": experiment_id,
        "seed": seed,
        "run_id": f"{experiment_id}_{experiment_name.lower()}",
        "notebook_name": exp_cfg["notebook_name"],
        "train_period": list(train_period),
        "val_test_period": list(val_test_period),
        "alpha": alpha,
        "p_tail": p_tail,
        "risk_state_names": risk_state_names,
        "n_risk_dims": n_risk_dims,
        "use_risk_observation": use_risk_observation,
        "reward_mode": "return",
        "lambda_value": None,
        "ppo_timesteps": PPO_TIMESTEPS,
        "account_value_rows": int(len(account_value)),
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    # Diagnostics: per-window training events recorded by the audit collector.
    diagnostics_path = run_dir / "audit" / "training_runs.csv"
    diagnostics = (
        pd.read_csv(diagnostics_path)
        if diagnostics_path.exists()
        else pd.DataFrame()
    )

    print(f"\nResults saved to: {run_dir}")
    print("=" * 60)

    return {
        "run_dir": run_dir,
        "account_value": account_value,
        "df_summary": mda_result.df_summary,
        "baseline": mda_result.baseline,
        "comparison": mda_result.comparison,
        "mda_stats": mda_result.mda_stats,
        "baseline_stats": mda_result.baseline_stats,
        "runner_result": runner_result,
        "diagnostics": diagnostics,
        "config": config_snapshot,
        "observation_contract": observation_contract,
        "q_by_date": q_by_date,
        "dji_return_by_date": dji_return_by_date,
        "window_dates": window_dates,
    }


# ---------------------------------------------------------------------------
# Convenience: run with full error capture (suitable for notebook cells)
# ---------------------------------------------------------------------------

def run_tail_gae_experiment_safe(experiment_id: str, seed: int) -> dict:
    """Wrapper that returns error info instead of raising.

    On success the dict matches ``run_tail_gae_experiment`` exactly;
    on failure an extra ``"error"`` key is present.
    """

    try:
        return run_tail_gae_experiment(experiment_id, seed)
    except Exception as exc:
        trace = traceback.format_exc()
        print(f"\n[FAILED] experiment_id={experiment_id} seed={seed}")
        print(trace)
        return {
            "run_dir": None,
            "account_value": pd.DataFrame(),
            "df_summary": pd.DataFrame(),
            "baseline": None,
            "comparison": None,
            "mda_stats": None,
            "baseline_stats": None,
            "runner_result": {},
            "diagnostics": pd.DataFrame(),
            "config": None,
            "observation_contract": {},
            "q_by_date": {},
            "dji_return_by_date": {},
            "window_dates": [],
            "error": str(exc),
            "traceback": trace,
        }
