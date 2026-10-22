#!/usr/bin/env python3
"""Compare a re-run against the archived run of the same (condition, seed).

The authoritative checks are the account-value hash written into
``*_metrics.json``, the account-value series, the observation contract, and the
run status.  The experiment config is compared on the intersection of keys so
that later additive configuration keys do not mask a mechanism difference.

Usage:
    python scripts/Tail_GAE/compare_runs.py --condition 1321b_hi --seed 0 \
        --old-root /path/to/archive/Results --new-root /path/to/scratch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [
        p
        for p in base.iterdir()
        if p.is_dir() and "superceded" not in p.name.lower()
    ]
    return max(dirs, key=lambda p: p.name) if dirs else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def one(paths: list[Path], what: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"expected one {what}, found {len(paths)}: {paths}")
    return paths[0]


def suite(run_dir: Path) -> dict:
    metrics = one(sorted((run_dir / "audit" / "results").glob("*_metrics.json")), "metrics json")
    account = one(
        sorted((run_dir / "audit" / "results").glob("df_account_value_*.csv")),
        "account value csv",
    )
    return {
        "metrics_path": metrics,
        "account_path": account,
        "metrics": load_json(metrics),
        "account": pd.read_csv(account),
        "observation": load_json(run_dir / "audit" / "observation_contract.json"),
        "config": load_json(run_dir / "audit" / "experiment_config.json"),
        "status": load_json(run_dir / "audit" / "run_status.json"),
    }


METRIC_KEYS = [
    "total_return",
    "annualized_volatility",
    "max_drawdown",
    "sharpe",
    "sortino",
    "account_value_hash",
    "end_value",
    "rows",
]


def compare_metrics(old: dict, new: dict, tol: float) -> dict:
    out = {}
    for key in METRIC_KEYS:
        a, b = old.get(key), new.get(key)
        if isinstance(a, float) and isinstance(b, float):
            out[key] = {
                "old": a,
                "new": b,
                "abs_diff": abs(a - b),
                "pass": math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol,
            }
        else:
            out[key] = {"old": a, "new": b, "pass": a == b}
    return out


def compare_account(old: pd.DataFrame, new: pd.DataFrame, tol: float) -> dict:
    if list(old.columns) != list(new.columns):
        return {"pass": False, "reason": f"columns differ: {list(old.columns)} vs {list(new.columns)}"}
    if len(old) != len(new):
        return {"pass": False, "reason": f"rows differ: {len(old)} vs {len(new)}"}
    date_ok = old.iloc[:, 0].astype(str).equals(new.iloc[:, 0].astype(str))
    old_values = old.iloc[:, 1].astype(float).to_numpy()
    new_values = new.iloc[:, 1].astype(float).to_numpy()
    max_diff = float(abs(old_values - new_values).max()) if len(old_values) else 0.0
    return {
        "pass": bool(date_ok and max_diff <= tol),
        "dates_equal": bool(date_ok),
        "rows": int(len(old)),
        "max_abs_diff": max_diff,
        "old_sha256": hashlib.sha256(old.to_csv(index=False).encode()).hexdigest(),
        "new_sha256": hashlib.sha256(new.to_csv(index=False).encode()).hexdigest(),
    }


def compare_contract(old: dict, new: dict) -> dict:
    """Compare contracts, allowing the Phase-9 metadata keys to default to zero.

    The 1270b archive predates the ``phase9_observation_*`` metadata keys.  A
    new run of a pre-Phase-9 condition carries them with zero/empty values;
    that is an additive schema change, not a mechanism change.
    """
    if old == new:
        return {"pass": True, "old": old, "new": new}
    extra = {k: new[k] for k in new if k not in old}
    missing = {k: old[k] for k in old if k not in new}
    changed = {k: {"old": old[k], "new": new[k]} for k in old if k in new and old[k] != new[k]}
    compatible_extra = all(
        (key == "phase9_observation_dim" and value == 0)
        or (key == "phase9_observation_fields" and value == [])
        for key, value in extra.items()
    )
    return {
        "pass": not missing and not changed and compatible_extra,
        "old": old,
        "new": new,
        "added_compatible": extra,
        "missing": missing,
        "changed": changed,
    }


def compare_config(old: dict, new: dict) -> dict:
    ignored = {"data_path", "dji_path", "output_root", "created_at_utc"}
    old_extra = {k: v for k, v in old.get("extra", {}).items() if k not in ignored}
    new_extra = {k: v for k, v in new.get("extra", {}).items() if k not in ignored}
    common = sorted(set(old_extra) & set(new_extra))
    diffs = {k: {"old": old_extra[k], "new": new_extra[k]} for k in common if old_extra[k] != new_extra[k]}
    added = sorted(set(new_extra) - set(old_extra))
    removed = sorted(set(old_extra) - set(new_extra))
    env_same = old.get("env_kwargs") == new.get("env_kwargs")
    model_same = old.get("model_kwargs") == new.get("model_kwargs")
    return {
        "pass": not diffs and env_same and model_same,
        "common_key_diffs": diffs,
        "keys_added_in_new": added,
        "keys_removed_in_new": removed,
        "env_kwargs_equal": env_same,
        "model_kwargs_equal": model_same,
    }


def compare_rng(old_dir: Path, new_dir: Path) -> dict:
    a = old_dir / "audit" / "rng_state_after_training.json"
    b = new_dir / "audit" / "rng_state_after_training.json"
    if not a.exists() or not b.exists():
        return {"pass": None, "reason": "missing"}
    ha = hashlib.sha256(json.dumps(load_json(a), sort_keys=True).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(load_json(b), sort_keys=True).encode()).hexdigest()
    return {"pass": ha == hb, "old_sha256": ha, "new_sha256": hb}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    old_dir = latest_run(args.old_root / "Tail_GAE" / args.condition / f"seed_{args.seed}")
    new_dir = latest_run(args.new_root / "Tail_GAE" / args.condition / f"seed_{args.seed}")
    if old_dir is None or new_dir is None:
        print(json.dumps({"condition": args.condition, "seed": args.seed, "pass": False,
                          "reason": "missing run dir", "old": str(old_dir), "new": str(new_dir)}))
        return 1
    old = suite(old_dir)
    new = suite(new_dir)
    result = {
        "condition": args.condition,
        "seed": args.seed,
        "old_run": str(old_dir),
        "new_run": str(new_dir),
        "metrics": compare_metrics(old["metrics"], new["metrics"], args.tol),
        "account": compare_account(old["account"], new["account"], args.tol),
        "observation": compare_contract(old["observation"], new["observation"]),
        "config": compare_config(old["config"], new["config"]),
        "rng": compare_rng(old_dir, new_dir),
        "old_status": old["status"].get("status"),
        "new_status": new["status"].get("status"),
    }
    result["pass"] = bool(
        all(v["pass"] for v in result["metrics"].values())
        and result["account"]["pass"]
        and result["observation"]["pass"]
        and result["config"]["pass"]
        and result["old_status"] == "completed"
        and result["new_status"] == "completed"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
