#!/usr/bin/env python3
"""Aggregate ten-seed metrics and compare them with the archived runs.

For each (condition, seed) the latest non-superseded run directory is used.
AR is annualised from CR with the paper's convention,
``(1 + CR) ** (252 / (rows - 1)) - 1`` (252 rows -> 251 return intervals).

Usage:
    python scripts/Tail_GAE/summarize_results.py \
        --new-root /path/to/scratch --seeds 0-9
    # optional archive comparison:
    #   --old-root /path/to/archive/Results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDITIONS = ["1250", "1270b", "1321b_ctrl", "1321b_lo", "1321b_hi"]
METRICS = ["CR", "AR", "AV", "SR", "MDD"]


def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir() and "superceded" not in p.name.lower()]
    return max(dirs, key=lambda p: p.name) if dirs else None


def read_metrics(run_dir: Path) -> dict:
    paths = sorted((run_dir / "audit" / "results").glob("*_metrics.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one metrics json in {run_dir}, found {len(paths)}")
    raw = json.loads(paths[0].read_text())
    rows = int(raw["rows"])
    cr = float(raw["total_return"])
    return {
        "CR": cr,
        "AR": (1.0 + cr) ** (252.0 / (rows - 1)) - 1.0,
        "AV": float(raw["annualized_volatility"]),
        "SR": float(raw["sharpe"]),
        "MDD": float(raw["max_drawdown"]),
        "account_value_hash": raw.get("account_value_hash"),
        "rows": rows,
    }


def collect(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        for seed in seeds:
            run_dir = latest_run(root / "Tail_GAE" / condition / f"seed_{seed}")
            if run_dir is None:
                continue
            metrics = read_metrics(run_dir)
            metrics.update({"condition": condition, "seed": seed})
            rows.append(metrics)
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, part in frame.groupby("condition", sort=False):
        row = {"condition": condition, "n": int(len(part))}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_var"] = float(part[metric].var(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=None)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", type=Path, default=Path("results_summary"))
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    new = collect(args.new_root, seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.out / "reproduction_per_seed.csv", index=False)
    new_agg = aggregate(new)
    new_agg.to_csv(args.out / "reproduction_aggregate.csv", index=False)
    print(new_agg.to_string(index=False))
    if args.old_root is None:
        print("archive comparison skipped: pass --old-root to compare against archived runs")
        return 0
    old = collect(args.old_root, seeds)
    old.to_csv(args.out / "archive_per_seed.csv", index=False)
    old_agg = aggregate(old)
    old_agg.to_csv(args.out / "archive_aggregate.csv", index=False)

    merged = old.merge(new, on=["condition", "seed"], suffixes=("_old", "_new"))
    per_seed = []
    for _, row in merged.iterrows():
        entry = {"condition": row["condition"], "seed": int(row["seed"]), "pass": True}
        for metric in METRICS:
            a, b = float(row[f"{metric}_old"]), float(row[f"{metric}_new"])
            diff = abs(a - b)
            entry[metric] = {"old": a, "new": b, "abs_diff": diff}
            if diff > args.tol:
                entry["pass"] = False
        if row.get("account_value_hash_old") != row.get("account_value_hash_new"):
            entry["pass"] = False
        per_seed.append(entry)

    agg_merged = old_agg.merge(new_agg, on="condition", suffixes=("_old", "_new"))
    aggregate_rows = []
    for _, row in agg_merged.iterrows():
        entry = {"condition": row["condition"], "n_old": int(row["n_old"]), "n_new": int(row["n_new"])}
        for metric in METRICS:
            a = float(row[f"{metric}_mean_old"])
            b = float(row[f"{metric}_mean_new"])
            entry[f"{metric}_old"] = a
            entry[f"{metric}_new"] = b
            entry[f"{metric}_diff"] = b - a
        aggregate_rows.append(entry)
    pd.DataFrame(aggregate_rows).to_csv(args.out / "aggregate_comparison.csv", index=False)
    report = {
        "old_root": "<archive>",
        "new_root": "<results_root>",
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate_rows,
        "pass": all(entry["pass"] for entry in per_seed) and len(per_seed) == len(CONDITIONS) * len(seeds),
    }
    (args.out / "reproduction_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(pd.DataFrame(aggregate_rows).to_string(index=False))
    print("per-seed pass:", sum(entry["pass"] for entry in per_seed), "/", len(per_seed))
    print("overall pass:", report["pass"])
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
