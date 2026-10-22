#!/usr/bin/env python3
"""Aggregate the ten 0101 (VSE) seeds and compare them with the archive.

The archived metrics are the full-precision values behind the 0101 row and the
per-seed table in ``Results_Analysis.md``.  The 0101 code reports
``annual_return == cumulative_return``; the table follows that convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["CR", "AR", "AV", "SR", "MDD"]


def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    dirs = [
        p
        for p in base.iterdir()
        if p.is_dir()
        and "superceded" not in p.name.lower()
        and (p / "audit").is_dir()
    ]
    return max(dirs, key=lambda p: p.name) if dirs else None


def read_metrics(run_dir: Path) -> dict:
    paths = sorted((run_dir / "audit" / "results").glob("*_metrics.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one metrics json in {run_dir}, found {len(paths)}")
    raw = json.loads(paths[0].read_text())
    cr = float(raw["total_return"])
    return {
        "CR": cr,
        "AR": cr,
        "AV": float(raw["annualized_volatility"]),
        "SR": float(raw["sharpe"]),
        "MDD": float(raw["max_drawdown"]),
        "account_value_hash": raw.get("account_value_hash"),
        "rows": int(raw["rows"]),
    }


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for seed in range(10):
        run_dir = latest_run(root / f"seed_{seed}")
        if run_dir is None:
            continue
        metrics = read_metrics(run_dir)
        metrics.update({"condition": "0101", "seed": seed})
        rows.append(metrics)
    return pd.DataFrame(rows)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    row = {"condition": "0101", "n": int(len(frame))}
    for metric in METRICS:
        row[f"{metric}_mean"] = float(frame[metric].mean())
        row[f"{metric}_var"] = float(frame[metric].var(ddof=1))
    return pd.DataFrame([row])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, default=None)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results_summary"))
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()
    new = collect(args.new_root)
    args.out.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.out / "0101_reproduction_per_seed.csv", index=False)
    new_agg = aggregate(new)
    new_agg.to_csv(args.out / "0101_reproduction_aggregate.csv", index=False)
    print(new_agg.to_string(index=False))
    if args.old_root is None:
        print("archive comparison skipped: pass --old-root to compare against the archived 0101 runs")
        return 0
    old = collect(args.old_root)
    old.to_csv(args.out / "0101_archive_per_seed.csv", index=False)
    old_agg = aggregate(old)
    old_agg.to_csv(args.out / "0101_archive_aggregate.csv", index=False)

    merged = old.merge(new, on=["condition", "seed"], suffixes=("_old", "_new")).sort_values("seed")
    rows = []
    for _, row in merged.iterrows():
        entry = {"condition": "0101", "seed": int(row.seed), "pass": True}
        for metric in METRICS:
            a, b = float(row[f"{metric}_old"]), float(row[f"{metric}_new"])
            entry[metric] = {"old": a, "new": b, "diff": b - a}
            if abs(a - b) > args.tol:
                entry["pass"] = False
        if row.get("account_value_hash_old") != row.get("account_value_hash_new"):
            entry["pass"] = False
        rows.append(entry)
    report = {
        "old_root": "<archive>",
        "new_root": "<results_root>",
        "per_seed": rows,
        "aggregate_archive": old_agg.to_dict(orient="records"),
        "aggregate_reproduction": new_agg.to_dict(orient="records"),
        "pass": all(entry["pass"] for entry in rows) and len(rows) == 10,
    }
    (args.out / "0101_reproduction_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(old_agg.to_string(index=False))
    print(new_agg.to_string(index=False))
    print("per-seed pass:", sum(entry["pass"] for entry in rows), "/", len(rows))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
