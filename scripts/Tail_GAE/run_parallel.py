#!/usr/bin/env python3
"""Run one notebook per (condition, seed) experiment, up to ``--jobs`` at once.

Each notebook is executed by ``exec_notebook.py`` in its own Jupyter kernel.
The kernel is started with the seed pinned in ``PYTHONHASHSEED`` and with the
frozen CUDA/thread environment; the notebook itself remains the experiment
definition (no temporary copies, no re-implementation of the experiment).

Example:
    python scripts/Tail_GAE/run_parallel.py \
        --conditions 1321b_hi --seeds 0 --jobs 1 \
        --results-root /path/to/scratch
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXEC = Path(__file__).resolve().parent / "exec_notebook.py"

CONDITIONS = {
    "1250": ("1250 Tail_GAE", "Tail_GAE_1250_r0_unified_1200_control_seed_{seed}.ipynb"),
    "1270b": ("1270b Tail_GAE", "Tail_GAE_1270b_s1_corrected_4c_seed_{seed}.ipynb"),
    "1321b_ctrl": ("1321b Tail_GAE", "Tail_GAE_1321b_ctrl_seed_{seed}.ipynb"),
    "1321b_lo": ("1321b Tail_GAE", "Tail_GAE_1321b_lo_seed_{seed}.ipynb"),
    "1321b_hi": ("1321b Tail_GAE", "Tail_GAE_1321b_hi_seed_{seed}.ipynb"),
    "0101": ("0101 AUST_ensemble + original", "AUST_seed_{seed}.ipynb"),
}


def parse_seeds(value: str) -> list[int]:
    seeds: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(part))
    invalid = sorted(s for s in seeds if s < 0 or s > 9)
    if invalid:
        raise SystemExit(f"seeds must be in 0..9: {invalid}")
    return sorted(seeds)


def notebook_path(condition: str, seed: int) -> Path:
    folder, pattern = CONDITIONS[condition]
    return ROOT / "ultimate_experiment" / folder / pattern.format(seed=seed)


def notebook_seed(path: Path) -> int:
    nb = json.loads(path.read_text())
    seed = nb.get("metadata", {}).get("rgppo", {}).get("seed")
    if seed is None and "phase7" in nb.get("metadata", {}):
        seed = nb["metadata"]["phase7"].get("seed")
    if seed is None:
        stem = path.stem
        seed = int(stem.rsplit("_seed_", 1)[1])
    return int(seed)


def launch_one(
    condition: str,
    seed: int,
    results_root: Path,
    timeout_s: int,
    force: bool,
) -> dict:
    path = notebook_path(condition, seed)
    if not path.exists():
        return {"condition": condition, "seed": seed, "status": "MISSING_NOTEBOOK", "path": str(path)}
    actual_seed = notebook_seed(path)
    if actual_seed != seed:
        return {
            "condition": condition,
            "seed": seed,
            "status": "SEED_MISMATCH",
            "path": str(path),
            "notebook_seed": actual_seed,
        }
    state_file = ROOT / "Results" / "Tail_GAE" / condition / "pilot_status" / f"seed_{seed}.json"
    if force and state_file.exists():
        backup = state_file.with_name(
            state_file.name + ".bak." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        state_file.rename(backup)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0",
            "OMP_NUM_THREADS": "12",
            "MKL_NUM_THREADS": "12",
            "RGPPO_RESULTS_ROOT": str(results_root),
            "RGPPO_ROOT": str(ROOT),
        }
    )
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if condition == "0101":
        env["RGPPO_NOTEBOOK"] = str(path)
    log_path = ROOT / "logs" / f"{condition}_seed_{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[launcher] condition={condition} seed={seed}\n")
        log.write(f"[launcher] notebook={path}\n")
        log.write(f"[launcher] results_root={results_root}\n")
        log.write(f"[launcher] PYTHONHASHSEED={seed}\n")
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, str(EXEC), str(path)],
            cwd=str(path.parent),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
            status = "OK" if returncode == 0 else f"FAILED(rc={returncode})"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            status = "TIMEOUT"
    return {
        "condition": condition,
        "seed": seed,
        "status": status,
        "returncode": returncode if status != "TIMEOUT" else None,
        "seconds": round(time.time() - started, 1),
        "notebook": str(path),
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--jobs", type=int, default=10)
    parser.add_argument("--results-root", type=Path, default=ROOT / "Results")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--force", action="store_true", help="move a completed Phase-9 claim aside")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conditions = (
        list(CONDITIONS)
        if args.conditions == "all"
        else [c.strip() for c in args.conditions.split(",") if c.strip()]
    )
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}; valid={list(CONDITIONS)}")
    seeds = parse_seeds(args.seeds)
    jobs = [
        (condition, seed)
        for condition in conditions
        for seed in seeds
    ]
    if args.dry_run:
        for condition, seed in jobs:
            print(condition, seed, notebook_path(condition, seed))
        return 0

    args.results_root.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[launcher] {len(jobs)} jobs, {args.jobs} concurrent, results={args.results_root}", flush=True)
    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                launch_one,
                condition,
                seed,
                args.results_root,
                args.timeout,
                args.force,
            ): (condition, seed)
            for condition, seed in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] {result['condition']} seed {result['seed']} "
                f"({result.get('seconds', '?')}s)",
                flush=True,
            )
    results.sort(key=lambda r: (r["condition"], r["seed"]))
    summary = {
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "results_root": str(args.results_root),
        "jobs": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = ROOT / "logs" / f"run_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    failed = [r for r in results if r["status"] != "OK"]
    print(f"[launcher] done in {summary['elapsed_seconds']}s; failures={len(failed)}; summary={summary_path}")
    for result in failed:
        print(f"  FAILED {result['condition']} seed {result['seed']}: {result['status']} log={result.get('log')}")
        log_path = Path(result.get("log", ""))
        if log_path.exists():
            tail = log_path.read_text(errors="replace").splitlines()[-12:]
            for line in tail:
                print("    " + line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
