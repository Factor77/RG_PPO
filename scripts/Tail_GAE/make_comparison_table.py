#!/usr/bin/env python3
"""Build the human-readable archive-vs-reproduction comparison tables.

Reads results_summary/{archive,reproduction}_per_seed.csv and the provenance
reports, and writes results_summary/comparison_table.md plus a long-format
comparison_long.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results_summary"
PROVENANCE = ROOT / "provenance"
CONDITIONS = ["1250", "1270b", "1321b_ctrl", "1321b_lo", "1321b_hi"]
METRICS = ["CR", "AR", "AV", "SR", "MDD"]


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def fmt_mean_var(mean: float, var: float, digits: int = 6) -> str:
    return f"{mean:.{digits}f} ({var:.{digits}f})"


def aggregate_section() -> str:
    archive = pd.read_csv(SUMMARY / "archive_aggregate.csv")
    repro = pd.read_csv(SUMMARY / "reproduction_aggregate.csv")
    lines = [
        "## Ten-seed aggregate",
        "",
        "Mean with sample variance (ddof = 1) over seeds 0-9. "
        "Archive = 4080 SUPER runs recorded in `Results_Analysis.md`; "
        "Repro = 2080 Ti re-run in this repository.",
        "",
        "| Condition | Metric | Archive mean (var) | Repro mean (var) | Diff (repro - archive) |",
        "|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        a = archive[archive.condition == condition].iloc[0]
        b = repro[repro.condition == condition].iloc[0]
        for metric in METRICS:
            am, av = float(a[f"{metric}_mean"]), float(a[f"{metric}_var"])
            bm, bv = float(b[f"{metric}_mean"]), float(b[f"{metric}_var"])
            lines.append(
                f"| {condition} | {metric} | {fmt_mean_var(am, av)} | "
                f"{fmt_mean_var(bm, bv)} | {fmt(bm - am)} |"
            )
    return "\n".join(lines)


def per_seed_section() -> str:
    archive = pd.read_csv(SUMMARY / "archive_per_seed.csv")
    repro = pd.read_csv(SUMMARY / "reproduction_per_seed.csv")
    merged = archive.merge(repro, on=["condition", "seed"], suffixes=("_a", "_r"))
    blocks = ["## Per-seed comparison", ""]
    for condition in CONDITIONS:
        part = merged[merged.condition == condition].sort_values("seed")
        blocks.append(f"### {condition}")
        blocks.append("")
        blocks.append(
            "| seed | CR arc | CR rep | dCR | MDD arc | MDD rep | dMDD | "
            "SR arc | SR rep | dSR | AV arc | AV rep | dAV |"
        )
        blocks.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in part.iterrows():
            blocks.append(
                "| {seed} | {cra} | {crr} | {dcr} | {mda} | {mdr} | {dmdd} | "
                "{sra} | {srr} | {dsr} | {ava} | {avr} | {dav} |".format(
                    seed=int(row.seed),
                    cra=fmt(row.CR_a), crr=fmt(row.CR_r), dcr=fmt(row.CR_r - row.CR_a),
                    mda=fmt(row.MDD_a), mdr=fmt(row.MDD_r), dmdd=fmt(row.MDD_r - row.MDD_a),
                    sra=fmt(row.SR_a), srr=fmt(row.SR_r), dsr=fmt(row.SR_r - row.SR_a),
                    ava=fmt(row.AV_a), avr=fmt(row.AV_r), dav=fmt(row.AV_r - row.AV_a),
                )
            )
        blocks.append("")
    return "\n".join(blocks)


def section_0101() -> str:
    archive_path = SUMMARY / "0101_archive_per_seed.csv"
    repro_path = SUMMARY / "0101_reproduction_per_seed.csv"
    if not (archive_path.exists() and repro_path.exists()):
        return ""
    archive = pd.read_csv(archive_path)
    repro = pd.read_csv(repro_path)
    merged = archive.merge(repro, on=["condition", "seed"], suffixes=("_a", "_r"))
    lines = ["## 0101 (VSE) comparison", ""]
    agg_a = SUMMARY / "0101_archive_aggregate.csv"
    agg_r = SUMMARY / "0101_reproduction_aggregate.csv"
    if agg_a.exists() and agg_r.exists():
        a = pd.read_csv(agg_a).iloc[0]
        b = pd.read_csv(agg_r).iloc[0]
        lines += [
            "| Metric | Archive mean (var) | Repro mean (var) | Diff |",
            "|---|---|---|---|",
        ]
        for metric in METRICS:
            am, av = float(a[f"{metric}_mean"]), float(a[f"{metric}_var"])
            bm, bv = float(b[f"{metric}_mean"]), float(b[f"{metric}_var"])
            lines.append(
                f"| {metric} | {fmt_mean_var(am, av)} | {fmt_mean_var(bm, bv)} | {fmt(bm - am)} |"
            )
        lines.append("")
    lines += [
        "| seed | CR arc | CR rep | dCR | MDD arc | MDD rep | dMDD | "
        "SR arc | SR rep | dSR | AV arc | AV rep | dAV |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in merged.sort_values("seed").iterrows():
        lines.append(
            "| {seed} | {cra} | {crr} | {dcr} | {mda} | {mdr} | {dmdd} | "
            "{sra} | {srr} | {dsr} | {ava} | {avr} | {dav} |".format(
                seed=int(row.seed),
                cra=fmt(row.CR_a), crr=fmt(row.CR_r), dcr=fmt(row.CR_r - row.CR_a),
                mda=fmt(row.MDD_a), mdr=fmt(row.MDD_r), dmdd=fmt(row.MDD_r - row.MDD_a),
                sra=fmt(row.SR_a), srr=fmt(row.SR_r), dsr=fmt(row.SR_r - row.SR_a),
                ava=fmt(row.AV_a), avr=fmt(row.AV_r), dav=fmt(row.AV_r - row.AV_a),
            )
        )
    lines.append("")
    lines.append(
        "0101 reports annual return equal to cumulative return; AR is omitted "
        "from the per-seed table."
    )
    return "\n".join(lines)


def gate_section() -> str:
    lines = [
        "## Seed-0 gate: archive vs re-run",
        "",
        "Full five-metric comparison for seed 0; the paired same-GPU test is the "
        "code-drift check.",
        "",
        "| Condition | CR arc | CR rep | MDD arc | MDD rep | SR arc | SR rep | "
        "AV arc | AV rep | Account hash |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        path = PROVENANCE / "gate1_seed0" / f"{condition}.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        m = report["metrics"]
        lines.append(
            "| {c} | {cra} | {crr} | {mda} | {mdr} | {sra} | {srr} | {ava} | {avr} | {h} |".format(
                c=condition,
                cra=fmt(m["total_return"]["old"]), crr=fmt(m["total_return"]["new"]),
                mda=fmt(m["max_drawdown"]["old"]), mdr=fmt(m["max_drawdown"]["new"]),
                sra=fmt(m["sharpe"]["old"]), srr=fmt(m["sharpe"]["new"]),
                ava=fmt(m["annualized_volatility"]["old"]),
                avr=fmt(m["annualized_volatility"]["new"]),
                h="same" if m["account_value_hash"]["pass"] else "different",
            )
        )
    return "\n".join(lines)


def paired_section() -> str:
    lines = [
        "## Paired same-GPU test (2080 Ti)",
        "",
        "Old code = the archived commit run on this host; New code = this "
        "repository. Identical hashes mean the release code is behavior-identical.",
        "",
        "| Pair | CR old | CR new | MDD old | MDD new | Account hash |",
        "|---|---|---|---|---|---|",
    ]
    pairs = [
        ("1250: d664ca9 vs release", "1250_old_vs_new.json"),
        ("1270b: 6d772e9 vs release", "1270b_old_vs_new.json"),
        ("1321b_hi: release vs release", "1321b_hi_determinism.json"),
    ]
    for label, filename in pairs:
        path = PROVENANCE / "paired" / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        m = report["metrics"]
        lines.append(
            "| {label} | {cra} | {crr} | {mda} | {mdr} | {h} |".format(
                label=label,
                cra=fmt(m["total_return"]["old"]), crr=fmt(m["total_return"]["new"]),
                mda=fmt(m["max_drawdown"]["old"]), mdr=fmt(m["max_drawdown"]["new"]),
                h="same" if m["account_value_hash"]["pass"] else "different",
            )
        )
    return "\n".join(lines)


def main() -> int:
    archive = pd.read_csv(SUMMARY / "archive_per_seed.csv")
    repro = pd.read_csv(SUMMARY / "reproduction_per_seed.csv")
    merged = archive.merge(repro, on=["condition", "seed"], suffixes=("_a", "_r"))
    rows = []
    for condition in CONDITIONS:
        for seed in range(10):
            row = merged[(merged.condition == condition) & (merged.seed == seed)].iloc[0]
            for metric in METRICS:
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "metric": metric,
                        "archive": float(row[f"{metric}_a"]),
                        "repro": float(row[f"{metric}_r"]),
                        "diff": float(row[f"{metric}_r"] - row[f"{metric}_a"]),
                    }
                )
    archive_0101 = SUMMARY / "0101_archive_per_seed.csv"
    repro_0101 = SUMMARY / "0101_reproduction_per_seed.csv"
    if archive_0101.exists() and repro_0101.exists():
        a = pd.read_csv(archive_0101)
        r = pd.read_csv(repro_0101)
        m = a.merge(r, on=["condition", "seed"], suffixes=("_a", "_r"))
        for _, row in m.iterrows():
            for metric in METRICS:
                rows.append(
                    {
                        "condition": "0101",
                        "seed": int(row.seed),
                        "metric": metric,
                        "archive": float(row[f"{metric}_a"]),
                        "repro": float(row[f"{metric}_r"]),
                        "diff": float(row[f"{metric}_r"] - row[f"{metric}_a"]),
                    }
                )
    pd.DataFrame(rows).to_csv(SUMMARY / "comparison_long.csv", index=False)

    header = [
        "# Archive vs reproduction",
        "",
        "Archive: 4080 SUPER runs recorded in the internal experiment log "
        "(full-precision values from the archived `*_metrics.json`).",
        "Repro: 2080 Ti ten-seed sweep in this repository.",
        "The two hosts differ; see `PROVENANCE.md` for the paired same-GPU proof "
        "that the code is equivalent.",
        "",
    ]
    body = "\n\n".join(
        [
            aggregate_section(),
            per_seed_section(),
            section_0101(),
            gate_section(),
            paired_section(),
        ]
    )
    (SUMMARY / "comparison_table.md").write_text("\n".join(header) + body + "\n")
    print(f"wrote {SUMMARY / 'comparison_table.md'}")
    print(f"wrote {SUMMARY / 'comparison_long.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
