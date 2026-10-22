# Results summary

`scripts/Tail_GAE/summarize_results.py` writes the single-agent files:

| File | Content |
|---|---|
| `archive_per_seed.csv` | Archived 4080 SUPER metrics for every `(condition, seed)` |
| `reproduction_per_seed.csv` | Re-run metrics for every `(condition, seed)` |
| `archive_aggregate.csv` | Ten-seed mean and sample variance (ddof = 1) of the archive |
| `reproduction_aggregate.csv` | Ten-seed mean and sample variance of the re-run |
| `aggregate_comparison.csv` | Side-by-side aggregate values and differences |
| `reproduction_comparison.json` | Per-seed and aggregate pass/fail report |

`scripts/summarize_0101.py` writes the VSE files:

| File | Content |
|---|---|
| `0101_archive_per_seed.csv` | Archived 0101 metrics per seed |
| `0101_reproduction_per_seed.csv` | Re-run 0101 metrics per seed |
| `0101_archive_aggregate.csv` | Archived 0101 ten-seed mean and variance |
| `0101_reproduction_aggregate.csv` | Re-run 0101 ten-seed mean and variance |
| `0101_reproduction_comparison.json` | Per-seed comparison |

`scripts/Tail_GAE/make_comparison_table.py` writes:

| File | Content |
|---|---|
| `comparison_table.md` | Aggregate, per-seed, 0101, seed-0 gate, and paired-test tables |
| `comparison_long.csv` | Long format: `condition, seed, metric, archive, repro, diff` |

The metrics are cumulative return (CR), annual return (AR), annual volatility
(AV), Sharpe ratio (SR), and maximum drawdown (MDD, signed negative). AR is
annualised from CR with `(1 + CR) ** (252 / (rows - 1)) - 1`; for 0101 the
archive reports AR = CR, and the 0101 table follows that convention.

The per-seed CSVs (`archive_per_seed.csv`, `reproduction_per_seed.csv`,
`0101_archive_per_seed.csv`, `0101_reproduction_per_seed.csv`) omit internal
absolute paths. They contain only `condition`, `seed`, the five metrics,
`account_value_hash`, and `rows`; run directories and metrics-file locations
are not published. The comparison JSONs likewise record the roots as
`<archive>` and `<results_root>` instead of host paths.

The archived numbers were produced on an RTX 4080 SUPER. The `archive_*` and
`0101_archive_*` CSVs are included as the reference. Run `summarize_results.py`
and `summarize_0101.py` with `--new-root` only to write the reproduction side;
pass `--old-root` only when maintaining the original raw archive.

The verification host is an RTX 2080 Ti; see `PROVENANCE.md` for the hardware
caveat and the paired same-GPU code-drift test.
