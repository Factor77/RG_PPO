# Risk-Guided PPO: code and reproduction package

This repository contains the code, data, frozen event calendar, and ten-seed
notebooks for the experiments in *Risk-Guided PPO for Reinforcement-Learning-
Based Trading under Persistent Market Declines* (manuscript in preparation).

The package covers six conditions: the AUST validation-selected ensemble
(VSE), the return-only single-agent baseline, the downside-penalty variant,
and three recovery-dose variants of the proposed method.

## Conditions

| Paper name | Experiment ID | Observation | Auxiliary terms |
|---|---|---|---|
| VSE (validation-selected ensemble) | `0101` | 301 | none (A2C/PPO/DDPG ensemble) |
| Return-PPO | `1250` | 301 | none |
| DP-PPO | `1270b` | 301 | downside penalty (beta = 0.03), stress latch |
| RG-PPO, beta_s = 0 | `1321b_ctrl` | 304 | downside penalty + recovery term off |
| RG-PPO, beta_s = 0.3 | `1321b_lo` | 304 | downside penalty + recovery term, low dose |
| RG-PPO, beta_s = 1.0 | `1321b_hi` | 304 | downside penalty + recovery term, high dose |

`1321b_hi` is the RG-PPO entry in the main comparison. The dose conditions
differ only in the recovery coefficient `beta_s`. `0101` is the original
AUST/FinRL ensemble benchmark; its code anchor is the last committed 0101
notebook (`ac2b3c9`), not an earlier ensemble variant.

## Repository layout

```
data/
  new_version_2026/data_30_20100101_20260601.csv   30 DJIA stocks, 2010-2026
  new_version_2026/DOW30_source.txt                stock-universe provenance
  DJI_2013_2026.csv                                DJIA benchmark (single-agent)
  DJI_2025_2026.csv                                benchmark used by 0101
  DJI_source.txt                                   benchmark provenance
  phase9/gated_rf20_calendar.csv                   frozen Phase 9 calendar
scripts/Tail_GAE/
  src/                                             single-agent training framework
  exec_notebook.py                                 kernel launcher (one notebook)
  run_parallel.py                                  seed sweep launcher (N notebooks)
  prepare_notebooks.py                             regenerate the single-agent seeds
  compare_runs.py                                  per-seed comparison with archive
  summarize_results.py                             ten-seed aggregation
  make_comparison_table.py                         combined comparison tables
scripts/
  prepare_0101_notebooks.py                        regenerate the VSE seeds
  summarize_0101.py                                VSE ten-seed aggregation
scripts/ensemble/                                  AUST ensemble audit utilities
ultimate_experiment/
  0101 AUST_ensemble + original/                   VSE, ten seeds
  1250 Tail_GAE/                                   ten seed notebooks
  1270b Tail_GAE/
  1321b Tail_GAE/                                  ctrl / lo / hi, ten seeds each
provenance/                                        run-to-commit audit trail
results_summary/                                   archive and reproduction tables
```

Each `(condition, seed)` pair is one notebook. The notebook is the experiment
definition: it sets the seed, loads the data, builds the runner, trains the
rolling windows, and writes the account-value and metric artifacts. The
notebooks are committed with the outputs of the verification run, so the
reported values can be inspected without rerunning.

## Environment

The archived runs used Python 3.10.18, PyTorch 2.9.1 (CUDA 12.8),
Stable-Baselines3 2.8.0a2, Gymnasium 1.2.2, NumPy 2.2.6, pandas 2.2.3, and
FinRL 0.3.8 on an NVIDIA RTX 4080 SUPER. The exact frozen environment is in
`provenance/pip_freeze.txt`.

A clean setup:

```bash
conda create -n rgppo python=3.10 -y
conda activate rgppo
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install --pre -r requirements.txt
python -m ipykernel install --user \
  --name finrl-aust-exploratory --display-name "FinRL AUST Exploratory"
```

`stable-baselines3==2.8.0a2` is a pre-release, hence `--pre`. The notebooks
declare the kernel `finrl-aust-exploratory`; the launcher reads that name, so
the `ipykernel install` step is required unless you change the notebook
metadata. The archived host used the Python interpreter of the FinRL
environment; `python` below means that interpreter, with the packages above.

## Data

All inputs are included; no download or preprocessing step is required.

- `data/new_version_2026/data_30_20100101_20260601.csv`: 30 DJIA stocks,
  2010-01-01 to 2026-06-01 (about 12 MB).
- `data/DJI_2013_2026.csv` and `data/DJI_2025_2026.csv`: DJIA benchmarks used
  by the single-agent and VSE conditions.
- `data/phase9/gated_rf20_calendar.csv`: frozen Phase 9 event calendar (about
  1.4 MB), SHA-256
  `061868855941cf013c5e976ce82c6cbd01598e5e7e8ed3995f71146047b269dc`.
  The notebook asserts this hash before training.

Training uses 2013-04-01 to 2024-12-31; validation and trading use
2025-01-01 to 2026-06-01, with 63-day rebalance and validation windows.

## Quick start: one seed

The launcher is the recommended entry point because it pins the seed before
the Python interpreter starts:

```bash
# One single-agent seed (RG-PPO high dose, seed 0)
python scripts/Tail_GAE/run_parallel.py \
  --conditions 1321b_hi --seeds 0 --jobs 1 \
  --results-root ./Results
```

Output: `./Results/Tail_GAE/1321b_hi/seed_0/<timestamp>/`, with
`audit/results/*_metrics.json` and the account-value series. The launcher log
is `logs/1321b_hi_seed_0.log`.

For the VSE benchmark:

```bash
python scripts/Tail_GAE/run_parallel.py \
  --conditions 0101 --seeds 0 --jobs 1 \
  --results-root ./Results
```

Output: `./Results/01 AUST/0101 AUST_data2026/seed_0/<timestamp>/`.

## Full reproduction: ten seeds

Single-agent conditions:

```bash
python scripts/Tail_GAE/run_parallel.py \
  --conditions 1250,1270b,1321b_ctrl,1321b_lo,1321b_hi \
  --seeds 0-9 --jobs 5 \
  --results-root ./Results
```

VSE benchmark:

```bash
python scripts/Tail_GAE/run_parallel.py \
  --conditions 0101 --seeds 0-9 --jobs 5 \
  --results-root ./Results
```

Use `--jobs 10` on a 16 GB or larger GPU. On the 11 GB RTX 2080 Ti
verification host, `--jobs 5` saturates the GPU without memory pressure.
`--conditions all` also includes `0101`; the explicit lists above keep the two
sweeps separate.

`--results-root` is optional; the default is `./Results`. Raw run directories
are about 5-6 GB for all sixty runs, so a scratch path is recommended. The
package itself is about 61 MB.

### Runtime on the verification host

| Sweep | Runs | Jobs | GPU | Wall time |
|---|---|---|---|---|
| Single-agent, five conditions | 50 | 5 | RTX 2080 Ti 11 GB | about 3 hours |
| VSE (`0101`) | 10 | 5 | RTX 2080 Ti 11 GB | about 1.8 hours |

A single single-agent seed takes about 20 minutes; a single VSE seed takes
about 33-39 minutes. Five concurrent single-agent runs use about 1.6 GB of GPU
memory; five concurrent VSE runs use about 1.3-1.6 GB.

## Launch contract and manual runs

The notebooks are the experiment definitions; the launcher does not modify
them. For every job it starts a fresh kernel and injects:

- `PYTHONHASHSEED=<seed>`
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `NVIDIA_TF32_OVERRIDE=0`,
  `OMP_NUM_THREADS=12`, `MKL_NUM_THREADS=12`
- `RGPPO_RESULTS_ROOT=<results root>`
- `RGPPO_ROOT=<repository>` and `PYTHONPATH=<repository>`
- for `0101` only: `RGPPO_NOTEBOOK=<notebook path>`

The path and launch adaptations are limited to:

- `RGPPO_RESULTS_ROOT` redirects outputs (default: `<repository>/Results`).
- `RGPPO_ROOT` and `PYTHONPATH` locate the repository root for data and script
  imports. The notebooks resolve the root from `RGPPO_ROOT`, or by walking up
  to the directory containing `data/new_version_2026`.
- `RGPPO_NOTEBOOK` is the notebook path that the ensemble audit archives
  (`0101` only).

In the VSE notebook the early `check_and_make_directories` call is dropped
because `audit_ensemble_run` creates and isolates the FinRL output directories
inside each run directory.

If you run a notebook manually in Jupyter, start the kernel with
`PYTHONHASHSEED=<seed>`; otherwise cell 0 prints a warning and bitwise
reproducibility is not guaranteed. Set `RGPPO_ROOT` and `PYTHONPATH` before
starting the kernel, or run from the repository root so the notebooks can
locate `data/new_version_2026`. For `0101`, also set `RGPPO_NOTEBOOK`.

The archive contains one launch-mode caveat: the `1270b` batch was started
from the JupyterLab UI, so its `PYTHONHASHSEED` was not pinned. The launcher
fixes this. See `PROVENANCE.md` before attributing any difference to code.

## Results and verification

The paper's table is the canonical result from the RTX 4080 SUPER. The
verification host is an RTX 2080 Ti; deterministic training is repeatable on
the same hardware, not bitwise across GPU architectures. On the same GPU, the
release code reproduced the archived code commits bitwise (1250, 1270b) and
the 1321b_hi run reproduced itself bitwise; see `provenance/paired/`.

Ten-seed results on the verification host:

| Condition | CR (repro) | CR (paper) | MDD (repro) | MDD (paper) |
|---|---|---|---|---|
| 1250 | 0.2681 | 0.2437 | -0.1105 | -0.0982 |
| 1270b | 0.1354 | 0.1546 | -0.0711 | -0.0605 |
| 1321b_ctrl | 0.1393 | 0.1208 | -0.0461 | -0.0419 |
| 1321b_lo | 0.1896 | 0.1811 | -0.0585 | -0.0721 |
| 1321b_hi | 0.2192 | 0.2234 | -0.0830 | -0.0755 |
| 0101 (VSE) | 0.1322 | 0.1268 | -0.1042 | -0.1086 |

The dose ordering (CR rises and MDD deepens with `beta_s`) and the RG-PPO
versus Return-PPO trade-off (lower return, shallower drawdown) reproduce. The
VSE aggregate is also close; its per-seed validation selection changes with
the hardware/launch environment even when the aggregate is stable. The full
five-metric and per-seed tables are in `results_summary/comparison_table.md`
and `results_summary/comparison_long.csv`.

### Regenerate the summary tables

After a sweep, aggregate the runs and rebuild the comparison tables:

```bash
# single-agent ten-seed aggregate
python scripts/Tail_GAE/summarize_results.py \
  --new-root ./Results --seeds 0-9 --out results_summary

# VSE ten-seed aggregate
python scripts/summarize_0101.py \
  --new-root "./Results/01 AUST/0101 AUST_data2026" \
  --out results_summary

# combined tables (archive CSVs included + new reproduction CSVs)
python scripts/Tail_GAE/make_comparison_table.py
```

The archive CSVs in `results_summary/` are included, so the comparison tables
can be rebuilt without the original raw runs. To compare against the original
raw archive, pass `--old-root`; this is only needed when maintaining the
archive. For the single-agent script it is the archived `Results` root; for
the VSE script it is the archived `01 AUST/0101 AUST_data2026` directory.
If you maintain the original raw archive (not included in this package), the
exact per-seed comparator is:

```bash
python scripts/Tail_GAE/compare_runs.py \
  --condition 1321b_hi --seed 0 \
  --old-root /path/to/archive/Results --new-root ./Results
```

It compares the account-value hash, the account-value series, the observation
contract, the experiment configuration, and the run status against the
archived run. Without the archive, use the included `results_summary/`
references instead.

## Regenerating the notebooks

The seed_0 notebook is the template. Edit it, then regenerate the ten seeds:

```bash
python scripts/Tail_GAE/prepare_notebooks.py
python scripts/prepare_0101_notebooks.py
```

The generators only change the seed constant, path-injection lines, kernel
metadata, and outputs. Manifests with SHA-256 hashes are written to
`provenance/notebook_manifest.tsv` and `provenance/notebook_manifest_0101.tsv`.

## Tests

```bash
python -m pytest -q scripts/Tail_GAE/tests
```

On the archived environment this reports 193 passed and 1 skipped. The tests
cover the single-agent framework; the VSE path is validated by the seed-0 and
ten-seed runs.

## Reproducibility checklist

| Item | Status |
|---|---|
| Training code | included |
| Evaluation code | included |
| Dependency specification | `requirements.txt`, `provenance/pip_freeze.txt` |
| Data | included |
| Frozen event calendar | included with SHA-256 |
| Expected results | `results_summary/` and the table above |
| Hardware / runtime | documented above |
| Known limitations | GPU-architecture caveat and launch-mode note below |

## Known limitations

1. Exact numerical reproduction of the paper's table requires an RTX 4080
   SUPER (or a GPU that produces the same kernels). Different GPU
   architectures can change low-order bits, which can change the selected
   policy over 4 x 40,000 training steps.
2. The archived `1270b` batch was UI-launched without a pinned
   `PYTHONHASHSEED`; the release launcher pins it.
3. The VSE condition selects one of A2C/PPO/DDPG per validation window; its
   per-seed selection is sensitive to the hardware and launch environment,
   although the ten-seed aggregate is stable.
4. The package includes the frozen data and calendar used in the paper but not
   the archived raw run directories; the aggregated reference values are in
   `results_summary/`.

## Citation

The manuscript is in preparation; citation details will be added upon publication.

## License and acknowledgements

MIT; see `LICENSE`. The AUST ensemble backbone and parts of the training and
evaluation utilities are adapted from FinRL/AUST (Yang et al., 2020), also
released under the MIT license; the upstream copyright and permission notice
are reproduced in `NOTICE`. The single-agent framework uses Stable-Baselines3.

For questions about the code, please open an issue.
