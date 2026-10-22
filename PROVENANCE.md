# Provenance

This file maps every reported condition to the original run artifacts, the
code snapshot that produced them, and the launch mode.  It is the audit trail
for the reproduction numbers; `README.md` is the user-facing guide.

## Conditions

| Paper name | Experiment ID | Role | Run batch (2026) | Code anchor |
|---|---|---|---|---|
| Return-PPO | `1250` | Single-agent return-only control (301-dim observation) | 08-20 16:54:37 – 17:12:44 | `ec6f2f8` (last pre-run commit); notebook first committed in `d664ca9` |
| DP-PPO | `1270b` | Downside-penalty PPO with the stress latch (301-dim) | 08-28 16:53:18 – 16:53:41 | `6d772e9` (first commit after the run) |
| RG-PPO, beta_s = 0 | `1321b_ctrl` | Recovery-term control (304-dim) | 09-13 19:40:06 – 19:40:22 | `f0d527b` (HEAD at release) |
| RG-PPO, beta_s = 0.3 | `1321b_lo` | Recovery-term low dose (304-dim) | 09-13 20:00:07 – 20:00:46 | `f0d527b` |
| RG-PPO, beta_s = 1.0 | `1321b_hi` | Recovery-term high dose, main RG-PPO result (304-dim) | 09-13 19:29:57 (seed 0); 20:19:51 – 20:20:47 (seeds 1–9) | `f0d527b` |
| VSE | `0101` | AUST validation-selected ensemble (301-dim) | 07-19 17:47:44 – 18:43:06 | `ac2b3c9` (last committed 0101 notebook; run-time copy matches) |

For `1321b_hi` seed 0 the run directory
`20260913_190304_superceded_nosnapshot` is a superseded attempt and is
excluded; the reported seed 0 is `20260913_192957`.

The original repository is the internal repository (absolute path omitted).
The archived run directories are
`Results/Tail_GAE/<experiment_id>/seed_<k>/<timestamp>/`; each contains a copy
of the executed notebook, `experiment_config.json`, `observation_contract.json`,
data-snapshot hashes, RNG states, and `audit/results/*_metrics.json`.

For `0101` the archived root is
`Results/01 AUST/0101 AUST_data2026/seed_<k>/<timestamp>/` (the notebook wrote
to `Results/AUST/AUST_data2026`; the folder was renamed after the run). The
run-time notebook copy
`seed_0/20260719_180055/audit/AUST_data2026_seed_1.ipynb` matches the current
`AUST_seed_1.ipynb` byte-for-byte; the ten seed notebooks differ only in the
seed constant. `0101` reports annual return equal to cumulative return, so the
table above uses AR = CR for this row.

## Reported metrics (ten seeds, mean with sample variance in parentheses)

Source: the internal experiment log, lines 113, 128, and 135–137.

| Condition | CR | AR | AV | SR | MDD |
|---|---|---|---|---|---|
| 1250 | 0.2437 (0.012435) | 0.2448 (0.012557) | 0.1367 (0.000493) | 1.6684 (0.479001) | -0.0982 (0.000752) |
| 1270b | 0.1546 (0.009042) | 0.1553 (0.009125) | 0.1115 (0.000065) | 1.3398 (0.621680) | -0.0605 (0.000197) |
| 1321b_ctrl | 0.1208 (0.001461) | 0.1213 (0.001474) | 0.0822 (0.000164) | 1.4554 (0.238655) | -0.0419 (0.000193) |
| 1321b_lo | 0.1811 (0.005999) | 0.1819 (0.006055) | 0.1127 (0.000115) | 1.5661 (0.460435) | -0.0721 (0.000454) |
| 1321b_hi | 0.2234 (0.003328) | 0.2243 (0.003360) | 0.1181 (0.000116) | 1.8056 (0.294772) | -0.0755 (0.000395) |
| 0101 | 0.1268 (0.004025) | 0.1268 (0.004025) | 0.1516 (0.000375) | 0.8701 (0.154898) | -0.1086 (0.000837) |

CR, cumulative return; AR, annual return; AV, annual volatility; SR, Sharpe
ratio; MDD, maximum drawdown (signed negative).

## Launch-mode caveats

- `1250`: the archived notebook output contains no
  `PYTHONHASHSEED was not set` warning, so the kernel was launched with the
  per-seed hash seed pinned.
- `1270b`: the archived notebook output contains that warning. The run was
  started from the JupyterLab UI; the interpreter-level hash seed was not
  pinned. The cell-level CUDA determinism settings still applied. A re-run
  through `exec_notebook.py` pins the hash seed. Any difference between the
  archived `1270b` numbers and a re-run must therefore first be attributed to
  this launch difference before claiming a code difference.
- `1321b_ctrl`, `1321b_lo`, `1321b_hi`: launched through `exec_notebook.py`
  with `metadata.phase7.seed`; both the hash seed and the CUDA environment were
  injected before the kernel started. No warning is present in the archived
  notebooks.
- `0101`: the archived `runtime.json` records `PYTHONHASHSEED=0` but
  `CUBLAS_WORKSPACE_CONFIG=null` and `CUDA_VISIBLE_DEVICES=null`; the run was
  launched from the repository root. The reproduction launcher pins all three.
  The ensemble's per-seed validation selection is sensitive to this difference.

## Code state in this repository

`scripts/Tail_GAE/src/` is copied from original commit `f0d527b` (the state
that produced the 1321b runs). The only modifications are path-only:

1. `notebook_config.py`: the output root honours
   `RGPPO_RESULTS_ROOT` when set; otherwise it is `<repo>/Results`.
2. The notebooks are rewritten by `scripts/Tail_GAE/prepare_notebooks.py`:
   the original absolute project root is replaced with this repository, the
   Phase 9 calendar is read from `data/phase9/gated_rf20_calendar.csv`, and
   `PHASE9_OUTPUT_ROOT` honours `RGPPO_RESULTS_ROOT`.

No training, environment, loss, or normalization code was changed.

## Data and calendar

- `data/new_version_2026/data_30_20100101_20260601.csv`: 30 DJIA stocks,
  2010-01-01 to 2026-06-01.
- `data/DJI_2013_2026.csv`: DJIA benchmark series.
- `data/phase9/gated_rf20_calendar.csv`: frozen Phase 9 event calendar,
  SHA-256
  `061868855941cf013c5e976ce82c6cbd01598e5e7e8ed3995f71146047b269dc`.
  It was previously stored in an internal directory outside the code
  repository; it is copied here unchanged and is covered by the manifest in
  `data/phase9/sha256_manifest.json`.

## Verification status

### Seed-0 gate against the archived runs

All five conditions were re-run on the 2080 Ti with the release code. Every
run completed, but none reproduced the archived 4080 SUPER account-value hash:

| Condition | archived CR | new CR | CR diff | archived MDD | new MDD |
|---|---|---|---|---|---|
| 1250 | 0.146465 | 0.171203 | +0.024738 | -0.076867 | -0.102858 |
| 1270b | 0.136614 | 0.089990 | -0.046624 | -0.059863 | -0.069363 |
| 1321b_ctrl | 0.146567 | 0.098002 | -0.048565 | -0.048920 | -0.035750 |
| 1321b_lo | 0.165835 | 0.174912 | +0.009076 | -0.058236 | -0.046177 |
| 1321b_hi | 0.198801 | 0.185248 | -0.013553 | -0.068932 | -0.080418 |

The full reports are in `provenance/gate1_seed0/<condition>.json`. The
differences are larger than low-order-bit noise, but the paired same-GPU test
below shows that they are hardware-induced: the release code and the archived
code produce identical account-value series on the same GPU.

### Paired same-GPU code-drift test (completed)

The archived 4080 SUPER is not available on the verification host, so the
code-drift question was answered on the same 2080 Ti:

| Pair | Account-value hash | CR (both sides) | MDD (both sides) |
|---|---|---|---|
| `1250`: commit `d664ca9` vs release | identical | 0.171203 | -0.102858 |
| `1270b`: commit `6d772e9` vs release | identical | 0.089990 | -0.069363 |
| `1321b_hi`: release run twice | identical | 0.185248 | -0.080418 |

Reports are in `provenance/paired/`. Every metric difference is exactly zero.
The release code is therefore behavior-identical to the archived code on this
host, and the host is deterministic across repeated runs.

The `rng_state_after_training.json` hash differs even when the account-value
series is identical (the snapshot includes state that is not part of the
training path). The authoritative check is the account-value hash, not the RNG
snapshot hash.

Conclusion: the seed-0 differences above are **hardware-induced**
(4080 SUPER vs 2080 Ti), not code drift.

### 0101 (VSE) seed-0 check

The archived 0101 seed 0 (4080 SUPER) reports CR 0.037965, AV 0.191243,
SR 0.290457, MDD -0.158655. The 2080 Ti re-run reports CR 0.126854,
AV 0.151494, SR 0.866982, MDD -0.130221. The new value is close to the archive
ten-seed mean (CR 0.126837, AV 0.151593, SR 0.870147, MDD -0.108611), so the
ensemble's per-seed validation selection is sensitive to the GPU/launch
environment even when the aggregate is stable.

The full ten-seed aggregate on the 2080 Ti is CR 0.132183 (0.012702),
AV 0.148769 (0.000123), SR 0.894894 (0.414186), MDD -0.104203 (0.000412),
close to the archive aggregate (CR 0.126837, AV 0.151593, SR 0.870147,
MDD -0.108611). The per-seed account hashes differ (0/10 exact), but the
aggregate is close in all four metrics. Full files are in
`results_summary/0101_*`; the per-seed table is in
`results_summary/comparison_table.md`.

### Full ten-seed reproduction (completed, 2080 Ti)

All fifty runs completed with zero failures. The aggregate uses the paper's
estimator: mean with sample variance (ddof = 1) over ten seeds.

| Condition | CR | AR | AV | SR | MDD |
|---|---|---|---|---|---|
| 1250 | 0.268087 (0.019720) | 0.269316 (0.019917) | 0.143955 (0.000330) | 1.718471 (0.665328) | -0.110508 (0.001795) |
| 1270b | 0.135350 (0.006387) | 0.135934 (0.006446) | 0.112845 (0.000071) | 1.173018 (0.419988) | -0.071117 (0.000347) |
| 1321b_ctrl | 0.139307 (0.002768) | 0.139904 (0.002793) | 0.088444 (0.000247) | 1.496465 (0.170408) | -0.046132 (0.000253) |
| 1321b_lo | 0.189648 (0.004185) | 0.190478 (0.004226) | 0.109941 (0.000116) | 1.654272 (0.290544) | -0.058502 (0.000165) |
| 1321b_hi | 0.219197 (0.003193) | 0.220165 (0.003223) | 0.119010 (0.000092) | 1.736888 (0.191464) | -0.082979 (0.000277) |

The archived 4080 SUPER aggregate above remains the canonical paper table.
The 2080 Ti sweep reproduces the qualitative conclusions:

- **Dose response.** As beta_s goes 0 -> 0.3 -> 1.0, CR rises
  0.139 -> 0.190 -> 0.219 and MDD deepens -0.046 -> -0.059 -> -0.083.
  Both orderings are monotone, as in the paper.
- **RG-PPO vs Return-PPO.** RG-PPO (hi) has lower CR (0.219 vs 0.268) and a
  shallower MDD (-0.083 vs -0.111), the same trade-off direction as the paper.
- **DP-PPO vs RG-PPO.** DP-PPO has lower CR (0.135 vs 0.219) and a shallower
  MDD (-0.071 vs -0.083), the same ordering as the paper.

Exact per-seed values do not match the archive because of the GPU change; the
paired test above isolates that cause. Full per-seed and aggregate files are in
`results_summary/`.

## Environment recorded by the archived runs

Python 3.10.18; PyTorch 2.9.1+cu128; Stable-Baselines3 2.8.0a2; Gymnasium
1.2.2; NumPy 2.2.6; pandas 2.2.3; FinRL 0.3.8; one NVIDIA RTX 4080 SUPER
(16 GB). The exact interpreter is the FinRL environment's Python.

## Hardware caveat for re-runs

The archived runs were produced on an **RTX 4080 SUPER**. The current
reproduction host is an **RTX 2080 Ti (11 GB)**. PyTorch's deterministic mode
guarantees repeatability on the same hardware and software stack, not across
GPU architectures: different CUDA kernels and accumulation orders can change
low-order bits, and those changes can compound over 4 × 40,000 training steps.

Consequences:

1. The seed-0 gate against the archived numbers answers "does the archived
   result recreate on this host?", not "is the code identical?". A mismatch
   must be split into hardware drift and code drift.
2. The code-drift test is a paired run on the **same** GPU: run the archived
   code commit and the release code with the same seed, then compare. This is
   the check used when a mismatch appears.
3. Exact numerical reproduction of the paper's table requires an RTX 4080
   SUPER (or a GPU that produces the same kernels). This limitation is stated
   in `README.md`.
4. The 11 GB card cannot host ten concurrent runs with the original memory
   headroom. The verification host ran five concurrent notebooks at about
   1.6 GB total GPU memory; the launcher uses `--jobs 5` on this host.
