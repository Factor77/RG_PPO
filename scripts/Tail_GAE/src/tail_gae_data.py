"""Tail_GAE V2 data layer: the single entry point that turns the raw CSV into
every downstream structure (env kwargs, tau thresholds, EW returns, risk
features, DJI returns, diagnostics windows).

All date keys are "YYYY-MM-DD" strings. Nothing here is forward-looking in
training: tau_* are training-window statistics, worst20_by_date is causal
(trailing 20-day mean return per stock), q_by_date is diagnostic-only.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_CSV = PROJECT_ROOT / "data" / "new_version_2026" / "data_30_20100101_20260601.csv"
DJI_CSV = PROJECT_ROOT / "data" / "DJI_2013_2026.csv"

TRAIN_START = "2013-04-01"
TRAIN_END = "2024-12-31"


@lru_cache(maxsize=None)
def _read_main_csv() -> pd.DataFrame:
    return pd.read_csv(MAIN_CSV)


@lru_cache(maxsize=None)
def _read_dji_csv() -> pd.DataFrame:
    return pd.read_csv(DJI_CSV)


def _ew_returns_from_close(close: pd.DataFrame) -> pd.Series:
    """Equal-weight daily portfolio returns from a flat date/tic/close panel."""
    ret = close.pivot(index="date", columns="tic", values="close").pct_change()
    return ret.mean(axis=1).dropna()


def _latch_i_by_date(ew: pd.Series, tau: float, *, k: int = 10, release_k: int = 5) -> dict[str, int]:
    """V4.2 latch state machine over the full EW series (spec §3/§5, 定稿口径).

    Replicates the settled scan (/tmp/tail_latch_final.py simulate()): for each
    trading day d, C10_d/C5_d are the k/release_k-day EW cumulative returns
    ending ON day d.  Unlatched: I_d = 1 iff C10_d <= tau (same-day entry).
    Latched: I_d = 1 unconditionally; a release condition (C5_d > 0) met on
    day d takes effect the NEXT day (the condition-holding day itself is still
    latched).  Simulation is continuous over the whole series; latch state
    carries across window boundaries (no re-init at window starts).
    """
    c_k = ew.rolling(k).sum()
    c_rel = ew.rolling(release_k).sum()
    out: dict[str, int] = {}
    latched = False
    for d in c_k.index:
        if latched:
            out[str(d)] = 1
            if c_rel[d] > 0.0:
                latched = False
        else:
            fired = bool(c_k[d] <= tau)
            out[str(d)] = int(fired)
            if fired:
                latched = True
    return out


def _ew_cum_dicts(ew: pd.Series, windows: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """Cumulative EW returns per date for each window (1216 obs features).

    Same rolling sums as the latch series: value at date d is the k-day EW
    cumulative ending ON day d.  Leading NaN days (fewer than k observations)
    are 0.0 so the wrapper's obs lookup never produces a NaN feature.
    """
    out: dict[str, dict[str, float]] = {}
    for k in windows:
        s = ew.rolling(k).sum()
        out[f"mkt_ew_{k}"] = {
            str(d): (0.0 if np.isnan(v) else float(v)) for d, v in s.items()
        }
    return out


def _worst20_from_close(close: pd.DataFrame) -> pd.Series:
    """Trailing-20-day mean daily return of the worst stock, in percent.

    Causal: each value uses only the past 20 days of prices.
    """
    ret = close.pivot(index="date", columns="tic", values="close").pct_change()
    return (ret.rolling(20).mean().min(axis=1) * 100).dropna()


def _dji_return_dict(dji: pd.DataFrame) -> dict[str, float]:
    s = dji.set_index("Date")["Close"].pct_change().dropna()
    return {str(k): float(v) for k, v in s.items()}


def _cdf_against(train_ew: pd.Series, ew: pd.Series) -> dict[str, float]:
    """CDF percentile of each EW return against the training distribution."""
    n = float(len(train_ew))
    return {d: float((train_ew <= r).sum() / n) for d, r in ew.items()}


def load_tail_data(p_tail: float = 0.10) -> dict:
    raw = _read_main_csv()

    # V3 D1 fix: align the 301-dim base state with the 1200 baseline. The
    # FeatureEngineer call (params verbatim from 1200 notebook cell 15) makes
    # the processed panel byte-identical to the 1200 processed snapshot
    # (hash-pinned by tests/test_tail_gae_data.py::test_processed_hash_matches_1200_snapshot).
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,  # ["macd","boll_ub","boll_lb","rsi_30","cci_30","dx_30","close_30_sma","close_60_sma"]
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(raw)
    processed = processed.copy().fillna(0).replace([float("inf")], 0)

    dates = sorted(raw["date"].unique())
    window_dates = [d for d in dates if TRAIN_START <= d <= TRAIN_END]

    ew_returns = _ew_returns_from_close(raw)
    train_ew = ew_returns[window_dates]
    tau_cumul = float(
        np.quantile(train_ew.rolling(10).sum().dropna().to_numpy(), p_tail)
    )
    # V4.2: the latch I_t series (market EW trigger, spec §3/§5) is precomputed
    # once over the full series so the training trigger rate matches the
    # settled scan (14.8%) by construction and window boundaries never reset
    # the latch state.
    i_by_date = _latch_i_by_date(ew_returns, tau_cumul)

    signal_table = pd.DataFrame({"date": dates, "risk_score": 0.0})

    return {
        "processed": processed,
        "ew_by_date": _ew_cum_dicts(ew_returns, windows=(10, 5)),
        "signal_table": signal_table,
        "env_kwargs": {
            "stock_dim": 30,
            "hmax": 100,
            "initial_amount": 1_000_000,
            "buy_cost_pct": [0.001] * 30,
            "sell_cost_pct": [0.001] * 30,
            "tech_indicator_list": INDICATORS,
            "turbulence_threshold": None,
            "reward_scaling": 1e-4,
            "state_space": 301,  # 1 + 2*30 + 8*30 (1200 baseline alignment)
            "action_space": 30,
        },
        "tau_cumul": tau_cumul,
        "i_by_date": i_by_date,
        "ew_returns": ew_returns,
        "q_by_date": _cdf_against(train_ew, ew_returns),
        "dji_return_by_date": _dji_return_dict(_read_dji_csv()),
        "worst20_by_date": {
            str(d): float(v) for d, v in _worst20_from_close(raw).items()
        },
        "window_dates": window_dates,
    }
