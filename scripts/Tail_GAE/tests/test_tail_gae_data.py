"""Acceptance tests for the Tail_GAE V3 data layer (tail_gae_data).

V3 (spec §2, D1 fix): the processed panel is now produced by the same
FeatureEngineer call as the 1200 baseline notebook (cell 15), so the base
state is 301-dim (1 + 2*30 + 8*30) and the processed panel is byte-identical
to the 1200 processed snapshot (hash-pinned below).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from finrl.config import INDICATORS

from scripts.ensemble.ensemble_audit import dataframe_hash
from tail_gae_data import (
    MAIN_CSV,
    TRAIN_END,
    TRAIN_START,
    _worst20_from_close,
    load_tail_data,
)

ENV_KWARGS_EXPECTED = {
    "stock_dim": 30,
    "hmax": 100,
    "initial_amount": 1_000_000,
    "buy_cost_pct": [0.001] * 30,
    "sell_cost_pct": [0.001] * 30,
    "tech_indicator_list": INDICATORS,
    "turbulence_threshold": None,
    "reward_scaling": 1e-4,
    "state_space": 301,  # 1 + 2*30 + 8*30
    "action_space": 30,
}

# 1200 processed snapshot (Results/Tail_GAE/1200 Baseline/seed_0/20260810_190207/
# audit/data/processed_snapshot.json): raw 8 cols + 8 indicators + turbulence,
# in this exact order, shape (123780, 17). Pinned here so the test stays
# hermetic (Results/ is gitignored).
PROCESSED_COLUMNS_1200 = [
    "date",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "tic",
    "day",
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "turbulence",
]

# dataframe_hash(processed) == snapshot's "hash" field (spec §2.2).
PROCESSED_SNAPSHOT_HASH_1200 = "8d65140845aaabdb385ed5109b372352bb08350b22d7900203a937d9d5eeafc6"


@pytest.fixture(scope="module")
def data():
    return load_tail_data(p_tail=0.10)


def test_tau_consistency(data):
    ew = data["ew_returns"]
    train_ew = ew[data["window_dates"]]
    assert np.isclose(
        data["tau_cumul"],
        np.quantile(train_ew.rolling(10).sum().dropna().to_numpy(), 0.10),
        atol=1e-8,
    )
    assert abs(data["tau_cumul"] - (-0.0254)) < 2e-3


def test_latch_i_by_date_consistency(data):
    """V4.2 spec §5: the latch series must match the settled scan
    (/tmp/tail_latch_final.py simulate(), rel5>0) — continuous simulation over
    the full EW series, same-day entry (C10 <= tau), next-day release (C5 > 0
    keeps the condition-holding day latched)."""
    i = data["i_by_date"]
    ew = data["ew_returns"]
    c10 = ew.rolling(10).sum()
    c5 = ew.rolling(5).sum()
    assert set(i) == set(c10.index)
    assert all(v in (0, 1) for v in i.values())

    # Replay the state machine independently; must agree at every date.
    latched = False
    for d in c10.index:
        k = str(d)
        if latched:
            assert i[k] == 1
            if c5[d] > 0.0:
                latched = False
        else:
            fired = bool(c10[d] <= data["tau_cumul"])
            assert i[k] == int(fired)
            if fired:
                latched = True

    days = sorted(i)
    tr = [x for x in days if TRAIN_START <= x <= TRAIN_END]
    assert abs(sum(i[x] for x in tr) / len(tr) - 0.1483) < 0.002
    te = [x for x in days if "2025-01-01" <= x <= "2026-06-01"]
    assert abs(sum(i[x] for x in te) / len(te) - 0.1222) < 0.002
    sb2 = [x for x in days if "2026-02-26" <= x <= "2026-04-07"]
    assert abs(sum(i[x] for x in sb2) / len(sb2) - 0.6071) < 0.01
    assert next(x for x in sb2 if i[x]) == "2026-03-06"


def test_ew_by_date_matches_latch_series(data):
    """1216: the obs EW tables (mkt_ew_10 / mkt_ew_5) must be the same
    cumulative series as the latch — values equal the rolling sums, key sets
    equal the latch's, leading NaN days are 0.0, and every day with
    mkt_ew_10 <= tau is latched (I=1), since C10 <= tau is the entry
    condition regardless of prior latch state."""
    ew = data["ew_returns"]
    c10 = ew.rolling(10).sum()
    c5 = ew.rolling(5).sum()
    ew10 = data["ew_by_date"]["mkt_ew_10"]
    ew5 = data["ew_by_date"]["mkt_ew_5"]
    assert set(ew10) == set(ew5) == set(data["i_by_date"])
    for d in c10.index[10:]:
        assert ew10[str(d)] == pytest.approx(c10[d], abs=1e-9)
    for d in c5.index[5:]:
        assert ew5[str(d)] == pytest.approx(c5[d], abs=1e-9)
    assert ew10[str(c10.index[0])] == 0.0  # leading NaN days -> 0.0
    assert ew5[str(c5.index[0])] == 0.0
    i = data["i_by_date"]
    tr = [str(d) for d in c10.index if TRAIN_START <= str(d) <= TRAIN_END]
    assert all(i[d] == 1 for d in tr if ew10[d] <= data["tau_cumul"])


def test_worst20_causality(data):
    dates = sorted(data["processed"]["date"].unique())
    d = min(x for x in dates if x >= "2025-01-01")
    assert d in data["worst20_by_date"]
    truncated = data["processed"][data["processed"]["date"] <= d]
    w = _worst20_from_close(truncated)
    assert np.isclose(w.iloc[-1], data["worst20_by_date"][d], atol=1e-12)


def test_worst20_key_coverage(data):
    dates = sorted(data["processed"]["date"].unique())
    assert set(data["window_dates"]) <= set(data["worst20_by_date"])
    for d in dates[:19]:
        assert d not in data["worst20_by_date"]
    for v in data["worst20_by_date"].values():
        assert np.isfinite(v)


def test_env_kwargs_contract(data):
    assert data["env_kwargs"] == ENV_KWARGS_EXPECTED


def test_synthetic_hand_check(data):
    np.random.seed(0)
    n_days = 50
    dates = [str(d.date()) for d in pd.bdate_range("2020-01-01", periods=n_days)]
    rows = []
    for i, d in enumerate(dates):
        rows.append((d, "A", 100 * 1.01**i))
        rows.append((d, "B", 100 * 0.99**i))
    panel = pd.DataFrame(rows, columns=["date", "tic", "close"])
    worst20 = _worst20_from_close(panel)
    mid = dates[30]
    assert abs(worst20[mid] - (-1.0)) < 0.1

    for q in data["q_by_date"].values():
        assert 0.0 <= q <= 1.0
    for k in data["dji_return_by_date"]:
        pd.to_datetime(k)


def test_processed_pass_through(data):
    raw = pd.read_csv(MAIN_CSV)
    # V3: processed is the FeatureEngineer output, not a raw passthrough —
    # columns must match the 1200 processed snapshot (8 indicators + turbulence).
    assert list(data["processed"].columns) == PROCESSED_COLUMNS_1200
    assert data["processed"]["tic"].nunique() == 30
    assert data["processed"].shape == (len(raw), len(PROCESSED_COLUMNS_1200))
    assert set(raw["date"].unique()) == set(data["processed"]["date"].unique())
    # preprocess_data may sort rows; market precomputations use `raw`, so the
    # date panel keyed checks below still hold on the returned data.
    window = data["window_dates"]
    assert set(window) <= set(data["ew_returns"].index)
    assert window == sorted(window)
    assert window[0] == TRAIN_START and window[-1] == TRAIN_END


def test_processed_hash_matches_1200_snapshot(data):
    """V3 spec §2.2: the recomputed processed panel must hash identically to
    the 1200 baseline processed snapshot. Any change to the data pipeline
    (indicator params, column order, fill strategy) fails here."""
    assert dataframe_hash(data["processed"]) == PROCESSED_SNAPSHOT_HASH_1200
