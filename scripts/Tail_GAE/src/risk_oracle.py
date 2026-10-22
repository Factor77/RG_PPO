"""Tail_GAE V2 risk-state registry: name -> (kind, formula, transform).

The wrapper (built by another agent) resolves dims through ``get_risk_dims``
and calls ``formula(base_state, accounting_state, stock_dim)`` for accounting entries,
``formula(window_returns)`` for portfolio_window entries, and looks up
``worst20_by_date[date]`` for market_lookup entries, or
``ew_by_date[name][date]`` for market_ew_lookup entries (1216: market EW
cumulative returns as policy-visible features).  This module is a leaf:
it imports nothing from the project.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

VALID_KINDS = {"accounting", "portfolio_window", "market_lookup", "market_ew_lookup"}


# ---------- portfolio_window formulas (pure functions of window returns) ----------

def _semi_vol20(win: Sequence[float]) -> float:
    # mean over ALL elements; positives and zeros contribute 0
    if not win:
        return 0.0
    return math.sqrt(sum(min(r, 0.0) ** 2 for r in win) / len(win))


def _streak20max(win: Sequence[float]) -> int:
    # longest run of strictly negative returns (r < 0); 0.0 breaks the streak
    best = run = 0
    for r in win:
        run = run + 1 if r < 0 else 0
        if run > best:
            best = run
    return best


def _ar1_20d(win: Sequence[float]) -> float:
    # lag-1 Pearson autocorrelation corr(r_t, r_{t-1}) over the window
    if len(win) < 4:
        return 0.0
    x, y = win[:-1], win[1:]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = sum((v - mx) ** 2 for v in x)
    sy = sum((v - my) ** 2 for v in y)
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sx * sy)


def _cumret20(win: Sequence[float]) -> float:
    # rolling SUM, not mean
    return sum(win)


def _streak40(win: Sequence[float]) -> int:
    return _streak20max(win)


# ---------- accounting extractors (base_state + duck-typed accounting_state) ----------

def _extract_current_drawdown(base_state: list[float], accounting_state, stock_dim) -> float:
    return float(accounting_state.current_drawdown)


def _extract_running_max_drawdown(base_state: list[float], accounting_state, stock_dim) -> float:
    return float(accounting_state.max_drawdown)


def _extract_drawdown_delta(base_state: list[float], accounting_state, stock_dim) -> float:
    return float(accounting_state.last_drawdown_delta)


def _extract_gross_exposure(base_state: list[float], accounting_state, stock_dim) -> float:
    # base_state 布局: [cash, price_1..price_n, share_1..share_n, <技术指标...>]
    # stock_dim 由调用方传入（真实股票数）。旧实现用 (len-1)//2 推断——state 追加
    # 240 个技术指标后 (301-1)//2=150，把技术指标当价格/股份相乘（旧 bug，曾产出
    # 仿射后观测 −731 类不可能值，corr 0.044）。
    prices = base_state[1 : 1 + stock_dim]
    shares = base_state[1 + stock_dim : 1 + 2 * stock_dim]
    gross = net = 0.0
    for p, s in zip(prices, shares):
        ps = p * s
        gross += abs(ps)   # 单边敞口 Σ|p·s|，做空不被净额抵消
        net += ps
    value = base_state[0] + net   # 净市值，与 _portfolio_value 同口径
    if value <= 1e-8:
        return 0.0
    # 单边敞口 ≤ 净值（spec §3.4 [0,1] 界）；clip 防未来布局回归再产出越界值
    return min(gross / value, 1.0)


# ---------- registry ----------

REGISTRY = {
    # name -> (kind, formula, transform); transform = (a, b): v' = v*a + b
    # kind: "accounting" | "portfolio_window" | "market_lookup"
    "current_drawdown":       ("accounting",       _extract_current_drawdown,    (2.0, -1.0)),
    "running_max_drawdown":   ("accounting",       _extract_running_max_drawdown,(2.0, -1.0)),
    "drawdown_delta":         ("accounting",       _extract_drawdown_delta,      (10.0, 0.0)),
    "gross_exposure":         ("accounting",       _extract_gross_exposure,      (2.0, -1.0)),
    "downside_volatility_20": ("portfolio_window", _semi_vol20,                  (20.0, 0.0)),
    "streak20max":            ("portfolio_window", _streak20max,                 (0.1, -1.0)),
    "ar1_20d":                ("portfolio_window", _ar1_20d,                     (1.0, 0.0)),
    "cumret20":               ("portfolio_window", _cumret20,                    (10.0, 0.0)),
    "streak40":               ("portfolio_window", _streak40,                    (0.05, -1.0)),
    "worst20":                ("market_lookup",    None,                         (1/3, 0.0)),  # formula None deliberately: kind decides lookup
    # 1216: market EW cumulative returns through the step's date, precomputed by
    # load_tail_data from the same EW series as the latch (pass-through scale).
    "mkt_ew_10":              ("market_ew_lookup", None,                         (1.0, 0.0)),
    "mkt_ew_5":               ("market_ew_lookup", None,                         (1.0, 0.0)),
}


def get_risk_dims(
    names: list[str],
    n_risk_dims: int | None = None,
) -> list[tuple[str, Callable | None, tuple[float, float]]]:
    """Resolve REGISTRY entries in the given order with fail-fast checks."""
    unknown = [n for n in names if n not in REGISTRY]
    assert not unknown, f"unknown risk dims: {unknown}"
    if n_risk_dims is not None:
        assert len(names) == n_risk_dims, f"expected {n_risk_dims} dims, got {len(names)}"
    entries = [REGISTRY[name] for name in names]
    assert all(kind in VALID_KINDS for kind, _, _ in entries), "invalid kind in registry entry"
    return entries
