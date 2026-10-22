"""Offline tests for risk_oracle: registry, transforms, pure formulas."""

import math

import pytest

from risk_oracle import (
    REGISTRY,
    _ar1_20d,
    _cumret20,
    _extract_current_drawdown,
    _extract_drawdown_delta,
    _extract_gross_exposure,
    _extract_running_max_drawdown,
    _semi_vol20,
    _streak20max,
    _streak40,
    get_risk_dims,
)


class _FakeAccounting:
    """Duck-typed reward_adapter.accounting_state (no project import)."""

    def __init__(self, current_drawdown, max_drawdown, last_drawdown_delta):
        self.current_drawdown = current_drawdown
        self.max_drawdown = max_drawdown
        self.last_drawdown_delta = last_drawdown_delta


def test_typo_name_raises():
    with pytest.raises(AssertionError):
        get_risk_dims(["streak20maxx"])


def test_n_risk_dims_mismatch():
    with pytest.raises(AssertionError):
        get_risk_dims(["current_drawdown", "worst20"], n_risk_dims=3)
    dims = get_risk_dims(["current_drawdown", "worst20"], n_risk_dims=2)
    assert dims == [REGISTRY["current_drawdown"], REGISTRY["worst20"]]


def test_invalid_kind_raises():
    import risk_oracle

    original = risk_oracle.REGISTRY["worst20"]
    risk_oracle.REGISTRY["worst20"] = ("bogus", None, (1.0, 0.0))
    try:
        with pytest.raises(AssertionError):
            risk_oracle.get_risk_dims(["worst20"])
    finally:
        risk_oracle.REGISTRY["worst20"] = original


def test_worst20_market_lookup():
    kind, formula, transform = get_risk_dims(["worst20"])[0]
    assert kind == "market_lookup"
    assert formula is None
    assert transform == pytest.approx((1 / 3, 0.0))


def test_semi_vol20():
    win = [-0.02, -0.01, 0, 0.01, 0]
    assert _semi_vol20(win) == pytest.approx(
        math.sqrt((0.02 ** 2 + 0.01 ** 2) / 5), rel=1e-9
    )
    assert _semi_vol20([]) == 0.0


def test_streak20max():
    # 0.0 BREAKS the streak: only strictly negative returns count.
    # Frozen acceptance list: max strictly-negative run is 2 ({-0.01,-0.02}
    # and {-0.01,-0.03}); the contract's "== 3" is unreachable under the
    # pinned r<0 rule, so assert the true value here.
    assert _streak20max([0.01, -0.01, -0.02, 0.0, -0.01, -0.03, 0.02]) == 2
    # contract value 3 reproduced with three consecutive negatives, 0.0 breaks:
    assert _streak20max([0.01, -0.01, -0.02, -0.01, 0.0, -0.03, 0.02]) == 3
    assert _streak20max([]) == 0
    assert _streak20max([0.0, -0.01]) == 1


def test_cumret20():
    assert _cumret20([0.01, -0.01, 0.005]) == 0.005
    assert _cumret20([]) == 0.0


def test_ar1_20d():
    assert _ar1_20d([0.01, 0.02, 0.01]) == 0.0  # len < 4
    assert _ar1_20d([]) == 0.0
    assert _ar1_20d([0.01]) == 0.0
    assert _ar1_20d([0.1, 0.1, 0.1, 0.1]) == 0.0  # zero variance
    assert _ar1_20d([0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_streak40_same_rule():
    assert _streak40([0.01, -0.01, -0.02, 0.0, -0.01, -0.03, 0.02]) == 2
    assert _streak40([-0.1, -0.2, -0.3, -0.4]) == 4


EXPECTED_KIND_AND_TRANSFORM = {
    "current_drawdown": ("accounting", (2.0, -1.0)),
    "running_max_drawdown": ("accounting", (2.0, -1.0)),
    "drawdown_delta": ("accounting", (10.0, 0.0)),
    "gross_exposure": ("accounting", (2.0, -1.0)),
    "downside_volatility_20": ("portfolio_window", (20.0, 0.0)),
    "streak20max": ("portfolio_window", (0.1, -1.0)),
    "ar1_20d": ("portfolio_window", (1.0, 0.0)),
    "cumret20": ("portfolio_window", (10.0, 0.0)),
    "streak40": ("portfolio_window", (0.05, -1.0)),
    "worst20": ("market_lookup", (1 / 3, 0.0)),
    "mkt_ew_10": ("market_ew_lookup", (1.0, 0.0)),
    "mkt_ew_5": ("market_ew_lookup", (1.0, 0.0)),
}


def test_registry_complete():
    assert len(REGISTRY) == 12
    assert set(REGISTRY) == set(EXPECTED_KIND_AND_TRANSFORM)
    for name, (kind, transform) in EXPECTED_KIND_AND_TRANSFORM.items():
        entry_kind, formula, entry_transform = REGISTRY[name]
        assert entry_kind == kind
        assert entry_transform == pytest.approx(transform)
        if kind in ("market_lookup", "market_ew_lookup"):
            assert formula is None
        else:
            assert callable(formula)


def test_accounting_extractors():
    base = [1000.0, 10, 20, 30, 5, 10, 15]
    acc = _FakeAccounting(0.25, 0.4, 0.02)
    exposure = (10 * 5 + 20 * 10 + 30 * 15) / (1000 + 10 * 5 + 20 * 10 + 30 * 15)
    assert _extract_gross_exposure(base, acc, 3) == pytest.approx(exposure)
    assert _extract_current_drawdown(base, acc, 3) == 0.25
    assert _extract_running_max_drawdown(base, acc, 3) == 0.4
    assert _extract_drawdown_delta(base, acc, 3) == 0.02
    # portfolio-value guard: value <= 1e-8 -> 0.0
    assert _extract_gross_exposure([1e-9, 0.0, 0.0, 0.0, 0.0, 0.0], acc, 3) == 0.0


def test_gross_exposure_301_layout_regression():
    # 旧实现用 (len-1)//2 推断 stock_dim：301 维 state 得 150，把 240 个技术
    # 指标当价格/股份相乘（曾产出仿射后观测 −731 类不可能值，corr 0.044）。
    # stock_dim 由调用方显式传入后必须复现正确敞口。
    base = [1000.0] + [10.0] * 30 + [5.0] * 30 + [1e6] * 240
    assert _extract_gross_exposure(base, _FakeAccounting(0.0, 0.0, 0.0), 30) == pytest.approx(
        1500.0 / 2500.0
    )


def test_gross_exposure_abs_and_clip():
    # 单边敞口 Σ|p·s|：做空股贡献分子但不被净额抵消
    base_short = [1000.0] + [10.0] * 30 + [5.0] * 29 + [-5.0]
    assert _extract_gross_exposure(
        base_short, _FakeAccounting(0.0, 0.0, 0.0), 30
    ) == pytest.approx(1500.0 / 2400.0)
    # clip：单边敞口超过净市值（重做空）时上限 1.0（spec §3.4 [0,1] 界）
    base_hi = [0.0] + [10.0] * 30 + [5.0] * 20 + [-5.0] * 10
    assert _extract_gross_exposure(base_hi, _FakeAccounting(0.0, 0.0, 0.0), 30) == 1.0


def test_transform_math():
    cases = [
        ("current_drawdown", 0.25, 0.25 * 2.0 - 1.0),
        ("running_max_drawdown", 0.4, 0.4 * 2.0 - 1.0),
        ("drawdown_delta", 0.02, 0.02 * 10.0),
        ("gross_exposure", 0.5, 0.5 * 2.0 - 1.0),
        ("downside_volatility_20", 0.03, 0.03 * 20.0),
        ("streak20max", 3, 3 * 0.1 - 1.0),
        ("ar1_20d", 0.5, 0.5 * 1.0),
        ("cumret20", 0.2, 0.2 * 10.0),
        ("streak40", 5, 5 * 0.05 - 1.0),
        ("worst20", 0.05, 0.05 * (1 / 3)),
    ]
    for name, v, expected in cases:
        a, b = REGISTRY[name][2]
        assert v * a + b == pytest.approx(expected)
