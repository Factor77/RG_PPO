"""Reusable backtest metric helpers for the ablation notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def load_trade_account_value(files: list[str | Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in files]
    if not frames:
        raise ValueError("no account_value files provided")
    return concatenate_continuous_account_values(frames)


def concatenate_continuous_account_values(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Join ordered rolling account paths without double-counting a boundary.

    Consecutive trading environments share their terminal/start timestamp.  A
    duplicate is valid only when both recordings agree on the equity value; it
    is then retained once.  Any date reversal or conflicting duplicate proves
    that the portfolio path is not continuous and is rejected.
    """

    if not frames:
        raise ValueError("no account-value frames provided")
    combined = pd.concat(frames, ignore_index=True)
    required = {"date", "account_value"}
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"account-value frames missing columns: {sorted(missing)}")

    dates = pd.to_datetime(combined["date"], errors="raise")
    if not dates.is_monotonic_increasing:
        raise ValueError("trade account-value dates must be monotonically increasing")
    values = pd.to_numeric(combined["account_value"], errors="raise")
    if values.isna().any():
        raise ValueError("account values must not contain missing values")

    keep = [True] * len(combined)
    for index in range(1, len(combined)):
        if dates.iloc[index] != dates.iloc[index - 1]:
            continue
        if float(values.iloc[index]) != float(values.iloc[index - 1]):
            raise ValueError("duplicate trade boundary has conflicting account values")
        keep[index] = False
    return combined.loc[keep].reset_index(drop=True)


def load_dji_baseline(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.lower() for column in df.columns]
    if "close" not in df.columns:
        raise ValueError(f"expected close column in {path}")
    return df


def compute_return_metrics(
    account_value: pd.DataFrame,
    *,
    value_col: str = "account_value",
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Compute cumulative and annual returns from account-value observations."""
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    series = pd.to_numeric(account_value[value_col], errors="raise")
    if series.isna().any():
        raise ValueError("account values must not contain missing values")
    if len(series) < 2:
        raise ValueError("at least two account-value observations are required")

    initial_value = float(series.iloc[0])
    final_value = float(series.iloc[-1])
    if initial_value <= 0 or final_value < 0:
        raise ValueError("account values must permit a non-negative growth ratio")

    periods = len(series) - 1
    growth_ratio = final_value / initial_value
    return {
        "cumulative_return": growth_ratio - 1.0,
        "annual_return": growth_ratio ** (annualization_factor / periods) - 1.0,
    }


def compute_performance_stats(
    account_value: pd.DataFrame,
    *,
    value_col: str = "account_value",
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    series = pd.to_numeric(account_value[value_col], errors="raise")
    return_metrics = compute_return_metrics(
        account_value,
        value_col=value_col,
        annualization_factor=annualization_factor,
    )
    daily_return = series.pct_change(fill_method=None).dropna()
    annual_volatility = daily_return.std() * (annualization_factor ** 0.5)
    sharpe = (
        (annualization_factor ** 0.5)
        * daily_return.mean()
        / daily_return.std()
    )
    rolling_max = series.cummax()
    mdd = (series / rolling_max - 1.0).min()
    return {
        "CR": float(return_metrics["cumulative_return"]),
        "AR": float(return_metrics["annual_return"]),
        "AV": float(annual_volatility),
        "SR": float(sharpe),
        "MDD": float(mdd),
    }
