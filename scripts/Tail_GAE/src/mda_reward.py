"""Authoritative simple-return plus maximum-drawdown-increment reward."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite, sqrt

DEFAULT_EPSILON = 1e-8
DEFAULT_CLIP_C = 0.02  # deprecated compatibility constant; MDA penalties are never clipped


@dataclass
class RewardAccountingState:
    """Continuous risk state shared by reward and observation construction.

    ``recent_portfolio_returns`` is runtime state, not an additional policy
    feature.  It is retained only so that the reported 20-day downside
    volatility remains continuous across consecutive trade windows.
    """
    peak_value: float
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    last_value: float = 0.0
    epsilon: float = DEFAULT_EPSILON
    last_drawdown_delta: float = 0.0
    recent_portfolio_returns: tuple[float, ...] = ()

    @classmethod
    def reset(cls, initial_value: float, *, epsilon: float = DEFAULT_EPSILON, clip_c: float | None = None) -> "RewardAccountingState":
        _validate_account_value(initial_value, "initial_value")
        _validate_positive(epsilon, "epsilon")
        return cls(float(initial_value), 0.0, 0.0, float(initial_value), float(epsilon))

    def reset_in_place(self, initial_value: float) -> None:
        _validate_account_value(initial_value, "initial_value")
        self.peak_value = self.last_value = float(initial_value)
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.last_drawdown_delta = 0.0
        self.recent_portfolio_returns = ()

    def copy(self) -> "RewardAccountingState":
        return type(self)(
            self.peak_value,
            self.current_drawdown,
            self.max_drawdown,
            self.last_value,
            self.epsilon,
            self.last_drawdown_delta,
            tuple(self.recent_portfolio_returns),
        )

    @property
    def downside_volatility_20(self) -> float:
        """Return unannualized RMS downside volatility of realised portfolio returns."""

        if not self.recent_portfolio_returns:
            return 0.0
        squared_downside = [min(float(value), 0.0) ** 2 for value in self.recent_portfolio_returns]
        return sqrt(sum(squared_downside) / len(squared_downside))

    def assert_continuous_with(self, previous_value: float, *, epsilon: float) -> None:
        _validate_account_value(previous_value, "previous_value")
        if not isclose(self.last_value, float(previous_value), rel_tol=1e-10, abs_tol=max(self.epsilon, abs(float(previous_value))*1e-12)):
            raise ValueError("carried reward accounting state is not continuous with previous_value")
        if not isclose(self.epsilon, float(epsilon), rel_tol=0.0, abs_tol=min(self.epsilon, float(epsilon))*1e-12):
            raise ValueError("carried reward accounting state has a different epsilon")


@dataclass(frozen=True)
class RewardComputationResult:
    raw_return: float
    prev_peak: float
    new_peak: float
    new_drawdown: float
    prev_drawdown: float
    drawdown_increase: float
    prev_max_drawdown: float
    new_max_drawdown: float
    c_t: float
    lambda_value: float
    final_reward: float
    def as_dict(self) -> dict[str, float]: return asdict(self)


def compute_return_reward(previous_value: float, next_value: float, *, epsilon: float = DEFAULT_EPSILON) -> float:
    _validate_account_value(previous_value, "previous_value")
    _validate_account_value(next_value, "next_value")
    _validate_positive(epsilon, "epsilon")
    return (float(next_value)-float(previous_value))/max(float(previous_value), float(epsilon))


def advance_mdd_accounting(previous_value: float, next_value: float, *, state: RewardAccountingState) -> tuple[float,float,float,float,float,float]:
    """Advance H/M after one actual environment transition.

    This function must never be called from a rolling-window reset.  A reset
    carries the state unchanged; an apparent boundary price move indicates an
    incorrectly partitioned transition and must fail continuity validation.
    """
    state.assert_continuous_with(previous_value, epsilon=state.epsilon)
    new_peak=max(float(state.peak_value), float(next_value))
    new_drawdown=max(0.0, 1.0-float(next_value)/max(new_peak, state.epsilon))
    previous_drawdown=float(state.current_drawdown)
    drawdown_increase=max(0.0, new_drawdown-previous_drawdown)
    previous_m=float(state.max_drawdown)
    new_m=max(previous_m, new_drawdown)
    c_t=new_m-previous_m
    state.peak_value, state.current_drawdown, state.max_drawdown, state.last_value=new_peak,new_drawdown,new_m,float(next_value)
    state.last_drawdown_delta = new_drawdown - previous_drawdown
    raw_return = compute_return_reward(previous_value, next_value, epsilon=state.epsilon)
    state.recent_portfolio_returns = (
        *state.recent_portfolio_returns,
        float(raw_return),
    )[-20:]
    return new_peak,previous_drawdown,new_drawdown,drawdown_increase,new_m,c_t


def compute_mda_reward(previous_value: float, next_value: float, *, lambda_value: float, state: RewardAccountingState) -> RewardComputationResult:
    _validate_account_value(previous_value, "previous_value")
    _validate_account_value(next_value, "next_value")
    _validate_non_negative(lambda_value, "lambda_value")
    if not isinstance(state, RewardAccountingState): raise TypeError("state must be a RewardAccountingState")
    raw_return=compute_return_reward(previous_value,next_value,epsilon=state.epsilon)
    prev_peak=float(state.peak_value); prev_m=float(state.max_drawdown)
    new_peak,prev_drawdown,new_drawdown,drawdown_increase,new_m,c_t=advance_mdd_accounting(previous_value,next_value,state=state)
    return RewardComputationResult(raw_return,prev_peak,new_peak,new_drawdown,prev_drawdown,drawdown_increase,prev_m,new_m,c_t,float(lambda_value),raw_return-float(lambda_value)*c_t)


def _validate_account_value(value: float, name: str) -> None:
    _validate_positive(value,name)
def _validate_positive(value: float, name: str) -> None:
    if not isfinite(float(value)) or float(value)<=0: raise ValueError(f"{name} must be positive")
def _validate_non_negative(value: float, name: str) -> None:
    if not isfinite(float(value)) or float(value)<0: raise ValueError(f"{name} must be non-negative")
