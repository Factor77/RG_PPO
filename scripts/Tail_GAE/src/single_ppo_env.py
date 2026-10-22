"""Repo-local PU-2 environment adapter for Return-PPO and MDA-PPO rewards.

This file does not patch FinRL. ``SinglePPOStockTradingEnv`` subclasses the
installed ``StockTradingEnv`` when available and takes reward authority only at
the local point where account values before and after trading are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Callable, Mapping

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces

try:
    from .mda_reward import (
        DEFAULT_CLIP_C,
        DEFAULT_EPSILON,
        RewardAccountingState,
        RewardComputationResult,
        advance_mdd_accounting,
        compute_mda_reward,
        compute_return_reward,
    )
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from mda_reward import (
        DEFAULT_CLIP_C,
        DEFAULT_EPSILON,
        RewardAccountingState,
        RewardComputationResult,
        advance_mdd_accounting,
        compute_mda_reward,
        compute_return_reward,
    )


try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
except ImportError:
    StockTradingEnv = None  # type: ignore[assignment]


SUPPORTED_REWARD_MODES = {"aust_original", "return", "mda"}


def gross_exposure_from_state(
    base_state: list[float],
    *,
    stock_dim: int,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Compute gross stock exposure from the canonical FinRL state prefix."""

    prices = np.asarray(base_state[1 : (stock_dim + 1)], dtype=float)
    shares = np.asarray(
        base_state[(stock_dim + 1) : (stock_dim * 2 + 1)], dtype=float
    )
    portfolio_value = float(base_state[0] + np.sum(prices * shares))
    if portfolio_value <= epsilon:
        raise ValueError("portfolio value must be positive when computing exposure")
    return float(np.sum(np.abs(prices * shares)) / portfolio_value)


@dataclass(frozen=True)
class RiskObservationFeature:
    """One ordered, policy-visible risk feature and its value extractor."""

    name: str
    extractor: Callable[[list[float], int, RewardAccountingState], float]


def _gross_exposure_feature(
    base_state: list[float],
    stock_dim: int,
    accounting_state: RewardAccountingState,
) -> float:
    return gross_exposure_from_state(
        base_state,
        stock_dim=stock_dim,
        epsilon=accounting_state.epsilon,
    )


RISK_OBSERVATION_SCHEMA = (
    RiskObservationFeature(
        "current_drawdown",
        lambda _base, _stock_dim, state: float(state.current_drawdown),
    ),
    RiskObservationFeature(
        "running_max_drawdown",
        lambda _base, _stock_dim, state: float(state.max_drawdown),
    ),
    RiskObservationFeature(
        "drawdown_delta",
        lambda _base, _stock_dim, state: float(state.last_drawdown_delta),
    ),
    RiskObservationFeature(
        "downside_volatility_20",
        lambda _base, _stock_dim, state: float(state.downside_volatility_20),
    ),
    RiskObservationFeature("gross_exposure", _gross_exposure_feature),
)
RISK_OBSERVATION_FIELDS = tuple(
    feature.name for feature in RISK_OBSERVATION_SCHEMA
)
# Backward-compatible export. Runtime logic derives the dimension from the
# schema instead of treating this value as an independently maintained setting.
RISK_OBSERVATION_DIM = len(RISK_OBSERVATION_SCHEMA)


def get_risk_observation_dim() -> int:
    """Return the number of fields in the authoritative risk schema."""

    return len(RISK_OBSERVATION_SCHEMA)


def risk_observation_values(
    base_state: list[float],
    *,
    stock_dim: int,
    accounting_state: RewardAccountingState,
) -> list[float]:
    """Build risk values in exactly the order declared by the schema."""

    values = [
        float(feature.extractor(base_state, stock_dim, accounting_state))
        for feature in RISK_OBSERVATION_SCHEMA
    ]
    if len(values) != get_risk_observation_dim():
        raise RuntimeError("risk observation values do not match the schema")
    return values


def append_risk_observation(
    base_state: list[float],
    *,
    stock_dim: int,
    accounting_state: RewardAccountingState,
) -> list[float]:
    """Append the schema-defined V2 risk fields to a FinRL base observation."""

    return [
        *base_state,
        *risk_observation_values(
            base_state,
            stock_dim=stock_dim,
            accounting_state=accounting_state,
        ),
    ]


class RewardComponentNormalizer:
    """Independent, train-fitted normalizers for return and MDD increment."""

    def __init__(self, state: Mapping[str, float] | None = None, *, training: bool = False) -> None:
        state = dict(state or {})
        self.training = bool(training)
        self.count = int(state.get("count", 0))
        self.return_mean = float(state.get("return_mean", 0.0))
        self.return_m2 = float(state.get("return_m2", 0.0))
        self.cost_mean = float(state.get("cost_mean", 0.0))
        self.cost_m2 = float(state.get("cost_m2", 0.0))

    def normalize(self, raw_return: float, mdd_cost: float) -> tuple[float, float]:
        normalized_return = self._normalize(raw_return, self.return_mean, self.return_m2)
        normalized_cost = self._normalize(mdd_cost, self.cost_mean, self.cost_m2)
        if self.training:
            self._update(raw_return, mdd_cost)
        return normalized_return, normalized_cost

    def as_dict(self) -> dict[str, float]:
        return {
            "count": float(self.count), "return_mean": self.return_mean,
            "return_m2": self.return_m2, "cost_mean": self.cost_mean,
            "cost_m2": self.cost_m2,
        }


    def _normalize(self, value: float, mean: float, m2: float) -> float:
        if self.count < 2:
            return float(value)
        variance = m2 / (self.count - 1)
        if variance <= self.epsilon:
            return float(value)
        return (float(value) - mean) / float(np.sqrt(variance + self.epsilon))

    def _update(self, raw_return: float, mdd_cost: float) -> None:
        self.count += 1
        return_delta = raw_return - self.return_mean
        self.return_mean += return_delta / self.count
        self.return_m2 += return_delta * (raw_return - self.return_mean)
        cost_delta = mdd_cost - self.cost_mean
        self.cost_mean += cost_delta / self.count
        self.cost_m2 += cost_delta * (mdd_cost - self.cost_mean)

    @property
    def epsilon(self) -> float:
        return 1e-8


class RewardModeAdapter:
    """Small adapter that keeps reward accounting outside observations."""

    def __init__(
        self,
        *,
        reward_mode: str,
        lambda_value: float = 0.0,
        epsilon: float = DEFAULT_EPSILON,
        clip_c: float = DEFAULT_CLIP_C,
        component_normalization_state: Mapping[str, float] | None = None,
        component_normalization_training: bool = False,
        enable_component_normalization: bool = False,
    ) -> None:
        if reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"unsupported reward_mode={reward_mode!r}; "
                f"expected one of {sorted(SUPPORTED_REWARD_MODES)}"
            )
        if reward_mode == "mda" and lambda_value < 0.0:
            raise ValueError("lambda_value must be non-negative")
        self.reward_mode = reward_mode
        self.lambda_value = float(lambda_value)
        self.epsilon = float(epsilon)
        self.clip_c = float(clip_c)
        self.accounting_state: RewardAccountingState | None = None
        self.last_result: RewardComputationResult | None = None
        self.enable_component_normalization = bool(enable_component_normalization)
        self.component_normalizer = RewardComponentNormalizer(
            component_normalization_state,
            training=component_normalization_training,
        )

    def reset(self, initial_value: float, *, carried_state: RewardAccountingState | None = None) -> RewardAccountingState:
        """Reset or restore risk state without manufacturing a bridge transition."""
        if carried_state is None:
            self.accounting_state = RewardAccountingState.reset(initial_value, epsilon=self.epsilon)
        else:
            carried_state.assert_continuous_with(initial_value, epsilon=self.epsilon)
            self.accounting_state = carried_state.copy()
        self.last_result = None
        return self.accounting_state

    def compute_reward(
        self,
        *,
        begin_total_asset: float,
        end_total_asset: float,
        original_reward_scaling: float,
    ) -> tuple[float, dict[str, float]]:
        if self.accounting_state is None:
            self.reset(begin_total_asset)

        if self.reward_mode == "aust_original":
            advance_mdd_accounting(
                begin_total_asset,
                end_total_asset,
                state=self.accounting_state,
            )
            final_reward = (end_total_asset - begin_total_asset) * original_reward_scaling
            return float(final_reward), {
                "raw_return": compute_return_reward(
                    begin_total_asset,
                    end_total_asset,
                    epsilon=self.epsilon,
                ),
                "final_reward": float(final_reward),
                "lambda_value": self.lambda_value,
            }

        if self.reward_mode == "return":
            advance_mdd_accounting(
                begin_total_asset,
                end_total_asset,
                state=self.accounting_state,
            )
            final_reward = compute_return_reward(
                begin_total_asset,
                end_total_asset,
                epsilon=self.epsilon,
            )
            return float(final_reward), {
                "raw_return": float(final_reward),
                "final_reward": float(final_reward),
                "lambda_value": self.lambda_value,
            }

        result = compute_mda_reward(
            begin_total_asset,
            end_total_asset,
            lambda_value=self.lambda_value,
            state=self.accounting_state,
        )
        self.last_result = result
        fields = result.as_dict()
        if self.enable_component_normalization:
            normalized_return, normalized_cost = self.component_normalizer.normalize(
                result.raw_return, result.c_t
            )
            final_reward = normalized_return - self.lambda_value * normalized_cost
            fields["normalized_return"] = normalized_return
            fields["normalized_mdd_cost"] = normalized_cost
        else:
            final_reward = result.raw_return - self.lambda_value * result.c_t
        fields["final_reward"] = final_reward
        return final_reward, fields


class SinglePPOStockTradingEnv(StockTradingEnv if StockTradingEnv is not None else object):  # type: ignore[misc,valid-type]
    """Local StockTradingEnv variant for Return-PPO and MDA-PPO.

    With ``use_risk_observation=True``, the returned state is the FinRL base
    state plus the fields in :data:`RISK_OBSERVATION_SCHEMA`. Reward accounting
    always advances on real transitions, independently of policy visibility.
    """

    def __init__(
        self,
        *args: Any,
        reward_mode: str,
        lambda_value: float = 0.0,
        clip_c: float = DEFAULT_CLIP_C,
        epsilon: float = DEFAULT_EPSILON,
        diagnostics_buffer: Any | None = None,
        reward_accounting_state: RewardAccountingState | None = None,
        expected_continuity_state: dict[str, object] | None = None,
        component_normalization_state: Mapping[str, float] | None = None,
        component_normalization_training: bool = False,
        enable_component_normalization: bool = False,
        use_risk_observation: bool = False,
        **kwargs: Any,
    ) -> None:
        if StockTradingEnv is None:
            raise ImportError("FinRL StockTradingEnv is required for SinglePPOStockTradingEnv")
        if not isinstance(use_risk_observation, (bool, np.bool_)):
            raise TypeError("use_risk_observation must be a boolean")
        self.use_risk_observation = bool(use_risk_observation)
        
        stock_dim = int(kwargs["stock_dim"])

        if "num_stock_shares" not in kwargs:
            kwargs["num_stock_shares"] = [0] * stock_dim

        if isinstance(kwargs.get("buy_cost_pct"), (int, float)):
            kwargs["buy_cost_pct"] = [float(kwargs["buy_cost_pct"])] * stock_dim

        if isinstance(kwargs.get("sell_cost_pct"), (int, float)):
            kwargs["sell_cost_pct"] = [float(kwargs["sell_cost_pct"])] * stock_dim

        super().__init__(*args, **kwargs)
        self._base_state_space = len(self.state)
        self._n_trade_days = len(self.df.index.unique())
        self._n_tics = len(self.df.tic.unique())
        self.reward_adapter = RewardModeAdapter(
            reward_mode=reward_mode,
            lambda_value=lambda_value,
            epsilon=epsilon,
            clip_c=clip_c,
            component_normalization_state=component_normalization_state,
            component_normalization_training=component_normalization_training,
            enable_component_normalization=enable_component_normalization,
        )
        self.reward_mode = reward_mode
        self.lambda_value = float(lambda_value)
        self.clip_c = float(clip_c)
        self.reward_epsilon = float(epsilon)
        self.diagnostics_buffer = diagnostics_buffer
        self._expected_continuity_state = expected_continuity_state
        self._initial_reward_accounting_state = reward_accounting_state.copy() if reward_accounting_state else None
        # FinRL initialises ``asset_memory`` from ``initial_amount`` even for
        # ``initial=False``.  The actual carried portfolio value is encoded in
        # the freshly initiated state and is the only valid reset value here.
        self.reward_adapter.reset(
            self._portfolio_value(list(self.state)),
            carried_state=self._initial_reward_accounting_state,
        )
        self._base_state = list(self.state)
        self._set_policy_observation()
        # The policy-visible observation may evolve independently of FinRL's
        # constructor argument. Derive its space from the actual state.
        self.state_space = len(self.state)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_space,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        state, info = super().reset(seed=seed, options=options)
        self._base_state = list(state)
        if self._expected_continuity_state is not None:
            self._assert_account_continuity(self._expected_continuity_state)
        self.reward_adapter.reset(self.asset_memory[0], carried_state=self._initial_reward_accounting_state)
        self._set_policy_observation()
        return self.state, info

    def get_reward_accounting_state(self) -> RewardAccountingState | None:
        state = self.reward_adapter.accounting_state
        return None if state is None else state.copy()

    def get_reward_component_normalization_state(self) -> dict[str, float]:
        return self.reward_adapter.component_normalizer.as_dict()


    def get_account_continuity_state(self) -> dict[str, object]:
        base_state = self._current_base_state()
        return {
            "date": str(self._get_date()),
            "cash": float(base_state[0]),
            "holdings": tuple(float(value) for value in base_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]),
            "portfolio_value": self._portfolio_value(base_state),
        }

    def get_base_state(self) -> list[float]:
        """Return only the FinRL-compatible state used for portfolio carry-over."""

        return list(self._current_base_state())

    def get_risk_observation_fields(self) -> tuple[str, ...]:
        """Return the ordered risk feature names, whether visible or ablated."""

        return RISK_OBSERVATION_FIELDS

    def get_risk_observation_values(self) -> list[float]:
        """Return current risk values without changing accounting state."""

        accounting_state = self.reward_adapter.accounting_state
        if accounting_state is None:
            raise RuntimeError("risk accounting state is not initialized")
        return risk_observation_values(
            self._current_base_state(),
            stock_dim=self.stock_dim,
            accounting_state=accounting_state,
        )

    def get_observation_metadata(self) -> dict[str, object]:
        """Return the actual policy observation contract for audit output."""

        risk_fields = self.get_risk_observation_fields()
        risk_dim = len(risk_fields)
        expected_policy_dim = self._base_state_space
        if self.use_risk_observation:
            expected_policy_dim += risk_dim
        policy_dim = int(self.observation_space.shape[0])
        if policy_dim != expected_policy_dim:
            raise RuntimeError(
                "policy observation space is inconsistent with the risk schema"
            )
        return {
            "use_risk_observation": self.use_risk_observation,
            "base_observation_dim": self._base_state_space,
            "risk_observation_dim": risk_dim,
            "risk_observation_fields": list(risk_fields),
            "policy_observation_dim": policy_dim,
        }

    def step(self, actions):
        self.state = self._current_base_state()
        self.terminal = self.day >= self._n_trade_days - 1
        if self.terminal:
            self._set_policy_observation()
            return self.state, self.reward, self.terminal, False, {}

        self._initial_reward_accounting_state = None
        
        actions = actions * self.hmax
        actions = actions.astype(int)
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            actions = np.array([-self.hmax] * self.stock_dim)

        begin_total_asset = self._portfolio_value()

        argsort_actions = np.argsort(actions)
        n_sell = np.count_nonzero(actions < 0)
        n_buy = np.count_nonzero(actions > 0)
        sell_index = argsort_actions[:n_sell]
        buy_index = argsort_actions[-n_buy:][::-1] if n_buy > 0 else argsort_actions[:0]

        for index in sell_index:
            actions[index] = self._sell_stock(index, actions[index]) * (-1)

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        self.actions_memory.append(actions)

        # Step 5 overlay hook: 买卖完成后、账户推进前（spec §6.2 侵入性正视）。
        # 默认 no-op —— risk_overlay=None 时行为逐位不变（1200 等价, R2）。
        self._apply_execution_overlay()

        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if self._n_tics == 1:
                self.turbulence = self.data[self.risk_indicator_col]
            elif self._n_tics > 1:
                self.turbulence = self.data[self.risk_indicator_col].values[0]
        self._base_state = list(self._update_state())
        self.state = self._base_state

        end_total_asset = self._portfolio_value()
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        self.reward, diagnostic_fields = self.reward_adapter.compute_reward(
            begin_total_asset=begin_total_asset,
            end_total_asset=end_total_asset,
            original_reward_scaling=self.reward_scaling,
        )
        self._set_policy_observation()
        self.rewards_memory.append(self.reward)
        self.state_memory.append(self.state)
        self._record_reward_diagnostics(
            begin_total_asset=begin_total_asset,
            end_total_asset=end_total_asset,
            fields=diagnostic_fields,
        )

        return self.state, self.reward, self.terminal, False, {}

    def _apply_execution_overlay(self) -> None:
        """执行层覆盖层 hook（spec §6.2）：默认 no-op，零开销。

        子类（DownsideBreadthStockTradingEnv）override 后在触发日投影缩仓；
        调用时机 = 买卖循环完成之后、day += 1 账户推进之前，self.data 仍是
        动作执行当日 d（causal：I_d 在线可得，I_{d+1} 不可用）。
        """

    def _portfolio_value(self, state: list[float] | None = None) -> float:
        values = self._current_base_state() if state is None else state
        prices = np.array(values[1 : (self.stock_dim + 1)])
        shares = np.array(values[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        return float(values[0] + np.sum(prices * shares))

    def _current_base_state(self) -> list[float]:
        return list(self.state[: self._base_state_space])

    # -- Performance: cached _n_tics for the per-step O(n) FinRL base calls
    # (df.tic.unique() builds a hash table over ~1,890 rows every step; see
    # the internal performance notes P0-2). Restored 2026-08-12 after being
    # dropped by the Tail_GAE_v1 rework (4dc2e3b). Both overrides are
    # numerically identical to the FinRL base implementations. --

    def _get_date(self):
        if not hasattr(self, "_n_tics"):
            # init path: _n_tics not yet set, run base logic once (O(n), negligible)
            if len(self.df.tic.unique()) > 1:
                return self.data.date.unique()[0]
            return self.data.date
        if self._n_tics > 1:
            return self.data.date.iloc[0]
        return self.data.date

    def _update_state(self):
        if not hasattr(self, "_n_tics"):
            # init path: fall back to base logic for one-shot call
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.state[0]]
                    + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
            return state

        if self._n_tics > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + [
                    v
                    for tech in self.tech_indicator_list
                    for v in self.data[tech].values.tolist()
                ]
            )
        else:
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + [self.data[tech] for tech in self.tech_indicator_list]
            )
        return state

    # -----------------------------------------------------------------

    # The FinRL parent's _sell_stock/_buy_stock gate on a "tradeable flag" at
    # state[index + 2*stock_dim + 1] — a slot that only exists when the tech
    # block is present. With tech_indicator_list=[] (V2 decision S7) any
    # nonzero action IndexErrors. The check is vestigial: with the tech block
    # present the slot holds a float that is never True, so it never blocks a
    # trade. These overrides drop it; behavior is identical for 1200.

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + self.stock_dim + 1] > 0:
                sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                self.trades += 1
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct[index])
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct[index]
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            available_amount = self.state[0] // (
                self.state[index + 1] * (1 + self.buy_cost_pct[index])
            )
            buy_num_shares = min(available_amount, action)
            buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct[index])
            self.state[0] -= buy_amount
            self.state[index + self.stock_dim + 1] += buy_num_shares
            self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
            self.trades += 1
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0

        return buy_num_shares

    def _set_policy_observation(self) -> None:
        accounting_state = self.reward_adapter.accounting_state
        if accounting_state is None:
            raise RuntimeError("risk accounting state must exist before building observation")
        base_state = self._current_base_state() if hasattr(self, "_base_state_space") else list(self.state)
        self._base_state = list(base_state)
        if self.use_risk_observation:
            self.state = append_risk_observation(
                base_state,
                stock_dim=self.stock_dim,
                accounting_state=accounting_state,
            )
        else:
            self.state = list(base_state)

    def _assert_account_continuity(self, expected: Mapping[str, object]) -> None:
        actual = self.get_account_continuity_state()
        if str(actual["date"]) != str(expected.get("date")):
            raise ValueError("trade window date is not continuous with the prior window")
        if not isclose(float(actual["cash"]), float(expected.get("cash")), rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("trade window cash is not continuous with the prior window")
        if tuple(actual["holdings"]) != tuple(expected.get("holdings", ())):
            raise ValueError("trade window holdings are not continuous with the prior window")
        if not isclose(float(actual["portfolio_value"]), float(expected.get("portfolio_value")), rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError("trade window portfolio value is not continuous with the prior window")

    def _record_reward_diagnostics(
        self,
        *,
        begin_total_asset: float,
        end_total_asset: float,
        fields: dict[str, float],
    ) -> None:
        if self.diagnostics_buffer is None:
            return
        record = {
            "V_t": float(begin_total_asset),
            "V_t_plus_1": float(end_total_asset),
            "reward_mode": self.reward_mode,
            **fields,
        }
        if hasattr(self.diagnostics_buffer, "record_step"):
            self.diagnostics_buffer.record_step(record)
