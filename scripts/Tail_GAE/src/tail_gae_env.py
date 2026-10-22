"""V2 environment base: pins reward mode to pure "return".

Thin subclass of ``SinglePPOStockTradingEnv``. The legacy DSB interface
parameters (``signal_table``, ``lambda_breadth``, ``use_breadth_observation``)
are accepted for backward compatibility and ignored.

Step 5 (spec §6): ``risk_overlay`` execution-layer overlay — a deterministic
projection applied inside ``_apply_execution_overlay()`` on trigger days
(I_t == 1). It never runs during training (``mode == "train"`` no-op), so
``risk_overlay=None`` keeps the 1200 behavior bit-identical.
"""

import numpy as np

try:
    from .single_ppo_env import (
        RISK_OBSERVATION_FIELDS,
        SinglePPOStockTradingEnv,
        get_risk_observation_dim,
    )
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from single_ppo_env import (
        RISK_OBSERVATION_FIELDS,
        SinglePPOStockTradingEnv,
        get_risk_observation_dim,
    )


RETURN_PPO = "return"
SUPPORTED_EXPERIMENTS = ("1200", "1205", "1207", "1210", "1215", "1216", "1217", "1218")
DOWNSIDE_BREADTH_PILOT = "downside_breadth_pilot"

# 覆盖层投影基准 ref（spec §6.2：A 目标 GE=ref；B 每股 cap=ref/n_stocks）
OVERLAY_REF = 0.85
SUPPORTED_OVERLAYS = (None, "proportional", "per_stock")


def _overlay_sell_quantities(
    *,
    prices: np.ndarray,
    shares: np.ndarray,
    value: float,
    mode: str,
    ref: float = OVERLAY_REF,
) -> np.ndarray:
    """投影应卖股数（纯数学，无交易成本，U1a/U4 直接可测）。

    A（proportional）：GE > ref 时同比例缩仓 w_i ← w_i·(ref/GE)，保持相对
    权重——目标股数 = floor(s_i·ref/GE)（向下取整，不超卖）。
    B（per_stock）：每股硬上限 ref/n，目标股数 = floor(ref·V/(n·p_i))。
    """
    weights = prices * shares / value
    if mode == "proportional":
        ge = float(np.sum(weights))
        if ge <= ref:
            return np.zeros(shares.shape, dtype=int)
        scale = ref / ge
        return (shares - np.floor(shares * scale)).astype(int)
    cap = ref / shares.shape[0]
    safe_prices = np.where(prices > 0, prices, np.inf)  # 0 价股目标 0 股
    target = np.floor(cap * value / safe_prices)
    return np.maximum(0, shares - target).astype(int)


class DownsideBreadthStockTradingEnv(SinglePPOStockTradingEnv):
    """V2 environment base pinning reward mode to pure "return"."""

    def __init__(
        self,
        df,
        signal_table=None,
        lambda_breadth=0.0,
        use_breadth_observation=False,
        reward_mode="return",
        use_risk_observation=False,
        risk_overlay=None,
        overlay_i_by_date=None,
        **kwargs,
    ):
        if reward_mode != "return":
            raise ValueError(
                f"V2 pins reward_mode='return', got {reward_mode!r}"
            )
        assert use_risk_observation is False, (
            "V2 risk state comes from RiskStateEnvWrapper; "
            "base-class use_risk_observation must stay False"
        )
        if risk_overlay not in SUPPORTED_OVERLAYS:
            raise ValueError(
                f"unsupported risk_overlay={risk_overlay!r}; "
                f"expected one of {SUPPORTED_OVERLAYS}"
            )
        super().__init__(
            df,
            reward_mode=reward_mode,
            use_risk_observation=False,
            **kwargs,
        )
        self.risk_overlay = risk_overlay
        # 触发源: date -> I_t（causal，I_d = 动作执行当日可得；None 默认不执行）
        self.overlay_i_by_date = dict(overlay_i_by_date or {})
        # C3 验收口径: hook 内记录 (date, GE_before, GE_after, w_max_after)
        self.overlay_log: list[tuple[str, float, float, float]] = []

    def _apply_execution_overlay(self) -> None:
        """Step 5 覆盖层（spec §6.2）：触发日按投影缩仓，只作用于执行层。

        调用点 = 买卖完成后、day += 1 前，self.data 仍是动作执行当日 d，
        查 overlay_i_by_date[date_d] = I_d（causal，非训练中的 hindsight I_{d+1}）。
        GE 用原生 state（shares·prices/V，非仿射）；卖出复用 _sell_stock
        （扣 sell_cost_pct、更新 cost/trades、现金实增 p·q·(1−c_s)）。
        """
        if self.risk_overlay is None:
            return
        if self.mode not in ("trade", "validation"):  # 训练期不作用
            return
        if not self.overlay_i_by_date:
            return
        if int(self.overlay_i_by_date.get(self._get_date(), 0)) != 1:
            return
        state = self._current_base_state()
        prices = np.asarray(state[1 : 1 + self.stock_dim], dtype=float)
        shares = np.asarray(state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=float)
        value = float(state[0] + np.sum(prices * shares))
        if value <= 0.0:
            return
        ge_before = float(np.sum(prices * shares / value))
        to_sell = _overlay_sell_quantities(
            prices=prices,
            shares=shares,
            value=value,
            mode=self.risk_overlay,
        )
        if not np.any(to_sell > 0):
            return
        for index in np.flatnonzero(to_sell):
            self._sell_stock(int(index), -int(to_sell[index]))
        state = self._current_base_state()
        prices = np.asarray(state[1 : 1 + self.stock_dim], dtype=float)
        shares = np.asarray(state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=float)
        value = float(state[0] + np.sum(prices * shares))
        w_after = prices * shares / value
        self.overlay_log.append(
            (
                str(self._get_date()),
                ge_before,
                float(np.sum(w_after)),
                float(np.max(w_after)),
            )
        )
