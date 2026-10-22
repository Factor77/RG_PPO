"""Risk-state observation wrapper for Tail_GAE V2 (spec 5.6).

Wraps a base trading env, appends a registry-driven risk block to the base
observation, and exposes the frozen per-step R_t/C_t/I_t protocol consumed by
TailGAERolloutBuffer: histories ``r_history``/``c_history``/``i_history``/
``date_history`` (one element per step(), persisting across episode
boundaries) plus current-step attributes ``last_R_t``/``last_C_t``/
``last_I_t``/``last_date``.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

try:
    from gymnasium import Env, spaces
except ImportError:
    from gym import Env, spaces

try:
    from .risk_oracle import _extract_gross_exposure, get_risk_dims
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from risk_oracle import _extract_gross_exposure, get_risk_dims


# portfolio_window formulas receive a tail of the live return buffer; ar1_20d
# needs one extra point for the lag-1 pairing, streak40 its own window.
_WINDOW_TAILS = {
    "downside_volatility_20": 20,
    "streak20max": 20,
    "ar1_20d": 21,
    "cumret20": 20,
    "streak40": 40,
}


class RiskStateEnvWrapper(Env):
    """Assemble the risk observation block and the R_t/C_t/I_t protocol.

    Subclasses gymnasium.Env so SB3's DummyVecEnv accepts it directly.
    """

    def __init__(self, env, names, *, i_by_date, worst20_by_date,
                 ew_by_date=None, ge_penalty_beta=0.0, ge_penalty_ref=0.95,
                 ge_penalty_mode="total", direct_penalty_ref=None,
                 direct_ref_exec_aligned=False, policy_names=None,
                 supervision_mode="legacy", supervision_weight_by_date=None,
                 phase9_by_date=None, use_phase9_observation=False,
                 phase9_obs_by_date=None):
        super().__init__()
        self.env = env
        self.names = list(names)
        self.policy_names = self.names if policy_names is None else list(policy_names)
        unknown_policy_names = [name for name in self.policy_names if name not in self.names]
        if unknown_policy_names:
            raise ValueError(
                "policy risk fields must be a subset of internal risk fields; "
                f"unknown={unknown_policy_names}"
            )
        self._policy_indices = [self.names.index(name) for name in self.policy_names]
        # V4.2: per-date latch indicator (market EW trigger, spec §3/§5),
        # precomputed by tail_gae_data.load_tail_data over the full series.
        self.i_by_date = dict(i_by_date)
        self.worst20_by_date = worst20_by_date
        # 1216: market EW cumulative-return lookup (name -> date -> value),
        # same causal calendar as worst20_by_date (tail_gae_data._ew_cum_dicts).
        self.ew_by_date = (
            {name: dict(by_date) for name, by_date in ew_by_date.items()}
            if ew_by_date
            else {}
        )
        # 1218: GE penalty (spec §1.1 环节④) -- reward is reduced by
        # beta * max(0, GE - ref) on trigger days (I_t == 1); GE is read
        # post-action with the same formula as the gross_exposure obs dim.
        # V4.4/1219: ge_penalty_mode "per_stock" replaces the total tax with a
        # per-stock tax beta * sum_i max(0, |w_i| - ref/30); "total" (default,
        # 1218) is unchanged.  Unknown modes fall back to "total".
        self.ge_penalty_beta = float(ge_penalty_beta)
        self.ge_penalty_ref = float(ge_penalty_ref)
        self.ge_penalty_mode = ge_penalty_mode
        self.last_ge_penalty = None
        self.ge_penalty_history = []
        # V4.5/1230: direct-gradient direction vector (spec §3.1).  None = 关闭
        # （1218/1219 等既有实验不计算）; 仅直通配置（beta != 0）时 runner 传入.
        self.direct_penalty_ref = direct_penalty_ref
        # V4.6 Step 4（spec §5.1）: d_i 执行对齐口径开关——True 时 d_i 用成交瞬间
        # 暴露（实际 holdings × pre-action 原成交价格估值）; 默认 False = 历史
        # post-action 新价格口径逐位不变（1230/1231/1232/3B 历史数据不受影响）.
        self.direct_ref_exec_aligned = bool(direct_ref_exec_aligned)
        # Phase 6（Trajectory-aware Direct Penalty §9.1/§9.2）: 连续监督权重
        # q_t^loss。legacy = I_t（默认, 与既有行为逐位一致）; trajectory =
        # 预生成 date→w_t 查表（模式互斥, 缺表/非法值 fail fast）. i_by_date 与
        # i_t 协议不受影响（trajectory 不覆盖 i_t）.
        if supervision_mode not in ("legacy", "trajectory"):
            raise ValueError(
                f"invalid supervision_mode={supervision_mode!r}; "
                "expected 'legacy' or 'trajectory'"
            )
        self.supervision_mode = supervision_mode
        self.supervision_weight_by_date = None
        if supervision_mode == "trajectory":
            table = dict(supervision_weight_by_date or {})
            if not table:
                raise ValueError(
                    "supervision_mode='trajectory' requires a non-empty "
                    "supervision_weight_by_date (date -> weight) table"
                )
            for d, w in table.items():
                if not np.isfinite(w) or not (0.0 <= w <= 1.0):
                    raise ValueError(
                        f"supervision weight out of range for {d!r}: {w!r} "
                        "(require finite value in [0, 1])"
                    )
            self.supervision_weight_by_date = table
        # Phase 9（contract §1.3/§1.4）: 当前 rolling window 的 gated 日历切片
        # （date -> row dict）。由 runner 按 window 取表后传入, 禁止跨窗口压平。
        # 存在时进入 Phase 9 模式: 监督权重按 transition_date（决策日）查表,
        # 修复 1305 的 step 后 date_memory[-1] 一日错位（§1.4）; 不存在时
        # 遗留路径逐位不变.
        self.phase9_by_date = None
        if phase9_by_date is not None:
            table9 = dict(phase9_by_date)
            if not table9:
                raise ValueError("phase9_by_date requires a non-empty table")
            required = ("w_down", "w_up", "up_days_remaining", "episode_age", "since_episode_end")
            for d, row in table9.items():
                missing = [k for k in required if k not in row]
                if missing:
                    raise ValueError(f"phase9 row for {d!r} missing {missing}")
            self.phase9_by_date = table9
        # Phase 9 D-observation (contract §2.1): 三个追加观测字段——r_t 恒等、
        # a_t/s_t 截断 [0,20] 后除以 20. 观测表与 loss 权重表解耦:
        # phase9_obs_by_date 始终承载 treatment 的真实状态机列（§1.3/§3.6:
        # 1324 dose-control 只置换 loss 权重, 观测不置换）, 每行含
        # rf20_descriptor/episode_age/since_episode_end. True 时两表都必须
        # 提供（缺表 fail fast）; False 时观测空间与 _assemble 输出逐位不变
        # （1320-1322 保持 301 维, M2）.
        self.use_phase9_observation = bool(use_phase9_observation)
        self.n_phase9_obs_dims = 0
        self.phase9_obs_by_date = None
        if self.use_phase9_observation:
            if self.phase9_by_date is None:
                raise ValueError(
                    "use_phase9_observation=True requires a non-empty "
                    "phase9_by_date loss-weight table"
                )
            obs9 = {str(d): dict(row) for d, row in dict(phase9_obs_by_date or {}).items()}
            if not obs9:
                raise ValueError(
                    "use_phase9_observation=True requires a non-empty "
                    "phase9_obs_by_date (date -> row) table"
                )
            required9 = ("rf20_descriptor", "episode_age", "since_episode_end")
            for d, row in obs9.items():
                missing = [k for k in required9 if k not in row]
                if missing:
                    raise ValueError(f"phase9 obs row for {d!r} missing {missing}")
                r_t = float(row["rf20_descriptor"])
                if not np.isfinite(r_t) or not (0.0 <= r_t <= 1.0):
                    raise ValueError(
                        f"rf20_descriptor out of range for {d!r}: {r_t!r} "
                        "(require finite value in [0, 1])"
                    )
            self.phase9_obs_by_date = obs9
            self.n_phase9_obs_dims = 3
        self.last_w_down = 0.0
        self.last_w_up = 0.0
        self.last_up_days_remaining = 0
        self.last_episode_age = 0
        self.last_since_episode_end = 0
        self.last_pre_cash = 0.0
        self.last_pre_value = 0.0
        self.last_pre_prices = None
        self.last_pre_holdings = None
        # Phase 9 M7 (contract §3.4): 成交瞬间组合 GE（pre 价格 × post 持仓）.
        self.last_real_exec_ge = 0.0
        self.last_transition_date = None
        self.last_supervision_weight = 0.0
        self.stock_dim = int(self.env.stock_dim)
        # V4.6 Step 3C/Step 4: 新协议字段（buffer 列数据源, spec §4.3/§5.1）.
        # last_constraint_cost = c_t = I_t·Σ max(0, w_i − w_ref)（每股阈值）;
        # last_mu_ref = 状态依赖执行对齐达标动作参考（pre-action 口径, 双向有符号）.
        # 直通配置未激活（direct_penalty_ref=None）时恒零, 既有实验行为不变.
        self.last_constraint_cost = 0.0
        self.last_mu_ref = np.zeros(self.stock_dim, dtype=float)

        self._dims = get_risk_dims(self.names)  # (kind, formula, transform) per name
        assert len(self._dims) == len(self.names)
        self.n_internal_risk_dims = len(self.names)
        self.n_risk_dims = len(self.policy_names)

        base_space = env.observation_space
        self.base_dim = int(base_space.shape[0])
        self.action_space = env.action_space
        self.last_direct_grad = np.zeros(self.stock_dim, dtype=int)
        # μ_ref 解析用整股化除数（env.hmax, 与动作 ×hmax 同一语义; 缺省 100）
        self.hmax = float(getattr(env, "hmax", 100))
        low = float(np.asarray(base_space.low).reshape(-1)[0])
        high = float(np.asarray(base_space.high).reshape(-1)[0])
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(self.base_dim + self.n_risk_dims + self.n_phase9_obs_dims,),
            dtype=base_space.dtype,
        )
        assert self.observation_space.shape[0] == (
            self.base_dim + self.n_risk_dims + self.n_phase9_obs_dims
        )

        # Sole live buffer: never cleared by reset() (portfolio value is
        # continuous across trade windows).
        # Primed from the accounting state so portfolio_window features
        # (downside_volatility_20 / streak20max / ar1_20d / cumret20 /
        # streak40) stay continuous across consecutive trade segments
        # (spec 3.2-5 / P1-7): reward_adapter.accounting_state
        # .recent_portfolio_returns is the last up-to-20 realized portfolio
        # returns carried over the window boundary (mda_reward.py:15-18, that
        # field's design intent is exactly this continuity).  A prefill shorter
        # than 40 days means streak40 remains a partial window for the first 40
        # steps of the segment -- accepted, trade segments are deterministic
        # backtests.
        # NOTE (spec 3.2-6 / P1-8): episode reset() below does not clear this
        # buffer, so C_t for the first 10-40 steps of a new training episode
        # mixes in the previous episode's returns (~1.3% of steps).  Accepted
        # and recorded: window 0 behaves exactly like the V2 single-segment
        # version, keeping comparisons valid.
        self._port_rets = list(
            self.env.reward_adapter.accounting_state.recent_portfolio_returns
        )
        self.c_history = []
        self.i_history = []
        self.date_history = []
        self.last_R_t = None
        self.last_C_t = None
        self.last_I_t = None
        self.last_date = None
        # drawdown_delta = current_drawdown minus the previous step's; None on
        # the first step of an episode yields 0. Cleared at reset() so a
        # boundary never leaks the previous episode's drawdown into the delta.
        self._prev_drawdown = None
        self._risk_block_history = []  # post-transform blocks, one per step
        # Keep the internal-only path total even before the first real
        # _assemble() call.  Production resets populate this immediately, but
        # direct step-level tests and duck-typed envs may enter step() first.
        self._last_internal_risk_block = [0.0] * self.n_internal_risk_dims

        # --- V4 diag layer (spec V4 §3.1) ---
        # G1a per-step CSV 句柄: open 时记录原始路径, close 时 gzip 需要 (§3.1.5).
        self._per_step_fh = None        # IO 句柄
        self._per_step_csv = None       # csv.writer
        # V4.13 P0（spec Part 7.1）: 方向标签逐 step 审计文件（date, I_t,
        # over_1..30, d_1..30）——修复后的标签生命周期逐行可查
        self._label_fh = None
        self._label_csv = None
        self._label_path: Path | None = None
        self._per_step_path: Path | None = None   # open 时记录原始路径, close 时 gzip 需要
        self._diag_csv_path: Path | None = None   # G2 统计 CSV 路径 (§3.1.3)
        # 市场等权日收益查表 (§3.1.4): 按 date 对个股 close pct_change 求均值,
        # 构造时预计算一次, 零实时计算; df 首日 pct_change 为 NaN → 该日期
        # mkt_ret 为 NaN, 属正常, 分析层 dropna 处理.
        df = self.env.df.copy()
        df["r"] = df.groupby("tic")["close"].pct_change()
        self._mkt_ret_by_date = df.groupby("date")["r"].mean().to_dict()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def __getattr__(self, name):
        """Delegate any missing attribute to the wrapped env (spec 3.2-3).

        Covers all five env_method names -- save_asset_memory /
        save_action_memory / get_base_state / get_reward_accounting_state /
        get_account_continuity_state -- and attribute passthrough (e.g. ``df``
        for trade-segment length checks).  FinRL's DRL_prediction calls both
        save_asset_memory and save_action_memory; missing either is an
        AttributeError (spec P1-3).  The ``env`` lookup goes through
        ``__dict__`` so pre-__init__ attribute access fails cleanly instead of
        recursing.
        """
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def get_observation_metadata(self) -> dict[str, object]:
        """Policy observation contract for audit output (spec 3.2-4).

        ``base_observation_dim`` derives from ``self.base_dim`` (no hardcoded
        301, spec P2-1).  In the rolling flow the runner registers the contract
        on the bare base env before the wrapper is created, so this method
        serves direct tests and standalone use; both derivations agree (base_dim
        + declared names, spec P1-13).
        """
        return {
            "use_risk_observation": bool(self.policy_names),
            "base_observation_dim": self.base_dim,
            "risk_observation_dim": self.n_risk_dims,
            "risk_observation_fields": list(self.policy_names),
            "phase9_observation_dim": self.n_phase9_obs_dims,
            "phase9_observation_fields": (
                ["rf20_descriptor", "episode_age", "since_episode_end"]
                if self.use_phase9_observation
                else []
            ),
            "policy_observation_dim": (
                self.base_dim + self.n_risk_dims + self.n_phase9_obs_dims
            ),
        }

    @property
    def r_history(self):
        return self._port_rets

    def reset(self, *, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self._prev_drawdown = None
        return self._assemble(state, track_drawdown=False), info

    def step(self, actions):
        # V4.6 Step 4（spec §5.1）: env.step(actions) 之前捕获 pre-action 状态
        # ——μ_ref 与当前动作 a_t 必须同一 transition（用执行后 shares 算出的
        # 是"下一次还需卖多少", 会错位一步）; V_pre = asset_memory[-1]（pre-action 资产）.
        # 基座 env 暴露 self.state 属性; 测试用 duck-typed env 无此属性时 pre_state=None
        # → μ_ref/d_i^exec 不计算（恒零）, 与既有行为一致.
        # Phase 9 §1.4: transition_date 必须在 env.step() 之前冻结（决策日 t）;
        # 监督权重与 pre-action 账户量全部挂该日期, 不得用 step 后的
        # date_memory[-1]（已是 t+1）.
        pre_state = getattr(self.env, "state", None)
        pre_asset = float(self.env.asset_memory[-1])
        pre_prices = pre_shares = None
        if pre_state is not None:
            pre_prices = np.asarray(pre_state[1 : 1 + self.stock_dim], dtype=float)
            pre_shares = np.asarray(pre_state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=float)
        self.last_transition_date = self.env.date_memory[-1]
        self.last_pre_value = pre_asset
        if pre_state is not None:
            self.last_pre_cash = float(pre_state[0])
            self.last_pre_prices = pre_prices
            self.last_pre_holdings = pre_shares
        state, reward, done, truncated, info = self.env.step(actions)
        # Phase 9 §3.4（含 2026-09-11 口径注记）: real_exec_ge = 成交瞬间组合 GE,
        # 分子用执行前价格 × 执行后持仓（排除下一日价格变化）, 分母统一取
        # 动作前净值 V_pre = pre_asset——与 §3.2 冻结的软执行模型
        # ĜE_post = Σ p·h̃_post / V_pre 同口径, 使 §3.6 的 soft/real 误差只含
        # 执行模型近似误差, 不混入手续费造成的分母差.
        # 无 phase9 表也计算（O(stock_dim), 零成本）; pre_prices=None（duck env
        # 无 state 属性）置 0.0. 与 loss 无梯度, 仅供 buffer 诊断列.
        if pre_prices is not None:
            post_shares_ge = np.asarray(
                state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=float
            )
            self.last_real_exec_ge = (
                float(np.sum(pre_prices * post_shares_ge)) / pre_asset
                if pre_asset > 0.0
                else 0.0
            )
        else:
            self.last_real_exec_ge = 0.0
        asset = self.env.asset_memory
        R_t = asset[-1] / asset[-2] - 1.0
        self._port_rets.append(R_t)
        C_t = float(sum(self._port_rets[-10:]))
        date = self.env.date_memory[-1]
        # V4.2: I_t is a per-date lookup of the precomputed latch series
        # (market EW trigger).  R_t/C_t above remain portfolio diagnostics
        # (G1a log); a missing date fails fast (missing-date lookup is
        # deliberate, same as worst20_by_date in _assemble).
        try:
            I_t = int(self.i_by_date[date])
        except KeyError:
            raise KeyError(
                f"i_by_date has no entry for {date!r}; the latch series is "
                "precomputed over the full market-EW calendar"
            ) from None
        self.c_history.append(C_t)
        self.i_history.append(I_t)
        self.date_history.append(date)
        self.last_R_t = R_t
        self.last_C_t = C_t
        self.last_I_t = I_t
        self.last_date = self.last_transition_date if self.phase9_by_date is not None else date
        # Phase 9（contract §1.4）: w_down/w_up 按决策日 transition_date 查表
        # （缺日期 fail fast）; 兼容列 supervision_weight 承载 w_down（§3.4）.
        # 非 Phase 9 模式 = 遗留路径（step 后 date 查表, 逐位不变）.
        if self.phase9_by_date is not None:
            t_date = self.last_transition_date
            try:
                row9 = self.phase9_by_date[t_date]
            except KeyError:
                raise KeyError(
                    f"phase9_by_date has no entry for transition date {t_date!r}; "
                    "the gated calendar must cover the full trading calendar"
                ) from None
            self.last_w_down = float(row9["w_down"])
            self.last_w_up = float(row9["w_up"])
            self.last_up_days_remaining = int(row9["up_days_remaining"])
            self.last_episode_age = int(row9["episode_age"])
            self.last_since_episode_end = int(row9["since_episode_end"])
            w_t = self.last_w_down
        elif self.supervision_mode == "trajectory":
            try:
                w_t = float(self.supervision_weight_by_date[date])
            except KeyError:
                raise KeyError(
                    f"supervision_weight_by_date has no entry for {date!r}; "
                    "the trajectory weight table must cover the full trading calendar"
                ) from None
        else:
            w_t = float(I_t)
        self.last_supervision_weight = w_t
        # 1218/1219: GE penalty on trigger days, post-action exposure.
        # "total" (1218): beta * max(0, GE - ref), GE same formula as the
        # gross_exposure obs dim (raw GE in [0,1], not the affine form).
        # "per_stock" (1219): beta * sum_i max(0, |w_i| - ref/30), w_i =
        # shares*close/asset, same weight basis as the G1a log.
        if self.ge_penalty_beta != 0.0 and I_t == 1:
            if self.ge_penalty_mode == "per_stock":
                asset = self.env.asset_memory
                # 权重口径与 G1a 日志同款 (w_i = shares*close/asset); 索引用
                # stock_dim 参数化, 生产 30 股时与 state[31:61]*state[1:31] 等价.
                prices = np.asarray(state[1 : 1 + self.stock_dim])
                shares = np.asarray(state[1 + self.stock_dim : 1 + 2 * self.stock_dim])
                w = prices * shares / asset[-1]
                penalty = self.ge_penalty_beta * float(
                    np.maximum(0.0, w - self.ge_penalty_ref / self.stock_dim).sum()
                )
            else:  # "total": 1218 原样
                ge = _extract_gross_exposure(
                    state,
                    self.env.reward_adapter.accounting_state,
                    self.stock_dim,
                )
                penalty = self.ge_penalty_beta * max(0.0, ge - self.ge_penalty_ref)
        else:
            penalty = 0.0
        reward = reward - penalty
        self.last_ge_penalty = penalty
        self.ge_penalty_history.append(penalty)
        obs = self._assemble(state)
        self._risk_block_history.append(list(self._last_internal_risk_block))
        # V4.5/1230: w 单点计算（spec §3.1）——G1a 日志与 last_direct_grad 共用,
        # 两处公式永不漂移. 每步无条件执行（O(30), 可忽略）.
        asset = self.env.asset_memory
        # state 是 Python list（非 ndarray）, 必须先 np.asarray 再运算
        weights = np.asarray(state[1 : 1 + self.stock_dim]) * \
            np.asarray(state[1 + self.stock_dim : 1 + 2 * self.stock_dim]) / asset[-1]
        # V4.6 Step 3C（spec §4.3）: c_t = I_t · Σ_i max(0, w_i − w_ref).
        # w_ref = direct_penalty_ref / stock_dim（每股阈值 = 0.85/30 = 0.0283,
        # 不是 0.85 单股阈值）; 与 d_i 同管线, 直通配置未激活时恒零.
        if self.direct_penalty_ref is not None:
            w_ref = self.direct_penalty_ref / self.stock_dim
            self.last_constraint_cost = float(I_t * np.maximum(0.0, weights - w_ref).sum())
        else:
            w_ref = 0.0
            self.last_constraint_cost = 0.0
        # V4.6 Step 4（spec §5.1）: 状态依赖 μ_ref（pre-action 口径, 双向有符号）.
        # h*_i = floor(w_ref·V_pre / p_i)（整股化）; μ_ref_i = clip((h*−h)/hmax, −1, 1):
        # 超仓需卖 → 负（h=100,h*=80,hmax=100 → −0.2）, 欠仓可买 → 正, 达标 → 0.
        if self.direct_penalty_ref is not None and pre_shares is not None:
            target_shares = np.floor(w_ref * pre_asset / np.maximum(pre_prices, 1e-12))
            self.last_mu_ref[:] = np.clip((target_shares - pre_shares) / self.hmax, -1.0, 1.0)
        else:
            self.last_mu_ref[:] = 0.0
        # V4.13 P0（spec Part 7.1）: 每步先清零方向标签, 再仅对当步 over 股票
        # 写入 ±1——修复历史实现"连续触发段陈旧标签残留"缺陷（触发日→触发日,
        # 股票由超限→未超限时旧 d_i 残留; 历史代码只在 I_t=0 时整体清零）。
        # 理想语义: d_i 恒等于当日超限集合的符号, 与历史触发段路径无关。
        self.last_direct_grad[:] = 0
        if self.last_supervision_weight > 0 and self.direct_penalty_ref is not None:
            if self.direct_ref_exec_aligned and pre_prices is not None:
                # V4.6 Step 4（spec §5.1）: d_i^exec = 成交瞬间暴露——实际 holdings
                # 按原成交价格（pre-action）估值: w^exec_i = h_post_i·p_pre_i / V^exec,
                # V^exec = cash_post + Σ h_post_j·p_pre_j. 历史 d_i（post-action 新
                # 价格）不变; 该口径仅 direct_ref_exec_aligned=True 时启用.
                post_shares = np.asarray(state[1 + self.stock_dim : 1 + 2 * self.stock_dim], dtype=float)
                v_exec = float(state[0]) + float(np.sum(pre_prices * post_shares))
                w_exec = pre_prices * post_shares / v_exec
                over = np.abs(w_exec) > w_ref
                self.last_direct_grad[over] = np.sign(w_exec[over]).astype(int)
            else:
                over = np.abs(weights) > w_ref
                self.last_direct_grad[over] = np.sign(weights[over]).astype(int)
            # V4.13 P0 运行期断言: 未超限股票的方向标签必须为零, 否则该 run 作废
            assert not np.any(self.last_direct_grad[~over] != 0), \
                "P0 assertion failed: d_i non-zero outside the over-threshold mask"
        else:
            over = np.zeros(self.stock_dim, dtype=bool)
        # V4.13 P0: 标签审计行（仅当 open_label_log 已打开时写）
        if self._label_fh is not None:
            self._label_csv.writerow(
                [self.last_date, int(I_t)]
                + [f"{int(o)}" for o in over.tolist()]
                + [f"{int(d)}" for d in self.last_direct_grad.tolist()]
            )
        # G1a: 组合层 per-step CSV (spec V4 §3.1.2)。只追加落盘, 不改 R_t/C_t/I_t
        # 计算与 _risk_block_history 语义; 假交易日末步照写, 由 tail_posthoc 剔除.
        if self._per_step_fh is not None:
            mkt_ret = self._mkt_ret_by_date.get(self.last_date, float("nan"))
            row = [
                self.last_date,
                f"{self.last_R_t:.6f}", f"{self.last_C_t:.6f}", int(self.last_I_t),
            ]
            if self.supervision_mode == "trajectory":
                # Phase 9: 9 位小数以满足 M1 的 EPS=1e-9 比对; 遗留模式保持 %.6f 格式逐位不变.
                row.append(
                    f"{self.last_supervision_weight:.9f}"
                    if self.phase9_by_date is not None
                    else f"{self.last_supervision_weight:.6f}"
                )
            if self.phase9_by_date is not None:
                row += [
                    f"{self.last_w_down:.9f}",
                    f"{self.last_w_up:.9f}",
                    int(self.last_up_days_remaining),
                ]
            row += [
                f"{self.last_ge_penalty:.6f}",
                f"{mkt_ret:.6f}",
                *[f"{float(x):.6f}" for x in self._risk_block_history[-1]],
                f"{asset[-2]:.2f}", f"{asset[-1]:.2f}",   # V_t / V_t_plus_1: %.2f, 其余 %.6f
                *[f"{w:.6f}" for w in weights.tolist()],
            ]
            self._per_step_csv.writerow(row)
        return obs, reward, done, truncated, info

    def print_diagnostics(self, iteration: int):
        """Per-dim mean/std/min/max over all steps so far (spec 3.2-1 / 3.3).

        All feature rows are joined into a single string emitted by one
        ``print(block, flush=True)`` so the block is atomic and cannot be
        interleaved by SB3 logger dumps or pipe buffering.  Driven by an
        external callback; the step() hook was removed (spec 3.2-2).
        """
        if not self._risk_block_history:
            return
        arr = np.asarray(self._risk_block_history, dtype=float)
        block = "\n".join(
            f"[risk_state_env] {name}: mean={arr[:, i].mean():.6f} "
            f"std={arr[:, i].std():.6f} min={arr[:, i].min():.6f} "
            f"max={arr[:, i].max():.6f} (iteration={iteration})"
            for i, name in enumerate(self.names)
        )
        print(block, flush=True)
        # G2: risk state 统计持久化 (spec V4 §3.1.3)。追加模式, 不维护句柄;
        # 表头按"文件不存在"判定, set_diag_csv_path 的重跑防护保证每窗口从 0 行开始.
        if self._diag_csv_path is not None:
            arr = np.asarray(self._risk_block_history, dtype=float)
            row = {"iteration": iteration}
            for i, name in enumerate(self.names):
                row[name] = float(arr[:, i].mean())
                row[f"{name}_std"] = float(arr[:, i].std())
                row[f"{name}_min"] = float(arr[:, i].min())
                row[f"{name}_max"] = float(arr[:, i].max())
            if not self._diag_csv_path.exists():  # 表头按"文件不存在"判定, 换路径重复调用安全
                self._diag_csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._diag_csv_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=list(row.keys())).writeheader()
            with open(self._diag_csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

    # ------------------------------------------------------------------
    # G1a/G2 文件管理 (spec V4 §3.1.1, 由 tail_gae_rolling_runner 接线调用)
    # ------------------------------------------------------------------

    def open_per_step_log(self, path: Path) -> None:
        """Open the G1a per-step CSV (spec V4 §3.1.1 / §3.1.2).

        已打开时先关闭、再以截断模式重开（重开 = 覆盖旧文件）。写表头:
        date, R_t, C_t, I_t, mkt_ret + 风险列(self.names 展开) + V_t, V_t_plus_1
        + w_1..w_30。句柄跨 rollout 保持打开, reset() 不关闭.
        """
        if self._per_step_fh is not None:
            self.close_per_step_log()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._per_step_path = path
        self._per_step_fh = open(path, "w", newline="")
        self._per_step_csv = csv.writer(self._per_step_fh)
        header = ["date", "R_t", "C_t", "I_t"]
        if self.supervision_mode == "trajectory":
            header.append("supervision_weight")
        if self.phase9_by_date is not None:
            header += ["w_down", "w_up", "up_days_remaining"]
        header += (
            ["ge_penalty", "mkt_ret"]
            + list(self.names)
            + ["V_t", "V_t_plus_1"]
            + [f"w_{i}" for i in range(1, 31)]
        )
        self._per_step_csv.writerow(header)

    def close_per_step_log(self) -> None:
        """Close the G1a CSV and gzip it (spec V4 §3.1.1 / §3.1.5).

        幂等, 句柄为 None 时 no-op。关闭后对原始 CSV gzip 压缩（level 6）并删除
        原文件; 压缩失败/文件不存在时 no-op, 保留原始 CSV（仍可读）。内部吞异常,
        防止 finally 中的异常掩盖原始 CUDA OOM 等错误.
        """
        fh, self._per_step_fh = self._per_step_fh, None
        self._per_step_csv = None
        if fh is None:
            return
        try:
            fh.close()
        except Exception:
            pass  # 磁盘满等 close 刷缓冲失败: 句柄由 GC 回收, 不掩盖原始错误
        path, self._per_step_path = self._per_step_path, None
        if path is None or not path.exists():
            return
        try:
            with open(path, "rb") as f_in, gzip.open(
                f"{path}.gz", "wb", compresslevel=6
            ) as f_out:
                while True:
                    chunk = f_in.read(1 << 16)
                    if not chunk:
                        break
                    f_out.write(chunk)
            path.unlink()  # 压缩成功后删除原文件
        except Exception:
            pass  # 压缩失败（磁盘满/IO 错）: no-op, 保留原始 CSV

    def flush_per_step_log(self) -> None:
        """Flush the G1a CSV OS buffer (spec V4 §3.1.2, 供 rollout 边界调用)."""
        if self._per_step_fh is not None:
            try:
                self._per_step_fh.flush()
            except Exception:
                pass

    def open_label_log(self, path: Path) -> None:
        """V4.13 P0（spec Part 7.1）: 打开方向标签审计 CSV。

        表头: date, I_t, over_1..over_30, d_1..d_30。纯 CSV（不 gzip, 无历史读者）;
        已打开时先关闭再截断重开。
        """
        if self._label_fh is not None:
            self.close_label_log()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._label_path = path
        self._label_fh = open(path, "w", newline="")
        self._label_csv = csv.writer(self._label_fh)
        self._label_csv.writerow(
            ["date", "I_t"]
            + [f"over_{i}" for i in range(1, self.stock_dim + 1)]
            + [f"d_{i}" for i in range(1, self.stock_dim + 1)]
        )

    def close_label_log(self) -> None:
        """V4.13 P0: 关闭标签审计 CSV（幂等）。"""
        fh, self._label_fh = self._label_fh, None
        self._label_csv = None
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass

    def set_diag_csv_path(self, path: Path) -> None:
        """Set the G2 risk-state diag CSV path (spec V4 §3.1.3).

        同路径重复设置（重跑同一窗口）时先删除旧文件, 防追加式写入累积旧行.
        """
        path = Path(path)
        if path.exists():
            path.unlink()
        self._diag_csv_path = path

    def _assemble(self, base_state, *, track_drawdown=True):
        accounting = self.env.reward_adapter.accounting_state
        date = self.env.date_memory[-1]
        block = []
        for name, (kind, formula, (a, b)) in zip(self.names, self._dims):
            if kind == "accounting":
                if name == "drawdown_delta":
                    current = float(accounting.current_drawdown)
                    v = 0.0 if self._prev_drawdown is None else current - self._prev_drawdown
                    if track_drawdown:
                        self._prev_drawdown = current
                else:
                    v = formula(base_state, accounting, self.stock_dim)
            elif kind == "portfolio_window":
                v = formula(self._port_rets[-_WINDOW_TAILS[name]:])
            elif kind == "market_ew_lookup":  # 1216: per-date EW feature
                v = self.ew_by_date[name][date]
            else:  # market_lookup; missing date fails fast
                v = self.worst20_by_date[date]
            block.append(v * a + b)
        self._last_internal_risk_block = block
        obs = list(base_state) + [block[index] for index in self._policy_indices]
        # Phase 9 D-observation (contract §2.1/§1.4): obs_{t+1} 装配 t+1 日的
        # 三个字段——date = date_memory[-1]（step 后即 t+1）, 与 transition_date
        # （决策日 t 的监督量）严格区分. r_t 恒等（init 已校验 [0,1]）;
        # a_t/s_t 截断 [0,20] 后除以 20（日历里 since_episode_end 可达数百,
        # 禁止 0-20 原值直接送入 MLP）. 观测恒读 phase9_obs_by_date（treatment
        # 列）, 即使 loss 表（phase9_by_date）为 dose-control 变体. 缺日期
        # fail fast（同 worst20 查表）, runner 须保证表覆盖窗口末日 + 1 个
        # 交易日（t+1 观测查表）.
        if self.use_phase9_observation:
            row9 = self.phase9_obs_by_date[date]
            r_t = float(row9["rf20_descriptor"])
            a_t = min(max(int(row9["episode_age"]), 0), 20) / 20.0
            s_t = min(max(int(row9["since_episode_end"]), 0), 20) / 20.0
            obs += [r_t, a_t, s_t]
        return obs
