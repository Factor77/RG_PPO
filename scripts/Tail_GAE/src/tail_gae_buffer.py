"""Tail-GAE rollout buffer (spec §3.3 / §5.7).

Overrides only the advantage/return computation of the SB3 RolloutBuffer.
The tail penalty reshapes only the GAE advantage recursion (tilde_delta);
the value target (returns) keeps the standard delta recursion (decision T1).
"""

import csv
import gzip
from collections import namedtuple
from pathlib import Path

import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

# V4.5/1230: 直通项样本类型（spec §3.2）。独立 namedtuple——namedtuple
# 子类的 __new__ 元组长度不变, 继承后再加字段会 TypeError; 字段名与 SB3 6 字段
# 完全一致（old_values/old_log_prob, type_aliases.py）保证 PPO.train() 属性访问兼容.
# V4.6 Step 3C/Step 4（spec §4.3/§5.1）: 追加 constraint_cost（1 维, c_t 样本值）
# 与 mu_ref（30 维, 状态依赖执行对齐达标动作参考）, 与 i_t/d_i 同管线扩展.
TailRolloutBufferSamples = namedtuple(
    "TailRolloutBufferSamples",
    ("observations", "actions", "old_values", "old_log_prob", "advantages",
     "returns", "i_t", "d_i", "constraint_cost", "mu_ref", "supervision_weight",
     "w_down", "w_up", "up_days_remaining", "episode_age", "old_mean_actions",
     "pre_cash", "pre_value", "pre_prices", "pre_holdings", "real_exec_ge"),
)


class TailGAERolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        device="cpu",
        gae_lambda=1.0,
        gamma=0.99,
        n_envs=1,
        tail_alpha=0.0,
        risk_wrapper=None,
        tilt_shift=False,
    ):
        if risk_wrapper is not None:
            assert n_envs == 1, "risk_wrapper protocol is single-env only"
        self.tail_alpha = tail_alpha
        self.risk_wrapper = risk_wrapper
        self.tilt_shift = bool(tilt_shift)
        # V4 G1b: RL 层逐日 CSV 句柄（open_rl_log/close_rl_log 独占管理;
        # reset() 不清——跨 rollout 保持打开续写, spec §3.2.1）
        self._rl_fh = None
        self._rl_csv = None
        self._rl_path = None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        # V4.5/1230: 直通列（spec §3.2）。存储形状遵循 SB3 约定 (buffer, n_envs, dim)
        # ——swap_and_flatten 按 3 维工作; n_envs=1 时 i_t 存 3 维避免形状歧义.
        self.i_t = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.d_i = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.i_t_flat = None
        self.d_i_flat = None
        # V4.6 Step 3C/Step 4（spec §4.3/§5.1）: 新列（同管线）.
        # constraint_cost: 1 维（c_t 标量）; mu_ref: action_dim 维（与 d_i 对齐）.
        # 无 wrapper 或配置关闭时恒零（初始零, add() 不写）.
        self.constraint_cost = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.mu_ref = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.constraint_cost_flat = None
        self.mu_ref_flat = None
        # Phase 6（Trajectory-aware Direct Penalty）: 连续监督权重列 q_t^loss
        # （legacy = I_t / trajectory = w_t, 值由 wrapper 的 last_supervision_weight
        # 提供）, 与 i_t 同管线存储; 无 wrapper 或旧式 duck-typed wrapper 时恒零.
        self.supervision_weight = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.supervision_weight_flat = None
        # Phase 9（contract §3.4）: gated 监督列与 pre-action 账户快照。
        # phase9_active 由 wrapper 是否挂 phase9_by_date 判定; 非 Phase 9
        # 运行全部恒零（getattr 缺省）, 旧实验逐位不变.
        self.phase9_active = bool(
            risk_wrapper is not None and getattr(risk_wrapper, "phase9_by_date", None) is not None
        )
        self.w_down = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.w_up = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.up_days_remaining = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        # Δ1321b（spec §1.1/§6.1）: 风险段龄列（0 = 健康期）, 与 up_days_remaining
        # 同管线; 缺省零——非 Phase 9 wrapper 无该属性, 旧实验逐位不变.
        self.episode_age = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.old_mean_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.pre_cash = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.pre_value = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.pre_prices = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.pre_holdings = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.real_exec_ge = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.w_down_flat = None
        self.w_up_flat = None
        self.up_days_remaining_flat = None
        self.episode_age_flat = None
        self.old_mean_actions_flat = None
        self.pre_cash_flat = None
        self.pre_value_flat = None
        self.pre_prices_flat = None
        self.pre_holdings_flat = None
        self.real_exec_ge_flat = None

    def reset(self):
        super().reset()
        self._i_hist = []
        self._computed_I = None

    def add(self, obs, action, reward, episode_start, value, log_prob):
        pos = self.pos  # SB3 add() 末尾递增 pos, 须在 super() 前取当前位
        super().add(obs, action, reward, episode_start, value, log_prob)
        if self.risk_wrapper is not None:
            self._i_hist.append(self.risk_wrapper.last_I_t)
            self.i_t[pos] = float(self.risk_wrapper.last_I_t)
            self.d_i[pos] = self.risk_wrapper.last_direct_grad
            # V4.6 Step 3C/Step 4: 新列与 i_t/d_i 同索引对齐（同一 add() 位置）;
            # getattr 缺省零——旧式 duck-typed wrapper（无新属性）或配置关闭时恒零.
            self.constraint_cost[pos] = float(getattr(self.risk_wrapper, "last_constraint_cost", 0.0))
            self.mu_ref[pos] = np.asarray(getattr(self.risk_wrapper, "last_mu_ref", 0.0), dtype=np.float32)
            self.supervision_weight[pos] = float(getattr(self.risk_wrapper, "last_supervision_weight", 0.0))
            # Phase 9 §3.4: 新列与 i_t/d_i 同索引对齐（同一 add() 位置）;
            # getattr 缺省零——非 Phase 9 wrapper 恒零.
            self.w_down[pos] = float(getattr(self.risk_wrapper, "last_w_down", 0.0))
            self.w_up[pos] = float(getattr(self.risk_wrapper, "last_w_up", 0.0))
            self.up_days_remaining[pos] = float(getattr(self.risk_wrapper, "last_up_days_remaining", 0))
            self.episode_age[pos] = float(getattr(self.risk_wrapper, "last_episode_age", 0))
            self.pre_cash[pos] = float(getattr(self.risk_wrapper, "last_pre_cash", 0.0))
            self.pre_value[pos] = float(getattr(self.risk_wrapper, "last_pre_value", 0.0))
            prices = getattr(self.risk_wrapper, "last_pre_prices", None)
            holdings = getattr(self.risk_wrapper, "last_pre_holdings", None)
            if prices is not None:
                self.pre_prices[pos] = np.asarray(prices, dtype=np.float32)
            if holdings is not None:
                self.pre_holdings[pos] = np.asarray(holdings, dtype=np.float32)
            # §3.4 real_exec_ge: wrapper 侧分母统一为动作前净值 V_pre（与 §3.2
            # 冻结软执行模型 ĜE_post = Σ p·h̃_post / V_pre 同口径, 差值只剩
            # 执行模型误差本身, 不含手续费造成的分母差）. 缺省零——非 Phase 9
            # wrapper 无该属性, 旧实验逐位不变.
            self.real_exec_ge[pos] = float(getattr(self.risk_wrapper, "last_real_exec_ge", 0.0))
        # --- G1b: RL 层逐日快照（spec §3.2.1）---
        # 守卫只用 _rl_fh——runner 仅在 wrapper 存在时 open_rl_log,
        # bridge 下 _rl_fh 恒 None, 写入块天然不执行
        if self._rl_fh is not None:
            row = [
                self.risk_wrapper.last_date,
                float(value.cpu().flatten()[0]),  # value 是 CUDA tensor: .cpu() 一次到位, 避免 float() 隐式二次 D2H
                float(np.mean(action)), float(episode_start), float(reward),
            ]
            if self._trajectory_supervision():
                row.append(float(getattr(self.risk_wrapper, "last_supervision_weight", 0.0)))
            self._rl_csv.writerow(row)

    def _trajectory_supervision(self) -> bool:
        """Phase 6: 当前 wrapper 是否处于 trajectory 监督模式（legacy 默认 = False）."""
        return bool(getattr(self.risk_wrapper, "supervision_mode", "legacy") == "trajectory")

    def open_rl_log(self, path: Path) -> None:
        """打开 G1b RL 层逐日 CSV（截断模式）; 已打开时先关闭、再重开（覆盖旧文件）。

        Phase 6: trajectory 模式下表头追加 supervision_weight 列; legacy 表头
        与既有输出逐字节一致.
        """
        if self._rl_fh is not None:
            self.close_rl_log()
        self._rl_path = Path(path)
        self._rl_path.parent.mkdir(parents=True, exist_ok=True)
        self._rl_fh = open(self._rl_path, "w", newline="")
        self._rl_csv = csv.writer(self._rl_fh)
        header = ["date", "critic_V_pre", "action_mean", "episode_start", "reward"]
        if self._trajectory_supervision():
            header.append("supervision_weight")
        self._rl_csv.writerow(header)

    def close_rl_log(self) -> None:
        """关闭 G1b 句柄并 gzip 压缩原始 CSV; 幂等, 句柄为 None 时 no-op。

        压缩成功删除原 .csv; 失败（磁盘满/IO 错）时 no-op 保留原始 CSV
        （spec §3.1.5）。异常一律吞掉——避免覆盖 finally 中原始的
        CUDA OOM（spec §4 异常掩盖防护）。
        """
        if self._rl_fh is None:
            return
        try:
            self._rl_fh.close()
        except Exception:
            pass
        self._rl_fh = None
        try:
            raw = self._rl_path
            if raw is not None and raw.exists():
                gz = str(raw) + ".gz"
                with open(raw, "rb") as f_in, gzip.open(gz, "wb", compresslevel=6) as f_out:
                    f_out.write(f_in.read())
                raw.unlink()
        except Exception:
            pass  # 压缩失败 no-op, 保留原始 CSV 仍可读

    def _flat_direct_cols(self):
        """直通列 flat 副本（spec §3.2）——compute_returns_and_advantage 两分支共用,
        rollout 末调用一次, 训练期不变（与 SB3 6 字段的 flat 时机语义等价）."""
        self.i_t_flat = self.swap_and_flatten(self.i_t)
        self.d_i_flat = self.swap_and_flatten(self.d_i)
        # V4.6 Step 3C/Step 4: 新列同管线 flat（constraint_cost 供 λ 的 Ĉ 计算,
        # mu_ref 供达标形式直通项）.
        self.constraint_cost_flat = self.swap_and_flatten(self.constraint_cost)
        self.mu_ref_flat = self.swap_and_flatten(self.mu_ref)
        self.supervision_weight_flat = self.swap_and_flatten(self.supervision_weight)
        self.w_down_flat = self.swap_and_flatten(self.w_down)
        self.w_up_flat = self.swap_and_flatten(self.w_up)
        self.up_days_remaining_flat = self.swap_and_flatten(self.up_days_remaining)
        self.episode_age_flat = self.swap_and_flatten(self.episode_age)
        self.old_mean_actions_flat = self.swap_and_flatten(self.old_mean_actions)
        self.pre_cash_flat = self.swap_and_flatten(self.pre_cash)
        self.pre_value_flat = self.swap_and_flatten(self.pre_value)
        self.pre_prices_flat = self.swap_and_flatten(self.pre_prices)
        self.pre_holdings_flat = self.swap_and_flatten(self.pre_holdings)
        self.real_exec_ge_flat = self.swap_and_flatten(self.real_exec_ge)

    def compute_returns_and_advantage(self, last_values, dones):
        if self.tail_alpha == 0.0:
            super().compute_returns_and_advantage(last_values, dones)
            self._flat_direct_cols()
            return
        last_values = last_values.clone().cpu().numpy().flatten()

        # Partial rollouts (total_timesteps % n_steps != 0): pos < buffer_size
        # and the unwritten tail slots must stay I=0 / standard delta, mirroring
        # the parent's full-buffer loop.
        n_valid = self.buffer_size if self.full else self.pos
        if self.risk_wrapper is not None:
            assert len(self._i_hist) == n_valid, (
                f"per-step tail history ({len(self._i_hist)}) must match "
                f"valid steps ({n_valid})"
            )
            # V4.2: I_t is the wrapper's per-date latch lookup (market EW
            # trigger, spec §3/§5); the recorded value is authoritative, no
            # recompute cross-check exists anymore.
            computed_I = np.zeros(self.buffer_size, dtype=np.int64)
            for step in range(n_valid):
                computed_I[step] = int(np.asarray(self._i_hist[step]).item())
            self._computed_I = computed_I
        else:
            computed_I = np.zeros(self.buffer_size, dtype=np.int64)
            self._computed_I = computed_I

        # --- Tail mechanism diagnostics: per-rollout accumulators ----------
        # Consumed by _DiagnosticsCallback._on_rollout_end via
        # model.rollout_buffer._tail_diag.  n_valid excludes unwritten slots
        # in partial (final) rollouts.
        _diag_sum_d0: float = 0.0
        _diag_sum_d1: float = 0.0
        _diag_sum_t1: float = 0.0
        _diag_n0: int = 0
        _diag_n1: int = 0

        # --- G3: δ 按 current_drawdown 三分位（spec §3.2.2）---
        _diag_delta_dd_lo = _diag_delta_dd_md = _diag_delta_dd_hi = 0.0
        _diag_n_dd_lo = _diag_n_dd_md = _diag_n_dd_hi = 0

        # --- D9: 标准 GAE advantage vs tail advantage ---
        _diag_adv_std_sum = 0.0
        _diag_adv_tail_sum = 0.0

        # --- G4: 动作分布按 I_t ---
        _diag_act_I0_sum = _diag_act_I1_sum = 0.0
        _diag_act_I0_sq = _diag_act_I1_sq = 0.0  # 用于算 std

        # 取当前 rollout 的 risk block; current_drawdown 按名索引（不硬编码第 0 维）
        if self.risk_wrapper is not None:
            _dd_idx = self.risk_wrapper.names.index("current_drawdown")
            _risk_blocks = self.risk_wrapper._risk_block_history[-n_valid:]
            _dd_values = [float(r[_dd_idx]) for r in _risk_blocks]
            _degenerate = len(_dd_values) < 3 or min(_dd_values) == max(_dd_values)
            if not _degenerate:
                _lo, _hi = np.percentile(_dd_values, [33.3, 66.7])
            else:  # 分位退化（如窗口前段 dd 全 0）: 按行索引等频三分, 避免静默全落 hi 桶
                _lo = _hi = 0.0
        else:  # 防御分支（正常不可达: runner 保证 tail_alpha>0 时 wrapper 存在）
            _dd_idx = 0
            _risk_blocks = []
            _degenerate = True
            _lo = _hi = 0.0

        last_gae_lam = 0
        last_gae_lam_ret = 0
        # D9: 第三条递推链（纯标准 δ, 不加 tail 惩罚）
        last_gae_lam_std = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            tilde_delta = delta + self.tail_alpha * np.float32(computed_I[step]) * min(delta, 0.0)
            if step < n_valid:
                if computed_I[step]:
                    _diag_sum_d1 += float(delta)
                    _diag_sum_t1 += float(tilde_delta)
                    _diag_n1 += 1
                else:
                    _diag_sum_d0 += float(delta)
                    _diag_n0 += 1
                # --- G3: δ 按 dd 分位（退化时按行索引三分; spec §3.2.3）---
                if _degenerate:
                    bucket = 2 if step % 3 == 2 else (0 if step % 3 == 0 else 1)
                else:
                    dd_val = float(_risk_blocks[step][_dd_idx])
                    bucket = 0 if dd_val < _lo else (1 if dd_val < _hi else 2)
                if bucket == 0:
                    _diag_delta_dd_lo += float(delta); _diag_n_dd_lo += 1
                elif bucket == 1:
                    _diag_delta_dd_md += float(delta); _diag_n_dd_md += 1
                else:
                    _diag_delta_dd_hi += float(delta); _diag_n_dd_hi += 1

                # --- G4: 动作按 I_t ---
                act = self.actions[step]  # (1, 30) numpy array（n_envs=1 时含前导维）
                act_mean = float(np.mean(act))
                if computed_I[step]:
                    _diag_act_I1_sum += act_mean; _diag_act_I1_sq += act_mean ** 2
                else:
                    _diag_act_I0_sum += act_mean; _diag_act_I0_sq += act_mean ** 2
            last_gae_lam = tilde_delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam_ret = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_ret
            # D9: 纯标准 δ 递推链, 无条件执行（与 last_gae_lam 一致）
            last_gae_lam_std = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam_std
            self.advantages[step] = last_gae_lam
            self.returns[step] = last_gae_lam_ret + self.values[step]
            # --- D9: 累加必须在 chain 更新之后、独立 guard（spec §3.2.4 P0）---
            # 若并入上面 step < n_valid 分支, 执行时 last_gae_lam/last_gae_lam_std
            # 还是 step+1 步的值, adv_tail_mean/adv_std_mean 会系统性错误
            if step < n_valid:
                _diag_adv_std_sum += float(last_gae_lam_std)
                _diag_adv_tail_sum += float(last_gae_lam)

        self._tail_diag = {
            # --- G0 现有 7 字段（不变）---
            "I_trigger_rate": float(_diag_n1) / max(n_valid, 1),
            "n_trigger": _diag_n1,
            "n_total": n_valid,
            "delta_I0_mean": float(_diag_sum_d0 / max(_diag_n0, 1)),
            "delta_I1_mean": float(_diag_sum_d1 / max(_diag_n1, 1)),
            "tilde_I1_mean": float(_diag_sum_t1 / max(_diag_n1, 1)),
            "penalty_per_step": float((_diag_sum_t1 - _diag_sum_d1) / max(n_valid, 1)),

            # --- 每 rollout 末尾的 bootstrap（离线 GAE 重算用, spec §3.3）---
            "last_values": float(np.asarray(last_values).flatten()[0]),
            "dones": float(np.asarray(dones).flatten()[0]),  # 与 G1b episode_start 列等价（§3.2.1）; next_non_terminal = 1 - dones

            # --- G3: δ by dd quantile（6 字段）---
            "delta_dd_lo": float(_diag_delta_dd_lo / max(_diag_n_dd_lo, 1)),
            "delta_dd_md": float(_diag_delta_dd_md / max(_diag_n_dd_md, 1)),
            "delta_dd_hi": float(_diag_delta_dd_hi / max(_diag_n_dd_hi, 1)),
            "n_dd_lo": _diag_n_dd_lo,
            "n_dd_md": _diag_n_dd_md,
            "n_dd_hi": _diag_n_dd_hi,

            # --- D9: advantage decomposition（2 字段）---
            "adv_std_mean": float(_diag_adv_std_sum / max(n_valid, 1)),
            "adv_tail_mean": float(_diag_adv_tail_sum / max(n_valid, 1)),

            # --- G4: action by I_t（4 字段）---
            "action_I0_mean": float(_diag_act_I0_sum / max(_diag_n0, 1)),
            "action_I1_mean": float(_diag_act_I1_sum / max(_diag_n1, 1)),
            "action_I0_std": float(np.sqrt(max(_diag_act_I0_sq / max(_diag_n0, 1) - (_diag_act_I0_sum / max(_diag_n0, 1)) ** 2, 0))),
            "action_I1_std": float(np.sqrt(max(_diag_act_I1_sq / max(_diag_n1, 1) - (_diag_act_I1_sum / max(_diag_n1, 1)) ** 2, 0))),
        }
        self._flat_direct_cols()

    def _get_samples(self, batch_inds, env=None):
        """SB3 get() 的现成钩子（buffers.py）: 6 字段 base + i_t/d_i/constraint_cost/
        mu_ref 追加（spec §3.2/§4.3/§5.1）."""
        base = super()._get_samples(batch_inds, env)
        return TailRolloutBufferSamples(
            *base,
            self.to_torch(self.i_t_flat[batch_inds]),
            self.to_torch(self.d_i_flat[batch_inds]),
            self.to_torch(self.constraint_cost_flat[batch_inds]),
            self.to_torch(self.mu_ref_flat[batch_inds]),
            self.to_torch(self.supervision_weight_flat[batch_inds]),
            self.to_torch(self.w_down_flat[batch_inds]),
            self.to_torch(self.w_up_flat[batch_inds]),
            self.to_torch(self.up_days_remaining_flat[batch_inds]),
            self.to_torch(self.episode_age_flat[batch_inds]),
            self.to_torch(self.old_mean_actions_flat[batch_inds]),
            self.to_torch(self.pre_cash_flat[batch_inds]),
            self.to_torch(self.pre_value_flat[batch_inds]),
            self.to_torch(self.pre_prices_flat[batch_inds]),
            self.to_torch(self.pre_holdings_flat[batch_inds]),
            self.to_torch(self.real_exec_ge_flat[batch_inds]),
        )
