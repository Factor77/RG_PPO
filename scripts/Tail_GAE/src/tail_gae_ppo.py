"""Tail-GAE PPO wiring (spec §5.8) + V4.5/1230 直通项（spec §3.3）+ V4.6 Step 3C/Step 4.

TailGAEPPO 在继承 SB3 PPO 训练循环的基础上:
- 通过 rollout_buffer_class/rollout_buffer_kwargs 注入 TailGAERolloutBuffer;
- V4.5: override train() 在 policy loss 上附加直通惩罚项
  β · (i_t ⊙ hinge(d_i · μ)).mean(触发样本) —— 不经 reward/GAE/优势/score.
  direct_penalty_beta=0.0（默认）时 train() 与 SB3 原式逐位等价（U5 守门）;
- V4.6 Step 3C（spec §4.3）: adaptive direct-penalty coefficient（自适应直通
  惩罚系数）——系数 λ 由 projected dual ascent 更新 λ ← clip(λ + η(Ĉ−ε), 0, λ_max)
  （每 train() 一次; 受 PPO-Lagrangian 启发但不声称其收敛性）, 默认关闭 =
  系数固定 β; 直通梯度非真实约束的 policy gradient, 名称一律不用 Lagrangian 术语;
- V4.6 Step 4（spec §5.1）: hinge 达标形式 max(0, d·(μ−μ_ref))（use_mu_ref=True
  时启用; μ_ref = 状态依赖执行对齐达标动作参考, 来自 buffer 新列, 非参数）,
  默认关闭 = v4.5 原式 max(0, d·μ).
- Phase 6（Trajectory-aware Direct Penalty）: direct loss 统一接收
  supervision_weight（legacy = i_t 逐位等价 / trajectory = 连续权重 w_t）;
  trajectory 模式追加监督剂量日志（legacy 零开销）.
"""

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

try:
    from .tail_gae_buffer import TailGAERolloutBuffer
except ImportError:  # flat-import mode (scripts/Tail_GAE/src on sys.path)
    from tail_gae_buffer import TailGAERolloutBuffer


def pcgrad_project(g_p: list[th.Tensor], g_r: list[th.Tensor],
                   eps: float = 1e-8) -> tuple[list[th.Tensor], float, bool]:
    """非对称 PCGrad 投影（v4.12 合同二 Part 3）: g_r' = g_r − (g_r·g_p/|g_p|²)·g_p
    （仅当方向余弦 < 0 时投影; g_p 原样不动——盈利是主目标）。

    纯函数、无 RNG（同输入两次调用逐位一致）。返回 (g_r_proj, cos, skipped):
    cos = 投影前方向余弦（float）; skipped = ‖g_p‖ < eps 时跳过投影、原样返回。
    输入为 autograd.grad 的产出（已 detach）; 投影为纯算术, 不保留计算图。
    """
    dot = float(sum((gp * gr).sum() for gp, gr in zip(g_p, g_r)))
    norm2_p = float(sum((gp * gp).sum() for gp in g_p))
    if norm2_p < eps * eps:
        return [gr.clone() for gr in g_r], 0.0, True
    norm2_r = float(sum((gr * gr).sum() for gr in g_r))
    cos = dot / (norm2_p ** 0.5 * (norm2_r ** 0.5 + 1e-30))
    if dot >= 0.0:
        return [gr.clone() for gr in g_r], cos, False
    alpha = dot / norm2_p
    return [gr - alpha * gp for gr, gp in zip(g_r, g_p)], cos, False


def _pcgrad_grads(policy, policy_loss: th.Tensor,
                  direct_loss_scaled: th.Tensor) -> list[tuple[th.Tensor, th.Tensor, th.Tensor]]:
    """1270 合同二：取 actor-mean 子空间的两路梯度对。

    返回 [(g_p, g_r, param)]：policy_loss 与 direct_loss 都不触 value 头参数
    （autograd.grad 返回 None, 按对过滤）; log_std 不参与（直接损失只经均值路径）。
    两损失的 None 模式一致（同为 policy_net + action_net）。
    """
    mu_params = [p for n, p in policy.named_parameters()
                 if n != "log_std" and p.requires_grad]
    g_p = th.autograd.grad(policy_loss, mu_params, retain_graph=True, allow_unused=True)
    g_r = th.autograd.grad(direct_loss_scaled, mu_params, retain_graph=True, allow_unused=True)
    return [(gp, gr, p) for gp, gr, p in zip(g_p, g_r, mu_params)
            if gp is not None and gr is not None]


def _aux_grad_diag(model, policy_loss: th.Tensor, aux_scaled: th.Tensor) -> list[float]:
    """Phase 9 §3.6: actor-mean 子空间上 PPO surrogate 与辅助项梯度的 norm/cos.

    每个 train() 对各自首个权重和大于零的 minibatch 调用一次（retain_graph,
    不释放主计算图; 每 train() 至多一次 extra backward 级开销）。返回
    [‖g_aux‖, cos(g_PPO, g_aux)]; None 模式须逐参数一致, 按对过滤;
    全空（无公共参数）时返回 [0.0, 0.0].
    """
    mu_params = [p for n, p in model.policy.named_parameters()
                 if n != "log_std" and p.requires_grad]
    g_p_all = th.autograd.grad(policy_loss, mu_params, retain_graph=True, allow_unused=True)
    g_a_all = th.autograd.grad(aux_scaled, mu_params, retain_graph=True, allow_unused=True)
    pairs = [(gp, ga) for gp, ga in zip(g_p_all, g_a_all)
             if gp is not None and ga is not None]
    if not pairs:
        return [0.0, 0.0]
    norm_p = float(th.sqrt(sum((gp * gp).sum() for gp, _ in pairs)).detach())
    norm_a = float(th.sqrt(sum((ga * ga).sum() for _, ga in pairs)).detach())
    dot = float(sum((gp * ga).sum() for gp, ga in pairs).detach())
    return [norm_a, dot / (norm_p * norm_a + 1e-30)]


# ---------------------------------------------------------------------------
# Phase 9 §3.2/§3.3（contract）: B 软执行回补项 + C actor-mean 近端裁剪的
# 纯函数。全部输入 (B,·) float32; 输出逐样本 (B,) 或标量。冻结参数
# GE_min=0.85、ε_GE=0.01、hmax=100 在调用方给定。
# ---------------------------------------------------------------------------


def ste_clipped(mu: th.Tensor) -> th.Tensor:
    """T_up(μ) = μ + stopgrad(clip(μ,−1,1) − μ): 前向 = 硬裁剪, 反向恒为 1."""
    return mu + (th.clamp(mu, -1.0, 1.0) - mu).detach()


def phase9_soft_ge_from_q(
    q: th.Tensor, cash: th.Tensor, prices: th.Tensor,
    holdings: th.Tensor, v_pre: th.Tensor,
    c_buy: th.Tensor, c_sell: th.Tensor,
) -> th.Tensor:
    """contract §3.2 软执行模型: 有符号交易量 q 上的执行后组合 GE.

    cash (B,), prices/holdings/q (B,30), v_pre (B,), c_buy/c_sell 标量.
    C_avail = cash + Σ p·s·(1−c^sell); λ = min(1, C_avail/(Σ p·b·(1+c^buy)+1e-8));
    h̃_post = h + q + (λ−1)b; ĜE_post = Σ p·h̃_post / V_pre. 逐项可微.
    """
    buys = th.clamp_min(q, 0.0)
    sells = th.clamp_min(-q, 0.0)
    cash_avail = cash + (prices * sells * (1.0 - c_sell)).sum(dim=1)
    buy_need = (prices * buys * (1.0 + c_buy)).sum(dim=1) + 1e-8
    scale = th.clamp(cash_avail / buy_need, max=1.0)
    post = holdings + q + (scale - 1.0).unsqueeze(1) * buys
    return (prices * post).sum(dim=1) / th.clamp_min(v_pre, 1e-8)


def phase9_up_hinge_per_sample(
    mean_actions: th.Tensor, pre_cash: th.Tensor, pre_value: th.Tensor,
    pre_prices: th.Tensor, pre_holdings: th.Tensor, up_days_remaining: th.Tensor,
    hmax: float, c_buy: float, c_sell: float,
) -> tuple[th.Tensor, th.Tensor]:
    """contract §3.2: 直通 T_up 上的逐样本 ℓ_up 与 ĜE_post(μ).

    ℓ_up,t = [GE_target − ĜE_post,t(T_up(μ_θ))]_+; GE_target = min(0.84,
    GE_pre + [0.84 − GE_pre]_+/R_t, GE_cap). GE_target/GE_cap/R_t 无梯度
    （detach）; 返回 (ℓ_up (B,), ĜE_post (B,), 后者供 soft/real 诊断) —
    诊断用采样动作时调用 phase9_soft_ge_from_q 而非本函数的 ĜE_post(μ).
    """
    cash = pre_cash[:, 0]
    v_pre = pre_value[:, 0]
    r_t = th.clamp_min(up_days_remaining[:, 0], 1.0)
    mu_hat = ste_clipped(mean_actions)
    ge_hat = phase9_soft_ge_from_q(
        th.maximum(-pre_holdings, hmax * mu_hat),
        cash, pre_prices, pre_holdings, v_pre, c_buy, c_sell,
    )
    ge_pre = (pre_prices * pre_holdings).sum(dim=1) / th.clamp_min(v_pre, 1e-8)
    ge_cap = phase9_soft_ge_from_q(
        th.maximum(-pre_holdings, hmax * th.ones_like(mu_hat)),
        cash, pre_prices, pre_holdings, v_pre, c_buy, c_sell,
    )
    floor = th.full_like(ge_pre, 0.84)
    target = th.minimum(
        th.minimum(floor, ge_pre + th.clamp_min(floor - ge_pre, 0.0) / r_t),
        ge_cap,
    ).detach()
    return th.clamp_min(target - ge_hat, 0.0), ge_hat


def phase9_cef_per_sample(
    mean_actions: th.Tensor, pre_cash: th.Tensor, pre_value: th.Tensor,
    pre_prices: th.Tensor, pre_holdings: th.Tensor, episode_age: th.Tensor,
    floor_g: float, band_delta: float, min_capacity: float,
    hmax: float, c_buy: float, c_sell: float,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """1321b spec §3.3: 条件化暴露地板（CEF）逐样本损失与掩码.

    M = {episode_age==0 ∧ gap>δ ∧ c_t>ε}; gap = G* − GE_pre;
    c_t = 软执行全买路径可达的 GE − GE_pre（单日容量）;
    a* = min(1, gap/c_t); ℓ = [a* − μ̄]_+², μ̄ = mean_i μ_i（动作空间,
    无需 STE——前向即真值）. 返回 (l_cef (B,), mask (B,) bool, a* (B,),
    gap (B,), c_t (B,)）. 全部无新参数梯度来源（pre_* 为冻结量）.
    """
    cash = pre_cash[:, 0]
    v_pre = pre_value[:, 0]
    ge_pre = (pre_prices * pre_holdings).sum(dim=1) / th.clamp_min(v_pre, 1e-8)
    gap = th.clamp_min(floor_g - ge_pre, 0.0)
    ge_cap = phase9_soft_ge_from_q(
        th.maximum(-pre_holdings, hmax * th.ones_like(pre_holdings)),
        cash, pre_prices, pre_holdings, v_pre, c_buy, c_sell,
    )
    c_t = ge_cap - ge_pre
    mask = (episode_age[:, 0] <= 0.0) & (gap > band_delta) & (c_t > min_capacity)
    a_star = th.clamp(gap / th.clamp_min(c_t, 1e-8), max=1.0)
    l_cef = th.clamp_min(a_star - mean_actions.mean(dim=1), 0.0) ** 2
    return l_cef, mask, a_star, gap, c_t


def direct_penalty_loss(mean_actions: th.Tensor, d_i: th.Tensor, supervision_weight: th.Tensor,
                        mu_ref: th.Tensor | None = None) -> th.Tensor:
    """直通惩罚项（spec §3.3）+ Phase 6 统一监督权重接口（Trajectory-aware
    Direct Penalty §9.3）.

    mean_actions (batch, 30), d_i (batch, 30), supervision_weight (batch, 1),
    mu_ref (batch, 30). 返回标量 direct_loss（未乘系数）——梯度方向由 d_i 决定
    （μ 数值不参与方向）, μ 越过 0 后 hinge 归零（不翻空）; 权重为 0 的样本
    贡献为零. supervision_weight 即 q_t^loss: legacy 模式由调用方传 i_t（与
    既有逐位等价）, trajectory 模式传 w_t（buffer 新列）.
    V4.6 Step 4（spec §5.1）: mu_ref 传入时改用达标形式 max(0, d·(μ−μ_ref))——
    平衡点从 μ→0 改为 μ→μ_ref（达标即停, 防降仓过头）; mu_ref=None 时保持
    v4.5 原式 max(0, d·μ)（默认, 逐位等价）.
    """
    mu = mean_actions - mu_ref if mu_ref is not None else mean_actions
    pen = th.clamp_min(d_i * mu, 0.0).sum(dim=1)  # (batch,)
    w_flat = supervision_weight[:, 0]  # (batch,) 展平, 防与 (batch,) 广播成 (batch, batch)
    return (w_flat * pen).sum() / (w_flat.sum() + 1e-8)


class TailGAEPPO(PPO):
    def __init__(
        self,
        policy,
        env,
        *,
        tail_alpha=0.0,
        risk_wrapper=None,
        tilt_shift=False,
        rollout_buffer=None,
        direct_penalty_beta=0.0,
        direct_penalty_ref=0.85,
        grad_diag=False,
        # V4.6 Step 3C（spec §4.3）: adaptive direct-penalty coefficient
        # （自适应直通惩罚系数——projected dual ascent 启发, 不声称
        # PPO-Lagrangian 收敛性）. lambda_eta=None（默认）= 关闭, 系数固定为
        # direct_penalty_beta, 行为与现有完全一致.
        lambda_eta=None,
        lambda_max=None,
        constraint_epsilon=None,
        lambda_init=None,
        # V4.6 Step 4（spec §5.1）: 直通项达标形式开关（μ_ref 来自 buffer 新列,
        # 非参数）; 默认关闭 = v4.5 原式 max(0, d·μ).
        use_mu_ref=False,
        # V4.12 合同二（spec Part 3）: 非对称 PCGrad——直通项梯度只在 actor-mean
        # 子空间投影到盈利梯度正交补（仅当方向余弦 < 0）。默认关闭 = 与现状
        # 逐位等价（loss.backward 直走, 零开销）。
        pcgrad=False,
        # Phase 9（contract §3.2/§3.3）: 回补项系数 β_up（0.0 = 关闭, 1321 起
        # 生效）; actor-mean 近端裁剪带 δ_μ（None = 关闭, 1322 起生效, 对
        # down/up 同时生效）; hmax/买卖费用率为 §3.4 只读常量——由 runner 从
        # env/config 解析后显式传入, 缺省时按 env 链回退解析; β_up>0 且解析
        # 失败 = fail fast.
        recovery_penalty_beta=0.0,
        aux_mean_prox_clip=None,
        # 1321b（spec §3.3）: 条件化暴露地板系数 β_s 与三个标量（目标线 G*、
        # 再平衡带 δ、容量下限 ε）。β_s=0.0 默认关闭 = 逐位等价（不进 aux_on）。
        cef_penalty_beta=0.0,
        cef_floor_G=0.90,
        cef_band_delta=0.02,
        cef_min_capacity=1e-3,
        hmax=None,
        buy_cost_pct=None,
        sell_cost_pct=None,
        **kwargs,
    ):
        if tail_alpha != 0.0 and risk_wrapper is None:
            raise AssertionError("tail_alpha > 0 requires risk_wrapper (I_t data source)")
        self.tail_alpha = tail_alpha
        self.direct_penalty_beta = float(direct_penalty_beta)
        self.direct_penalty_ref = float(direct_penalty_ref)
        # V4.6 B0 梯度诊断开关（spec §3.3）: 默认关闭 = 与 SB3 train() 逐位等价（U5）.
        self.grad_diag = bool(grad_diag)
        # V4.6 Step 3C: λ 配置（全部 None/0 = 关闭, 行为等价现有; 开启判定 =
        # lambda_eta is not None, 其余参数由配置条目一并给出）.
        self.lambda_eta = None if lambda_eta is None else float(lambda_eta)
        self.lambda_max = 0.0 if lambda_max is None else float(lambda_max)
        self.constraint_epsilon = 0.0 if constraint_epsilon is None else float(constraint_epsilon)
        self.lambda_init = 0.0 if lambda_init is None else float(lambda_init)
        self._lambda = self.lambda_init
        # V4.6 Step 4: 达标形式开关（默认关闭）.
        self.use_mu_ref = bool(use_mu_ref)
        # V4.12 合同二（spec Part 3）: 非对称 PCGrad 开关（默认关闭）.
        self.pcgrad = bool(pcgrad)
        # 2.8.0a2: OnPolicyAlgorithm.__init__ unconditionally reassigns
        # rollout_buffer_class/kwargs, so the injection must flow through
        # PPO's own keyword params rather than pre-super instance attributes.
        kwargs["rollout_buffer_class"] = TailGAERolloutBuffer
        kwargs["rollout_buffer_kwargs"] = dict(
            tail_alpha=tail_alpha,
            risk_wrapper=risk_wrapper,
            tilt_shift=tilt_shift,
        )
        super().__init__(policy, env, **kwargs)
        if rollout_buffer is not None:
            self.rollout_buffer = rollout_buffer
        # Phase 9（contract §3.4）: β_up/δ_μ 与软执行常量在 super().__init__
        # 之后解析（需 self.device/self.env 已就位）。旧配置不传新键 = 全部
        # 关闭, 与既有行为逐位一致。
        self.recovery_penalty_beta = float(recovery_penalty_beta)
        self.aux_mean_prox_clip = None if aux_mean_prox_clip is None else float(aux_mean_prox_clip)
        self.cef_penalty_beta = float(cef_penalty_beta)
        self.cef_floor_G = float(cef_floor_G)
        self.cef_band_delta = float(cef_band_delta)
        self.cef_min_capacity = float(cef_min_capacity)
        dev = next(iter(self.policy.parameters())).device

        def _resolve(name: str, given: object) -> float | None:
            if given is not None:
                return float(given)
            node = getattr(self, "env", None)
            seen: set[int] = set()
            while node is not None and id(node) not in seen:
                seen.add(id(node))
                try:
                    val = getattr(node, name, None)
                except Exception:
                    val = None
                if val is not None:
                    return float(np.asarray(val).reshape(-1)[0])
                if hasattr(node, "envs") and getattr(node, "envs"):
                    node = getattr(node, "envs")[0]
                else:
                    node = getattr(node, "venv", None)
            return None

        self._up_hmax = _resolve("hmax", hmax) or 100.0
        self._up_c_buy = _resolve("buy_cost_pct", buy_cost_pct)
        self._up_c_sell = _resolve("sell_cost_pct", sell_cost_pct)
        if (self.recovery_penalty_beta != 0.0 or self.cef_penalty_beta != 0.0) and (
            self._up_c_buy is None or self._up_c_sell is None
        ):
            raise ValueError(
                "recovery_penalty_beta > 0 requires buy_cost_pct/sell_cost_pct "
                "from env or explicit kwargs (contract §3.2 c^buy/c^sell)"
            )
        self._up_c_buy = 0.0 if self._up_c_buy is None else self._up_c_buy
        self._up_c_sell = 0.0 if self._up_c_sell is None else self._up_c_sell
        self._aux_hmax = th.tensor(self._up_hmax, dtype=th.float32, device=dev)
        self._aux_c_buy = th.tensor(self._up_c_buy, dtype=th.float32, device=dev)
        self._aux_c_sell = th.tensor(self._up_c_sell, dtype=th.float32, device=dev)

    def _trajectory_supervision(self) -> bool:
        """Phase 6: 当前 rollout buffer 是否处于 trajectory 监督模式
        （经 wrapper 读取; 无 wrapper/legacy = False, 零开销）."""
        wrapper = getattr(self.rollout_buffer, "risk_wrapper", None)
        return bool(getattr(wrapper, "supervision_mode", "legacy") == "trajectory")

    def _snapshot_old_mean_actions(self) -> None:
        """§3.3/§3.4: train() 入口旧均值快照。

        buffer 全部有效 observation 用当前冻结策略做一次 no_grad 前向, 按原始
        buffer 索引写入 old_mean_actions_flat（get() 只读 flat 副本）。调用点
        在 train() 入口——早于任何 minibatch 打乱与 optimizer.step(), 快照前
        后参数 checksum 天然一致; 该策略即 rollout 策略（两次 rollout 之间无
        optimizer step）。
        """
        buf = self.rollout_buffer
        assert buf.n_envs == 1, "old-mean snapshot protocol is single-env (§3.4)"
        n_valid = buf.buffer_size if buf.full else buf.pos
        obs = np.asarray(buf.swap_and_flatten(buf.observations)[:n_valid], dtype=np.float32)
        with th.no_grad():
            dist = self.policy.get_distribution(th.as_tensor(obs, device=self.device))
            mean = dist.distribution.mean.cpu().numpy()  # (n_valid, action_dim)
        buf.old_mean_actions_flat[:n_valid] = mean

    def train(self) -> None:
        """
        SB3 PPO.train() 逐行复制 + 直通项（spec §3.3）+ V4.6 Step 3C λ 更新。

        β=0.0 / λ 关闭时 loss 表达式与 SB3 原式字面相同（base_loss 分支不加
        直通项）, 保证回归逐位等价（U5）。直通项梯度不经 PPO clip（clip 只截
        ratio 项）, 幅度闸只有系数 + target_kl 整体早停。λ 对偶上升在 train()
        末尾执行一次（spec §4.3 更新契约）。
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        direct_losses = []
        mu_trigger_means = []
        # V4.6 B0: 每 train() 只在第一个 minibatch 计算一次梯度诊断（开销 ≈
        # 1 次额外 backward / train(), 增幅 < 5% 约束内）
        grad_diag_done = not self.grad_diag
        grad_logstd_ent = grad_logstd_pg = grad_mu_pg = None
        # V4.12 合同二（spec Part 3）: pcgrad 仪表（关闭时零开销）
        pcgrad_cos_list: list[float] = []
        pcgrad_skip_count = 0
        cum_gp: th.Tensor | None = None
        cum_gr: th.Tensor | None = None
        # V4.13 P0（spec Part 7.1）: 训练期审计仪表——hinge 激活率、裁剪前后
        # 梯度范数、KL 早停次数（仅记 logger 标量, 不改训练数学）
        hinge_active_list: list[float] = []
        grad_norm_pre_list: list[float] = []
        grad_norm_post_list: list[float] = []
        kl_early_stops = 0
        # V4.13 P0: g_C 范数与方向余弦（grad_diag 开启时, 每 train() 首 minibatch）
        grad_mu_pc = grad_cos_pc = None
        # Phase 6（Trajectory-aware Direct Penalty）: trajectory 模式监督剂量
        # 仪表（legacy 零开销）——sum_weight / effective_weighted_samples 逐
        # minibatch, direct gradient norm 每 train() 首个 minibatch 一次.
        super_w_sums: list[float] = []
        super_w_ess: list[float] = []
        super_grad_norm: float | None = None
        super_grad_done = False
        # Phase 9（contract §3.6）: 门后 buffer 才有 B/C 日志键与梯度 diag;
        # legacy buffer 这些列恒零, 全部门控关闭 = 旧行为逐位等价.
        phase9_buf = bool(getattr(self.rollout_buffer, "phase9_active", False))
        down_loss_vals: list[float] = []
        down_hinge_vals: list[float] = []
        up_losses: list[float] = []
        up_hinge_vals: list[float] = []
        cef_losses: list[float] = []
        cef_hinge_vals: list[float] = []
        cef_mu_means: list[float] = []
        cef_a_stars: list[float] = []
        cef_mask_counts_list: list[float] = []
        clip_exceed_list: list[float] = []
        soft_real_errs: list[th.Tensor] = []
        down_diag_done = False
        up_diag_done = False
        down_diag_vals: list[float] = []
        up_diag_vals: list[float] = []
        down_diag_counts = 0
        up_diag_counts = 0
        cef_diag_done = False
        cef_diag_vals: list[float] = []
        cef_diag_counts = 0
        # §3.3/§3.4: actor-mean prox 开启时, train() 入口对全部有效样本做一次
        # no_grad 旧均值快照（任何 minibatch 打乱 / optimizer.step() 之前）.
        if self.aux_mean_prox_clip is not None:
            self._snapshot_old_mean_actions()

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # ---- V4.5/1230: 直通项（spec §3.3）+ Phase 9 §3.2/§3.3 ---------
                # hinge 形式 max(0, d·μ)：μ 越过 0 后归零（防翻空）; 触发样本
                # 归一化：β = "每触发样本强度", 与 batch 内触发率无关. i_t 形状
                # (batch,1) 须 [:,0] 展平, 否则与 (batch,) 广播成 (batch,batch).
                # V4.6 Step 3C: 系数 = λ（自适应, lambda_eta 开启时）或固定 β
                # （默认, 与现有逐位等价）; Step 4: use_mu_ref=True 时传 μ_ref
                # 列改用达标形式 max(0, d·(μ−μ_ref))（μ_ref 非参数, 来自 buffer）.
                # Phase 9 §3.2: up 项在 w_up>0 子集上用软执行模型 + 进度地板;
                # §3.3: aux_mean_prox_clip 开启时 down/up 均取
                # max(ℓ(T_k(μ_θ)), ℓ(T_k(μ^prox)))（μ^prox = μ^old + clip(μ−μ^old,
                # −δ,δ)）。全部旧开关关闭（λ/beta/up/prox 均零）时 loss 表达式
                # 与 SB3 原式相同（回归逐位等价）。
                base_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                direct_coef = self._lambda if self.lambda_eta is not None else self.direct_penalty_beta
                prox_on = self.aux_mean_prox_clip is not None
                up_coef = self.recovery_penalty_beta
                cef_coef = self.cef_penalty_beta
                aux_on = direct_coef != 0.0 or up_coef != 0.0 or cef_coef != 0.0 or prox_on
                # V4.12 合同二（spec Part 3）: 本 minibatch 的投影覆盖信息（None =
                # 不覆盖 = 与关闭逐位等价）。loss.backward() 后把 actor-mean 参数
                # 的 .grad 替换为 g_P + g_R'。
                pcgrad_override = None
                up_loss = None
                cef_loss = None
                if aux_on:
                    # Phase 9 §3.4: 兼容列 supervision_weight 必须与 w_down 逐位相等
                    if getattr(self.rollout_buffer, "phase9_active", False):
                        assert th.equal(rollout_data.supervision_weight, rollout_data.w_down), \
                            "Phase 9 §3.4 violation: supervision_weight != w_down"
                    dist = self.policy.get_distribution(rollout_data.observations)
                    mean_actions = dist.distribution.mean  # (batch, 30)
                    mu_ref = rollout_data.mu_ref if self.use_mu_ref else None
                    mu_d = mean_actions - mu_ref if mu_ref is not None else mean_actions
                    l_down_cur = th.clamp_min(rollout_data.d_i * mu_d, 0.0).sum(dim=1)
                    if prox_on:
                        delta_mu = float(self.aux_mean_prox_clip)
                        old_mu = rollout_data.old_mean_actions
                        mu_px = old_mu + th.clamp(mean_actions - old_mu, -delta_mu, delta_mu)
                        mu_px_d = mu_px - mu_ref if mu_ref is not None else mu_px
                        l_down_px = th.clamp_min(rollout_data.d_i * mu_px_d, 0.0).sum(dim=1)
                        l_down = th.maximum(l_down_cur, l_down_px)
                        clip_exceed_list.append(
                            float(th.mean((th.abs(mean_actions - old_mu) > delta_mu).float())))
                    else:
                        l_down = l_down_cur
                    q_flat = rollout_data.supervision_weight[:, 0]
                    # §3.4 断言 w_down 承载监督（仅 phase9 buffer; 旧 buffer 该
                    # 列恒零, w_flat 即 legacy 权重, 行为不变）
                    direct_loss = (q_flat * l_down).sum() / (q_flat.sum() + 1e-8)
                    direct_losses.append(direct_loss.item())
                    down_loss_vals.append(direct_loss.item())
                    # μ 层日志（M1.2/M1.3 判据数据源）: 触发样本 Σ|μ| 均值
                    mu_trig = (q_flat * mean_actions.abs().sum(dim=1)).sum() / (q_flat.sum() + 1e-8)
                    mu_trigger_means.append(mu_trig.item())
                    # V4.13 P0 / §3.6: down hinge 激活率——有效代理项（prox 开启
                    # 时取 max(ℓ(μ_θ), ℓ(μ^prox)) 后的 l_down）> 0 的权重占比
                    hinge_val = float((q_flat * (l_down > 0).float()).sum() / (q_flat.sum() + 1e-8))
                    hinge_active_list.append(hinge_val)
                    down_hinge_vals.append(hinge_val)
                    # ---- Phase 9 §3.2 up 项: w_up>0 子集上的软执行 hinge ----
                    # 无 up 样本的 minibatch 跳过重算（mask 前检查, 零开销）;
                    # 整批零权重时按合同返回与参数相连但数值/梯度严格为零的
                    # 标量, 不记录 loss/active 列表（logging 按空列表跳过）。
                    if up_coef != 0.0:
                        wu_flat = rollout_data.w_up[:, 0]
                        up_mask = wu_flat > 0
                        if bool(up_mask.any()):
                            mu_up = mean_actions[up_mask]
                            pre_cash_s = rollout_data.pre_cash[up_mask]
                            pre_value_s = rollout_data.pre_value[up_mask]
                            pre_prices_s = rollout_data.pre_prices[up_mask]
                            pre_hold_s = rollout_data.pre_holdings[up_mask]
                            up_remain_s = rollout_data.up_days_remaining[up_mask]
                            l_up_cur, _ = phase9_up_hinge_per_sample(
                                mu_up, pre_cash_s, pre_value_s, pre_prices_s, pre_hold_s,
                                up_remain_s, self._up_hmax, self._up_c_buy, self._up_c_sell)
                            if prox_on:
                                old_mu_s = rollout_data.old_mean_actions[up_mask]
                                mu_px_s = old_mu_s + th.clamp(mu_up - old_mu_s,
                                                              -delta_mu, delta_mu)
                                l_up_px, _ = phase9_up_hinge_per_sample(
                                    mu_px_s, pre_cash_s, pre_value_s, pre_prices_s,
                                    pre_hold_s, up_remain_s, self._up_hmax,
                                    self._up_c_buy, self._up_c_sell)
                                l_up = th.maximum(l_up_cur, l_up_px)
                            else:
                                l_up = l_up_cur
                            wu_pos = wu_flat[up_mask]
                            up_loss = (wu_pos * l_up).sum() / (wu_pos.sum() + 1e-8)
                            up_losses.append(up_loss.item())
                            up_hinge_vals.append(
                                float((wu_pos * (l_up > 0).float()).sum() / (wu_pos.sum() + 1e-8)))
                            # §3.6 soft/real GE 误差: 实际采样动作经环境边界裁剪
                            # 后的软执行 GE vs 同一 transition 的 real_exec_ge
                            act_s = th.clamp(rollout_data.actions[up_mask], -1.0, 1.0)
                            ge_act = phase9_soft_ge_from_q(
                                th.maximum(-pre_hold_s, self._aux_hmax * act_s),
                                pre_cash_s[:, 0], pre_prices_s, pre_hold_s,
                                pre_value_s[:, 0], self._aux_c_buy, self._aux_c_sell)
                            soft_real_errs.append(
                                (ge_act - rollout_data.real_exec_ge[up_mask, 0]).abs().detach())
                            # §3.6 梯度诊断: 每 train() 首个 up 权重和>0 的 minibatch 一次
                            if phase9_buf and not up_diag_done:
                                up_diag_done = True
                                up_diag_vals.extend(_aux_grad_diag(
                                    self, policy_loss, up_coef * up_loss))
                                up_diag_counts = int(wu_pos.sum())
                        else:
                            up_loss = (0.0 * mean_actions).sum()
                    # ---- 1321b: 条件化暴露地板（CEF）项 ----
                    # 掩码态上的单边 hinge 平方; 按掩码均值归一（|M|=0 时返回
                    # 与参数相连的零标量, 不记录列表）。β_s=0 时整段不执行,
                    # 与旧行为逐位等价。
                    if cef_coef != 0.0:
                        age_col = rollout_data.episode_age
                        l_cef_all, cef_mask, a_star_all, _, _ = phase9_cef_per_sample(
                            mean_actions, rollout_data.pre_cash, rollout_data.pre_value,
                            rollout_data.pre_prices, rollout_data.pre_holdings, age_col,
                            self.cef_floor_G, self.cef_band_delta, self.cef_min_capacity,
                            self._up_hmax, self._up_c_buy, self._up_c_sell)
                        n_cef = int(cef_mask.sum())
                        if n_cef > 0:
                            l_cef = l_cef_all[cef_mask]
                            cef_loss = l_cef.mean()
                            cef_losses.append(float(cef_loss.detach()))
                            cef_hinge_vals.append(float((l_cef > 0).float().mean().detach()))
                            cef_mu_means.append(
                                float(mean_actions.mean(dim=1)[cef_mask].mean().detach()))
                            cef_a_stars.append(float(a_star_all[cef_mask].mean().detach()))
                            cef_mask_counts_list.append(float(n_cef))
                            if phase9_buf and not cef_diag_done:
                                cef_diag_done = True
                                cef_diag_vals.extend(_aux_grad_diag(
                                    self, policy_loss, cef_coef * cef_loss))
                                cef_diag_counts = n_cef
                        else:
                            cef_loss = (0.0 * mean_actions).sum()
                    # down 梯度诊断（§3.6: 每 train() 首个 down 权重和>0 的 minibatch 一次）
                    if phase9_buf and direct_coef != 0.0 and not down_diag_done and q_flat.sum() > 0:
                        down_diag_done = True
                        down_diag_vals.extend(_aux_grad_diag(
                            self, policy_loss, direct_coef * direct_loss))
                        down_diag_counts = int(q_flat.sum())
                    # Phase 6: trajectory 模式剂量日志（legacy 不执行）
                    if self._trajectory_supervision():
                        w_sum = float(q_flat.sum())
                        w_sq = float((q_flat * q_flat).sum())
                        super_w_sums.append(w_sum)
                        super_w_ess.append(w_sum * w_sum / w_sq if w_sq > 0 else 0.0)
                        if not super_grad_done:
                            super_grad_done = True
                            mu_params = [p for n, p in self.policy.named_parameters()
                                         if n != "log_std" and p.requires_grad]
                            g_mc = th.autograd.grad(direct_coef * direct_loss, mu_params,
                                                    retain_graph=True, allow_unused=True)
                            g_mc = [g for g in g_mc if g is not None]
                            super_grad_norm = float(
                                th.sqrt(sum((g * g).sum() for g in g_mc)).detach())
                    aux_scaled = direct_coef * direct_loss
                    if up_loss is not None:
                        aux_scaled = aux_scaled + up_coef * up_loss
                    if cef_loss is not None:
                        aux_scaled = aux_scaled + cef_coef * cef_loss
                    loss = base_loss + aux_scaled
                    if self.pcgrad:
                        pairs = _pcgrad_grads(self.policy, policy_loss, aux_scaled)
                        g_r_proj, cos, skipped = pcgrad_project(
                            [gp for gp, _, _ in pairs], [gr for _, gr, _ in pairs])
                        pcgrad_override = [(p, gp, grp) for (gp, _, p), grp in zip(pairs, g_r_proj)]
                        pcgrad_cos_list.append(cos)
                        pcgrad_skip_count += int(skipped)
                        # cos_cum 累计向量（窗口口径, 必记字段）: 逐 minibatch 拼接累加
                        flat_gp = th.cat([gp.reshape(-1) for gp, _, _ in pairs])
                        flat_gr = th.cat([gr.reshape(-1) for _, gr, _ in pairs])
                        cum_gp = flat_gp if cum_gp is None else cum_gp + flat_gp
                        cum_gr = flat_gr if cum_gr is None else cum_gr + flat_gr
                else:
                    loss = base_loss
                # ------------------------------------------------------------------

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    kl_early_stops += 1
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # ---- V4.6 B0: 梯度实测（spec §3.3）---------------------------
                # 在 loss.backward() 之前用 autograd.grad 分离计算（backward 后
                # 计算图释放, 顺序错则拿不到梯度）。默认关闭, 零开销; 每 train()
                # 仅第一个 minibatch 计算一次（性能约束 <5%）.
                # 理论基准: 熵项对 log_std 每维梯度 = -ent_coef（与 σ 无关）,
                # 30 维 L2 = sqrt(30) * ent_coef ≈ 0.0548（ent_coef=0.01）.
                if not grad_diag_done:
                    grad_diag_done = True
                    log_std = [p for n, p in self.policy.named_parameters() if n == "log_std"][0]
                    mu_params = [p for n, p in self.policy.named_parameters()
                                 if n != "log_std" and p.requires_grad]
                    g_ent = th.autograd.grad(self.ent_coef * entropy_loss, log_std, retain_graph=True)[0]
                    g_pg = th.autograd.grad(policy_loss, log_std, retain_graph=True)[0]
                    # allow_unused: mu_params 含 value head（不参与 policy_loss 图）,
                    # 未用参数返回 None——过滤后再算范数
                    g_mu = th.autograd.grad(policy_loss, mu_params, retain_graph=True,
                                            allow_unused=True)
                    g_mu = [g for g in g_mu if g is not None]
                    grad_logstd_ent = g_ent.detach().norm().item()
                    grad_logstd_pg = g_pg.detach().norm().item()
                    grad_mu_pg = th.sqrt(sum((g * g).sum() for g in g_mu)).detach().item()
                    # V4.13 P0（spec Part 7.1）: g_C 范数与 g_P/g_C 方向余弦——
                    # 只在直通项激活时计算（None 模式与 policy_loss 一致, 按对过滤）
                    if direct_coef != 0.0:
                        g_mc = th.autograd.grad(direct_coef * direct_loss, mu_params,
                                                retain_graph=True, allow_unused=True)
                        g_mc = [g for g in g_mc if g is not None]
                        if g_mu and g_mc:
                            norm_p = float(th.sqrt(sum((g * g).sum() for g in g_mu)).detach())
                            norm_c = float(th.sqrt(sum((g * g).sum() for g in g_mc)).detach())
                            dot = float(sum((gp * gc).sum()
                                            for gp, gc in zip(g_mu, g_mc)).detach())
                            grad_mu_pc = norm_c
                            grad_cos_pc = dot / (norm_p * norm_c + 1e-30)
                # ------------------------------------------------------------------

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # V4.12 合同二: 投影覆盖（仅 actor-mean 参数; log_std/critic 保留
                # loss.backward 的原梯度）。pcgrad_override=None 时零开销。
                if pcgrad_override is not None:
                    for p, gp, grp in pcgrad_override:
                        p.grad = gp + grp
                # Clip grad norm
                # V4.13 P0（spec Part 7.1）: 裁剪前范数（clip 的返回值）与裁剪后
                # 范数（grad_diag 开启时多扫一遍, 不影响训练数学）
                total_norm_pre = th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
                grad_norm_pre_list.append(float(total_norm_pre))
                if self.grad_diag:
                    total_norm_post = float(
                        sum((p.grad.norm().item() ** 2)
                            for p in self.policy.parameters() if p.grad is not None) ** 0.5)
                    grad_norm_post_list.append(total_norm_post)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if direct_losses:
            self.logger.record("train/direct_loss", np.mean(direct_losses))
            self.logger.record("train/mu_trigger", np.mean(mu_trigger_means))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        # V4.6 B0 梯度诊断标量（spec §3.3; grad_diag=False 时不记录）
        if grad_logstd_ent is not None:
            self.logger.record("train/grad_logstd_ent", grad_logstd_ent)
            self.logger.record("train/grad_logstd_pg", grad_logstd_pg)
            self.logger.record("train/grad_mu_pg", grad_mu_pg)
        # V4.13 P0（spec Part 7.1）: g_C 范数/余弦（grad_diag 且直通激活时）
        if grad_mu_pc is not None:
            self.logger.record("train/grad_mu_pc", grad_mu_pc)
            self.logger.record("train/grad_cos_pc", grad_cos_pc)
        if grad_norm_pre_list:
            self.logger.record("train/grad_norm_pre", np.mean(grad_norm_pre_list))
            self.logger.record("train/kl_early_stops", kl_early_stops)
        if grad_norm_post_list:
            self.logger.record("train/grad_norm_post", np.mean(grad_norm_post_list))
        if hinge_active_list:
            self.logger.record("train/hinge_active", np.mean(hinge_active_list))
        # Phase 9 §3.6: B/C 训练日志（键名冻结, 禁另起同义键; 仅门后 buffer 记录,
        # legacy 旧日志逐位不变）。整轮无有效样本时 grad norm/cos 记录 0 并另记
        # active-sample count=0（不得沿用上轮值）; loss/hinge 空列表时不记录.
        if phase9_buf:
            if self.direct_penalty_beta != 0.0:
                self.logger.record("train/down_loss",
                                   float(np.mean(down_loss_vals)) if down_loss_vals else 0.0)
                self.logger.record("train/down_hinge_active",
                                   float(np.mean(down_hinge_vals)) if down_hinge_vals else 0.0)
                self.logger.record("train/down_grad_norm",
                                   down_diag_vals[0] if down_diag_vals else 0.0)
                self.logger.record("train/down_grad_cos_ppo",
                                   down_diag_vals[1] if len(down_diag_vals) > 1 else 0.0)
                self.logger.record("train/down_active_samples", down_diag_counts)
            if self.recovery_penalty_beta != 0.0:
                self.logger.record("train/up_loss",
                                   float(np.mean(up_losses)) if up_losses else 0.0)
                self.logger.record("train/up_hinge_active",
                                   float(np.mean(up_hinge_vals)) if up_hinge_vals else 0.0)
                self.logger.record("train/up_grad_norm",
                                   up_diag_vals[0] if up_diag_vals else 0.0)
                self.logger.record("train/up_grad_cos_ppo",
                                   up_diag_vals[1] if len(up_diag_vals) > 1 else 0.0)
                self.logger.record("train/up_active_samples", up_diag_counts)
                if soft_real_errs:
                    _soft_real = th.cat(soft_real_errs)
                    self.logger.record("train/soft_real_ge_mae", float(_soft_real.mean()))
                    # R8 v2 判据用中位数（contract §7 R8 v2 解释）; mae 是均值,
                    # 不可互相替代, 故并列记录.
                    self.logger.record("train/soft_real_ge_median", float(_soft_real.median()))
                    self.logger.record("train/soft_real_ge_p95",
                                       float(th.quantile(_soft_real, 0.95)))
                else:
                    self.logger.record("train/soft_real_ge_mae", 0.0)
                    self.logger.record("train/soft_real_ge_median", 0.0)
                    self.logger.record("train/soft_real_ge_p95", 0.0)
            # 1321b（spec §4.2）: CEF 训练日志（仅 β_s>0 记录; 空列表记 0 并
            # 另记 active-sample count=0, 不得沿用上轮值）
            if self.cef_penalty_beta != 0.0:
                self.logger.record("train/cef_loss",
                                   float(np.mean(cef_losses)) if cef_losses else 0.0)
                self.logger.record("train/cef_hinge_active",
                                   float(np.mean(cef_hinge_vals)) if cef_hinge_vals else 0.0)
                self.logger.record("train/cef_grad_norm",
                                   cef_diag_vals[0] if cef_diag_vals else 0.0)
                self.logger.record("train/cef_grad_cos_ppo",
                                   cef_diag_vals[1] if len(cef_diag_vals) > 1 else 0.0)
                self.logger.record("train/cef_active_samples", cef_diag_counts)
                self.logger.record("train/cef_mask_count",
                                   float(np.mean(cef_mask_counts_list)) if cef_mask_counts_list else 0.0)
                self.logger.record("train/cef_mu_mean_mask",
                                   float(np.mean(cef_mu_means)) if cef_mu_means else 0.0)
                self.logger.record("train/cef_a_star_mean",
                                   float(np.mean(cef_a_stars)) if cef_a_stars else 0.0)
            if self.aux_mean_prox_clip is not None:
                self.logger.record("train/aux_mean_clip_exceed_rate",
                                   float(np.mean(clip_exceed_list)) if clip_exceed_list else 0.0)
        # Phase 6: trajectory 模式监督剂量日志（legacy 不记录）
        if super_w_sums:
            self.logger.record("train/supervision_sum_weight", np.mean(super_w_sums))
            self.logger.record("train/supervision_eff_samples", np.mean(super_w_ess))
        if super_grad_norm is not None:
            self.logger.record("train/direct_gradient_norm", super_grad_norm)
        # V4.12 合同二（spec Part 3）: pcgrad 仪表（关闭时不记录）
        if self.pcgrad and pcgrad_cos_list:
            self.logger.record("train/pcgrad_cos", float(np.mean(pcgrad_cos_list)))
            self.logger.record("train/pcgrad_n_skip", pcgrad_skip_count)
            if cum_gp is not None and cum_gp.numel() > 0:
                dot = float((cum_gp * cum_gr).sum())
                n_p = float((cum_gp * cum_gp).sum()) ** 0.5
                n_r = float((cum_gr * cum_gr).sum()) ** 0.5
                self.logger.record(
                    "train/pcgrad_cos_cum",
                    dot / (n_p * n_r + 1e-30) if n_p > 0 and n_r > 0 else 0.0,
                )

        # ---- V4.6 Step 3C（spec §4.3）: 对偶上升 λ 更新——每 train() 调用
        # （每个 rollout）末尾更新一次, 不是每 minibatch（同一约束违反重复累计
        # 会使 λ 虚增）. λ ← clip(λ + η·(Ĉ − ε), 0, λ_max); Ĉ = 该 rollout 的
        # constraint_cost 无条件时间平均（partial rollout 只算有效步）——
        # 无触发 rollout → Ĉ=0 → λ 按 −η·ε 自然衰减（软预算语义）.
        # 关闭（lambda_eta=None）时零开销, 不新增日志.
        if self.lambda_eta is not None:
            buf = self.rollout_buffer
            n_valid = buf.buffer_size if buf.full else buf.pos
            c_hat = float(buf.constraint_cost[:n_valid].mean())
            self._lambda = float(
                np.clip(self._lambda + self.lambda_eta * (c_hat - self.constraint_epsilon),
                        0.0, self.lambda_max)
            )
            self.logger.record("train/constraint_cost", c_hat)
            self.logger.record("train/lambda", self._lambda)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
