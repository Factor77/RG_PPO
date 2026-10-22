"""
Experiment configuration, seed control, and path management for Tail_GAE notebooks.

Usage in a notebook cell (call BEFORE importing numpy/torch for strict determinism):

    from notebook_config import set_reproducible_seed, get_experiment_config
    set_reproducible_seed(0)
    config = get_experiment_config("1200", seed=0)
"""

from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Project root and data paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root
DATA_FILE = PROJECT_ROOT / "data" / "new_version_2026" / "data_30_20100101_20260601.csv"
DJI_FILE = PROJECT_ROOT / "data" / "DJI_2013_2026.csv"

# ---------------------------------------------------------------------------
# PPO model hyper-parameters (shared across all Tail_GAE experiments)
# ---------------------------------------------------------------------------

PPO_MODEL_KWARGS: Dict[str, Any] = {
    "ent_coef": 0.01,
    "n_steps": 1024,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "target_kl": 0.015,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
}

PPO_TIMESTEPS: int = 40_000

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

_EXPERIMENT_TABLE: Dict[str, Dict[str, Any]] = {
    "1200": {
        "experiment_name": "Tail_GAE_Baseline",
        "n_risk_dims": 0,
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.025,
    },
    "1205": {
        "experiment_name": "Tail_GAE_V2_5Risk",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure"],
        "use_risk_observation": True,
        "alpha": 0.5,
        "p_tail": 0.10,
    },
    "1207": {
        "experiment_name": "Tail_GAE_V2_7Risk",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "streak20max", "worst20"],
        "use_risk_observation": True,
        "alpha": 1.0,
        "p_tail": 0.10,
    },
    "1210": {
        "experiment_name": "Tail_GAE_V2_10Risk",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "streak20max", "worst20", "ar1_20d", "cumret20", "streak40"],
        "use_risk_observation": True,
        "alpha": 1.0,
        "p_tail": 0.10,
    },
    "1215": {
        "experiment_name": "Tail_GAE_V4_2_MktLatch",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure"],
        "use_risk_observation": True,
        "alpha": 0.35,
        "p_tail": 0.10,
    },
    "1216": {
        "experiment_name": "Tail_GAE_V4_3_MktEWObs",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.35,
        "p_tail": 0.10,
    },
    "1217": {
        "experiment_name": "Tail_GAE_V4_4_MktShift",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.35,
        "p_tail": 0.10,
        "tilt_shift": True,
    },
    "1218": {
        "experiment_name": "Tail_GAE_V4_5_GePenalty",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # tilt isolated: the ge penalty is the sole mechanism
        "p_tail": 0.10,
        "ge_penalty_beta": 0.10,
        "ge_penalty_ref": 0.85,
    },
    "1219": {
        "experiment_name": "Tail_GAE_V4_6_PerStockPenalty",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # 同 1218: 惩罚为唯一机制
        "p_tail": 0.10,
        "ge_penalty_mode": "per_stock",
        "ge_penalty_beta": 0.10,
        "ge_penalty_ref": 0.85,
    },
    "1230": {
        "experiment_name": "Tail_GAE_V4_7_DirectPenalty",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # Tail-GAE 关闭：惩罚唯一机制 = 直通项（spec v4.5 §4）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,      # 惩罚退出 reward（reward 通道关闭）
        "direct_penalty_beta": 0.10, # 直通项强度（β = 每触发样本强度, μ 单位, §5.4 扫描定）
        "direct_penalty_ref": 0.85,  # 逐股上限 = ref/30
    },
    "1231": {
        "experiment_name": "Tail_GAE_V4_7_DirectPenalty_B03",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03, # β 扫描下档（spec §5.4）
        "direct_penalty_ref": 0.85,
    },
    "1232": {
        "experiment_name": "Tail_GAE_V4_7_DirectPenalty_B30",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.30, # β 扫描上档（spec §5.4）
        "direct_penalty_ref": 0.85,
    },
    # ---- V4.6 Step 2: 基座修复（spec §3）——全部 β=0 纯收益 -------------
    # 编号 1233 起连续（用户规定，不用 spec 预留的 1238/1239）:
    # 1233 = 2A 七档扫描基础配置（ent_coef/log_std_init 由 /tmp/s0a_scan.py 覆盖）,
    # 1234-0 = 2B 对照臂（现状配置 β=0）, 1234 = 2B 修复臂（2A 胜出档）.
    "1233": {
        "experiment_name": "Tail_GAE_V4_8_BaseScan",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
    },
    "1234-0": {
        "experiment_name": "Tail_GAE_V4_8_Base2B_Control",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
    },
    "1234": {
        "experiment_name": "Tail_GAE_V4_8_Base2B_Win",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
    },
    # ---- V4.6 Step 3/Step 4: 约束交付裁决与结构修复（spec §4/§5）---------
    # 编号 1235 起连续: 1235 = 3A-1（reward 总量税, 修复基座）, 1236 = 3A-2
    # （reward 逐股税）, 1237 = 3B（直通固定 β=0.03 重测）, 1238 = 3C
    # （adaptive direct-penalty coefficient——λ 由 projected dual ascent 更新,
    # 不声称 PPO-Lagrangian 收敛性）, 1239 = 4B/4C（hinge 达标形式 + μ_ref +
    # d_i^exec 执行对齐）, 1240 = 4D（λ + 达标形式叠加）. 修复基座档
    # （ent_coef/log_std_init, 2B-win 结果）由运行脚本覆盖（同 1233/1234 约定）.
    "1235": {
        "experiment_name": "Tail_GAE_V4_9_3A1_RewardTotal",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # 惩罚唯一机制 = reward 总量税（1218 参数, 修复基座上重测）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.10,
        "ge_penalty_ref": 0.85,
        "ge_penalty_mode": "total",
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
    },
    "1236": {
        "experiment_name": "Tail_GAE_V4_9_3A2_RewardPerStock",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # 逐股税（1219 参数）: 状态内协方差断点检验
        "p_tail": 0.10,
        "ge_penalty_beta": 0.10,
        "ge_penalty_ref": 0.85,
        "ge_penalty_mode": "per_stock",
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
    },
    "1237": {
        "experiment_name": "Tail_GAE_V4_9_3B_DirectB03",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # 直通固定 β=0.03（同 1231 力度, 修复基座上重测）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
    },
    "1238": {
        "experiment_name": "Tail_GAE_V4_9_3C_LambdaAdapt",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # λ 自适应直通（β 通道关闭, 系数由 λ 接管）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
        "lambda_init": 0.03,          # λ_0 = β 扫描最优档等效（spec §4.3）
        "lambda_eta": 0.3,            # η 尺度按 Ĉ 量级校准（每 train() 一次）
        "lambda_max": 0.30,           # 上界 = β 扫描上档
        "constraint_epsilon": 0.005,  # 约束目标水平（Ĉ < ε 时 λ 衰减）
    },
    "1239": {
        "experiment_name": "Tail_GAE_V4_9_4BC_HingeMuRef",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # 直通达标形式（β=0.03 固定 + μ_ref + d_i^exec, spec §5）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.7 Priority 0: 1239 strict deterministic rerun -----------------
    # 仅修复启动时机：CUBLAS/threads 等确定性环境变量必须在导入 numpy/pandas/
    # torch 之前由 run_1245_seed.py 设置。学习、观测、归一化和 4C 约束均不变。
    "1245": {
        "experiment_name": "Tail_GAE_V4_7_Strict1239Rerun",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.7 Priority 1 / N1: 1200-exact reward normalization only ------
    # The notebook keeps 1239's risk state, direct loss, PPO settings and
    # per-window observation fitting.  Only its environment reward path is
    # restored to 1200's mda(lambda=0) component-normalized protocol.
    "1246": {
        "experiment_name": "Tail_GAE_V4_7_N1_RewardOnly",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.7 Priority 2 / N2: first-window base-observation stats only ---
    # Reward stays raw return.  The notebook enables observation fit-once;
    # every model/risk/direct-loss field remains the strict 1239 configuration.
    "1247": {
        "experiment_name": "Tail_GAE_V4_7_N2_ObsFreezeOnly",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.7 Priority 3 / N3: complete 1200 normalization protocol ------
    # Combines 1246's first-window-frozen return-component normalizer with
    # 1247's first-window-frozen 301-dimensional base-observation normalizer.
    # The seven risk features and the complete 1239 4C loss remain unchanged.
    "1248": {
        "experiment_name": "Tail_GAE_V4_7_N3_Full1200Normalization",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.7 Stage B0 / R0: unified-code historical-1200 control --------
    # The notebook supplies the complete 1200 normalization protocol.  This
    # registry entry removes every Tail-GAE/risk/direct-loss mechanism so the
    # remaining H0->R0 difference is the unified runner/env/PPO code path.
    "1250": {
        "experiment_name": "Tail_GAE_V4_7_R0_Unified1200Control",
        "risk_state_names": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.025,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
        "use_mu_ref": False,
        "direct_ref_exec_aligned": False,
    },
    # ---- V4.7 Stage B1 / W0: internal wrapper, policy remains 301-dim -----
    "1251": {
        "experiment_name": "Tail_GAE_V4_7_W0_InternalWrapperControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": False,
        "direct_ref_exec_aligned": False,
    },
    # ---- V4.7 Stage B2 / S1: direct loss without policy risk state -------
    # Keep W0's complete internal seven-field wrapper and 301-dimensional
    # policy observation, then enable only the 1239 4C direct-loss path.
    "1253": {
        "experiment_name": "Tail_GAE_V4_7_S1_DirectLossOnly",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.12 合同二（spec Part 3）: 非对称 PCGrad 正式实验 1270 ---------
    # 1253 配置 + pcgrad=True（直通项梯度仅在 actor-mean 子空间投影到盈利
    # 梯度正交补, 仅当方向余弦 < 0）。零扰动: pcgrad=False 时与 1253 逐位一致。
    "1270": {
        "experiment_name": "Tail_GAE_V4_12_NonSymmetricPCGrad",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        "pcgrad": True,
    },
    # V4.12 前置门快速版（1270 专用）: 零扰动验收——1253 配置逐位克隆
    # （pcgrad 默认关闭）, max_windows=1 由 notebook 侧控制。
    "1270_accept_w1": {
        "experiment_name": "Tail_GAE_V4_7_S1_DirectLossOnly",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.13 P0 / Experiment A（spec Part 7.1）: corrected-4C -----------
    # 唯一实验变量 = 方向标签生命周期修复（每步先清零再写当日超限集合, 全局
    # 生效无需配置键）; 其余与 1253 严格一致。审计: notebook 侧开 grad_diag
    # （train/grad_mu_pc、train/grad_cos_pc、裁剪前后范数、KL 早停、hinge 激活率）。
    "1270b": {
        "experiment_name": "Tail_GAE_V4_13_Corrected4C",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # Phase 7 Identification Pilot: three independent experiment identities.
    # Training mechanisms intentionally clone 1270b; 1301/1302 receive their
    # frozen trajectory calendars through runner_kwargs, not this base table.
    "1300": {
        "experiment_name": "Tail_GAE_Phase7_LegacyControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1301": {
        "experiment_name": "Tail_GAE_Phase7_Density5LinearTreatment",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1302": {
        "experiment_name": "Tail_GAE_Phase7_Density5DoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1303": {
        "experiment_name": "Tail_GAE_Phase7_Density8LinearTreatment",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1304": {
        "experiment_name": "Tail_GAE_Phase7_Density8DoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # Phase 7 follow-on lines.  These identities intentionally preserve the
    # 1270b training mechanism; the frozen per-window calendars are attached
    # by their notebooks at runtime.
    "1305": {
        "experiment_name": "Tail_GAE_Phase7_RecoveryShortfall20LinearTreatment",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1320": {
        "experiment_name": "Tail_GAE_Phase9_GatedRF20_AOnly",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.0, None, False, treatment)
        "recovery_penalty_beta": 0.0,
        "aux_mean_prox_clip": None,
        "use_phase9_observation": False,
        "phase9_calendar_variant": "treatment",
    },
    "1321": {
        "experiment_name": "Tail_GAE_Phase9_GatedRF20_AB",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.03, None, False, treatment): 1320 + B
        "recovery_penalty_beta": 0.03,
        "aux_mean_prox_clip": None,
        "use_phase9_observation": False,
        "phase9_calendar_variant": "treatment",
    },
    "1322": {
        "experiment_name": "Tail_GAE_Phase9_GatedRF20_ABC",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.03, 0.20, False, treatment): 1321 + C
        "recovery_penalty_beta": 0.03,
        "aux_mean_prox_clip": 0.20,
        "use_phase9_observation": False,
        "phase9_calendar_variant": "treatment",
    },
    "1323": {
        "experiment_name": "Tail_GAE_Phase9_GatedRF20_Full",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.03, 0.20, True, treatment): 全套 Treatment
        "recovery_penalty_beta": 0.03,
        "aux_mean_prox_clip": 0.20,
        "use_phase9_observation": True,
        "phase9_calendar_variant": "treatment",
    },
    "1324": {
        "experiment_name": "Tail_GAE_Phase9_GatedRF20_FullDoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.03, 0.20, True, dose_control): 全套 Dose-control
        # (loss 权重读置换列, 304 维 obs 仍读 treatment 列)
        "recovery_penalty_beta": 0.03,
        "aux_mean_prox_clip": 0.20,
        "use_phase9_observation": True,
        "phase9_calendar_variant": "dose_control",
    },
    # ---- 1321b 条件化暴露地板（CEF, spec 1321b_spec.md §3.1/§6.1）----
    # 三条条件 = 1320（A, 无 B/C）逐字段复制 + D（304 维观测）全开, 唯一旋钮
    # `cef_penalty_beta`（旧 ℓ_up 关闭: recovery_penalty_beta=0）.
    # `cef_penalty_beta`=0.0 时 CEF 分支不进入 aux_on, 与旧行为逐位等价.
    "1321b_ctrl": {
        "experiment_name": "Tail_GAE_1321b_CTRL",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.0, None, True, treatment): 1320 + D（观测扩展）
        "recovery_penalty_beta": 0.0,
        "aux_mean_prox_clip": None,
        "use_phase9_observation": True,
        "phase9_calendar_variant": "treatment",
        # CEF 剂量与冻结参数（spec §3.3/§3.6）: 控制条件 β_s=0（地板关）
        "cef_penalty_beta": 0.0,
        "cef_floor_G": 0.90,
        "cef_band_delta": 0.02,
        "cef_min_capacity": 1e-3,
        # §4.3 Stage-0 诊断依赖策略快照; 磁盘核算: 每 iter ≈1–2 MB
        "save_policy_snapshot": True,
    },
    "1321b_lo": {
        "experiment_name": "Tail_GAE_1321b_LO",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.0, None, True, treatment): 1320 + D（观测扩展）
        "recovery_penalty_beta": 0.0,
        "aux_mean_prox_clip": None,
        "use_phase9_observation": True,
        "phase9_calendar_variant": "treatment",
        # CEF 低档剂量 β_s=0.3（spec §9 第 3 项）
        "cef_penalty_beta": 0.3,
        "cef_floor_G": 0.90,
        "cef_band_delta": 0.02,
        "cef_min_capacity": 1e-3,
        # §4.3 Stage-0 诊断依赖策略快照; 磁盘核算: 每 iter ≈1–2 MB
        "save_policy_snapshot": True,
    },
    "1321b_hi": {
        "experiment_name": "Tail_GAE_1321b_HI",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
        # Phase 9 五元组 (0.03, 0.0, None, True, treatment): 1320 + D（观测扩展）
        "recovery_penalty_beta": 0.0,
        "aux_mean_prox_clip": None,
        "use_phase9_observation": True,
        "phase9_calendar_variant": "treatment",
        # CEF 高档剂量 β_s=1.0（spec §9 第 3 项）
        "cef_penalty_beta": 1.0,
        "cef_floor_G": 0.90,
        "cef_band_delta": 0.02,
        "cef_min_capacity": 1e-3,
        # §4.3 Stage-0 诊断依赖策略快照; 磁盘核算: 每 iter ≈1–2 MB
        "save_policy_snapshot": True,
    },
    "1306": {
        "experiment_name": "Tail_GAE_Phase7_RecoveryShortfall20DoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1307": {
        "experiment_name": "Tail_GAE_Phase7_Density10LinearTreatment",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1308": {
        "experiment_name": "Tail_GAE_Phase7_Density10DoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # Phase 8 Formal: Phase 7.5 froze Density8 Linear as the first passing
    # trajectory candidate.  The notebooks bind the same frozen treatment /
    # dose-control calendar used by 1303/1304; these registry entries only
    # assign new output identities for the clean 10-seed formal batch.
    "1309": {
        "experiment_name": "Tail_GAE_Phase8_Density8LinearTreatment",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    "1310": {
        "experiment_name": "Tail_GAE_Phase8_Density8DoseControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.03,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # V4.13 P0 前置门: 无 4C 路径零扰动验收——1251 配置逐位克隆（β=0,
    # 标签修复不得扰动无 4C 训练轨迹）, max_windows=1 由 notebook 侧控制。
    "1270b_accept_w1": {
        "experiment_name": "Tail_GAE_V4_7_W0_InternalWrapperControl",
        "risk_state_names": [],
        "internal_risk_fields": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "policy_risk_fields": [],
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
        "compute_direct_diagnostics": True,
        "use_mu_ref": False,
        "direct_ref_exec_aligned": False,
    },
    "1240": {
        "experiment_name": "Tail_GAE_V4_9_4D_LambdaHinge",
        "risk_state_names": ["current_drawdown", "running_max_drawdown", "drawdown_delta", "downside_volatility_20", "gross_exposure", "mkt_ew_10", "mkt_ew_5"],
        "use_risk_observation": True,
        "alpha": 0.0,  # λ 自适应 + 达标形式叠加（spec §5.2 实验 4D）
        "p_tail": 0.10,
        "ge_penalty_beta": 0.0,
        "direct_penalty_beta": 0.0,
        "direct_penalty_ref": 0.85,
        "lambda_init": 0.03,
        "lambda_eta": 0.3,
        "lambda_max": 0.30,
        "constraint_epsilon": 0.005,
        "use_mu_ref": True,
        "direct_ref_exec_aligned": True,
    },
    # ---- V4.6 Step 5: 风险覆盖层（spec §6）——1200 基线配置 + 执行时投影 ----
    # 5A/5B 用 1200 训练协议（p_tail=0.025、无风险 obs、alpha=0），仅执行层
    # 加投影规则；训练确定性保证与 1200 逐位相同（v4.5 已验证确定性等价）。
    "1242": {
        "experiment_name": "Tail_GAE_V5_0_OverlayProportional",
        "n_risk_dims": 0,
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.025,
        "risk_overlay": "proportional",
        "trade_log": True,
    },
    "1243": {
        "experiment_name": "Tail_GAE_V5_0_OverlayPerStock",
        "n_risk_dims": 0,
        "use_risk_observation": False,
        "alpha": 0.0,
        "p_tail": 0.025,
        "risk_overlay": "per_stock",
        "trade_log": True,
    },
}

# Shared date ranges for all experiments
_TRAIN_START = "2013-04-01"
_TRAIN_END = "2024-12-31"
_TEST_START = "2025-01-01"
_TEST_END = "2026-06-01"


# ---------------------------------------------------------------------------
# 1. Reproducible seed bootstrap
# ---------------------------------------------------------------------------

def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds and (optionally) torch deterministic flags.

    Matches the bootstrap cell from the 0900 DSB-PPO reference notebook
    (AUST_DSB_PPO_V1_lam=0_seed_0.ipynb, Cell 0).

    Parameters
    ----------
    seed : int
        The integer seed used for PYTHONHASHSEED, random, numpy, and torch.
    deterministic : bool
        If True (default), enforce strict GPU determinism:
        cudnn.deterministic, no benchmark, no TF32, single OMP/MKL threads,
        highest matmul precision, and deterministic algorithms.
        Set to False to relax these constraints while still seeding RNGs.

    Raises
    ------
    RuntimeError
        If *deterministic=True* and numpy, pandas, torch, stable_baselines3,
        or finrl are already imported.  The bootstrap must run **before** any
        of these libraries are loaded to guarantee determinism.
    """
    # ---- guard: premature imports can break determinism --------------------
    _premature = sorted(
        name
        for name in ("numpy", "pandas", "torch", "stable_baselines3", "finrl")
        if name in sys.modules
    )
    if _premature:
        msg = (
            "Restart the kernel and run this cell first; already imported: "
            + ", ".join(_premature)
        )
        if deterministic:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)

    # ---- environment variables --------------------------------------------
    # 线程数与 1200 基线 notebook 一致（torch.set_num_threads(12)、OMP/MKL=12）;
    # 原 1 线程设定来自 0900 参考 notebook, 与项目 1200 惯例不符（用户 2026-08-18 指出）。
    _required_env = {
        "PYTHONHASHSEED": str(seed),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "12",
        "MKL_NUM_THREADS": "12",
    }
    for _name, _value in _required_env.items():
        os.environ[_name] = _value

    # ---- Python random ----------------------------------------------------
    random.seed(seed)

    # ---- numpy ------------------------------------------------------------
    import numpy as np

    np.random.seed(seed)

    # ---- PyTorch ----------------------------------------------------------
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.set_num_threads(12)
        torch.set_num_interop_threads(12)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)

        # GPU is required for strict determinism in this project
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Tail_GAE requires an NVIDIA GPU, but torch.cuda.is_available() is False"
            )
        torch.cuda.set_device(0)

        # ---- assertions (match reference notebook) -------------------------
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert torch.get_num_threads() == 12
        assert all(
            os.environ[name] == value for name, value in _required_env.items()
        )

    # ---- informational summary --------------------------------------------
    print(f"Tail_GAE deterministic bootstrap: seed={seed}, deterministic={deterministic}")
    if torch.cuda.is_available():
        print(f"  device = {torch.device('cuda:0')}  {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch/CUDA = {torch.__version__}  {torch.version.cuda}")


# ---------------------------------------------------------------------------
# 2. Experiment config factory
# ---------------------------------------------------------------------------

def get_experiment_config(experiment_id: str, seed: int) -> Dict[str, Any]:
    """Return a complete configuration dict for a single notebook run.

    Parameters
    ----------
    experiment_id : str
        One of ``"1200"`` (Baseline), ``"1205"`` (V2 5Risk), ``"1207"`` (V2 7Risk),
        ``"1210"`` (V2 10Risk).
    seed : int
        Random seed for this run (used in the notebook name and output path).

    Returns
    -------
    dict
        Configuration with keys: experiment_id, experiment_name, seed, alpha,
        p_tail, risk_state_names, n_risk_dims, use_risk_observation,
        train_start, train_end, val_test_start, val_test_end, output_root,
        artifact_id, notebook_name.

    Raises
    ------
    ValueError
        If *experiment_id* is not one of the valid values.
    """
    if experiment_id not in _EXPERIMENT_TABLE:
        raise ValueError(
            f"Unknown experiment_id {experiment_id!r}. "
            f"Valid values: {sorted(_EXPERIMENT_TABLE.keys())}"
        )

    entry = _EXPERIMENT_TABLE[experiment_id]
    experiment_name = entry["experiment_name"]
    risk_state_names = entry.get("risk_state_names", [])
    internal_risk_fields = entry.get("internal_risk_fields", risk_state_names)
    policy_risk_fields = entry.get("policy_risk_fields", risk_state_names)

    results_root = Path(os.environ.get("RGPPO_RESULTS_ROOT", PROJECT_ROOT / "Results"))
    output_root = results_root / "Tail_GAE" / experiment_id

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "alpha": entry["alpha"],
        "p_tail": entry["p_tail"],
        "risk_state_names": risk_state_names,
        "internal_risk_fields": internal_risk_fields,
        "policy_risk_fields": policy_risk_fields,
        "n_risk_dims": len(risk_state_names),
        "use_risk_observation": entry["use_risk_observation"],
        "tilt_shift": bool(entry.get("tilt_shift", False)),
        "ge_penalty_beta": float(entry.get("ge_penalty_beta", 0.0)),
        "ge_penalty_ref": float(entry.get("ge_penalty_ref", 0.95)),
        "ge_penalty_mode": entry.get("ge_penalty_mode", "total"),
        "direct_penalty_beta": float(entry.get("direct_penalty_beta", 0.0)),
        "direct_penalty_ref": float(entry.get("direct_penalty_ref", 0.85)),
        "compute_direct_diagnostics": bool(entry.get("compute_direct_diagnostics", False)),
        # V4.6 Step 3C/Step 4（spec §4.3/§5.1）: λ 自适应直通与达标形式开关
        # （默认关闭 = 全部 None/False, 既有实验配置不受影响）.
        "lambda_init": float(entry.get("lambda_init", 0.0)),
        "lambda_eta": (
            None if entry.get("lambda_eta", None) is None else float(entry["lambda_eta"])
        ),
        "lambda_max": float(entry.get("lambda_max", 0.0)),
        "constraint_epsilon": float(entry.get("constraint_epsilon", 0.0)),
        "use_mu_ref": bool(entry.get("use_mu_ref", False)),
        "direct_ref_exec_aligned": bool(entry.get("direct_ref_exec_aligned", False)),
        # Phase 9（contract §3.6/§2.1）: B/C/D 配置键。默认值全关 = 旧配置
        # （无这些键）行为逐位不变。五元组顺序 =
        # (direct_penalty_beta, recovery_penalty_beta, aux_mean_prox_clip,
        #  use_phase9_observation, phase9_calendar_variant).
        "recovery_penalty_beta": float(entry.get("recovery_penalty_beta", 0.0)),
        "aux_mean_prox_clip": (
            None
            if entry.get("aux_mean_prox_clip", None) is None
            else float(entry["aux_mean_prox_clip"])
        ),
        "use_phase9_observation": bool(entry.get("use_phase9_observation", False)),
        "phase9_calendar_variant": entry.get("phase9_calendar_variant", "treatment"),
        # 1321b CEF（spec 1321b_spec.md §6.1）: 新语义键，默认值 = 关闭/冻结参数
        # ⇒ 既有实验（无这些键）的返回值与运行行为逐位不变.
        "cef_penalty_beta": float(entry.get("cef_penalty_beta", 0.0)),
        "cef_floor_G": float(entry.get("cef_floor_G", 0.90)),
        "cef_band_delta": float(entry.get("cef_band_delta", 0.02)),
        "cef_min_capacity": float(entry.get("cef_min_capacity", 1e-3)),

        "save_policy_snapshot": bool(entry.get("save_policy_snapshot", False)),
        # V4.12 合同二（spec Part 3）: 非对称 PCGrad（默认关闭 = 与现状逐位一致）
        "pcgrad": bool(entry.get("pcgrad", False)),
        # V4.6 Step 5（spec §6）: 风险覆盖层与 trade 段日志（默认关闭）
        "risk_overlay": entry.get("risk_overlay"),
        "trade_log": bool(entry.get("trade_log", False)),
        "train_start": _TRAIN_START,
        "train_end": _TRAIN_END,
        "val_test_start": _TEST_START,
        "val_test_end": _TEST_END,
        "output_root": output_root,
        "artifact_id": f"{experiment_id} {experiment_name.lower().replace('_', ' ')}",
        "notebook_name": f"{experiment_name}_seed_{seed}",
    }
