"""Tail-GAE V4 离线分析工具（spec §3.3）: per-step CSV 合并 + 离线 GAE 重算。

**不进训练路径**——本模块只被分析脚本 / jupyter 手动调用, 训练代码不 import;
依赖仅 numpy + pandas（均已有, 无新依赖）。

职责:
- load_per_step: 读 G1a（组合层 per-step）+ G1b（RL 层 per-step）, 按行对齐合并,
  断言一致, 标记假交易日（is_fake 列）;
- recompute_gae: 按 rollout 分段（每 n_steps 行一段）离线重算
  delta / tilde_delta / advantage, 支持换 γ/λ 对比惩罚 credit 落点（D7）;
- cross_validate: mean(A_t) vs diag CSV adv_tail_mean 交叉验证
  （40 行/段简单均值口径, spec §3.3 要点 7）。

用法示例（spec §6）::

    from pathlib import Path
    from tail_posthoc import load_per_step, recompute_gae, cross_validate

    df = load_per_step("iter_126", seed=0,
                       root=Path("Results/Tail_GAE/1205/seed_0/20260812_164506"))
    df_adv = recompute_gae(df, diag_df, gamma=0.99, gae_lambda=0.98, tail_alpha=0.5)
    cross_validate(df_adv, diag_df, "iter_126")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_csv(csv_path: Path, window_label: str, root: Path, subdir: str) -> Path:
    """探测 .csv.gz 优先, 不存在回退 .csv（crash 残留）, 两者皆缺抛错。"""
    gz_path = csv_path.with_suffix(".csv.gz")
    if gz_path.exists():
        return gz_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(
        f"load_per_step: 未找到 {subdir}/{window_label} 的逐日记录"
        f"（root={root}, 探测 .csv.gz 与 .csv 均缺失）——数据未产出或布局不符"
    )


def _assert_newline_terminated(path: Path, window_label: str, side: str) -> None:
    """末字节 \\n 检查：仅对未压缩 .csv 残留执行（kill -9 截断检测）。

    .csv.gz 不查——gzip trailer 的末字节不是 \\n, 检查会误杀正常压缩文件。
    截断行会被 pandas 静默读成 NaN 行（行数够时完全静默）, 必须先于 read_csv 拦截。
    失败即"崩溃 run 数据作废"。
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
    except OSError as exc:
        raise ValueError(
            f"load_per_step({window_label}): {side} 文件 {path} 读取末字节失败"
            f"——崩溃 run 数据作废"
        ) from exc
    if last != b"\n":
        raise ValueError(
            f"load_per_step({window_label}): {side} 文件 {path} 末字节非换行符"
            f"（{last!r}）——kill -9 截断残留, 崩溃 run 数据作废"
        )


def load_per_step(window_label: str, seed: int, root: Path) -> pd.DataFrame:
    """读 G1a + G1b, 按行对齐合并并断言一致, 返回逐日全表（含 is_fake 列）。

    root 为 run 根（时间戳目录, 如 .../seed_8/20260812_164427/）, 函数在其下找
    tail_mechanism/per_step/{window_label}.csv.gz 与
    tail_mechanism/rl/{window_label}.csv.gz（crash 残留为 .csv,
    pd.read_csv 对两者透明）;
    seed 不参与路径拼装（文件名不含 seed）, 仅用于校验 root 形如 .../seed_{seed}/,
    防配错 root 静默读错数据。
    """
    root = Path(root)
    # seed 校验：root 的父目录必须是 seed_{seed}
    if root.parent.name != f"seed_{seed}":
        raise ValueError(
            f"load_per_step({window_label}): root 必须形如 .../seed_{seed}/<时间戳>,"
            f" 现为 {root}——seed 不参与路径拼装, 仅用于防配错"
        )

    g1a_path = _resolve_csv(
        root / "tail_mechanism" / "per_step" / f"{window_label}.csv",
        window_label, root, "per_step",
    )
    g1b_path = _resolve_csv(
        root / "tail_mechanism" / "rl" / f"{window_label}.csv",
        window_label, root, "rl",
    )
    # 末字节 \\n 检查仅对未压缩 .csv 残留执行（.csv.gz 的 gzip trailer 末字节非 \\n）
    for side, path in (("G1a", g1a_path), ("G1b", g1b_path)):
        if path.suffix == ".csv":
            _assert_newline_terminated(path, window_label, side)

    g1a = pd.read_csv(g1a_path)
    g1b = pd.read_csv(g1b_path)

    # 完整性断言：同一 env step 先写 G1a 再写 G1b, 行数差 1 即两文件间崩溃或丢步;
    # 行数不等时 pd.concat(axis=1) 会静默 NaN 填充, 必须先于合并暴露。
    assert len(g1a) == len(g1b), (
        f"load_per_step({window_label}): G1a 行数 {len(g1a)} != G1b 行数 {len(g1b)}"
        f"——两文件间崩溃或丢步"
    )
    # 语义断言：date 列逐行一致（训练段同一日期重复多次, 不能用 date 做 join 键）
    assert (g1a["date"].values == g1b["date"].values).all(), (
        f"load_per_step({window_label}): G1a/G1b 的 date 列逐行不一致——丢步"
    )

    # 按行号位置合并（两文件每步一行、写入顺序锁步）; date 列已断言逐行相等,
    # 先 drop g1b 的 date 再合并——否则 df["date"] 返回 2 列 DataFrame 而非
    # Series, 破坏 §6 分析代码（D1/D2 等取列即错）
    g1a = g1a.reset_index(drop=True)
    g1b = g1b.reset_index(drop=True).drop(columns=["date"])
    merged = pd.concat([g1a, g1b], axis=1)
    # 假交易日标记：episode 末步 base env 提前返回, date 与上一行相同
    merged["is_fake"] = g1a["date"] == g1a["date"].shift(1)
    return merged


def recompute_gae(
    df: pd.DataFrame,
    diag_df: pd.DataFrame,
    gamma: float,
    gae_lambda: float,
    tail_alpha: float,
    n_steps: int = 1024,
) -> pd.DataFrame:
    """按 rollout 分段（每 n_steps 行一段）, 离线重算 delta / tilde_delta /
    逐日 advantage, 追加为返回表的列。

    段内第 t 步（与 compute_returns_and_advantage() 逐项一致）::

        delta_t       = r_t + γ·V_pre(t+1)·(1 − ep_start(t+1)) − V_pre(t)
        tilde_delta_t = delta_t + α·I_t·min(delta_t, 0)
        A_t           = Σ_l (γλ)^l · tilde_delta_{t+l}     （段内反向递归）

    每段最后一步的 bootstrap 用 diag_df 对应行的 last_values 与 dones:
    next_values = last_values, next_non_terminal = 1 − dones
    （dones 与 G1b episode_start 列逐行等价, spec §3.2.1）。
    diag_df 行数必须等于段数（每窗口恰 40 段, PPO_TIMESTEPS=40_000 / 1024）。

    df 需为 load_per_step 的合并结果（含 reward / critic_V_pre / episode_start /
    I_t 列）。返回 df 的副本 + delta / tilde_delta / adv 三列, 不修改入参。
    """
    total = len(df)
    if total % n_steps != 0:
        raise ValueError(
            f"recompute_gae: 行数 {total} 不是段长 {n_steps} 的整数倍"
            f"（每窗口应恰 {total // n_steps} 个完整 rollout）"
        )
    n_segments = total // n_steps
    if len(diag_df) != n_segments:
        raise ValueError(
            f"recompute_gae: diag_df 行数 {len(diag_df)} 与段数 {n_segments} 不一致"
        )
    missing = {"reward", "critic_V_pre", "episode_start", "I_t"} - set(df.columns)
    if missing:
        raise ValueError(
            f"recompute_gae: df 缺少列 {sorted(missing)}——需 load_per_step 合并后的全表"
        )
    missing_diag = {"last_values", "dones"} - set(diag_df.columns)
    if missing_diag:
        raise ValueError(f"recompute_gae: diag_df 缺少列 {sorted(missing_diag)}")

    v_pre = df["critic_V_pre"].to_numpy(dtype=np.float64)
    rewards = df["reward"].to_numpy(dtype=np.float64)
    ep_start = df["episode_start"].to_numpy(dtype=np.float64)
    i_t = df["I_t"].to_numpy(dtype=np.float64)
    last_values = diag_df["last_values"].to_numpy(dtype=np.float64)
    dones = diag_df["dones"].to_numpy(dtype=np.float64)

    deltas = np.empty(total, dtype=np.float64)
    tilde_deltas = np.empty(total, dtype=np.float64)
    advs = np.empty(total, dtype=np.float64)

    for seg in range(n_segments):
        lo = seg * n_steps
        hi = lo + n_steps

        # 段内 t 步的 next 量：t+1 行数据; 段末一步用 diag bootstrap
        next_values = np.empty(n_steps, dtype=np.float64)
        next_non_terminal = np.empty(n_steps, dtype=np.float64)
        next_values[:-1] = v_pre[lo + 1:hi]
        next_values[-1] = last_values[seg]
        next_non_terminal[:-1] = 1.0 - ep_start[lo + 1:hi]
        next_non_terminal[-1] = 1.0 - dones[seg]

        delta = rewards[lo:hi] + gamma * next_values * next_non_terminal - v_pre[lo:hi]
        tilde_delta = delta + tail_alpha * i_t[lo:hi] * np.minimum(delta, 0.0)

        # 标准 GAE(λ) 反向递归
        adv = np.empty(n_steps, dtype=np.float64)
        acc = 0.0
        for t in range(n_steps - 1, -1, -1):
            acc = tilde_delta[t] + gamma * gae_lambda * next_non_terminal[t] * acc
            adv[t] = acc

        deltas[lo:hi] = delta
        tilde_deltas[lo:hi] = tilde_delta
        advs[lo:hi] = adv

    out = df.copy()
    out["delta"] = deltas
    out["tilde_delta"] = tilde_deltas
    out["adv"] = advs
    return out


def cross_validate(df_adv: pd.DataFrame, diag_df: pd.DataFrame, window_label: str) -> bool:
    """交叉验证：离线重算 mean(A_t) == diag CSV 对应窗口的 adv_tail_mean。

    聚合口径写死（spec §3.3 要点 7）: 两侧均为 40 行/段的简单均值——
    离线侧对重算表全量取均值, diag 侧对 40 个 adv_tail_mean（每 rollout 一行）
    取均值; 每段恒满 n_steps 行时, 段均值之和 = 全量均值, 两口径等价。

    判定：绝对误差 ≤ 1e-3 且相对误差 < 5%。不一致说明记录管线或分段有误,
    先修管线再信分析——不通过直接抛错, 防止静默忽略。
    """
    offline_mean = float(df_adv["adv"].mean())
    diag_mean = float(diag_df["adv_tail_mean"].mean())
    abs_err = abs(offline_mean - diag_mean)
    rel_err = abs_err / max(abs(diag_mean), 1e-12)
    if abs_err > 1e-3 or rel_err >= 0.05:
        raise ValueError(
            f"cross_validate({window_label}): 离线 adv 均值 {offline_mean:.6e} vs "
            f"diag adv_tail_mean 均值 {diag_mean:.6e}, 绝对误差 {abs_err:.3e}"
            f"（容差 1e-3）, 相对误差 {rel_err:.2%}（容差 5%）"
            f"——记录管线或分段有误"
        )
    return True
