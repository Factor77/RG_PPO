"""Unit tests for notebook_config.get_experiment_config (Tail_GAE V2)."""

from __future__ import annotations

import json

import pytest

from notebook_config import (
    PROJECT_ROOT,
    _EXPERIMENT_TABLE,
    get_experiment_config,
)

_V2_IDS = ("1205", "1207", "1210", "1215", "1216", "1217", "1218", "1219")
_LEGACY_IDS = ("1201", "1202")


def test_v2_entries_present_legacy_absent():
    assert all(i in _EXPERIMENT_TABLE for i in _V2_IDS)
    assert all(i not in _EXPERIMENT_TABLE for i in _LEGACY_IDS)
    for i in _LEGACY_IDS:
        with pytest.raises(KeyError):
            _EXPERIMENT_TABLE[i]
        with pytest.raises(ValueError):
            get_experiment_config(i, 0)


def test_1207_config():
    cfg = get_experiment_config("1207", 0)
    assert cfg["alpha"] == 1.0
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == [
        "current_drawdown",
        "running_max_drawdown",
        "drawdown_delta",
        "downside_volatility_20",
        "gross_exposure",
        "streak20max",
        "worst20",
    ]
    assert len(cfg["risk_state_names"]) == 7
    assert cfg["n_risk_dims"] == 7
    assert cfg["use_risk_observation"] is True


def test_ablation_bridge_prefix_structure():
    names = {i: get_experiment_config(i, 0)["risk_state_names"] for i in _V2_IDS}
    assert names["1205"] == names["1207"][:5]
    assert names["1207"] == names["1210"][:7]
    assert len(names["1205"]) == 5
    assert len(names["1210"]) == 10


def test_1251_w0_internal_wrapper_config():
    cfg = get_experiment_config("1251", 0)
    assert cfg["risk_state_names"] == []
    assert len(cfg["internal_risk_fields"]) == 7
    assert cfg["policy_risk_fields"] == []
    assert cfg["n_risk_dims"] == 0
    assert cfg["use_risk_observation"] is False
    assert cfg["alpha"] == 0.0
    assert cfg["direct_penalty_beta"] == 0.0
    assert cfg["compute_direct_diagnostics"] is True


def test_1253_s1_direct_loss_without_policy_risk_state_config():
    cfg = get_experiment_config("1253", 0)
    assert cfg["risk_state_names"] == []
    assert len(cfg["internal_risk_fields"]) == 7
    assert cfg["policy_risk_fields"] == []
    assert cfg["n_risk_dims"] == 0
    assert cfg["use_risk_observation"] is False
    assert cfg["alpha"] == 0.0
    assert cfg["p_tail"] == 0.10
    assert cfg["ge_penalty_beta"] == 0.0
    assert cfg["direct_penalty_beta"] == 0.03
    assert cfg["direct_penalty_ref"] == 0.85
    assert cfg["compute_direct_diagnostics"] is True
    assert cfg["use_mu_ref"] is True
    assert cfg["direct_ref_exec_aligned"] is True
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1253"


def test_1270_is_1253_plus_pcgrad():
    """V4.12 合同二（spec Part 3）: 1270 = 1253 配置 + pcgrad=True。"""
    cfg = get_experiment_config("1270", 0)
    base = get_experiment_config("1253", 0)
    for k in ("risk_state_names", "internal_risk_fields", "policy_risk_fields",
              "n_risk_dims", "use_risk_observation", "alpha", "p_tail",
              "ge_penalty_beta", "direct_penalty_beta", "direct_penalty_ref",
              "compute_direct_diagnostics", "use_mu_ref", "direct_ref_exec_aligned"):
        assert cfg[k] == base[k], f"1270: {k} 与 1253 不一致"
    assert cfg["pcgrad"] is True
    assert base["pcgrad"] is False  # 1253 默认关闭（零扰动口径）
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1270"


def test_1270_accept_w1_is_1253_bitwise_clone():
    """V4.12 前置门: 1270_accept_w1 = 1253 逐位克隆（pcgrad 关闭）。"""
    cfg = get_experiment_config("1270_accept_w1", 0)
    base = get_experiment_config("1253", 0)
    for k in ("risk_state_names", "internal_risk_fields", "policy_risk_fields",
              "n_risk_dims", "use_risk_observation", "alpha", "p_tail",
              "ge_penalty_beta", "direct_penalty_beta", "direct_penalty_ref",
              "compute_direct_diagnostics", "use_mu_ref", "direct_ref_exec_aligned"):
        assert cfg[k] == base[k], f"1270_accept_w1: {k} 与 1253 不一致"
    assert cfg["pcgrad"] is False
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1270_accept_w1"


def test_1270b_is_1253_config_with_fixed_labels():
    """V4.13 P0（spec Part 7.1）: 1270b = 1253 配置（标签修复为全局代码变更,
    不设配置键）。"""
    cfg = get_experiment_config("1270b", 0)
    base = get_experiment_config("1253", 0)
    for k in ("risk_state_names", "internal_risk_fields", "policy_risk_fields",
              "n_risk_dims", "use_risk_observation", "alpha", "p_tail",
              "ge_penalty_beta", "direct_penalty_beta", "direct_penalty_ref",
              "compute_direct_diagnostics", "use_mu_ref", "direct_ref_exec_aligned"):
        assert cfg[k] == base[k], f"1270b: {k} 与 1253 不一致"
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1270b"


def test_phase8_density8_formal_configs_match_phase7_candidates():
    """1309/1310 only create clean formal identities for 1303/1304."""
    for formal_id, pilot_id in (("1309", "1303"), ("1310", "1304")):
        formal = get_experiment_config(formal_id, 9)
        pilot = get_experiment_config(pilot_id, 9)
        for key in (
            "risk_state_names",
            "internal_risk_fields",
            "policy_risk_fields",
            "n_risk_dims",
            "use_risk_observation",
            "alpha",
            "p_tail",
            "ge_penalty_beta",
            "direct_penalty_beta",
            "direct_penalty_ref",
            "compute_direct_diagnostics",
            "use_mu_ref",
            "direct_ref_exec_aligned",
        ):
            assert formal[key] == pilot[key], f"{formal_id}: {key} drifted"
        assert formal["seed"] == 9
        assert formal["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / formal_id


def test_1270b_accept_w1_is_1251_bitwise_clone():
    """V4.13 P0 前置门: 1270b_accept_w1 = 1251 逐位克隆（无 4C 路径零扰动）。"""
    cfg = get_experiment_config("1270b_accept_w1", 0)
    base = get_experiment_config("1251", 0)
    for k in ("risk_state_names", "internal_risk_fields", "policy_risk_fields",
              "n_risk_dims", "use_risk_observation", "alpha", "p_tail",
              "ge_penalty_beta", "direct_penalty_beta", "direct_penalty_ref",
              "compute_direct_diagnostics", "use_mu_ref", "direct_ref_exec_aligned"):
        assert cfg[k] == base[k], f"1270b_accept_w1: {k} 与 1251 不一致"
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1270b_accept_w1"


def test_phase7_arms_clone_1270b_training_config():
    """Phase 7 uses independent IDs while preserving the 1270b mechanism."""
    baseline = get_experiment_config("1270b", 0)
    behavior_keys = (
        "alpha", "p_tail", "risk_state_names", "internal_risk_fields",
        "policy_risk_fields", "use_risk_observation", "ge_penalty_beta",
        "direct_penalty_beta", "direct_penalty_ref",
        "compute_direct_diagnostics", "use_mu_ref", "direct_ref_exec_aligned",
        "train_start", "train_end", "val_test_start", "val_test_end",
    )
    for experiment_id in ("1300", "1301", "1302", "1303", "1304", "1305", "1306", "1307", "1308"):
        candidate = get_experiment_config(experiment_id, 0)
        for key in behavior_keys:
            assert candidate[key] == baseline[key], f"{experiment_id}: {key} differs from 1270b"
        assert candidate["experiment_id"] == experiment_id
        assert candidate["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / experiment_id
def test_1253_s1_notebooks_are_clean_and_match_the_control_contract():
    notebook_root = PROJECT_ROOT / "ultimate_experiment" / "1253 Tail_GAE"
    if not notebook_root.exists():
        pytest.skip("1253 notebooks are not part of this reproduction package")
    paths = sorted(notebook_root.glob("Tail_GAE_1253_s1_direct_loss_only_seed_*.ipynb"))
    assert len(paths) == 10

    for seed, path in enumerate(paths):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert len(notebook["cells"]) == 41
        assert all(not cell.get("outputs") for cell in notebook["cells"])
        assert all(
            cell.get("execution_count") is None
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        source = "".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        assert f"EXPERIMENT_SEED = {seed}" in source
        assert 'EXPERIMENT_ID = "1253"' in source
        assert 'CUBLAS_WORKSPACE_CONFIG": ":4096:8"' in source
        assert 'internal_risk_fields = list(exp_cfg["internal_risk_fields"])' in source
        assert 'policy_risk_fields = list(exp_cfg["policy_risk_fields"])' in source
        assert 'assert policy_risk_fields == []' in source
        assert 'assert ppo_model_kwargs["use_mu_ref"] is True' in source
        assert '"direct_penalty_beta": float(exp_cfg["direct_penalty_beta"])' in source
        assert '"direct_ref_exec_aligned": bool(exp_cfg["direct_ref_exec_aligned"])' in source
        assert '"fit_once_on_meta_train": True' in source
        assert '"fit_mda_components_once_on_meta_train": True' in source
        assert 'assert observation_contract["risk_observation_dim"] == 0' in source
        assert 'assert observation_contract["policy_observation_dim"] == policy_state_space == 301' in source
        assert "1251" not in source
        assert "W0" not in source


def test_1215_config():
    cfg = get_experiment_config("1215", 0)
    assert cfg["alpha"] == 0.35
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == [
        "current_drawdown",
        "running_max_drawdown",
        "drawdown_delta",
        "downside_volatility_20",
        "gross_exposure",
    ]
    assert cfg["n_risk_dims"] == 5
    assert cfg["use_risk_observation"] is True


def test_1216_config():
    cfg = get_experiment_config("1216", 0)
    assert cfg["alpha"] == 0.35
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == [
        "current_drawdown",
        "running_max_drawdown",
        "drawdown_delta",
        "downside_volatility_20",
        "gross_exposure",
        "mkt_ew_10",
        "mkt_ew_5",
    ]
    assert cfg["n_risk_dims"] == 7
    assert cfg["use_risk_observation"] is True
    assert cfg["tilt_shift"] is False
    assert get_experiment_config("1215", 0)["risk_state_names"][:5] == cfg["risk_state_names"][:5]


def test_1217_config():
    cfg = get_experiment_config("1217", 0)
    assert cfg["alpha"] == 0.35
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == get_experiment_config("1216", 0)["risk_state_names"]
    assert cfg["n_risk_dims"] == 7
    assert cfg["use_risk_observation"] is True
    assert cfg["tilt_shift"] is True  # 惩罚前移一日 (buffer-side flag)


def test_1218_config():
    cfg = get_experiment_config("1218", 0)
    assert cfg["alpha"] == 0.0  # tilt isolated: ge penalty is the sole mechanism
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == get_experiment_config("1216", 0)["risk_state_names"]
    assert cfg["n_risk_dims"] == 7
    assert cfg["use_risk_observation"] is True
    assert cfg["tilt_shift"] is False
    assert cfg["ge_penalty_beta"] == 0.10
    assert cfg["ge_penalty_ref"] == 0.85
    # 1216/1217/1200 keep the ge-penalty defaults (no-op legacy behavior).
    for i in ("1200", "1215", "1216", "1217"):
        assert get_experiment_config(i, 0)["ge_penalty_beta"] == 0.0


def test_1219_config():
    cfg = get_experiment_config("1219", 0)
    assert cfg["alpha"] == 0.0  # 同 1218: 惩罚为唯一机制
    assert cfg["p_tail"] == 0.10
    assert cfg["risk_state_names"] == get_experiment_config("1216", 0)["risk_state_names"]
    assert cfg["n_risk_dims"] == 7
    assert cfg["use_risk_observation"] is True
    assert cfg["ge_penalty_mode"] == "per_stock"
    assert cfg["ge_penalty_beta"] == 0.10
    assert cfg["ge_penalty_ref"] == 0.85
    # 其余实验 mode 默认 total（1218 条目不写该字段 -> 默认值）
    for i in ("1200", "1215", "1216", "1217", "1218"):
        assert get_experiment_config(i, 0)["ge_penalty_mode"] == "total"


def test_all_v2_use_risk_observation():
    for i in _V2_IDS:
        assert get_experiment_config(i, 0)["use_risk_observation"] is True


def test_1200_baseline():
    cfg = get_experiment_config("1200", 3)
    assert cfg["risk_state_names"] == []
    assert cfg["n_risk_dims"] == 0
    assert cfg["alpha"] == 0.0
    assert cfg["p_tail"] == 0.025
    assert cfg["use_risk_observation"] is False


def test_val_test_key_names():
    cfg = get_experiment_config("1205", 0)
    assert cfg["val_test_start"] == "2025-01-01"
    assert cfg["val_test_end"] == "2026-06-01"
    assert "test_start" not in cfg
    assert "test_end" not in cfg


def test_unknown_id_raises_value_error():
    with pytest.raises(ValueError) as exc:
        get_experiment_config("9999", 0)
    msg = str(exc.value)
    assert "Unknown experiment_id" in msg
    for condition in ("1250", "1270b", "1321b_ctrl", "1321b_lo", "1321b_hi"):
        assert condition in _EXPERIMENT_TABLE
    for i in sorted(_EXPERIMENT_TABLE.keys()):
        assert i in msg


def test_1245_is_a_strict_1239_4c_config_rerun():
    rerun = get_experiment_config("1245", 0)
    original = get_experiment_config("1239", 0)
    for key in (
        "risk_state_names",
        "use_risk_observation",
        "alpha",
        "p_tail",
        "ge_penalty_beta",
        "direct_penalty_beta",
        "direct_penalty_ref",
        "use_mu_ref",
        "direct_ref_exec_aligned",
    ):
        assert rerun[key] == original[key], key
    assert rerun["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1245"


def test_1246_n1_retains_the_1239_mechanism_config():
    n1 = get_experiment_config("1246", 0)
    original = get_experiment_config("1239", 0)
    for key in (
        "risk_state_names",
        "use_risk_observation",
        "alpha",
        "p_tail",
        "ge_penalty_beta",
        "direct_penalty_beta",
        "direct_penalty_ref",
        "use_mu_ref",
        "direct_ref_exec_aligned",
    ):
        assert n1[key] == original[key], key
    assert n1["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1246"


def test_1247_n2_retains_the_1239_mechanism_config():
    n2 = get_experiment_config("1247", 0)
    original = get_experiment_config("1239", 0)
    for key in (
        "risk_state_names",
        "use_risk_observation",
        "alpha",
        "p_tail",
        "ge_penalty_beta",
        "direct_penalty_beta",
        "direct_penalty_ref",
        "use_mu_ref",
        "direct_ref_exec_aligned",
    ):
        assert n2[key] == original[key], key
    assert n2["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1247"


def test_1248_n3_retains_the_1239_mechanism_config():
    n3 = get_experiment_config("1248", 0)
    original = get_experiment_config("1239", 0)
    for key in (
        "risk_state_names",
        "use_risk_observation",
        "alpha",
        "p_tail",
        "ge_penalty_beta",
        "direct_penalty_beta",
        "direct_penalty_ref",
        "use_mu_ref",
        "direct_ref_exec_aligned",
    ):
        assert n3[key] == original[key], key
    assert n3["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1248"


def test_1250_r0_is_the_unified_code_301_dim_control():
    r0 = get_experiment_config("1250", 0)
    assert r0["risk_state_names"] == []
    assert r0["n_risk_dims"] == 0
    assert r0["use_risk_observation"] is False
    assert r0["alpha"] == 0.0
    assert r0["p_tail"] == 0.025
    assert r0["ge_penalty_beta"] == 0.0
    assert r0["direct_penalty_beta"] == 0.0
    assert r0["use_mu_ref"] is False
    assert r0["direct_ref_exec_aligned"] is False
    assert r0["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1250"


def test_output_and_artifact_fields():
    cfg = get_experiment_config("1207", 42)
    assert cfg["output_root"] == PROJECT_ROOT / "Results" / "Tail_GAE" / "1207"
    assert cfg["artifact_id"] == "1207 tail gae v2 7risk"
    assert cfg["notebook_name"] == "Tail_GAE_V2_7Risk_seed_42"
    assert cfg["train_start"] == "2013-04-01"
    assert cfg["train_end"] == "2024-12-31"


def test_registry_cross_check():
    risk_oracle = pytest.importorskip("risk_oracle")
    from risk_oracle import get_risk_dims

    for i in _V2_IDS:
        cfg = get_experiment_config(i, 0)
        assert len(get_risk_dims(cfg["risk_state_names"])) == len(
            cfg["risk_state_names"]
        ), f"{i}: unregistered risk state name"
