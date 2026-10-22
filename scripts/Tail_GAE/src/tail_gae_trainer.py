"""Repo-local single-PPO rolling runner for AUST-compatible experiments.

The runner keeps the original AUST-Ensemble path isolated. ``dry_run=True``
only builds a plan. ``dry_run=False`` executes a PPO-only rolling train,
validation, and trade path under scripts/MDA_PPO/runs.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from .output_paths import OutputPathManager
    from tail_gae_env import (
        RISK_OBSERVATION_FIELDS,
        get_risk_observation_dim,
    )
except ImportError:
    from .output_paths import OutputPathManager
    from .single_ppo_env import (
        RISK_OBSERVATION_FIELDS,
        get_risk_observation_dim,
    )


FORBIDDEN_SINGLE_PPO_MODELS = {"a2c", "ddpg", "sac", "td3"}
SUPPORTED_REWARD_MODES = {"aust_original", "return", "mda"}
SUPPORTED_AUST_RUN_MODES = {"ensemble", "single_ppo"}
DEFAULT_AUST_LAMBDA_GRID = "0,0.03,0.1,0.3,1.0"
DEFAULT_OUTPUT_ROOT = "./scripts/MDA_PPO/runs"


@dataclass(frozen=True)
class RollingWindow:
    """AUST-style rolling window descriptor."""

    index: int
    iteration: int
    label: str


@dataclass(frozen=True)
class TrainingBudgetSummary:
    """Training-level budget accounting required before full experiments."""

    lambda_grid_size: int
    seed_count: int
    rolling_window_count: int
    ppo_timesteps_per_train: int
    candidate_validation_train_count: int
    candidate_validation_total_timesteps: int
    baseline_train_count: int
    final_trade_or_test_train_count: int
    estimated_total_ppo_train_count: int
    estimated_total_ppo_timesteps: int


class SinglePPORollingRunner:
    """Run an isolated PPO-only rolling experiment path."""

    def __init__(
        self,
        *,
        processed_data: Any,
        env_kwargs: dict[str, Any],
        train_period: tuple[str, str] | list[str],
        val_test_period: tuple[str, str] | list[str],
        rebalance_window: int,
        validation_window: int,
        ppo_model_kwargs: dict[str, Any],
        ppo_timesteps: int,
        reward_mode: str,
        lambda_value: float | None,
        lambda_grid: Iterable[float],
        clip_c: float,
        run_id: str,
        seed: int | Iterable[int],
        output_root: str | Path,
        use_risk_observation: bool = False,
        experiment_name: str | None = None,
        rolling_window_count: int | None = None,
        max_smoke_windows: int | None = None,
        drl_agent_cls: type[Any] | None = None,
        data_split_fn: Callable[..., Any] | None = None,
        env_cls: type[Any] | None = None,
        training_callback_factory: Callable[..., Any] | None = None,
        training_event_callback: Callable[[Mapping[str, Any]], None] | None = None,
        diagnostics_buffer_factory: Callable[..., Any] | None = None,
        normalization_config: Mapping[str, Any] | None = None,
        quality_gate_config: Mapping[str, Any] | None = None,
    ) -> None:
        if reward_mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(
                f"unsupported reward_mode={reward_mode!r}; "
                f"expected one of {sorted(SUPPORTED_REWARD_MODES)}"
            )
        if rebalance_window != 63:
            raise ValueError("PU-6 must preserve rebalance_window = 63")
        if validation_window != 63:
            raise ValueError("PU-6 must preserve validation_window = 63")
        if ppo_timesteps <= 0:
            raise ValueError("ppo_timesteps must be positive")
        if not run_id:
            raise ValueError("run_id must be explicit")
        if not isinstance(use_risk_observation, bool):
            raise TypeError("use_risk_observation must be a boolean")

        self.processed_data = processed_data
        self.env_kwargs = dict(env_kwargs)
        self.train_period = tuple(train_period)
        self.val_test_period = tuple(val_test_period)
        self.rebalance_window = int(rebalance_window)
        self.validation_window = int(validation_window)
        self.ppo_model_kwargs = dict(ppo_model_kwargs)
        self.ppo_timesteps = int(ppo_timesteps)
        self.reward_mode = reward_mode
        self.lambda_value = None if lambda_value is None else float(lambda_value)
        self.lambda_grid = [float(value) for value in lambda_grid]
        self.clip_c = float(clip_c)
        self.run_id = run_id
        self.seeds = self._normalize_seeds(seed)
        self.output_root = Path(output_root)
        self.use_risk_observation = use_risk_observation
        self.experiment_name = experiment_name or self._default_experiment_name()
        self.rolling_window_count = rolling_window_count
        self.max_smoke_windows = max_smoke_windows
        self.drl_agent_cls = drl_agent_cls
        self.data_split_fn = data_split_fn
        self.env_cls = env_cls
        self.training_callback_factory = training_callback_factory
        self.training_event_callback = training_event_callback
        self.diagnostics_buffer_factory = diagnostics_buffer_factory
        self.normalization_config = self._normalize_normalization_config(normalization_config)
        self.quality_gate_config = self._normalize_quality_gate_config(quality_gate_config)
        self._cached_unique_trade_dates: list[str] | None = None
        self._normalization_paths: dict[tuple[int, int], Path] = {}
        self._shared_normalization_paths: dict[int, Path] = {}
        self._reward_component_states: dict[int, dict[str, float]] = {}
        self._training_quality: dict[tuple[int, int], dict[str, float]] = {}
        self._observation_metadata: dict[str, Any] = {
            "use_risk_observation": self.use_risk_observation,
            "base_observation_dim": None,
            "risk_observation_dim": get_risk_observation_dim(),
            "risk_observation_fields": list(RISK_OBSERVATION_FIELDS),
            "policy_observation_dim": None,
        }
        self.path_manager = OutputPathManager(output_root=self.output_root)

        if self.reward_mode == "mda" and not self.lambda_grid:
            raise ValueError("lambda_grid must not be empty")
        self._assert_safe_output_root()

    def build_rolling_windows(self) -> list[RollingWindow]:
        """Build AUST-style rolling windows for planning and execution."""

        if self.rolling_window_count is not None:
            count = int(self.rolling_window_count)
            if self.max_smoke_windows is not None:
                count = min(count, int(self.max_smoke_windows))
            return [
                RollingWindow(index=index, iteration=index, label=f"dry_window_{index}")
                for index in range(count)
            ]

        dates = self._extract_dates(self.processed_data)
        if not dates:
            raise ValueError(
                "processed_data did not provide dates; set rolling_window_count "
                "for dry-run planning"
            )

        unique_trade_dates = [
            date
            for date in sorted(set(dates))
            if self.val_test_period[0] <= str(date) <= self.val_test_period[1]
        ]
        start = self.rebalance_window + self.validation_window
        iterations = range(start, len(unique_trade_dates), self.rebalance_window)
        windows = [
            RollingWindow(index=index, iteration=iteration, label=f"iter_{iteration}")
            for index, iteration in enumerate(iterations)
        ]
        if self.max_smoke_windows is not None:
            windows = windows[: int(self.max_smoke_windows)]
        return windows

    def estimate_training_budget(
        self, windows: list[RollingWindow] | None = None
    ) -> TrainingBudgetSummary:
        """Return PPO-only training budget accounting."""

        window_count = len(windows if windows is not None else self.build_rolling_windows())
        lambda_grid_size = len(self.lambda_grid) if self.reward_mode == "mda" else 0
        seed_count = len(self.seeds)
        candidate_train_count = lambda_grid_size * seed_count * window_count
        baseline_train_count = 0
        final_train_count = seed_count * window_count
        total_train_count = candidate_train_count + baseline_train_count + final_train_count
        return TrainingBudgetSummary(
            lambda_grid_size=lambda_grid_size,
            seed_count=seed_count,
            rolling_window_count=window_count,
            ppo_timesteps_per_train=self.ppo_timesteps,
            candidate_validation_train_count=candidate_train_count,
            candidate_validation_total_timesteps=candidate_train_count * self.ppo_timesteps,
            baseline_train_count=baseline_train_count,
            final_trade_or_test_train_count=final_train_count,
            estimated_total_ppo_train_count=total_train_count,
            estimated_total_ppo_timesteps=total_train_count * self.ppo_timesteps,
        )

    def dry_run(self) -> dict[str, Any]:
        """Return a serializable run plan without starting training."""

        windows = self.build_rolling_windows()
        budget = self.estimate_training_budget(windows)
        planned_output_paths = self._planned_output_paths(windows)
        plan = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "reward_mode": self.reward_mode,
            "planned_models": ["ppo"],
            "forbidden_models_present": False,
            "uses_ensemble_sharpe_selection": False,
            "modifies_aust_py": False,
            "modifies_site_packages": False,
            "rebalance_window": self.rebalance_window,
            "validation_window": self.validation_window,
            "ppo_timesteps": self.ppo_timesteps,
            "lambda_grid": self.lambda_grid,
            "seeds": self.seeds,
            "rolling_windows": [asdict(window) for window in windows],
            "training_budget": asdict(budget),
            "output_root": str(self.output_root),
            "planned_output_paths": planned_output_paths,
            "writes_original_results": False,
            "contract_safe": True,
            "normalization": self.normalization_config,
            "quality_gate": self.quality_gate_config,
            "observation": dict(self._observation_metadata),
        }
        self._assert_plan_contract(plan)
        return plan

    def run(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Run a dry plan or the full PPO-only rolling training path."""

        if dry_run:
            return self.dry_run()
        return self._run_full_training()

    def _run_full_training(self) -> dict[str, Any]:
        self._print_run_start()
        windows = self._iter_windows()
        if not windows:
            raise ValueError("no rolling windows available for full training")

        records: list[dict[str, Any]] = []
        quality_gate_results: list[dict[str, Any]] = []
        quality_gate_failure: dict[str, Any] | None = None
        for seed in self.seeds:
            previous_trade_state: Any | None = None
            previous_reward_state: Any | None = None
            previous_continuity_state: Any | None = None
            for window in windows:
                self._print_window_header(window=window, seed=seed)
                self._print_training_header(window=window)
                train_env = self._build_train_env(window, seed=seed)
                trained_model = self._train_ppo_for_window(
                    window=window,
                    train_env=train_env,
                    seed=seed,
                )
                validation_env = self._build_validation_env(window, seed=seed)
                self._print_validation_header(window=window)
                validation_path = self._run_validation_for_window(
                    window=window,
                    model=trained_model,
                    env=validation_env,
                    seed=seed,
                )
                self._print_validation_result(window=window, validation_path=validation_path)
                quality = self._evaluate_quality_gate(
                    window=window,
                    seed=seed,
                    validation_path=validation_path,
                )
                quality_gate_results.append(
                    {"seed": seed, "window_index": window.index, "iteration": window.iteration, **quality}
                )
                if not quality["passed"]:
                    blocked_record = {
                        "run_id": self.run_id,
                        "experiment_name": self.experiment_name,
                        "reward_mode": self.reward_mode,
                        "lambda_value": self._reported_lambda_value(),
                        "seed": seed,
                        "window_index": window.index,
                        "iteration": window.iteration,
                        "window_label": window.label,
                        "model_name": "ppo",
                        "ppo_timesteps": self.ppo_timesteps,
                        "validation_account_value_path": str(validation_path),
                        "trade_account_value_path": "",
                        "quality_gate_status": "blocked",
                        "quality_gate_reason": quality["reason"],
                        **self._observation_record_fields(),
                    }
                    if quality_gate_failure is None:
                        quality_gate_failure = blocked_record
                    if self.quality_gate_config["enforcement"] == "block":
                        records.append(blocked_record)
                        print(f"PPO quality gate blocked trade: {quality['reason']}")
                        break
                    print(
                        "PPO quality gate recorded an exploratory failure; "
                        f"continuing trade: {quality['reason']}"
                    )
                trade_env = self._build_trade_env(
                    window,
                    seed=seed,
                    previous_state=previous_trade_state,
                    previous_reward_state=previous_reward_state,
                    previous_continuity_state=previous_continuity_state,
                )
                self._print_trading_header(window=window)
                trade_path, previous_trade_state, previous_reward_state, previous_continuity_state = self._run_trade_for_window(
                    window=window,
                    model=trained_model,
                    env=trade_env,
                    seed=seed,
                )
                self._print_trade_result(window=window, trade_path=trade_path)
                records.append(
                    {
                        "run_id": self.run_id,
                        "experiment_name": self.experiment_name,
                        "reward_mode": self.reward_mode,
                        "lambda_value": self._reported_lambda_value(),
                        "seed": seed,
                        "window_index": window.index,
                        "iteration": window.iteration,
                        "window_label": window.label,
                        "model_name": "ppo",
                        "ppo_timesteps": self.ppo_timesteps,
                        "validation_account_value_path": str(validation_path),
                        "trade_account_value_path": str(trade_path),
                        "quality_gate_status": (
                            "passed" if quality["passed"] else "observed_failure"
                        ) if self.quality_gate_config["enabled"] else "disabled",
                        "quality_gate_reason": quality["reason"],
                        **self._observation_record_fields(),
                    }
                )
            if (
                quality_gate_failure is not None
                and self.quality_gate_config["enforcement"] == "block"
            ):
                break

        summary_path = self._write_runner_summary(records)
        self._print_run_end(summary_path=summary_path, windows=len(windows))
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "reward_mode": self.reward_mode,
            "planned_models": ["ppo"],
            "trained_models": ["ppo"],
            "forbidden_models_present": False,
            "uses_ensemble_sharpe_selection": False,
            "rolling_window_count": len(windows),
            "seed_count": len(self.seeds),
            "ppo_timesteps": self.ppo_timesteps,
            "output_root": str(self.output_root),
            "runner_summary_path": str(summary_path),
            "windows": records,
            "writes_original_results": False,
            "contract_safe": True,
            "normalization": self.normalization_config,
            "quality_gate": self.quality_gate_config,
            "quality_gate_failed": quality_gate_failure is not None,
            "quality_gate_failure": quality_gate_failure,
            "quality_gate_results": quality_gate_results,
            "observation": dict(self._observation_metadata),
        }

    def _print_run_start(self) -> None:
        print("============Start Single PPO Strategy============")
        print("============================================")
        print(f"run_id: {self.run_id}")
        print(f"experiment_name: {self.experiment_name}")
        print(f"reward_mode: {self.reward_mode}")
        print(f"ppo_timesteps: {self.ppo_timesteps}")
        print(f"seed(s): {self.seeds}")
        print(
            "risk_observation: "
            f"enabled={self.use_risk_observation}, "
            f"dim={get_risk_observation_dim()}, "
            f"fields={list(RISK_OBSERVATION_FIELDS)}"
        )

    def _print_window_header(self, *, window: RollingWindow, seed: int) -> None:
        print("============================================")
        print(f"window_index: {window.index}")
        print(f"iteration: {window.iteration}")
        print(f"window_label: {window.label}")
        print(f"seed: {seed}")

    def _print_training_header(self, *, window: RollingWindow) -> None:
        print("======PPO Training========")
        print(
            f"======Model training for window: {window.label} "
            f"({self.train_period[0]} -> {self._window_end_label(window=window, mode='train')})"
        )

    def _print_validation_header(self, *, window: RollingWindow) -> None:
        print(
            f"======PPO Validation from: "
            f"{self._window_start_label(window=window, mode='validation')} to "
            f"{self._window_end_label(window=window, mode='validation')}"
        )

    def _print_validation_result(self, *, window: RollingWindow, validation_path: Path) -> None:
        print(f"Validation account_value path: {validation_path}")

    def _print_trading_header(self, *, window: RollingWindow) -> None:
        print(
            f"======Trading from: "
            f"{self._window_start_label(window=window, mode='trade')} to "
            f"{self._window_end_label(window=window, mode='trade')}"
        )
        print("Used Model: PPO")

    def _print_trade_result(self, *, window: RollingWindow, trade_path: Path) -> None:
        print(f"Trade account_value path: {trade_path}")

    def _print_run_end(self, *, summary_path: Path, windows: int) -> None:
        print(f"Runner summary path: {summary_path}")
        print(f"Single PPO Strategy took: {windows} window(s)")
        print("============================================")

    def _window_start_label(self, *, window: RollingWindow, mode: str) -> str:
        if self.rolling_window_count is not None:
            if mode == "train":
                return str(self.train_period[0])
            return str(self.val_test_period[0])
        if mode == "validation":
            unique_trade_dates = self._unique_trade_dates()
            train_end_index = int(window.iteration) - self.rebalance_window - self.validation_window
            return str(unique_trade_dates[train_end_index])
        if mode == "trade":
            unique_trade_dates = self._unique_trade_dates()
            validation_end_index = int(window.iteration) - self.rebalance_window
            if window.index > 0:
                # FinRL's ``data_split`` has an exclusive end date.  Reusing
                # the previous terminal date makes the next action transition
                # start at exactly the carried portfolio valuation instead of
                # inventing an unobserved reset-time price bridge.
                validation_end_index -= 1
            return str(unique_trade_dates[validation_end_index])
        raise ValueError(f"unsupported mode={mode!r}")

    def _window_end_label(self, *, window: RollingWindow, mode: str) -> str:
        if self.rolling_window_count is not None:
            if mode == "train":
                return str(self.train_period[1])
            return str(self.val_test_period[1])
        unique_trade_dates = self._unique_trade_dates()
        if mode == "train":
            end_index = int(window.iteration) - self.rebalance_window - self.validation_window
        elif mode == "validation":
            end_index = int(window.iteration) - self.rebalance_window
        elif mode == "trade":
            end_index = int(window.iteration)
        else:
            raise ValueError(f"unsupported mode={mode!r}")
        return str(unique_trade_dates[end_index])

    def _iter_windows(self) -> list[RollingWindow]:
        return self.build_rolling_windows()

    def _train_ppo_for_window(
        self,
        *,
        window: RollingWindow,
        train_env: Any,
        seed: int,
    ) -> Any:
        drl_agent_cls = self._resolve_drl_agent_cls()
        agent = drl_agent_cls(env=train_env)
        model_kwargs = dict(self.ppo_model_kwargs)
        # FinRL passes policy_kwargs separately to SB3.  Keeping it inside
        # model_kwargs would pass the keyword twice and fail at construction.
        policy_kwargs = model_kwargs.pop("policy_kwargs", None)
        model = agent.get_model(
            "ppo",
            model_kwargs=model_kwargs,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )
        tb_log_name = f"ppo_{self.run_id}_{window.label}_seed_{seed}"
        callback = None
        if self.training_callback_factory is not None:
            callback = self.training_callback_factory(
                model_name="ppo",
                window=window,
                seed=seed,
            )

        started = time.time()
        event = {
            "model_name": "ppo",
            "window_index": window.index,
            "iteration": window.iteration,
            "window_label": window.label,
            "seed": seed,
            "tb_log_name": tb_log_name,
            "total_timesteps": self.ppo_timesteps,
            "status": "running",
        }
        try:
            train_kwargs = {
                "model": model,
                "tb_log_name": tb_log_name,
                "total_timesteps": self.ppo_timesteps,
            }
            if callback is not None:
                train_kwargs["callbacks"] = [callback]
            trained_model = agent.train_model(**train_kwargs)
            self._save_training_normalizer(train_env=train_env, window=window, seed=seed)
            self._save_reward_component_normalizer(train_env=train_env, seed=seed)
            event["status"] = "completed"
            return trained_model
        except Exception as exc:
            event["status"] = "failed"
            event["error"] = repr(exc)
            raise
        finally:
            flush = getattr(callback, "flush", None)
            if callable(flush):
                flush()
            event["elapsed_seconds"] = time.time() - started
            event["num_timesteps_after_training"] = int(
                getattr(model, "num_timesteps", 0)
            )
            quality_metrics = getattr(callback, "training_quality_metrics", None)
            if callable(quality_metrics):
                event["training_quality"] = quality_metrics()
                self._training_quality[(seed, window.index)] = dict(event["training_quality"])
            if self.training_event_callback is not None:
                self.training_event_callback(event)

    def _run_validation_for_window(
        self,
        *,
        window: RollingWindow,
        model: Any,
        env: Any,
        seed: int,
    ) -> Path:
        output_path = self._account_value_output_path(
            mode="validation",
            window=window,
            seed=seed,
        )
        account_value = self._predict_account_value(model=model, env=env)
        self._write_account_value(account_value, output_path)
        return output_path

    def _run_trade_for_window(
        self,
        *,
        window: RollingWindow,
        model: Any,
        env: Any,
        seed: int,
    ) -> tuple[Path, Any, Any, Any]:
        output_path = self._account_value_output_path(
            mode="trade",
            window=window,
            seed=seed,
        )
        account_value, last_state, reward_state, continuity_state = self._predict_trade_account_value_and_state(
            model=model,
            env=env,
        )
        self._write_account_value(account_value, output_path)
        return output_path, last_state, reward_state, continuity_state

    def _build_train_env(self, window: RollingWindow, *, seed: int) -> Any:
        raw_env = self._build_env(window=window, mode="train", seed=seed)
        if hasattr(raw_env, "get_sb_env"):
            sb_env = raw_env.get_sb_env()
            if isinstance(sb_env, tuple):
                vec_env = sb_env[0]
            else:
                vec_env = sb_env
            return self._wrap_training_env(vec_env=vec_env, seed=seed)
        return raw_env

    def _build_validation_env(self, window: RollingWindow, *, seed: int) -> Any:
        env = self._build_env(window=window, mode="validation", seed=seed)
        return self._wrap_evaluation_env(env=env, window=window, seed=seed)

    def _build_trade_env(
        self,
        window: RollingWindow,
        *,
        seed: int,
        previous_state: Any | None,
        previous_reward_state: Any | None,
        previous_continuity_state: Any | None,
    ) -> Any:
        env = self._build_env(
            window=window,
            mode="trade",
            seed=seed,
            previous_state=previous_state,
            previous_reward_state=previous_reward_state,
            previous_continuity_state=previous_continuity_state,
        )
        return self._wrap_evaluation_env(env=env, window=window, seed=seed)

    def _build_env(
        self,
        *,
        window: RollingWindow,
        mode: str,
        seed: int,
        previous_state: Any | None = None,
        previous_reward_state: Any | None = None,
        previous_continuity_state: Any | None = None,
    ) -> Any:
        env_cls = self._resolve_env_cls()
        data = self._window_data(window=window, mode=mode)
        kwargs = self._env_kwargs_for_mode(
            mode=mode,
            seed=seed,
            previous_state=previous_state,
        )
        if mode == "trade":
            kwargs["reward_accounting_state"] = previous_reward_state
            kwargs["expected_continuity_state"] = previous_continuity_state
        diagnostics_buffer = None
        if self.diagnostics_buffer_factory is not None:
            diagnostics_buffer = self.diagnostics_buffer_factory(
                mode=mode,
                window=window,
                seed=seed,
            )
        env = env_cls(
            df=data,
            reward_mode=self.reward_mode,
            lambda_value=self._effective_lambda_value(),
            clip_c=self.clip_c,
            diagnostics_buffer=diagnostics_buffer,
            use_risk_observation=self.use_risk_observation,
            **kwargs,
        )
        self._register_observation_metadata(env)
        return env

    def _window_data(self, *, window: RollingWindow, mode: str) -> Any:
        start, end = self._window_period(window=window, mode=mode)
        data_split_fn = self._resolve_data_split_fn()
        return data_split_fn(self.processed_data, start, end)

    def _window_period(self, *, window: RollingWindow, mode: str) -> tuple[str, str]:
        if self.rolling_window_count is not None:
            if mode == "train":
                return str(self.train_period[0]), str(self.train_period[1])
            return str(self.val_test_period[0]), str(self.val_test_period[1])

        unique_trade_dates = self._unique_trade_dates()
        iteration = int(window.iteration)
        train_end_index = iteration - self.rebalance_window - self.validation_window
        validation_end_index = iteration - self.rebalance_window
        trade_end_index = iteration
        if train_end_index < 0 or trade_end_index >= len(unique_trade_dates):
            raise ValueError(f"rolling window {window.label} is outside available dates")

        train_end = unique_trade_dates[train_end_index]
        validation_end = unique_trade_dates[validation_end_index]
        trade_end = unique_trade_dates[trade_end_index]
        if mode == "train":
            return str(self.train_period[0]), str(train_end)
        if mode == "validation":
            return str(train_end), str(validation_end)
        if mode == "trade":
            if window.index > 0:
                if validation_end_index <= 0:
                    raise ValueError("trade continuity start is outside available dates")
                return str(unique_trade_dates[validation_end_index - 1]), str(trade_end)
            return str(validation_end), str(trade_end)
        raise ValueError(f"unsupported mode={mode!r}")

    def _predict_account_value(self, *, model: Any, env: Any) -> Any:
        drl_agent_cls = self._resolve_drl_agent_cls()
        if not hasattr(drl_agent_cls, "DRL_prediction"):
            raise AttributeError("DRLAgent must provide DRL_prediction for validation/trade")
        prediction = drl_agent_cls.DRL_prediction(model=model, environment=env)
        if isinstance(prediction, tuple):
            return prediction[0]
        return prediction

    def _predict_trade_account_value_and_state(
        self,
        *,
        model: Any,
        env: Any,
    ) -> tuple[Any, Any, Any, Any]:
        if not hasattr(env, "get_sb_env"):
            raise AttributeError("trade environment must provide get_sb_env")
        if not hasattr(env, "df"):
            raise AttributeError("trade environment must expose df for trade length")

        test_env, test_obs = env.get_sb_env()
        trade_day_count = len(env.df.index.unique())
        last_state = None
        last_reward_state = None
        last_continuity_state = None
        account_memory = None

        for i in range(trade_day_count):
            action, _states = model.predict(test_obs, deterministic=True)
            test_obs, rewards, dones, info = test_env.step(action)

            if i == trade_day_count - 2:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                # ``render`` may include the schema-defined risk observation.
                # FinRL's next trade environment must receive only its original
                # portfolio state; risk continuity travels separately below.
                last_state = test_env.env_method(method_name="get_base_state")
                last_reward_state = test_env.env_method(method_name="get_reward_accounting_state")
                last_continuity_state = test_env.env_method(method_name="get_account_continuity_state")

            if dones[0]:
                break

        if account_memory is None or last_state is None:
            raise RuntimeError(
                "failed to capture trade terminal state at the expected step; "
                "trade loop may have ended early or the capture index is wrong"
            )

        return account_memory[0], last_state[0], None if last_reward_state is None else last_reward_state[0], None if last_continuity_state is None else last_continuity_state[0]

    def _write_account_value(self, account_value: Any, path: Path) -> None:
        self._assert_safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(account_value, "to_csv"):
            account_value.to_csv(path, index=False)
            return
        rows = self._normalize_account_value_rows(account_value)
        fieldnames = sorted({key for row in rows for key in row}) or ["account_value"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_runner_summary(self, records: list[dict[str, Any]]) -> Path:
        path = self.path_manager.runner_summary_path(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
        )
        self._assert_safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "run_id",
            "experiment_name",
            "reward_mode",
            "lambda_value",
            "seed",
            "window_index",
            "iteration",
            "window_label",
            "model_name",
            "ppo_timesteps",
            "validation_account_value_path",
            "trade_account_value_path",
            "quality_gate_status",
            "quality_gate_reason",
            "use_risk_observation",
            "base_observation_dim",
            "risk_observation_dim",
            "risk_observation_fields",
            "policy_observation_dim",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return path

    def _observation_record_fields(self) -> dict[str, Any]:
        metadata = self._observation_metadata
        return {
            "use_risk_observation": metadata["use_risk_observation"],
            "base_observation_dim": metadata["base_observation_dim"],
            "risk_observation_dim": metadata["risk_observation_dim"],
            "risk_observation_fields": json.dumps(
                metadata["risk_observation_fields"],
                separators=(",", ":"),
            ),
            "policy_observation_dim": metadata["policy_observation_dim"],
        }

    def _register_observation_metadata(self, env: Any) -> None:
        if not hasattr(env, "get_observation_metadata"):
            return
        metadata = dict(env.get_observation_metadata())
        expected_fields = list(RISK_OBSERVATION_FIELDS)
        if metadata.get("use_risk_observation") is not self.use_risk_observation:
            raise RuntimeError("environment did not apply use_risk_observation")
        if metadata.get("risk_observation_fields") != expected_fields:
            raise RuntimeError("environment risk observation fields do not match the schema")
        if metadata.get("risk_observation_dim") != len(expected_fields):
            raise RuntimeError("environment risk observation dimension does not match the schema")
        base_dim = int(metadata["base_observation_dim"])
        policy_dim = int(metadata["policy_observation_dim"])
        expected_policy_dim = base_dim
        if self.use_risk_observation:
            expected_policy_dim += len(expected_fields)
        if policy_dim != expected_policy_dim:
            raise RuntimeError("environment policy observation dimension is inconsistent")
        normalized = {
            "use_risk_observation": self.use_risk_observation,
            "base_observation_dim": base_dim,
            "risk_observation_dim": len(expected_fields),
            "risk_observation_fields": expected_fields,
            "policy_observation_dim": policy_dim,
        }
        current_base_dim = self._observation_metadata["base_observation_dim"]
        if current_base_dim is not None and self._observation_metadata != normalized:
            raise RuntimeError(
                "training, validation, and trade observation contracts differ"
            )
        self._observation_metadata = normalized

    def _account_value_output_path(
        self,
        *,
        mode: str,
        window: RollingWindow,
        seed: int,
    ) -> Path:
        iteration = window.label if len(self.seeds) == 1 else f"{window.label}_seed_{seed}"
        return self.path_manager.account_value_path(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            mode=mode,
            iteration=iteration,
        )

    def _planned_output_paths(self, windows: list[RollingWindow]) -> list[str]:
        paths: list[str] = []
        for window in windows:
            for seed in self.seeds:
                paths.append(str(self._account_value_output_path(mode="validation", window=window, seed=seed)))
                paths.append(str(self._account_value_output_path(mode="trade", window=window, seed=seed)))
        paths.append(
            str(
                self.path_manager.runner_summary_path(
                    run_id=self.run_id,
                    experiment_name=self.experiment_name,
                )
            )
        )
        for path in paths:
            self._assert_safe_path(Path(path))
        return paths

    def _assert_plan_contract(self, plan: dict[str, Any]) -> None:
        planned_models = set(plan["planned_models"])
        forbidden = planned_models.intersection(FORBIDDEN_SINGLE_PPO_MODELS)
        if forbidden:
            raise AssertionError(f"dry-run planned forbidden models: {sorted(forbidden)}")
        if plan["uses_ensemble_sharpe_selection"]:
            raise AssertionError("single-PPO dry-run must not use ensemble Sharpe selection")
        if plan["writes_original_results"]:
            raise AssertionError("single-PPO dry-run must not write original results/")

    def _assert_safe_output_root(self) -> None:
        parts = self.output_root.as_posix().split("/")
        if "results" in parts and "MDA-PPO" not in parts and "MDA_PPO" not in parts:
            raise ValueError("output_root must not target the original results/ directory")
        if "runs" not in parts:
            raise ValueError("output_root must point under scripts/MDA_PPO/runs")

    def _assert_safe_path(self, path: Path) -> None:
        normalized = path.as_posix()
        if normalized.startswith("results/") or "/results/" in normalized:
            raise ValueError(f"planned path targets original results/: {normalized}")
        if "scripts/MDA-PPO/runs" not in normalized and "scripts/MDA_PPO/runs" not in normalized:
            raise ValueError(f"planned path is outside scripts/MDA_PPO/runs: {normalized}")

    def _default_experiment_name(self) -> str:
        if self.reward_mode == "aust_original":
            return "aust_ppo"
        if self.reward_mode == "return":
            return "return_ppo"
        return "mda_ppo"

    def _effective_lambda_value(self) -> float:
        if self.reward_mode == "mda":
            if self.lambda_value is None:
                raise ValueError("MDA full training requires fixed lambda_value")
            return float(self.lambda_value)
        return 0.0 if self.lambda_value is None else float(self.lambda_value)

    def _reported_lambda_value(self) -> float | None:
        if self.reward_mode == "mda":
            return self._effective_lambda_value()
        return self.lambda_value

    def _env_kwargs_for_mode(
        self,
        *,
        mode: str,
        seed: int,
        previous_state: Any | None = None,
    ) -> dict[str, Any]:
        kwargs = dict(self.env_kwargs)
        if "stock_dim" in kwargs:
            stock_dim = int(kwargs["stock_dim"])
            if "num_stock_shares" not in kwargs:
                kwargs["num_stock_shares"] = [0] * stock_dim
            if "buy_cost_pct" in kwargs:
                kwargs["buy_cost_pct"] = self._normalize_per_stock_float_arg(
                    kwargs["buy_cost_pct"],
                    stock_dim=stock_dim,
                    name="buy_cost_pct",
                )
            if "sell_cost_pct" in kwargs:
                kwargs["sell_cost_pct"] = self._normalize_per_stock_float_arg(
                    kwargs["sell_cost_pct"],
                    stock_dim=stock_dim,
                    name="sell_cost_pct",
                )
        kwargs.setdefault("mode", mode)
        kwargs.setdefault("iteration", 0)
        kwargs.setdefault("model_name", "ppo")
        kwargs.setdefault("print_verbosity", 0)
        if self.normalization_config["normalize_mda_components"] and self.reward_mode == "mda":
            kwargs["component_normalization_state"] = self._reward_component_states.get(seed)
            kwargs["component_normalization_training"] = (
                mode == "train"
                and (
                    not self.normalization_config["fit_mda_components_once_on_meta_train"]
                    or seed not in self._reward_component_states
                )
            )
            kwargs["enable_component_normalization"] = True

        if mode in {"train", "validation"}:
            kwargs["initial"] = True
            kwargs.pop("previous_state", None)
            return kwargs

        if mode == "trade":
            if previous_state is None:
                kwargs["initial"] = True
                kwargs.pop("previous_state", None)
            else:
                kwargs["initial"] = False
                kwargs["previous_state"] = previous_state
            return kwargs

        raise ValueError(f"unsupported mode={mode!r}")

    def _wrap_training_env(self, *, vec_env: Any, seed: int) -> Any:
        if not self.normalization_config["enabled"]:
            return vec_env
        shared_path = self._shared_normalization_paths.get(seed)
        if self.normalization_config["fit_once_on_meta_train"] and shared_path is not None:
            try:
                from .ppo_normalization import load_frozen_normalizer
            except ImportError:
                from .ppo_normalization import load_frozen_normalizer
            return load_frozen_normalizer(vec_env, shared_path)
        try:
            from .ppo_normalization import make_training_normalizer
        except ImportError:
            from .ppo_normalization import make_training_normalizer
        return make_training_normalizer(
            vec_env,
            normalize_observations=self.normalization_config["normalize_observations"],
            normalize_rewards=self.normalization_config["normalize_rewards"],
        )

    def _save_training_normalizer(self, *, train_env: Any, window: RollingWindow, seed: int) -> None:
        if not self.normalization_config["enabled"]:
            return
        save = getattr(train_env, "save", None)
        if not callable(save):
            raise TypeError("normalization enabled but training environment cannot save statistics")
        path = self.output_root / "normalization" / f"ppo_{self.run_id}_seed_{seed}.pkl"
        self._assert_safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save(str(path))
        self._normalization_paths[(seed, window.index)] = path
        if self.normalization_config["fit_once_on_meta_train"]:
            self._shared_normalization_paths.setdefault(seed, path)
        else:
            self._shared_normalization_paths[seed] = path

    def _save_reward_component_normalizer(self, *, train_env: Any, seed: int) -> None:
        if not self.normalization_config["normalize_mda_components"] or self.reward_mode != "mda":
            return
        if (
            self.normalization_config["fit_mda_components_once_on_meta_train"]
            and seed in self._reward_component_states
        ):
            return
        env_method = getattr(train_env, "env_method", None)
        if not callable(env_method):
            raise TypeError("component normalization requires VecEnv.env_method")
        states = env_method("get_reward_component_normalization_state")
        if not states or not isinstance(states[0], Mapping):
            raise ValueError("training environment did not return component normalization state")
        state = {str(key): float(value) for key, value in states[0].items()}
        self._reward_component_states[seed] = state
        path = self.output_root / "normalization" / f"mda_components_{self.run_id}_seed_{seed}.json"
        self._assert_safe_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def _wrap_evaluation_env(self, *, env: Any, window: RollingWindow, seed: int) -> Any:
        if not self.normalization_config["enabled"]:
            return env
        path = self._shared_normalization_paths.get(seed)
        if path is None or not path.exists():
            raise FileNotFoundError("missing training-fitted normalization statistics")
        try:
            from .ppo_normalization import FrozenNormalizationEnv
        except ImportError:
            from .ppo_normalization import FrozenNormalizationEnv
        return FrozenNormalizationEnv(env, path)

    def _evaluate_quality_gate(self, *, window: RollingWindow, seed: int, validation_path: Path) -> dict[str, Any]:
        if not self.quality_gate_config["enabled"]:
            return {"passed": True, "reason": "quality gate disabled", "metrics": {}, "thresholds": self.quality_gate_config}
        metrics = self._training_quality.get((seed, window.index), {})
        required = (
            "late_explained_variance_median",
            "late_explained_variance_negative_share",
            "late_approx_kl_median",
            "late_clip_fraction_median",
        )
        if not all(key in metrics for key in required):
            return {"passed": False, "reason": "incomplete PPO training diagnostics", "metrics": metrics, "thresholds": self.quality_gate_config}
        ev = metrics["late_explained_variance_median"]
        negative_share = metrics["late_explained_variance_negative_share"]
        kl = metrics["late_approx_kl_median"]
        if ev < self.quality_gate_config["min_explained_variance"]:
            return {"passed": False, "reason": f"late EV median {ev:.4g} below threshold", "metrics": metrics, "thresholds": self.quality_gate_config}
        if negative_share > self.quality_gate_config["max_negative_ev_share"]:
            return {"passed": False, "reason": f"late negative EV share {negative_share:.2%} above threshold", "metrics": metrics, "thresholds": self.quality_gate_config}
        if kl > self.quality_gate_config["max_approx_kl"]:
            return {"passed": False, "reason": f"late KL median {kl:.4g} above threshold", "metrics": metrics, "thresholds": self.quality_gate_config}
        # clip fraction remains in `metrics` for monitoring only. It is not
        # an eligibility criterion.
        validation_return = self._validation_total_return(validation_path)
        if validation_return is None:
            return {"passed": False, "reason": "missing validation account values", "metrics": metrics, "thresholds": self.quality_gate_config}
        metrics = {**metrics, "validation_total_return": validation_return}
        if validation_return < self.quality_gate_config["min_validation_total_return"]:
            return {"passed": False, "reason": f"validation total return {validation_return:.2%} below threshold", "metrics": metrics, "thresholds": self.quality_gate_config}
        return {"passed": True, "reason": "training and validation quality checks passed", "metrics": metrics, "thresholds": self.quality_gate_config}

    @staticmethod
    def _validation_total_return(path: Path) -> float | None:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < 2:
            return None
        try:
            start = float(rows[0]["account_value"])
            end = float(rows[-1]["account_value"])
        except (KeyError, TypeError, ValueError):
            return None
        return None if start == 0 else end / start - 1.0

    @staticmethod
    def _normalize_normalization_config(config: Mapping[str, Any] | None) -> dict[str, bool]:
        values = dict(config or {})
        return {
            "enabled": bool(values.get("enabled", False)),
            "normalize_observations": bool(values.get("normalize_observations", True)),
            "normalize_rewards": bool(values.get("normalize_rewards", False)),
            "fit_once_on_meta_train": bool(values.get("fit_once_on_meta_train", True)),
            "normalize_mda_components": bool(values.get("normalize_mda_components", False)),
            "fit_mda_components_once_on_meta_train": bool(
                values.get(
                    "fit_mda_components_once_on_meta_train",
                    values.get("fit_once_on_meta_train", True),
                )
            ),
        }

    @staticmethod
    def _normalize_quality_gate_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
        values = dict(config or {})
        enforcement = str(values.get("enforcement", "block")).strip().lower()
        if enforcement not in {"block", "observe"}:
            raise ValueError("quality gate enforcement must be 'block' or 'observe'")
        return {
            "enabled": bool(values.get("enabled", False)),
            "enforcement": enforcement,
            "min_explained_variance": float(values.get("min_explained_variance", 0.0)),
            "max_negative_ev_share": float(values.get("max_negative_ev_share", 0.5)),
            "max_approx_kl": float(values.get("max_approx_kl", 0.03)),
            "min_validation_total_return": float(values.get("min_validation_total_return", 0.0)),
        }

    @staticmethod
    def _normalize_per_stock_float_arg(
        value: Any,
        *,
        stock_dim: int,
        name: str,
    ) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)] * stock_dim
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            if len(value) != stock_dim:
                raise ValueError(
                    f"{name} length must equal stock_dim; "
                    f"got len={len(value)} stock_dim={stock_dim}"
                )
            return [float(item) for item in value]
        raise TypeError(
            f"{name} must be a float or a list/tuple of length stock_dim; "
            f"got {type(value).__name__}"
        )

    def _resolve_drl_agent_cls(self) -> type[Any]:
        if self.drl_agent_cls is not None:
            return self.drl_agent_cls
        try:
            from finrl.agents.stablebaselines3.models import DRLAgent
        except ImportError as exc:
            raise ImportError("FinRL DRLAgent is required for dry_run=False") from exc
        return DRLAgent

    def _resolve_data_split_fn(self) -> Callable[..., Any]:
        if self.data_split_fn is not None:
            return self.data_split_fn
        try:
            from finrl.meta.preprocessor.preprocessors import data_split
        except ImportError as exc:
            raise ImportError("FinRL data_split is required for dry_run=False") from exc
        return data_split

    def _resolve_env_cls(self) -> type[Any]:
        if self.env_cls is not None:
            return self.env_cls
        try:
            from tail_gae_env import SinglePPOStockTradingEnv
        except ImportError:
            from .single_ppo_env import SinglePPOStockTradingEnv
        return SinglePPOStockTradingEnv

    def _unique_trade_dates(self) -> list[str]:
        if self._cached_unique_trade_dates is None:
            dates = self._extract_dates(self.processed_data)
            self._cached_unique_trade_dates = [
                date
                for date in sorted(set(dates))
                if self.val_test_period[0] <= str(date) <= self.val_test_period[1]
            ]
        return self._cached_unique_trade_dates

    @staticmethod
    def _normalize_seeds(seed: int | Iterable[int]) -> list[int]:
        if isinstance(seed, int):
            return [seed]
        seeds = [int(value) for value in seed]
        if not seeds:
            raise ValueError("seed list must not be empty")
        return seeds

    @staticmethod
    def _extract_dates(processed_data: Any) -> list[str]:
        if processed_data is None:
            return []
        if isinstance(processed_data, dict) and "date" in processed_data:
            return [str(value) for value in processed_data["date"]]
        if hasattr(processed_data, "__getitem__"):
            try:
                return [str(value) for value in processed_data["date"]]
            except Exception:
                return []
        return []

    @staticmethod
    def _normalize_account_value_rows(account_value: Any) -> list[dict[str, Any]]:
        if account_value is None:
            return []
        if hasattr(account_value, "to_dict"):
            try:
                return [dict(row) for row in account_value.to_dict("records")]
            except TypeError:
                pass
        if isinstance(account_value, list):
            if not account_value:
                return []
            if isinstance(account_value[0], dict):
                return [dict(row) for row in account_value]
            return [{"account_value": value} for value in account_value]
        if isinstance(account_value, dict):
            return [dict(account_value)]
        return [{"account_value": account_value}]

def parse_aust_run_mode(environ: dict[str, str] | None = None) -> str:
    """Parse the optional AUST entrypoint mode without importing MDA modules."""

    import os

    env = os.environ if environ is None else environ
    mode = env.get("AUST_RUN_MODE", "ensemble").strip().lower()
    if mode not in SUPPORTED_AUST_RUN_MODES:
        raise ValueError(
            f"unsupported AUST_RUN_MODE={mode!r}; "
            f"expected one of {sorted(SUPPORTED_AUST_RUN_MODES)}"
        )
    return mode


def build_single_ppo_runner_from_aust_context(
    *,
    processed_data: Any,
    env_kwargs: dict[str, Any],
    train_period: tuple[str, str],
    val_test_period: tuple[str, str],
    rebalance_window: int,
    validation_window: int,
    ppo_model_kwargs: dict[str, Any],
    ppo_timesteps: int,
    environ: dict[str, str] | None = None,
    runner_cls: type[SinglePPORollingRunner] = SinglePPORollingRunner,
) -> SinglePPORollingRunner:
    """Build the single-PPO runner from explicit AUST entrypoint settings."""

    import os

    env = os.environ if environ is None else environ
    reward_mode = env.get("MDA_PPO_REWARD_MODE", "mda").strip().lower()
    lambda_value = _parse_optional_float(env.get("MDA_PPO_LAMBDA"))
    lambda_grid = _parse_lambda_grid(
        env.get("MDA_PPO_LAMBDA_GRID", DEFAULT_AUST_LAMBDA_GRID)
    )
    clip_c = float(env.get("MDA_PPO_CLIP_C", "0.02"))
    run_id = env.get("MDA_PPO_RUN_ID", "aust_entry_smoke").strip()
    seed = int(env.get("MDA_PPO_SEED", "0"))
    max_windows = _parse_optional_int(env.get("MDA_PPO_MAX_WINDOWS"))
    effective_timesteps = int(env.get("MDA_PPO_PPO_TIMESTEPS", ppo_timesteps))
    output_root = env.get("MDA_PPO_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
    use_risk_observation = parse_env_bool(
        env.get("MDA_PPO_USE_RISK_OBSERVATION"),
        default=True,
    )

    return runner_cls(
        processed_data=processed_data,
        env_kwargs=env_kwargs,
        train_period=train_period,
        val_test_period=val_test_period,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
        ppo_model_kwargs=ppo_model_kwargs,
        ppo_timesteps=effective_timesteps,
        reward_mode=reward_mode,
        lambda_value=lambda_value,
        lambda_grid=lambda_grid,
        clip_c=clip_c,
        run_id=run_id,
        seed=seed,
        output_root=output_root,
        use_risk_observation=use_risk_observation,
        max_smoke_windows=max_windows,
    )


def parse_env_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("MDA_PPO_MAX_WINDOWS must be positive")
    return parsed


def _parse_lambda_grid(value: str) -> list[float]:
    grid = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not grid:
        raise ValueError("MDA_PPO_LAMBDA_GRID must contain at least one value")
    return grid
