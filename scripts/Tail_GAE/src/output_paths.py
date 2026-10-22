"""PU-4 output path isolation for MDA-PPO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


# Legacy standard names retained for callers that import this constant.
# OutputPathManager also accepts explicitly named, safe experiment IDs.
EXPERIMENT_NAMES = {
    "aust_ppo",
    "return_ppo",
    "mda_ppo",
    "aust_ensemble_reference",
}

_EXPERIMENT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class OutputPathManager:
    """Generate isolated run paths and reject original results/ writes."""

    output_root: str | Path = Path("scripts/MDA_PPO/runs")

    def __post_init__(self) -> None:
        root = Path(self.output_root)
        _assert_not_original_results(root)
        _assert_under_mda_runs(root)
        object.__setattr__(self, "output_root", root)

    def experiment_dir(self, *, run_id: str, experiment_name: str) -> Path:
        self._validate_run_id(run_id)
        self._validate_experiment_name(experiment_name)
        return self._checked_path(Path(self.output_root) / run_id / experiment_name)

    def account_value_path(
        self,
        *,
        run_id: str,
        experiment_name: str,
        mode: str,
        iteration: str | int,
    ) -> Path:
        mode = self._validate_mode(mode)
        iteration = self._validate_iteration(iteration)
        path = (
            self.experiment_dir(run_id=run_id, experiment_name=experiment_name)
            / f"account_value_{mode}_{experiment_name}_{iteration}.csv"
        )
        return self._checked_path(path)

    def diagnostics_path(
        self,
        *,
        run_id: str,
        experiment_name: str,
        mode: str,
        iteration: str | int,
    ) -> Path:
        mode = self._validate_mode(mode)
        iteration = self._validate_iteration(iteration)
        path = (
            Path(self.output_root)
            / run_id
            / "diagnostics"
            / experiment_name
            / f"diagnostics_{mode}_{experiment_name}_{iteration}.csv"
        )
        self._validate_run_id(run_id)
        self._validate_experiment_name(experiment_name)
        return self._checked_path(path)

    def runner_summary_path(self, *, run_id: str, experiment_name: str) -> Path:
        return self._checked_path(
            self.experiment_dir(run_id=run_id, experiment_name=experiment_name)
            / "runner_summary.csv"
        )

    def lambda_selection_dir(self, *, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self._checked_path(Path(self.output_root) / run_id / "lambda_selection")

    def lambda_selection_summary_path(self, *, run_id: str) -> Path:
        return self._checked_path(
            self.lambda_selection_dir(run_id=run_id) / "lambda_selection_summary.csv"
        )

    def selected_lambda_path(self, *, run_id: str) -> Path:
        return self._checked_path(
            self.lambda_selection_dir(run_id=run_id) / "selected_lambda.json"
        )

    def _checked_path(self, path: Path) -> Path:
        _assert_not_original_results(path)
        _assert_under_mda_runs(path)
        return path

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty path segment")

    @staticmethod
    def _validate_experiment_name(experiment_name: str) -> None:
        if not _EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name):
            raise ValueError(
                "experiment_name must contain only lowercase letters, digits, "
                "underscores, or hyphens; it must start with a lowercase "
                f"letter or digit, got {experiment_name!r}"
            )

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in {"validation", "trade"}:
            raise ValueError("mode must be 'validation' or 'trade'")
        return mode

    @staticmethod
    def _validate_iteration(iteration: str | int) -> str:
        value = str(iteration)
        if not value or "/" in value or "\\" in value:
            raise ValueError("iteration must be a non-empty path segment")
        return value


def _assert_not_original_results(path: Path) -> None:
    parts = path.parts
    if path.as_posix() == "results" or (parts and parts[0] == "results"):
        raise ValueError(f"refusing to write under original results/: {path}")
    if "results" in parts and "MDA-PPO" not in parts and "MDA_PPO" not in parts:
        raise ValueError(f"refusing to write under original results/: {path}")


def _assert_under_mda_runs(path: Path) -> None:
    parts = path.as_posix().split("/")
    for index in range(0, len(parts) - 2):
        if parts[index : index + 3] in (
            ["scripts", "MDA-PPO", "runs"],
            ["scripts", "MDA_PPO", "runs"],
        ):
            return
    raise ValueError(f"path must be under scripts/MDA_PPO/runs: {path}")
