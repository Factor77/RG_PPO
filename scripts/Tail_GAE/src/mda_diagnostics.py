"""PU-4 diagnostics buffer and explicit CSV flushing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence


COMMON_REWARD_RESULT_FIELDS = (
    "raw_return",
    "lambda_value",
    "final_reward",
)

MDA_REWARD_RESULT_FIELDS = (
    *COMMON_REWARD_RESULT_FIELDS[:1],
    "prev_peak",
    "new_peak",
    "new_drawdown",
    "prev_drawdown",
    "drawdown_increase",
    "prev_max_drawdown",
    "new_max_drawdown",
    "c_t",
    *COMMON_REWARD_RESULT_FIELDS[1:],
)

# Backward-compatible default schema for existing MDA diagnostics callers.
REWARD_RESULT_FIELDS = MDA_REWARD_RESULT_FIELDS

REWARD_RESULT_FIELDS_BY_MODE = {
    "aust_original": COMMON_REWARD_RESULT_FIELDS,
    "return": COMMON_REWARD_RESULT_FIELDS,
    "mda": MDA_REWARD_RESULT_FIELDS,
}


def required_reward_fields(reward_mode: str) -> tuple[str, ...]:
    try:
        return REWARD_RESULT_FIELDS_BY_MODE[reward_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported reward_mode={reward_mode!r}") from exc


class DiagnosticsBuffer:
    """Collect diagnostics records without changing reward/state/action."""

    def __init__(self, required_fields: Sequence[str] = REWARD_RESULT_FIELDS) -> None:
        self.required_fields = tuple(required_fields)
        self._records: list[dict[str, object]] = []

    @property
    def records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._records)

    def record_step(self, record: Mapping[str, object]) -> Mapping[str, object]:
        missing = [field for field in self.required_fields if field not in record]
        if missing:
            raise ValueError(f"diagnostic record missing fields: {missing}")
        copied = dict(record)
        self._records.append(copied)
        return record

    def clear(self) -> None:
        self._records.clear()

    def flush_csv(
        self,
        path: str | Path,
        *,
        fieldnames: Sequence[str] | None = None,
    ) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        names = list(fieldnames) if fieldnames is not None else self._fieldnames()
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for record in self._records:
                writer.writerow(record)
        return output_path

    def _fieldnames(self) -> list[str]:
        seen: list[str] = []
        for field in self.required_fields:
            if field not in seen:
                seen.append(field)
        for record in self._records:
            for field in record:
                if field not in seen:
                    seen.append(field)
        return seen
