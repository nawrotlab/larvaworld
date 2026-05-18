from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModuleInspection:
    module_id: str
    display_name: str
    present: bool
    mode: str | None
    parameters: dict[str, Any]
    is_baseline: bool


@dataclass(frozen=True)
class ModelInspection:
    model_id: str
    baseline_modules: tuple[ModuleInspection, ...]
    optional_modules: tuple[ModuleInspection, ...]


@dataclass(frozen=True)
class ModuleComparison:
    module_id: str
    primary: ModuleInspection
    comparison: ModuleInspection
    changed_fields: tuple[str, ...]
    equal: bool


@dataclass(frozen=True)
class ProbeIssue:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    model_id: str
    steps: int
    dt: float
    a_in: float
    dataframe: pd.DataFrame
    reporter_paths: dict[str, str]
    reporter_available: dict[str, bool]
    issues: tuple[ProbeIssue, ...] = ()


class ModelInspectorError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = context or {}
