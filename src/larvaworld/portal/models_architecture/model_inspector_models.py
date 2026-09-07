"""
Records describing an inspected model and its modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class ModuleInspection:
    """One module of an inspected model, with its parameters."""

    module_id: str
    display_name: str
    present: bool
    mode: str | None
    parameters: dict[str, Any]
    is_baseline: bool


@dataclass(frozen=True)
class ModelInspection:
    """An inspected model, module by module."""

    model_id: str
    baseline_modules: tuple[ModuleInspection, ...]
    optional_modules: tuple[ModuleInspection, ...]


ModuleKind = Literal["brain", "memory", "larva"]
DraftValidationSeverity = Literal["warning", "error"]


@dataclass(frozen=True)
class ModelModuleSpec:
    """The modes and parameters one model module offers."""

    module_id: str
    display_name: str
    group: str
    subgroup: str
    module_kind: ModuleKind
    present: bool
    enabled: bool
    current_mode: str | None
    mode_options: tuple[str, ...]
    mode_labels: dict[str, str]
    parameters: dict[str, Any]
    current_modality: str | None = None
    modality_options_by_mode: dict[str, tuple[str, ...]] = field(default_factory=dict)
    is_core: bool = False


@dataclass(frozen=True)
class DraftValidationIssue:
    """One problem found while validating a model draft."""

    code: str
    severity: DraftValidationSeverity
    module_id: str
    path: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class ModuleComparison:
    """The differences between two versions of one module."""

    module_id: str
    primary: ModuleInspection
    comparison: ModuleInspection
    changed_fields: tuple[str, ...]
    equal: bool


@dataclass(frozen=True)
class ModuleComparisonMany:
    """The differences across several models' modules."""

    module_id: str
    #: One inspection per compared model, in the same order as the models
    #: passed to compare_model_inspections_many (primary first).
    inspections: tuple[ModuleInspection, ...]
    #: True if any non-primary model differs from the primary in
    #: presence, mode, or parameters.
    changed: bool


@dataclass(frozen=True)
class ProbeIssue:
    """One problem encountered while probing a model's behaviour."""

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeResult:
    """The outcome of previewing a model's behaviour."""

    model_id: str
    steps: int
    dt: float
    a_in: float
    dataframe: pd.DataFrame
    reporter_paths: dict[str, str]
    reporter_available: dict[str, bool]
    issues: tuple[ProbeIssue, ...] = ()


class ModelInspectorError(RuntimeError):
    """Raised when a model cannot be inspected or built."""

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
