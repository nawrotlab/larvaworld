"""Pure data layer for the portal Module Inspector (no Panel/Bokeh).

Builds standalone crawler/turner instances from ``moduleDB``, steps them with
``step(A_in=...)``, and records whichever of ``input`` / ``activation`` / ``phi`` /
``output`` exist on the instance.

Determinism:
    Initial oscillator phase is randomized by default; after construction this layer
    sets ``phi = 0.0`` when the attribute exists so repeated traces match for the
    same parameters (except ``NeuralOscillator`` warm-up, which uses stdlib ``random``;
    tests should seed RNGs and assert structure, not exact floats).
"""

from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd

from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.models_architecture.module_inspector_models import (
    ModuleInspectorError,
    ModuleTraceResult,
    ModuleVariantSpec,
)

__all__ = [
    "CANDIDATE_SIGNALS",
    "DEFAULT_A_IN",
    "DEFAULT_DT",
    "DEFAULT_STEPS",
    "EXCLUDED_MODES",
    "FALLBACK_INPUT_RANGE",
    "INSPECTABLE_MODULES",
    "build_standalone_module",
    "default_module_config",
    "detect_signals",
    "list_inspectable_modules",
    "mode_label",
    "module_input_range",
    "module_modes",
    "run_module_trace",
]

INSPECTABLE_MODULES: tuple[str, ...] = ("crawler", "turner")
EXCLUDED_MODES: frozenset[str] = frozenset({"nengo"})
CANDIDATE_SIGNALS: tuple[str, ...] = ("input", "activation", "phi", "output")
DEFAULT_STEPS: int = 100
DEFAULT_DT: float = 0.1
DEFAULT_A_IN: float = 0.0
FALLBACK_INPUT_RANGE: tuple[float, float] = (-1.0, 1.0)


def module_modes(module_id: str) -> tuple[str, ...]:
    if module_id not in INSPECTABLE_MODULES:
        raise ModuleInspectorError(
            "invalid_module",
            f'Module "{module_id}" is not inspectable.',
            context={"module_id": module_id},
        )
    modes = MD.mod_modes(module_id) or ()
    return tuple(m for m in modes if m not in EXCLUDED_MODES)


def mode_label(module_id: str, mode: str) -> str:
    short = MD.brainDB[module_id].ModeShortNames.get(mode)
    return f"{mode} ({short})" if short else mode


def list_inspectable_modules() -> tuple[ModuleVariantSpec, ...]:
    specs: list[ModuleVariantSpec] = []
    for module_id in INSPECTABLE_MODULES:
        for mode in module_modes(module_id):
            module = build_standalone_module(module_id, mode, dt=DEFAULT_DT)
            specs.append(
                ModuleVariantSpec(
                    module_id=module_id,
                    mode=mode,
                    display_name=f"{module_id.title()} / {mode}",
                    available_signals=detect_signals(module),
                )
            )
    return tuple(specs)


def default_module_config(module_id: str, mode: str) -> Any:
    if mode not in module_modes(module_id):
        raise ModuleInspectorError(
            "invalid_mode",
            f'Mode "{mode}" is not available for module "{module_id}".',
            context={"module_id": module_id, "mode": mode},
        )
    conf = MD.module_conf(mID=module_id, mode=mode, as_entry=False)
    if conf is None:
        raise ModuleInspectorError(
            "module_config_unavailable",
            f'Could not build default config for "{module_id}" mode "{mode}".',
            context={"module_id": module_id, "mode": mode},
        )
    return conf


def build_standalone_module(
    module_id: str, mode: str, conf: Any | None = None, *, dt: float = DEFAULT_DT
) -> Any:
    if conf is None:
        conf = default_module_config(module_id, mode)
    module = MD.brainDB[module_id].build_module(conf, dt=dt)
    if module is None:
        raise ModuleInspectorError(
            "module_build_failed",
            f'Failed to build standalone module "{module_id}" mode "{mode}".',
            context={"module_id": module_id, "mode": mode},
        )
    if hasattr(module, "phi"):
        module.phi = 0.0
    return module


def detect_signals(module: Any) -> tuple[str, ...]:
    return tuple(name for name in CANDIDATE_SIGNALS if hasattr(module, name))


def module_input_range(module: Any) -> tuple[float, float]:
    rng = getattr(module, "input_range", None)
    if rng is not None and len(rng) == 2 and rng[0] is not None and rng[1] is not None:
        return (float(rng[0]), float(rng[1]))
    return FALLBACK_INPUT_RANGE


def run_module_trace(
    module_id: str,
    mode: str,
    conf: Any | None = None,
    *,
    steps: int = DEFAULT_STEPS,
    dt: float = DEFAULT_DT,
    a_in: float = DEFAULT_A_IN,
) -> ModuleTraceResult:
    if steps <= 0:
        raise ModuleInspectorError(
            "invalid_trace_steps",
            "steps must be > 0.",
            context={"steps": steps},
        )
    if dt <= 0:
        raise ModuleInspectorError(
            "invalid_trace_dt",
            "dt must be > 0.",
            context={"dt": dt},
        )
    module = build_standalone_module(module_id, mode, conf, dt=dt)
    signals = detect_signals(module)
    rows: list[dict[str, Any]] = []
    for tick in range(steps):
        module.step(A_in=a_in)
        row: dict[str, Any] = {"time": tick * dt}
        for sig in signals:
            row[sig] = _coerce_float(getattr(module, sig))
        rows.append(row)
    dataframe = pd.DataFrame(rows, columns=["time", *signals])
    return ModuleTraceResult(
        module_id=module_id,
        mode=mode,
        steps=steps,
        dt=dt,
        a_in=a_in,
        signals=signals,
        dataframe=dataframe,
        input_range=module_input_range(module),
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    return None
