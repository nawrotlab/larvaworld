"""Pure data layer for the portal Module Inspector (no Panel/Bokeh).

Builds standalone module instances from ``moduleDB`` and steps them per "kind":

- "effector" (crawler, turner): ``step(A_in=<scalar>)`` with a constant A_in.
- "feeder": ``start_effector()`` once, then ``step()`` (self-oscillator, no input).
- "sensor" (olfactor, toucher, windsensor, thermosensor): driven by a
  time-varying stimulus converted to the per-sensor dict input, ``step(A_in=<dict>)``.

Determinism:
    Initial oscillator phase is randomized by default; after construction this layer
    sets ``phi = 0.0`` when the attribute exists so repeated traces match for the
    same parameters (except ``NeuralOscillator`` warm-up, which uses stdlib
    ``random``; tests should seed RNGs and assert structure, not exact floats).
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import pandas as pd

from larvaworld.lib import util
from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.models_architecture._module_config_utils import (
    WINDSENSOR_DEFAULT_WEIGHTS,
    copy_config_value,
)
from larvaworld.portal.models_architecture.module_inspector_models import (
    ModuleInspectorError,
    ModuleTraceResult,
    ModuleVariantSpec,
    StimulusSpec,
)

__all__ = [
    "ALL_SIGNALS",
    "CANDIDATE_SIGNALS",
    "DEFAULT_A_IN",
    "DEFAULT_DT",
    "DEFAULT_STEPS",
    "DEFAULT_STIMULUS",
    "EFFECTOR_SIGNALS",
    "EXCLUDED_MODES",
    "FALLBACK_INPUT_RANGE",
    "FEEDER_SIGNALS",
    "INSPECTABLE_MODULES",
    "MODULE_KINDS",
    "SENSOR_SIGNALS",
    "build_standalone_module",
    "default_module_config",
    "detect_signals",
    "list_inspectable_modules",
    "mode_label",
    "module_input_range",
    "module_kind",
    "module_modes",
    "run_module_trace",
    "signals_for_kind",
    "stimulus_series",
    "stimulus_to_input",
]

INSPECTABLE_MODULES: tuple[str, ...] = (
    "crawler",
    "turner",
    "feeder",
    "olfactor",
    "toucher",
    "windsensor",
    "thermosensor",
)
EXCLUDED_MODES: frozenset[str] = frozenset({"nengo", "osn"})

MODULE_KINDS: dict[str, str] = {
    "crawler": "effector",
    "turner": "effector",
    "feeder": "feeder",
    "olfactor": "sensor",
    "toucher": "sensor",
    "windsensor": "sensor",
    "thermosensor": "sensor",
}

EFFECTOR_SIGNALS: tuple[str, ...] = ("input", "activation", "phi", "output")
# Backward-compatible alias (the original effectors-only API name).
CANDIDATE_SIGNALS: tuple[str, ...] = EFFECTOR_SIGNALS
FEEDER_SIGNALS: tuple[str, ...] = ("phi", "complete_iteration")
SENSOR_SIGNALS: tuple[str, ...] = ("stimulus", "output")
# Union of every signal any kind may emit (used for stable plot data sources).
ALL_SIGNALS: tuple[str, ...] = (
    "input",
    "activation",
    "phi",
    "output",
    "complete_iteration",
    "stimulus",
)

DEFAULT_STEPS: int = 100
DEFAULT_DT: float = 0.1
DEFAULT_A_IN: float = 0.0
FALLBACK_INPUT_RANGE: tuple[float, float] = (-1.0, 1.0)

# Stimulus input key per sensor (must match a gain_dict key on the instance).
_SENSOR_STIMULUS_KEY: dict[str, str] = {
    "olfactor": "odor",
    "toucher": "touch",
    "windsensor": "windsensor",
    "thermosensor": "warm",
}
# Non-canonical preview gain injected for sensors whose default gain_dict is empty.
_PREVIEW_SENSOR_GAIN: float = 1.0

# Default time-varying stimulus for sensors (baseline != 0 so log/linear sensors
# produce a non-zero perceived change after the first step).
DEFAULT_STIMULUS: StimulusSpec = StimulusSpec(
    waveform="sinusoid",
    baseline=1.0,
    amplitude=0.25,
    frequency=1.0,
    onset=1.0,
)


def module_kind(module_id: str) -> str:
    if module_id not in MODULE_KINDS:
        raise ModuleInspectorError(
            "invalid_module",
            f'Module "{module_id}" is not inspectable.',
            context={"module_id": module_id},
        )
    return MODULE_KINDS[module_id]


def signals_for_kind(kind: str) -> tuple[str, ...]:
    if kind == "effector":
        return EFFECTOR_SIGNALS
    if kind == "feeder":
        return FEEDER_SIGNALS
    if kind == "sensor":
        return SENSOR_SIGNALS
    raise ModuleInspectorError(
        "invalid_kind",
        f'Unknown module kind "{kind}".',
        context={"kind": kind},
    )


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
    # Never mutate the (possibly shared) config returned by moduleDB.
    conf = copy_config_value(conf)

    kind = module_kind(module_id)
    if module_id == "windsensor" and "weights" not in conf:
        conf["weights"] = util.AttrDict(dict(WINDSENSOR_DEFAULT_WEIGHTS))
    if kind == "sensor":
        gain_dict = conf.get("gain_dict", None)
        if not gain_dict:
            key = _SENSOR_STIMULUS_KEY[module_id]
            conf["gain_dict"] = util.AttrDict({key: _PREVIEW_SENSOR_GAIN})

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


def detect_signals(module: Any, kind: str) -> tuple[str, ...]:
    if kind == "sensor":
        return SENSOR_SIGNALS
    candidates = signals_for_kind(kind)
    return tuple(name for name in candidates if hasattr(module, name))


def module_input_range(module: Any) -> tuple[float, float]:
    rng = getattr(module, "input_range", None)
    if rng is not None and len(rng) == 2 and rng[0] is not None and rng[1] is not None:
        return (float(rng[0]), float(rng[1]))
    return FALLBACK_INPUT_RANGE


def list_inspectable_modules() -> tuple[ModuleVariantSpec, ...]:
    specs: list[ModuleVariantSpec] = []
    for module_id in INSPECTABLE_MODULES:
        kind = module_kind(module_id)
        for mode in module_modes(module_id):
            module = build_standalone_module(module_id, mode, dt=DEFAULT_DT)
            specs.append(
                ModuleVariantSpec(
                    module_id=module_id,
                    mode=mode,
                    kind=kind,
                    display_name=f"{module_id.title()} / {mode}",
                    available_signals=detect_signals(module, kind),
                )
            )
    return tuple(specs)


def stimulus_series(stim: StimulusSpec, steps: int, dt: float) -> list[float]:
    values: list[float] = []
    for tick in range(steps):
        t = tick * dt
        if stim.waveform == "sinusoid":
            value = stim.baseline + stim.amplitude * math.sin(
                2.0 * math.pi * stim.frequency * t
            )
        elif stim.waveform == "step":
            value = stim.baseline + (stim.amplitude if t >= stim.onset else 0.0)
        else:
            raise ModuleInspectorError(
                "invalid_stimulus_waveform",
                f'Unknown stimulus waveform "{stim.waveform}".',
                context={"waveform": stim.waveform},
            )
        values.append(float(value))
    return values


def stimulus_to_input(
    module_id: str, value: float, baseline: float
) -> dict[str, float]:
    if module_id == "thermosensor":
        return {"warm": float(value), "cool": float(baseline)}
    key = _SENSOR_STIMULUS_KEY[module_id]
    return {key: float(value)}


def run_module_trace(
    module_id: str,
    mode: str,
    conf: Any | None = None,
    *,
    steps: int = DEFAULT_STEPS,
    dt: float = DEFAULT_DT,
    a_in: float = DEFAULT_A_IN,
    stimulus: StimulusSpec | None = None,
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

    kind = module_kind(module_id)
    module = build_standalone_module(module_id, mode, conf, dt=dt)
    rows: list[dict[str, Any]] = []
    stim_used: StimulusSpec | None = None

    if kind == "effector":
        signals = detect_signals(module, kind)
        for tick in range(steps):
            module.step(A_in=a_in)
            row: dict[str, Any] = {"time": tick * dt}
            for sig in signals:
                row[sig] = _coerce_float(getattr(module, sig))
            rows.append(row)
    elif kind == "feeder":
        module.start_effector()
        signals = detect_signals(module, kind)
        for tick in range(steps):
            module.step()
            row = {"time": tick * dt}
            for sig in signals:
                row[sig] = _coerce_float(getattr(module, sig))
            rows.append(row)
    else:  # sensor
        stim_used = stimulus or DEFAULT_STIMULUS
        series = stimulus_series(stim_used, steps, dt)
        signals = SENSOR_SIGNALS
        for tick in range(steps):
            value = series[tick]
            module.step(A_in=stimulus_to_input(module_id, value, stim_used.baseline))
            rows.append(
                {
                    "time": tick * dt,
                    "stimulus": float(value),
                    "output": _coerce_float(module.output),
                }
            )

    dataframe = pd.DataFrame(rows, columns=["time", *signals])
    return ModuleTraceResult(
        module_id=module_id,
        mode=mode,
        kind=kind,
        steps=steps,
        dt=dt,
        a_in=a_in,
        signals=tuple(signals),
        dataframe=dataframe,
        input_range=module_input_range(module),
        stimulus=stim_used,
    )


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        return float(value)
    return None
