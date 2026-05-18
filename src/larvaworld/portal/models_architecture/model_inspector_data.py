from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd

from larvaworld.lib import reg, util
from larvaworld.lib.model import DefaultBrain
from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.models_architecture.model_inspector_models import (
    ModelInspection,
    ModelInspectorError,
    ModuleInspection,
    ModuleComparison,
    ProbeIssue,
    ProbeResult,
)


BASELINE_MODULES: tuple[str, ...] = tuple(MD.LocoModsBasic)
OPTIONAL_MODULES: tuple[str, ...] = (
    "feeder",
    "olfactor",
    "toucher",
    "windsensor",
    "thermosensor",
    "memory",
)
PROBE_REPORTER_KEYS: tuple[str, ...] = ("A_T", "A_C")


def list_model_ids() -> list[str]:
    return list(reg.conf.Model.confIDs)


def inspect_model(model_id: str) -> ModelInspection:
    model = _get_model_conf(model_id)
    brain_conf = _get_brain_conf(model_id, model)

    baseline: list[ModuleInspection] = []
    for module_id in BASELINE_MODULES:
        baseline.append(
            _build_module_inspection(
                brain_conf=brain_conf,
                module_id=module_id,
                is_baseline=True,
            )
        )

    optional: list[ModuleInspection] = []
    for module_id in OPTIONAL_MODULES:
        module_conf = _module_conf(brain_conf, module_id)
        if module_conf is None:
            continue
        optional.append(
            _build_module_inspection(
                brain_conf=brain_conf,
                module_id=module_id,
                is_baseline=False,
            )
        )

    return ModelInspection(
        model_id=model_id,
        baseline_modules=tuple(baseline),
        optional_modules=tuple(optional),
    )


def build_inspection_brain(model_id: str, *, dt: float = 0.1) -> DefaultBrain:
    model = _get_model_conf(model_id)
    brain_conf = _get_brain_conf(model_id, model)
    return DefaultBrain(conf=brain_conf, agent=_inspection_agent(model_id, dt), dt=dt)


def compare_model_inspections(
    primary: ModelInspection,
    comparison: ModelInspection,
) -> list[ModuleComparison]:
    module_ids = _ordered_module_ids(primary, comparison)
    primary_map = _module_map(primary)
    comparison_map = _module_map(comparison)

    diffs: list[ModuleComparison] = []
    for module_id in module_ids:
        p = primary_map.get(module_id) or _missing_module_inspection(module_id)
        c = comparison_map.get(module_id) or _missing_module_inspection(module_id)
        changed_fields: list[str] = []
        if p.present != c.present:
            changed_fields.append("presence")
        if p.mode != c.mode:
            changed_fields.append("mode")
        if p.parameters != c.parameters:
            changed_fields.append("parameters")
        diffs.append(
            ModuleComparison(
                module_id=module_id,
                primary=p,
                comparison=c,
                changed_fields=tuple(changed_fields),
                equal=len(changed_fields) == 0,
            )
        )
    return diffs


def run_model_probe(
    model_id: str,
    *,
    steps: int = 501,
    dt: float = 0.1,
    a_in: float = 0.0,
) -> ProbeResult:
    if steps <= 0:
        raise ModelInspectorError(
            "invalid_probe_steps",
            "Probe steps must be > 0.",
            context={"steps": steps},
        )
    if dt <= 0:
        raise ModelInspectorError(
            "invalid_probe_dt",
            "Probe dt must be > 0.",
            context={"dt": dt},
        )
    model = _get_model_conf(model_id)
    brain_conf = _get_brain_conf(model_id, model)
    brain = DefaultBrain(conf=brain_conf, agent=_inspection_agent(model_id, dt), dt=dt)
    runtime = SimpleNamespace(brain=brain)

    available_from_registry = reg.par.output_reporters(
        ks=list(PROBE_REPORTER_KEYS), agents=[runtime]
    )
    available_paths = set(available_from_registry.values())
    issues: list[ProbeIssue] = []
    reporter_paths: dict[str, str] = {}
    for key in PROBE_REPORTER_KEYS:
        try:
            reporter_paths[key] = reg.par.kdict[key].codename
        except Exception:
            reporter_paths[key] = ""
            issues.append(
                ProbeIssue(
                    code="reporter_key_missing",
                    message=f'Reporter key "{key}" is not registered.',
                    context={"model_id": model_id, "reporter": key},
                )
            )
    reporter_available: dict[str, bool] = {}
    for key, path in reporter_paths.items():
        available = bool(path) and path in available_paths
        # Keep legacy reporter mechanism, but tolerate registry lookup drift by
        # verifying direct runtime path resolution when needed.
        if not available and path:
            try:
                util.rgetattr(runtime, path)
                available = True
            except Exception:
                available = False
        reporter_available[key] = available
        if not reporter_available[key]:
            issues.append(
                ProbeIssue(
                    code="reporter_unavailable",
                    message=f'Reporter "{key}" is unavailable for model "{model_id}".',
                    context={"model_id": model_id, "reporter": key, "path": path},
                )
            )

    rows: list[dict[str, Any]] = []
    for tick in range(steps):
        lin, ang, feed_motion = brain.locomotor.step(A_in=a_in)
        row: dict[str, Any] = {
            "time": tick * dt,
            "lin": lin,
            "ang": ang,
            "feed_motion": bool(feed_motion),
        }
        for key, path in reporter_paths.items():
            if not reporter_available[key]:
                row[key] = None
                continue
            try:
                row[key] = util.rgetattr(runtime, path)
            except Exception as exc:
                reporter_available[key] = False
                row[key] = None
                issues.append(
                    ProbeIssue(
                        code="reporter_resolution_failed",
                        message=f'Could not resolve reporter "{key}".',
                        context={
                            "model_id": model_id,
                            "reporter": key,
                            "path": path,
                            "error": str(exc),
                        },
                    )
                )
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    return ProbeResult(
        model_id=model_id,
        steps=steps,
        dt=dt,
        a_in=a_in,
        dataframe=dataframe,
        reporter_paths=reporter_paths,
        reporter_available=reporter_available,
        issues=tuple(issues),
    )


def _get_model_conf(model_id: str) -> Any:
    if model_id not in reg.conf.Model.confIDs:
        raise ModelInspectorError(
            "model_not_found",
            f'Model "{model_id}" does not exist in registry.',
            context={"model_id": model_id},
        )
    return reg.conf.Model.getID(model_id)


def _get_brain_conf(model_id: str, model_conf: Any) -> Any:
    try:
        brain_conf = model_conf.brain
    except Exception as exc:
        raise ModelInspectorError(
            "model_brain_missing",
            f'Model "{model_id}" has no valid brain configuration.',
            context={"model_id": model_id, "error": str(exc)},
        ) from exc
    return brain_conf.get_copy() if hasattr(brain_conf, "get_copy") else brain_conf


def _inspection_agent(model_id: str, dt: float) -> SimpleNamespace:
    model_stub = SimpleNamespace(id=model_id, dt=dt)
    return SimpleNamespace(
        model=model_stub,
        radius=0.0,
        pos=(0.0, 0.0),
        olfactor_pos=(0.0, 0.0),
        touch_sensorIDs=(),
        add_touch_sensors=lambda *_args, **_kwargs: None,
        get_sensor_position=lambda *_args, **_kwargs: (0.0, 0.0),
    )


def _module_conf(brain_conf: Any, module_id: str) -> Any | None:
    try:
        if module_id not in brain_conf:
            return None
        return brain_conf[module_id]
    except Exception:
        return None


def _build_module_inspection(
    *, brain_conf: Any, module_id: str, is_baseline: bool
) -> ModuleInspection:
    module_conf = _module_conf(brain_conf, module_id)
    if module_conf is None:
        return _missing_module_inspection(module_id, is_baseline=is_baseline)
    mode = None
    parameters: dict[str, Any] = {}
    if hasattr(module_conf, "items"):
        mode = module_conf.get("mode")
        parameters = {
            str(k): module_conf[k]
            for k in module_conf.keys()
            if str(k) not in {"mode", "name"}
        }
    return ModuleInspection(
        module_id=module_id,
        display_name=module_id.replace("_", " ").title(),
        present=True,
        mode=str(mode) if mode is not None else None,
        parameters=parameters,
        is_baseline=is_baseline,
    )


def _missing_module_inspection(
    module_id: str, *, is_baseline: bool = False
) -> ModuleInspection:
    return ModuleInspection(
        module_id=module_id,
        display_name=module_id.replace("_", " ").title(),
        present=False,
        mode=None,
        parameters={},
        is_baseline=is_baseline,
    )


def _module_map(inspection: ModelInspection) -> dict[str, ModuleInspection]:
    modules = inspection.baseline_modules + inspection.optional_modules
    return {module.module_id: module for module in modules}


def _ordered_module_ids(
    primary: ModelInspection, comparison: ModelInspection
) -> list[str]:
    ids: list[str] = list(BASELINE_MODULES)
    seen = set(ids)
    for inspection in (primary, comparison):
        for module in inspection.optional_modules:
            if module.module_id in seen:
                continue
            ids.append(module.module_id)
            seen.add(module.module_id)
    return ids
