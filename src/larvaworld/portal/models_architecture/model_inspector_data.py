from __future__ import annotations

from numbers import Real
from types import SimpleNamespace
from typing import Any

import pandas as pd

from larvaworld.lib import reg, util
from larvaworld.lib.model import DefaultBrain
from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.models_architecture.model_inspector_models import (
    DraftValidationIssue,
    ModelInspection,
    ModelModuleSpec,
    ModelInspectorError,
    ModuleInspection,
    ModuleComparison,
    ProbeIssue,
    ProbeResult,
)
from larvaworld.portal.models_architecture._module_config_utils import (
    WINDSENSOR_DEFAULT_WEIGHTS,
    copy_config_value,
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
# Default probe / live-preview reporter keys (full set selectable in UI).
LIVE_PREVIEW_REPORTER_KEYS: tuple[str, ...] = (
    "A_T",
    "A_C",
    "A_F",
    "I_T",
    "I_C",
    "I_F",
    "phi_T",
    "phi_C",
    "phi_F",
)
DEFAULT_LIVE_PREVIEW_REPORTER_KEYS: tuple[str, ...] = ("A_T", "A_C")
PROBE_REPORTER_KEYS: tuple[str, ...] = LIVE_PREVIEW_REPORTER_KEYS
LOCOMOTION_MODULES: tuple[str, ...] = (
    "crawler",
    "interference",
    "intermitter",
    "turner",
    "feeder",
)
SENSATION_MODULES: tuple[str, ...] = (
    "olfactor",
    "toucher",
    "windsensor",
    "thermosensor",
)
# Model Inspector: Mode dropdown beside "Enabled" in narrow cards (match Memory mode width).
BRAIN_MODE_SELECT_FIXED_WIDTH_MODULES: frozenset[str] = frozenset(
    (*SENSATION_MODULES, "feeder")
)
MEMORY_MODULES: tuple[str, ...] = ("memory",)
LARVA_CORE_MODULES: tuple[str, ...] = ("body", "physics")
LARVA_OPTIONAL_MODULES: tuple[str, ...] = ("energetics", "sensorimotor", "Box2D")
MODEL_MODULE_ORDER: tuple[str, ...] = (
    *LOCOMOTION_MODULES,
    *SENSATION_MODULES,
    *MEMORY_MODULES,
    *LARVA_CORE_MODULES,
    *LARVA_OPTIONAL_MODULES,
)
_MODE_SHORT_NAMES: dict[str, str] = dict(
    getattr(MD.brainDB["crawler"], "ModeShortNames")
)
_MEMORY_DEFAULT_MODE = "RL"
_MEMORY_DEFAULT_MODALITY = "olfaction"
_BRAIN_OPTIONAL_NON_MEMORY_MODULES: tuple[str, ...] = (
    "feeder",
    "olfactor",
    "toucher",
    "windsensor",
    "thermosensor",
)
_WINDSENSOR_DEFAULT_WEIGHTS: dict[str, float] = WINDSENSOR_DEFAULT_WEIGHTS


def load_model_draft(model_id: str) -> Any:
    model = _get_model_conf(model_id)
    return _copy_config_value(model)


def default_brain_module_config(module_id: str, mode: str) -> Any:
    if module_id == "memory":
        raise ModelInspectorError(
            "invalid_brain_module",
            'Use default_memory_config for module "memory".',
            context={"module_id": module_id},
        )
    if module_id not in MD.BrainMods:
        raise ModelInspectorError(
            "invalid_brain_module",
            f'Module "{module_id}" is not a brain module.',
            context={"module_id": module_id},
        )
    modes = tuple(MD.mod_modes(module_id) or ())
    if mode not in modes:
        raise ModelInspectorError(
            "invalid_brain_mode",
            f'Mode "{mode}" is not supported for module "{module_id}".',
            context={"module_id": module_id, "mode": mode, "modes": modes},
        )
    conf = MD.module_conf(mID=module_id, mode=mode, as_entry=False)
    if conf is None:
        raise ModelInspectorError(
            "brain_default_config_unavailable",
            f'Could not build canonical defaults for "{module_id}" mode "{mode}".',
            context={"module_id": module_id, "mode": mode},
        )
    copied = _copy_config_value(conf)
    if module_id == "windsensor" and "weights" not in copied:
        copied["weights"] = util.AttrDict(_WINDSENSOR_DEFAULT_WEIGHTS.copy())
    copied["mode"] = mode
    return copied


def default_memory_config(mode: str, modality: str) -> Any:
    conf = MD.memory_kws(mode=mode, modality=modality, as_entry=False)
    if conf is None:
        raise ModelInspectorError(
            "invalid_memory_mode_modality",
            f'Unsupported memory mode/modality "{mode}/{modality}".',
            context={"mode": mode, "modality": modality},
        )
    copied = _copy_config_value(conf)
    copied["mode"] = mode
    copied["modality"] = modality
    return copied


def default_larva_module_config(module_id: str) -> Any:
    if module_id not in MD.LarvaMods:
        raise ModelInspectorError(
            "invalid_larva_module",
            f'Module "{module_id}" is not a larva module.',
            context={"module_id": module_id},
        )
    conf_builder = MD.LarvaModsConfDict.get(module_id)
    if conf_builder is None:
        raise ModelInspectorError(
            "larva_default_config_unavailable",
            f'No canonical default builder for larva module "{module_id}".',
            context={"module_id": module_id},
        )
    return _copy_config_value(conf_builder())


def set_draft_module_enabled(model_conf: Any, module_id: str, enabled: bool) -> None:
    brain_conf = _get_brain_conf("<draft>", model_conf)
    if module_id in LOCOMOTION_MODULES:
        if not enabled:
            raise ModelInspectorError(
                "core_module_disable_forbidden",
                f'Core module "{module_id}" cannot be disabled.',
                context={"module_id": module_id},
            )
        module_conf = _module_conf(brain_conf, module_id)
        if module_conf is None:
            first_mode = _first_mode_or_raise(module_id)
            brain_conf[module_id] = default_brain_module_config(module_id, first_mode)
        model_conf.brain = brain_conf
        return
    if module_id in _BRAIN_OPTIONAL_NON_MEMORY_MODULES:
        if not enabled:
            brain_conf[module_id] = None
            model_conf.brain = brain_conf
            return
        module_conf = _module_conf(brain_conf, module_id)
        if module_conf is None:
            first_mode = _first_mode_or_raise(module_id)
            brain_conf[module_id] = default_brain_module_config(module_id, first_mode)
        elif module_id == "windsensor" and "weights" not in module_conf:
            module_conf["weights"] = util.AttrDict(_WINDSENSOR_DEFAULT_WEIGHTS.copy())
        model_conf.brain = brain_conf
        return
    if module_id == "memory":
        if not enabled:
            brain_conf["memory"] = None
            model_conf.brain = brain_conf
            return
        memory_conf = _module_conf(brain_conf, "memory")
        if _is_valid_memory_conf(memory_conf):
            model_conf.brain = brain_conf
            return
        brain_conf["memory"] = default_memory_config(
            _MEMORY_DEFAULT_MODE, _MEMORY_DEFAULT_MODALITY
        )
        model_conf.brain = brain_conf
        return
    if module_id in LARVA_CORE_MODULES:
        if not enabled:
            raise ModelInspectorError(
                "core_module_disable_forbidden",
                f'Core module "{module_id}" cannot be disabled.',
                context={"module_id": module_id},
            )
        if not _has_module_config(model_conf, module_id):
            model_conf[module_id] = default_larva_module_config(module_id)
        return
    if module_id in LARVA_OPTIONAL_MODULES:
        if not enabled:
            model_conf[module_id] = None
            return
        if not _has_module_config(model_conf, module_id):
            model_conf[module_id] = default_larva_module_config(module_id)
        return
    raise ModelInspectorError(
        "unknown_module",
        f'Unknown module "{module_id}".',
        context={"module_id": module_id},
    )


def set_draft_brain_module_mode(model_conf: Any, module_id: str, mode: str) -> None:
    if module_id == "memory":
        raise ModelInspectorError(
            "invalid_brain_module",
            'Use set_draft_memory_config for module "memory".',
            context={"module_id": module_id},
        )
    if module_id not in MD.BrainMods:
        raise ModelInspectorError(
            "invalid_brain_module",
            f'Module "{module_id}" is not a non-memory brain module.',
            context={"module_id": module_id},
        )
    brain_conf = _get_brain_conf("<draft>", model_conf)
    brain_conf[module_id] = default_brain_module_config(module_id, mode)
    model_conf.brain = brain_conf


def set_draft_memory_config(
    model_conf: Any,
    *,
    enabled: bool,
    mode: str | None = None,
    modality: str | None = None,
) -> None:
    brain_conf = _get_brain_conf("<draft>", model_conf)
    if not enabled:
        brain_conf["memory"] = None
        model_conf.brain = brain_conf
        return
    current_memory = _module_conf(brain_conf, "memory")
    resolved_mode = _resolve_memory_mode(mode, current_memory)
    supported_modalities = _memory_modalities_for_mode(resolved_mode)
    resolved_modality = _resolve_memory_modality(
        modality, current_memory, resolved_mode, supported_modalities
    )
    brain_conf["memory"] = default_memory_config(resolved_mode, resolved_modality)
    model_conf.brain = brain_conf


def set_draft_module_parameter(
    model_conf: Any,
    module_id: str,
    parameter_path: tuple[str, ...],
    value: Any,
) -> None:
    if not parameter_path:
        raise ModelInspectorError(
            "invalid_parameter_path",
            "Parameter path cannot be empty.",
            context={"module_id": module_id},
        )
    first = str(parameter_path[0])
    if first in {"mode", "modality", "name"}:
        raise ModelInspectorError(
            "protected_parameter_path",
            f'Cannot mutate protected parameter path "{first}".',
            context={"module_id": module_id, "parameter_path": parameter_path},
        )

    module_conf = _draft_module_conf(model_conf, module_id)
    cursor = module_conf
    for segment in parameter_path[:-1]:
        key = str(segment)
        if not hasattr(cursor, "__contains__") or key not in cursor:
            raise ModelInspectorError(
                "parameter_path_missing",
                f'Path segment "{key}" does not exist.',
                context={"module_id": module_id, "parameter_path": parameter_path},
            )
        cursor = cursor[key]
        if not hasattr(cursor, "__contains__") and not hasattr(cursor, "__setitem__"):
            raise ModelInspectorError(
                "parameter_path_not_mapping",
                f'Path segment "{key}" is not dict-like.',
                context={"module_id": module_id, "parameter_path": parameter_path},
            )
    leaf = str(parameter_path[-1])
    if not hasattr(cursor, "__contains__") or leaf not in cursor:
        raise ModelInspectorError(
            "parameter_path_missing",
            f'Leaf "{leaf}" does not exist.',
            context={"module_id": module_id, "parameter_path": parameter_path},
        )
    cursor[leaf] = value


def validate_draft_module_config(model_conf: Any) -> tuple[DraftValidationIssue, ...]:
    issues: list[DraftValidationIssue] = []
    brain_conf = _get_brain_conf("<draft>", model_conf)
    memory_conf = _module_conf(brain_conf, "memory")
    if memory_conf is not None:
        memory_mode = _conf_get(memory_conf, "mode")
        memory_modality = _conf_get(memory_conf, "modality")
        if memory_mode is None:
            issues.append(
                DraftValidationIssue(
                    code="memory_mode_missing",
                    severity="error",
                    module_id="memory",
                    path=("brain", "memory", "mode"),
                    message='Memory configuration is missing required "mode".',
                )
            )
        elif memory_mode not in MD.BrainModuleModes["memory"]:
            issues.append(
                DraftValidationIssue(
                    code="memory_mode_unsupported",
                    severity="error",
                    module_id="memory",
                    path=("brain", "memory", "mode"),
                    message=f'Unsupported memory mode "{memory_mode}".',
                )
            )
        elif memory_modality is None:
            issues.append(
                DraftValidationIssue(
                    code="memory_modality_missing",
                    severity="error",
                    module_id="memory",
                    path=("brain", "memory", "modality"),
                    message='Memory configuration is missing required "modality".',
                )
            )
        else:
            supported_modalities = _memory_modalities_for_mode(memory_mode)
            if memory_modality not in supported_modalities:
                issues.append(
                    DraftValidationIssue(
                        code="memory_modality_unsupported",
                        severity="error",
                        module_id="memory",
                        path=("brain", "memory", "modality"),
                        message=(
                            f'Unsupported memory modality "{memory_modality}" for mode "{memory_mode}".'
                        ),
                    )
                )
            elif (
                memory_modality == "olfaction"
                and _module_conf(brain_conf, "olfactor") is None
            ):
                issues.append(
                    DraftValidationIssue(
                        code="memory_sensor_missing",
                        severity="warning",
                        module_id="memory",
                        path=("brain", "memory", "modality"),
                        message="Memory modality requires enabled sensor module (olfactor).",
                    )
                )
            elif (
                memory_modality == "touch"
                and _module_conf(brain_conf, "toucher") is None
            ):
                issues.append(
                    DraftValidationIssue(
                        code="memory_sensor_missing",
                        severity="warning",
                        module_id="memory",
                        path=("brain", "memory", "modality"),
                        message="Memory modality requires enabled sensor module (toucher).",
                    )
                )

    intermitter_conf = _module_conf(brain_conf, "intermitter")
    intermitter_mode = _conf_get(intermitter_conf, "mode")
    intermitter_beta = _conf_get(intermitter_conf, "beta")
    if intermitter_mode == "branch" and not _is_positive_real(intermitter_beta):
        issues.append(
            DraftValidationIssue(
                code="intermitter_branch_beta_invalid",
                severity="error",
                module_id="intermitter",
                path=("brain", "intermitter", "beta"),
                message='Branch intermitter requires a positive numeric "beta" before live preview can run.',
            )
        )
    return tuple(issues)


def _is_positive_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value > 0


def inspect_model_modules_from_config(
    model_id: str,
    model_conf: Any,
) -> tuple[ModelModuleSpec, ...]:
    brain_conf = _get_brain_conf(model_id, model_conf)

    specs: list[ModelModuleSpec] = []
    for module_id in MODEL_MODULE_ORDER:
        if module_id in MEMORY_MODULES:
            specs.append(_memory_module_spec(brain_conf=brain_conf))
            continue
        if module_id in LARVA_CORE_MODULES or module_id in LARVA_OPTIONAL_MODULES:
            specs.append(_larva_module_spec(model_conf=model_conf, module_id=module_id))
            continue
        specs.append(_brain_module_spec(brain_conf=brain_conf, module_id=module_id))
    return tuple(specs)


def inspect_model_modules(model_id: str) -> tuple[ModelModuleSpec, ...]:
    return inspect_model_modules_from_config(model_id, load_model_draft(model_id))


def list_model_ids() -> list[str]:
    return list(reg.conf.Model.confIDs)


def inspect_model_from_config(model_id: str, model_conf: Any) -> ModelInspection:
    brain_conf = _get_brain_conf(model_id, model_conf)

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


def inspect_model(model_id: str) -> ModelInspection:
    return inspect_model_from_config(model_id, load_model_draft(model_id))


def build_inspection_brain_from_config(
    model_id: str, model_conf: Any, *, dt: float = 0.1
) -> DefaultBrain:
    brain_conf = _get_brain_conf(model_id, model_conf)
    return DefaultBrain(conf=brain_conf, agent=_inspection_agent(model_id, dt), dt=dt)


def build_inspection_brain(model_id: str, *, dt: float = 0.1) -> DefaultBrain:
    return build_inspection_brain_from_config(
        model_id, load_model_draft(model_id), dt=dt
    )


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
    reporter_keys: tuple[str, ...] | None = None,
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

    keys = tuple(reporter_keys) if reporter_keys is not None else PROBE_REPORTER_KEYS
    available_from_registry = reg.par.output_reporters(ks=list(keys), agents=[runtime])
    available_paths = set(available_from_registry.values())
    issues: list[ProbeIssue] = []
    reporter_paths: dict[str, str] = {}
    for key in keys:
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


def _copy_config_value(value: Any) -> Any:
    return copy_config_value(value)


def _has_module_config(model_conf: Any, module_id: str) -> bool:
    try:
        return module_id in model_conf and model_conf[module_id] is not None
    except Exception:
        return False


def _first_mode_or_raise(module_id: str) -> str:
    modes = tuple(MD.mod_modes(module_id) or ())
    if not modes:
        raise ModelInspectorError(
            "missing_module_modes",
            f'No canonical modes are available for module "{module_id}".',
            context={"module_id": module_id},
        )
    return str(modes[0])


def _conf_get(conf: Any, key: str) -> Any:
    if conf is None:
        return None
    if hasattr(conf, "get"):
        return conf.get(key)
    try:
        return conf[key]
    except Exception:
        return None


def _memory_modalities_for_mode(mode: str) -> tuple[str, ...]:
    modal_map = MD.BrainModuleModes["memory"].get(mode)
    if modal_map is None:
        raise ModelInspectorError(
            "invalid_memory_mode",
            f'Unsupported memory mode "{mode}".',
            context={"mode": mode},
        )
    return tuple(str(k) for k in modal_map.keys())


def _is_valid_memory_conf(conf: Any) -> bool:
    if conf is None:
        return False
    mode = _conf_get(conf, "mode")
    modality = _conf_get(conf, "modality")
    if mode is None or modality is None:
        return False
    try:
        supported = _memory_modalities_for_mode(str(mode))
    except ModelInspectorError:
        return False
    return str(modality) in supported


def _resolve_memory_mode(requested_mode: str | None, current_memory: Any) -> str:
    if requested_mode is not None:
        if requested_mode not in MD.BrainModuleModes["memory"]:
            raise ModelInspectorError(
                "invalid_memory_mode",
                f'Unsupported memory mode "{requested_mode}".',
                context={"mode": requested_mode},
            )
        return requested_mode
    current_mode = _conf_get(current_memory, "mode")
    if current_mode in MD.BrainModuleModes["memory"]:
        return str(current_mode)
    return _MEMORY_DEFAULT_MODE


def _resolve_memory_modality(
    requested_modality: str | None,
    current_memory: Any,
    mode: str,
    supported_modalities: tuple[str, ...],
) -> str:
    if requested_modality is not None:
        if requested_modality not in supported_modalities:
            raise ModelInspectorError(
                "invalid_memory_modality",
                f'Unsupported memory modality "{requested_modality}" for mode "{mode}".',
                context={
                    "mode": mode,
                    "modality": requested_modality,
                    "supported_modalities": supported_modalities,
                },
            )
        return requested_modality
    current_modality = _conf_get(current_memory, "modality")
    if current_modality in supported_modalities:
        return str(current_modality)
    return str(supported_modalities[0])


def _draft_module_conf(model_conf: Any, module_id: str) -> Any:
    if module_id in MD.BrainMods:
        brain_conf = _get_brain_conf("<draft>", model_conf)
        module_conf = _module_conf(brain_conf, module_id)
        if module_conf is None:
            raise ModelInspectorError(
                "module_missing_or_disabled",
                f'Module "{module_id}" is missing or disabled in draft brain config.',
                context={"module_id": module_id},
            )
        model_conf.brain = brain_conf
        return model_conf.brain[module_id]
    if module_id in MD.LarvaMods:
        if not _has_module_config(model_conf, module_id):
            raise ModelInspectorError(
                "module_missing_or_disabled",
                f'Module "{module_id}" is missing or disabled in draft larva config.',
                context={"module_id": module_id},
            )
        return model_conf[module_id]
    raise ModelInspectorError(
        "unknown_module",
        f'Unknown module "{module_id}".',
        context={"module_id": module_id},
    )


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


def _mode_labels(mode_options: tuple[str, ...]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for mode in mode_options:
        acronym = _MODE_SHORT_NAMES.get(mode)
        labels[mode] = f"{mode} ({acronym})" if acronym else mode
    return labels


def _spec_parameters(module_conf: Any, *, excluded: set[str]) -> dict[str, Any]:
    if module_conf is None or not hasattr(module_conf, "keys"):
        return {}
    return {
        str(k): module_conf[k] for k in module_conf.keys() if str(k) not in excluded
    }


def _module_group(module_id: str) -> tuple[str, str, bool]:
    if module_id == "feeder":
        return ("Nervous System", "Locomotion", False)
    if module_id in LOCOMOTION_MODULES:
        return ("Nervous System", "Locomotion", True)
    if module_id in SENSATION_MODULES:
        return ("Nervous System", "Sensation", False)
    if module_id in MEMORY_MODULES:
        return ("Nervous System", "Memory", False)
    if module_id in LARVA_CORE_MODULES:
        return ("Body and Metabolism", "Core", True)
    return ("Body and Metabolism", "Optional", False)


def _brain_module_spec(*, brain_conf: Any, module_id: str) -> ModelModuleSpec:
    module_conf = _module_conf(brain_conf, module_id)
    present = module_conf is not None
    group, subgroup, is_core = _module_group(module_id)
    mode_options = tuple(MD.mod_modes(module_id) or ())
    current_mode = (
        module_conf.get("mode") if present and hasattr(module_conf, "get") else None
    )
    return ModelModuleSpec(
        module_id=module_id,
        display_name=module_id.replace("_", " ").title(),
        group=group,
        subgroup=subgroup,
        module_kind="brain",
        present=present,
        enabled=present,
        current_mode=str(current_mode) if current_mode is not None else None,
        mode_options=mode_options,
        mode_labels=_mode_labels(mode_options),
        parameters=_spec_parameters(module_conf, excluded={"mode", "name"}),
        current_modality=None,
        modality_options_by_mode={},
        is_core=is_core,
    )


def _memory_module_spec(*, brain_conf: Any) -> ModelModuleSpec:
    module_conf = _module_conf(brain_conf, "memory")
    present = module_conf is not None
    memory_modes = tuple(MD.BrainModuleModes["memory"].keys())
    modality_options_by_mode: dict[str, tuple[str, ...]] = {
        str(mode): tuple(
            str(modality) for modality in MD.BrainModuleModes["memory"][mode].keys()
        )
        for mode in memory_modes
    }
    current_mode = (
        module_conf.get("mode") if present and hasattr(module_conf, "get") else None
    )
    current_modality = (
        module_conf.get("modality") if present and hasattr(module_conf, "get") else None
    )
    group, subgroup, is_core = _module_group("memory")
    display_name = (
        f"Memory ({current_modality})"
        if present and current_modality is not None
        else "Memory"
    )
    return ModelModuleSpec(
        module_id="memory",
        display_name=display_name,
        group=group,
        subgroup=subgroup,
        module_kind="memory",
        present=present,
        enabled=present,
        current_mode=str(current_mode) if current_mode is not None else None,
        mode_options=memory_modes,
        mode_labels=_mode_labels(memory_modes),
        parameters=_spec_parameters(module_conf, excluded={"mode", "modality", "name"}),
        current_modality=(
            str(current_modality) if current_modality is not None else None
        ),
        modality_options_by_mode=modality_options_by_mode,
        is_core=is_core,
    )


def _larva_module_spec(*, model_conf: Any, module_id: str) -> ModelModuleSpec:
    module_conf = None
    try:
        if module_id in model_conf:
            module_conf = model_conf[module_id]
    except Exception:
        module_conf = getattr(model_conf, module_id, None)
    present = module_conf is not None
    group, subgroup, is_core = _module_group(module_id)
    return ModelModuleSpec(
        module_id=module_id,
        display_name=module_id.replace("_", " ").title(),
        group=group,
        subgroup=subgroup,
        module_kind="larva",
        present=present,
        enabled=present,
        current_mode=None,
        mode_options=(),
        mode_labels={},
        parameters=_spec_parameters(module_conf, excluded={"name"}),
        current_modality=None,
        modality_options_by_mode={},
        is_core=is_core,
    )


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
