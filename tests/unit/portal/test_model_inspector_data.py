from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.model import DefaultBrain
from larvaworld.lib.model import moduleDB as MD
from larvaworld.portal.models_architecture import model_inspector_data as data
from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    MODEL_MODULE_ORDER,
    build_inspection_brain_from_config,
    compare_model_inspections,
    default_brain_module_config,
    default_larva_module_config,
    default_memory_config,
    inspect_model,
    inspect_model_from_config,
    inspect_model_modules,
    inspect_model_modules_from_config,
    list_model_ids,
    load_model_draft,
    set_draft_brain_module_mode,
    set_draft_memory_config,
    set_draft_module_enabled,
    set_draft_module_parameter,
    run_model_probe,
    validate_draft_module_config,
)
from larvaworld.portal.models_architecture.model_inspector_models import (
    ModelInspectorError,
)
from larvaworld.portal.workspace import clear_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def _legacy_equivalent_probe(
    model_id: str, *, steps: int = 501, dt: float = 0.1, a_in: float = 0.0
) -> tuple[pd.DataFrame, dict[str, bool]]:
    model = reg.conf.Model.getID(model_id)
    brain = DefaultBrain(conf=model.brain.get_copy(), dt=dt)
    runtime = SimpleNamespace(brain=brain)
    available_from_registry = reg.par.output_reporters(
        ks=["A_T", "A_C"], agents=[runtime]
    )
    available_paths = set(available_from_registry.values())
    reporter_paths: dict[str, str] = {}
    for key in ("A_T", "A_C"):
        try:
            reporter_paths[key] = reg.par.kdict[key].codename
        except Exception:
            reporter_paths[key] = ""
    reporter_available: dict[str, bool] = {}
    for key, path in reporter_paths.items():
        available = bool(path) and path in available_paths
        if not available and path:
            try:
                util.rgetattr(runtime, path)
                available = True
            except Exception:
                available = False
        reporter_available[key] = available

    rows = []
    for tick in range(steps):
        lin, ang, feed_motion = brain.locomotor.step(A_in=a_in)
        row = {
            "time": tick * dt,
            "lin": lin,
            "ang": ang,
            "feed_motion": bool(feed_motion),
        }
        for key, path in reporter_paths.items():
            row[key] = util.rgetattr(runtime, path) if reporter_available[key] else None
        rows.append(row)
    return pd.DataFrame(rows), reporter_available


def test_list_model_ids_returns_registry_ids() -> None:
    ids = list_model_ids()
    assert ids == list(reg.conf.Model.confIDs)
    assert len(ids) > 0


def test_inspect_model_explorer_includes_baseline_modules() -> None:
    if "explorer" not in reg.conf.Model.confIDs:
        pytest.skip('"explorer" model is not available in this environment.')
    inspection = inspect_model("explorer")
    baseline_ids = [module.module_id for module in inspection.baseline_modules]
    assert tuple(baseline_ids) == BASELINE_MODULES
    assert all(module.is_baseline for module in inspection.baseline_modules)


def test_inspect_model_marks_absent_baseline_modules_explicitly() -> None:
    if "explorer" not in reg.conf.Model.confIDs:
        pytest.skip('"explorer" model is not available in this environment.')
    inspection = inspect_model("explorer")
    assert any(module.present for module in inspection.baseline_modules)
    for module in inspection.baseline_modules:
        assert module.present in {True, False}


def test_optional_module_detection_when_configured() -> None:
    for model_id in reg.conf.Model.confIDs:
        inspection = inspect_model(model_id)
        if inspection.optional_modules:
            assert all(not module.is_baseline for module in inspection.optional_modules)
            return
    pytest.skip("No model with configured optional modules was found.")


def test_inspect_model_modules_returns_all_slots_in_canonical_order() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = inspect_model_modules(model_id)
    assert tuple(spec.module_id for spec in specs) == MODEL_MODULE_ORDER


def test_inspect_model_modules_grouping_and_kinds() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    assert specs["crawler"].group == "Nervous System"
    assert specs["crawler"].subgroup == "Locomotion"
    assert specs["crawler"].module_kind == "brain"
    assert specs["olfactor"].subgroup == "Sensation"
    assert specs["feeder"].subgroup == "Feeding"
    assert specs["memory"].subgroup == "Memory"
    assert specs["body"].group == "Larva Modules"
    assert specs["body"].subgroup == "Core"
    assert specs["energetics"].subgroup == "Optional"
    assert specs["body"].module_kind == "larva"
    assert specs["memory"].module_kind == "memory"


def test_inspect_model_modules_enabled_matches_present() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = inspect_model_modules(model_id)
    assert all(spec.enabled == spec.present for spec in specs)


def test_inspect_model_modules_brain_modes_and_labels_use_moduledb() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    crawler = specs["crawler"]
    assert crawler.mode_options == tuple(MD.mod_modes("crawler") or ())
    assert crawler.mode_labels["constant"] == "constant (CON)"


def test_inspect_model_modules_memory_mode_and_modality_options() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    memory = {spec.module_id: spec for spec in inspect_model_modules(model_id)}[
        "memory"
    ]
    assert set(memory.mode_options) >= {"RL", "MB"}
    assert "RL" in memory.modality_options_by_mode
    assert "MB" in memory.modality_options_by_mode
    assert set(memory.modality_options_by_mode["RL"]) >= {"olfaction", "touch"}


def test_inspect_model_modules_larva_specs_have_no_mode_semantics() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    for module_id in ("body", "physics", "energetics", "sensorimotor", "Box2D"):
        spec = specs[module_id]
        assert spec.module_kind == "larva"
        assert spec.mode_options == ()
        assert spec.current_mode is None
        assert spec.mode_labels == {}


def test_inspect_model_modules_core_flags() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    for module_id in (
        "crawler",
        "interference",
        "intermitter",
        "turner",
        "body",
        "physics",
    ):
        assert specs[module_id].is_core is True
    for module_id in (
        "feeder",
        "olfactor",
        "toucher",
        "windsensor",
        "thermosensor",
        "memory",
        "energetics",
        "sensorimotor",
        "Box2D",
    ):
        assert specs[module_id].is_core is False


def test_inspect_model_modules_marks_absent_optional_slots() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    optional_larva = ("energetics", "sensorimotor", "Box2D")
    assert any(specs[module_id].present is False for module_id in optional_larva)


def test_inspect_model_modules_marks_configured_modules_present() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    assert specs["crawler"].present is True
    assert specs["turner"].present is True
    assert specs["body"].present is True
    assert specs["physics"].present is True


def test_load_model_draft_returns_deep_independent_copy() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    before = reg.conf.Model.getID(model_id).brain["crawler"]["mode"]
    draft.brain["crawler"]["mode"] = "constant"
    after = reg.conf.Model.getID(model_id).brain["crawler"]["mode"]
    assert before == after


def test_inspect_model_from_config_reads_draft_not_registry() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["crawler"]["mode"] = "constant"
    draft_inspection = inspect_model_from_config(model_id, draft)
    canonical = inspect_model(model_id)
    draft_mode = next(
        module.mode
        for module in draft_inspection.baseline_modules
        if module.module_id == "crawler"
    )
    canonical_mode = next(
        module.mode
        for module in canonical.baseline_modules
        if module.module_id == "crawler"
    )
    assert draft_mode == "constant"
    assert canonical_mode != "constant"


def test_inspect_model_modules_from_config_reads_draft_not_registry() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["crawler"]["mode"] = "constant"
    specs = {
        spec.module_id: spec
        for spec in inspect_model_modules_from_config(model_id, draft)
    }
    assert specs["crawler"].current_mode == "constant"
    canonical_specs = {spec.module_id: spec for spec in inspect_model_modules(model_id)}
    assert canonical_specs["crawler"].current_mode != "constant"


def test_from_config_helpers_do_not_reload_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)

    def _boom(*_args, **_kwargs):
        raise AssertionError("registry lookup should not run for *_from_config")

    monkeypatch.setattr(data, "_get_model_conf", _boom, raising=True)

    inspect_model_from_config(model_id, draft)
    inspect_model_modules_from_config(model_id, draft)


def test_build_inspection_brain_from_config_does_not_mutate_draft_brain() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    before = repr(draft.brain)
    build_inspection_brain_from_config(model_id, draft, dt=0.1)
    after = repr(draft.brain)
    assert before == after


def test_build_inspection_brain_from_config_reads_draft_not_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    canonical_mode = draft.brain["crawler"]["mode"]
    if canonical_mode == "constant":
        pytest.skip("Need non-constant canonical crawler mode for this assertion.")
    draft.brain["crawler"]["mode"] = "constant"

    captured: dict[str, object] = {}

    class _FakeBrain:
        def __init__(self, *, conf, agent, dt):
            captured["conf"] = conf
            captured["agent"] = agent
            captured["dt"] = dt

    monkeypatch.setattr(data, "DefaultBrain", _FakeBrain, raising=True)
    build_inspection_brain_from_config(model_id, draft, dt=0.1)
    assert captured["conf"]["crawler"]["mode"] == "constant"
    assert load_model_draft(model_id).brain["crawler"]["mode"] == canonical_mode


def test_build_inspection_brain_from_config_does_not_reload_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["crawler"]["mode"] = "constant"
    captured: dict[str, object] = {}

    class _FakeBrain:
        def __init__(self, *, conf, agent, dt):
            captured["conf"] = conf
            captured["agent"] = agent
            captured["dt"] = dt

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "registry lookup should not run for build_inspection_brain_from_config"
        )

    monkeypatch.setattr(data, "_get_model_conf", _boom, raising=True)
    monkeypatch.setattr(data, "DefaultBrain", _FakeBrain, raising=True)
    build_inspection_brain_from_config(model_id, draft, dt=0.1)
    assert captured["conf"]["crawler"]["mode"] == "constant"


def test_build_inspection_brain_wrapper_remains_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    captured: dict[str, object] = {}
    canonical_mode = load_model_draft(model_id).brain["crawler"]["mode"]

    def _fake_builder(model_id_arg: str, model_conf, *, dt: float = 0.1):
        captured["model_id"] = model_id_arg
        captured["model_conf"] = model_conf
        captured["dt"] = dt
        return SimpleNamespace(
            locomotor=SimpleNamespace(step=lambda **_kwargs: (0.0, 0.0, False))
        )

    monkeypatch.setattr(
        data, "build_inspection_brain_from_config", _fake_builder, raising=True
    )
    data.build_inspection_brain(model_id, dt=0.1)
    assert captured["model_id"] == model_id
    assert captured["model_conf"] is not None
    assert captured["model_conf"].brain["crawler"]["mode"] == canonical_mode


def test_default_brain_module_config_uses_canonical_mode_defaults() -> None:
    conf = default_brain_module_config("crawler", "constant")
    assert conf["mode"] == "constant"
    assert "amp" in conf


def test_default_brain_module_config_rejects_invalid_inputs() -> None:
    with pytest.raises(ModelInspectorError):
        default_brain_module_config("memory", "RL")
    with pytest.raises(ModelInspectorError):
        default_brain_module_config("unknown_module", "constant")
    with pytest.raises(ModelInspectorError):
        default_brain_module_config("crawler", "unsupported_mode")


def test_default_brain_module_config_returns_independent_copy() -> None:
    a = default_brain_module_config("crawler", "constant")
    b = default_brain_module_config("crawler", "constant")
    a["amp"] = 123.0
    assert b["amp"] != 123.0


def test_default_memory_config_uses_mode_and_modality() -> None:
    conf = default_memory_config("RL", "olfaction")
    assert conf["mode"] == "RL"
    assert conf["modality"] == "olfaction"


def test_default_memory_config_rejects_invalid_pair() -> None:
    with pytest.raises(ModelInspectorError):
        default_memory_config("INVALID", "olfaction")
    with pytest.raises(ModelInspectorError):
        default_memory_config("RL", "invalid_modality")


def test_default_larva_module_config_uses_canonical_defaults() -> None:
    conf = default_larva_module_config("body")
    assert "Nsegs" in conf


def test_default_larva_module_config_rejects_invalid_module() -> None:
    with pytest.raises(ModelInspectorError):
        default_larva_module_config("crawler")
    with pytest.raises(ModelInspectorError):
        default_larva_module_config("unknown_module")


def test_set_draft_module_enabled_disables_optional_brain_module() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_module_enabled(draft, "olfactor", False)
    assert draft.brain["olfactor"] is None


def test_set_draft_module_enabled_enables_optional_brain_module_with_defaults() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_module_enabled(draft, "olfactor", False)
    set_draft_module_enabled(draft, "olfactor", True)
    assert draft.brain["olfactor"] is not None
    assert draft.brain["olfactor"]["mode"] == (MD.mod_modes("olfactor") or [None])[0]


def test_set_draft_module_enabled_enables_windsensor_with_required_weights() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["windsensor"] = None
    set_draft_module_enabled(draft, "windsensor", True)
    assert draft.brain["windsensor"] is not None
    assert "weights" in draft.brain["windsensor"]
    assert set(draft.brain["windsensor"]["weights"].keys()) == {
        "hunch_lin",
        "hunch_ang",
        "bend_lin",
        "bend_ang",
    }


def test_set_draft_module_enabled_disables_optional_larva_module() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_module_enabled(draft, "Box2D", True)
    set_draft_module_enabled(draft, "Box2D", False)
    assert draft["Box2D"] is None


def test_set_draft_module_enabled_enables_optional_larva_module_with_defaults() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft["Box2D"] = None
    set_draft_module_enabled(draft, "Box2D", True)
    assert "joint_types" in draft["Box2D"]


def test_set_draft_module_enabled_refuses_to_disable_core_modules() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    with pytest.raises(ModelInspectorError):
        set_draft_module_enabled(draft, "crawler", False)
    with pytest.raises(ModelInspectorError):
        set_draft_module_enabled(draft, "body", False)
    with pytest.raises(ModelInspectorError):
        set_draft_module_enabled(draft, "physics", False)


def test_set_draft_brain_module_mode_replaces_config_with_defaults() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["crawler"]["bogus"] = 123
    set_draft_brain_module_mode(draft, "crawler", "constant")
    assert draft.brain["crawler"]["mode"] == "constant"
    assert "bogus" not in draft.brain["crawler"]


def test_set_draft_brain_module_mode_rejects_memory_and_larva_modules() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    with pytest.raises(ModelInspectorError):
        set_draft_brain_module_mode(draft, "memory", "RL")
    with pytest.raises(ModelInspectorError):
        set_draft_brain_module_mode(draft, "body", "constant")


def test_set_draft_memory_config_disables_memory() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=False)
    assert draft.brain["memory"] is None


def test_set_draft_memory_config_writes_valid_memory_config() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=True, mode="RL", modality="touch")
    assert draft.brain["memory"]["mode"] == "RL"
    assert draft.brain["memory"]["modality"] == "touch"


def test_set_draft_memory_config_preserves_valid_modality_when_mode_changes() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=True, mode="RL", modality="touch")
    set_draft_memory_config(draft, enabled=True, mode="MB", modality=None)
    assert draft.brain["memory"]["modality"] == "touch"


def test_set_draft_module_parameter_writes_nested_value() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_module_enabled(draft, "energetics", True)
    set_draft_module_parameter(draft, "energetics", ("DEB", "assimilation_mode"), "gut")
    assert draft["energetics"]["DEB"]["assimilation_mode"] == "gut"


def test_set_draft_module_parameter_rejects_protected_paths() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    with pytest.raises(ModelInspectorError):
        set_draft_module_parameter(draft, "crawler", ("mode",), "constant")
    with pytest.raises(ModelInspectorError):
        set_draft_module_parameter(draft, "memory", ("modality",), "touch")
    with pytest.raises(ModelInspectorError):
        set_draft_module_parameter(draft, "body", ("name",), "x")


def test_set_draft_module_parameter_rejects_disabled_module() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_module_enabled(draft, "olfactor", False)
    with pytest.raises(ModelInspectorError):
        set_draft_module_parameter(draft, "olfactor", ("perception",), "log")


def test_validate_draft_module_config_returns_empty_for_valid_draft() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=False)
    assert validate_draft_module_config(draft) == ()


def test_validate_draft_module_config_reports_memory_missing_sensor_warning() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=True, mode="RL", modality="olfaction")
    set_draft_module_enabled(draft, "olfactor", False)
    issues = validate_draft_module_config(draft)
    assert len(issues) == 1
    assert issues[0].code == "memory_sensor_missing"
    assert issues[0].severity == "warning"
    assert "Memory modality requires enabled sensor module" in issues[0].message


def test_validate_draft_module_config_reports_invalid_memory_mode_error() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    draft.brain["memory"] = {"mode": "INVALID", "modality": "olfaction"}
    issues = validate_draft_module_config(draft)
    assert len(issues) == 1
    assert issues[0].code == "memory_mode_unsupported"
    assert issues[0].severity == "error"


def test_validate_draft_module_config_reports_branch_intermitter_invalid_beta_error() -> (
    None
):
    model_id = (
        "CON_CON_SQ_BR"
        if "CON_CON_SQ_BR" in reg.conf.Model.confIDs
        else (
            "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
        )
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=False)
    draft.brain["intermitter"]["mode"] = "branch"
    draft.brain["intermitter"]["beta"] = None
    issues = validate_draft_module_config(draft)
    target = next(
        (issue for issue in issues if issue.code == "intermitter_branch_beta_invalid"),
        None,
    )
    assert target is not None
    assert target.severity == "error"
    assert target.module_id == "intermitter"
    assert target.path == ("brain", "intermitter", "beta")
    assert 'positive numeric "beta"' in target.message


def test_validate_draft_module_config_allows_branch_intermitter_with_positive_beta() -> (
    None
):
    model_id = (
        "CON_CON_SQ_BR"
        if "CON_CON_SQ_BR" in reg.conf.Model.confIDs
        else (
            "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
        )
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=False)
    draft.brain["intermitter"]["mode"] = "branch"
    draft.brain["intermitter"]["beta"] = 4.7
    issues = validate_draft_module_config(draft)
    assert not any(issue.code == "intermitter_branch_beta_invalid" for issue in issues)


def test_validate_draft_module_config_reports_memory_and_intermitter_issues_together() -> (
    None
):
    model_id = (
        "CON_CON_SQ_BR"
        if "CON_CON_SQ_BR" in reg.conf.Model.confIDs
        else (
            "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
        )
    )
    draft = load_model_draft(model_id)
    set_draft_memory_config(draft, enabled=True, mode="RL", modality="olfaction")
    set_draft_module_enabled(draft, "olfactor", False)
    draft.brain["intermitter"]["mode"] = "branch"
    draft.brain["intermitter"]["beta"] = None
    issues = validate_draft_module_config(draft)
    codes = {issue.code for issue in issues}
    assert "memory_sensor_missing" in codes
    assert "intermitter_branch_beta_invalid" in codes


def test_validate_draft_module_config_does_not_mutate_draft() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    draft = load_model_draft(model_id)
    before = repr(draft.brain)
    validate_draft_module_config(draft)
    after = repr(draft.brain)
    assert before == after


def test_compare_model_inspections_same_model_is_equal() -> None:
    model_id = list_model_ids()[0]
    a = inspect_model(model_id)
    b = inspect_model(model_id)
    diffs = compare_model_inspections(a, b)
    assert len(diffs) >= len(BASELINE_MODULES)
    assert all(diff.equal for diff in diffs)


def test_compare_model_inspections_detects_changes() -> None:
    ids = list_model_ids()
    if len(ids) < 2:
        pytest.skip("Need at least two models to compare.")
    primary = "explorer" if "explorer" in ids else ids[0]
    comparison = (
        "navigator" if "navigator" in ids and primary != "navigator" else ids[1]
    )
    diffs = compare_model_inspections(inspect_model(primary), inspect_model(comparison))
    assert any(not diff.equal for diff in diffs)


def test_probe_output_schema_and_reporter_metadata() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    result = run_model_probe(model_id, steps=5, dt=0.1, a_in=0.0)
    assert list(result.dataframe.columns)[:4] == ["time", "lin", "ang", "feed_motion"]
    assert "A_T" in result.dataframe.columns
    assert "A_C" in result.dataframe.columns
    assert set(result.reporter_available.keys()) == {"A_T", "A_C"}


def test_probe_parity_with_legacy_equivalent_runner() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    steps = 25
    dt = 0.1
    a_in = 0.0

    random.seed(1234)
    np.random.seed(1234)
    reference_df, reference_available = _legacy_equivalent_probe(
        model_id, steps=steps, dt=dt, a_in=a_in
    )

    random.seed(1234)
    np.random.seed(1234)
    probe = run_model_probe(model_id, steps=steps, dt=dt, a_in=a_in)

    assert probe.reporter_available == reference_available
    assert probe.dataframe.shape == reference_df.shape
    assert list(probe.dataframe.columns) == list(reference_df.columns)
    assert np.array_equal(
        probe.dataframe["feed_motion"].to_numpy(),
        reference_df["feed_motion"].to_numpy(),
    )
    for column in ["time", "lin", "ang"]:
        left = probe.dataframe[column].to_numpy()
        right = reference_df[column].to_numpy()
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)

    for column in ["A_T", "A_C"]:
        left = probe.dataframe[column].to_numpy()
        right = reference_df[column].to_numpy()
        if reference_available.get(column, False):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
        else:
            assert pd.isna(left).all()
            assert pd.isna(right).all()


def test_new_data_module_does_not_import_legacy_dashboard_source() -> None:
    source = Path(data.__file__).read_text(encoding="utf-8")
    assert "larvaworld.dashboards.model_inspector" not in source
    assert "import panel" not in source
    assert "import bokeh" not in source
