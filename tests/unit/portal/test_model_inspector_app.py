from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import larvaworld
import pandas as pd
import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.model import moduleDB as MD
from larvaworld.lib.reg import config as reg_config
from larvaworld.portal.config_widgets.preset_controls import PresetSource
from larvaworld.portal.models_architecture.model_inspector_app import (
    _ModelInspectorController,
    _reporter_key_columns,
    model_inspector_app,
)
from larvaworld.portal.models_architecture.model_inspector_data import (
    DEFAULT_LIVE_PREVIEW_REPORTER_KEYS,
    LIVE_PREVIEW_REPORTER_KEYS,
    MODEL_MODULE_ORDER,
    default_brain_module_config,
    load_model_draft,
    inspect_model,
    inspect_model_modules,
)
from larvaworld.portal.workspace import clear_active_workspace_path
from larvaworld.portal.workspace import initialize_workspace, set_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


@pytest.fixture()
def isolated_model_conf_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Opt-in isolation for the handful of tests that actually exercise
    # registry save/delete/reset on reg.conf.Model (now unconditionally
    # available -- see _build_model_preset_controls). CONF_DIR is shared
    # by every conftype, so every conftype gets seeded and restored here,
    # not just Model (see the equivalent fixture in
    # test_single_experiment_app.py for why: a partial seed would
    # silently wipe any other conftype's in-memory dict to empty on its
    # next .load()).
    original_conf_dir = reg_config.CONF_DIR
    original_dicts = {
        conftype: reg.conf[conftype].dict for conftype in larvaworld.CONFTYPES
    }
    tmp_conf_dir = tmp_path / "confDicts"
    tmp_conf_dir.mkdir()
    for conftype, conf_dict in original_dicts.items():
        util.save_dict(
            util.AttrDict(conf_dict).get_copy(), f"{tmp_conf_dir}/{conftype}.txt"
        )
    monkeypatch.setattr(reg_config, "CONF_DIR", str(tmp_conf_dir))
    try:
        yield tmp_conf_dir
    finally:
        monkeypatch.setattr(reg_config, "CONF_DIR", original_conf_dir)
        for conftype, conf_dict in original_dicts.items():
            reg.conf[conftype].dict = conf_dict


def _guard_registry_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("registry write path must not be called")

    monkeypatch.setattr(reg.conf.Model, "save", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "setID", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "delete", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "set_dict", _boom, raising=True)
    monkeypatch.setattr(reg.conf.Model, "reset", _boom, raising=True)


def _activate_test_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace, name="Test Workspace")
    set_active_workspace_path(workspace)
    return workspace.resolve()


def _workspace_preset_token(controller: _ModelInspectorController, name: str) -> str:
    controller.model_preset_controls.refresh_list()
    label = f"Workspace / {name}"
    assert label in controller.model_preset_controls.preset_select.options
    return controller.model_preset_controls.preset_select.options[label]


def _load_registry_model(controller: _ModelInspectorController, model_id: str) -> None:
    """Load `model_id` as the primary model via the top Stored
    Configurations panel -- the replacement for the old
    `controller.primary_select.value = model_id`."""
    controller.model_preset_controls.refresh_list()
    label = f"Registry / {model_id}"
    assert label in controller.model_preset_controls.preset_select.options
    controller.model_preset_controls.preset_select.value = (
        controller.model_preset_controls.preset_select.options[label]
    )
    assert controller.model_preset_controls.load_selected() is True


def _collect_cards(viewable) -> list[pn.Card]:
    cards: list[pn.Card] = []
    if isinstance(viewable, pn.Card):
        cards.append(viewable)
    children = getattr(viewable, "objects", [])
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        cards.extend(_collect_cards(child))
    return cards


def _collect_card_titles(viewable) -> list[str]:
    return [
        card.title for card in _collect_cards(viewable) if getattr(card, "title", "")
    ]


def _collect_module_card_titles(viewable) -> list[str]:
    return [title for title in _collect_card_titles(viewable) if " | " in title]


def _collect_text(viewable) -> str:
    bits: list[str] = []
    if hasattr(viewable, "object") and isinstance(getattr(viewable, "object"), str):
        bits.append(getattr(viewable, "object"))
    if hasattr(viewable, "title") and isinstance(getattr(viewable, "title"), str):
        bits.append(getattr(viewable, "title"))
    children = getattr(viewable, "objects", [])
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        bits.append(_collect_text(child))
    return "\n".join(bit for bit in bits if bit)


def _find_card(viewable, module_id: str) -> pn.Card:
    for card in _collect_cards(viewable):
        title = getattr(card, "title", "")
        if title.split(" | ", 1)[0] == module_id:
            return card
    raise AssertionError(f'Card for module "{module_id}" not found.')


def _collect_widgets(
    viewable, widget_type: type | None = None, name: str | None = None
) -> list:
    widgets: list = []
    if isinstance(viewable, pn.widgets.Widget):
        if (widget_type is None or isinstance(viewable, widget_type)) and (
            name is None or getattr(viewable, "name", None) == name
        ):
            widgets.append(viewable)
    children = getattr(viewable, "objects", [])
    if isinstance(children, dict):
        children = children.values()
    for child in children:
        widgets.extend(_collect_widgets(child, widget_type=widget_type, name=name))
    return widgets


def _find_widget_in_card(card: pn.Card, widget_type: type, name: str):
    widgets = _collect_widgets(card, widget_type=widget_type, name=name)
    if not widgets:
        raise AssertionError(f'Widget "{name}" not found in card "{card.title}".')
    return widgets[0]


def _card_slot(controller: _ModelInspectorController, module_id: str) -> pn.Column:
    slot = controller._module_card_slots.get(module_id)
    assert slot is not None
    return slot


def _status_text(controller: _ModelInspectorController) -> str:
    return (
        _collect_text(controller.status_pane)
        + "\n"
        + _collect_text(controller.validation_pane)
    )


def _fake_preview_builder(captured: dict):
    def _builder(model_id: str, model_conf, *, dt: float = 0.1):
        captured.setdefault("calls", []).append(
            {
                "model_id": model_id,
                "model_conf": model_conf,
                "dt": dt,
                "crawler_mode": model_conf.brain["crawler"]["mode"],
                "crawler_amp": model_conf.brain["crawler"].get("amp"),
            }
        )
        return SimpleNamespace(
            locomotor=SimpleNamespace(step=lambda **_kwargs: (0.0, 0.0, False))
        )

    return _builder


def test_model_inspector_app_returns_viewable(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    view = model_inspector_app()
    assert view is not None


def test_controller_initializes_primary_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert controller._draft_model_id is not None
    assert controller.compare_models_select.value == [controller._draft_model_id]
    assert controller._draft_model is not None
    assert controller.module_sections_box.objects


def test_model_preset_select_defaults_to_explorer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # PresetControlsController.refresh_list() alone defaults preset_select
    # to whichever catalog entry sorts first, with no awareness of the
    # "explorer" default this controller's own draft loads on construction
    # -- the visible dropdown must agree with what's actually loaded.
    _guard_registry_writes(monkeypatch)
    if "explorer" not in reg.conf.Model.confIDs:
        pytest.skip('"explorer" model is not available in this environment.')
    controller = _ModelInspectorController()
    assert controller._draft_model_id == "explorer"
    selected_label = next(
        label
        for label, token in controller.model_preset_controls.preset_select.options.items()
        if token == controller.model_preset_controls.preset_select.value
    )
    assert selected_label == "Registry / explorer"


def test_module_cards_have_draft_backed_parameter_editors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    assert _collect_widgets(crawler_card, pn.widgets.Select, "Mode")
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    assert _collect_widgets(olfactor_card, pn.widgets.Checkbox, "Enabled")
    assert _collect_widgets(olfactor_card, pn.widgets.Select, "Mode")
    memory_card = _find_card(controller.module_sections_box, "memory")
    assert _collect_widgets(memory_card, pn.widgets.Checkbox, "Enabled")
    assert _collect_widgets(memory_card, pn.widgets.Select, "Memory mode")
    assert _collect_widgets(memory_card, pn.widgets.Select, "Memory modality")
    assert _collect_widgets(crawler_card, pn.widgets.Widget, "amp")
    body_card = _find_card(controller.module_sections_box, "body")
    assert _collect_widgets(body_card, pn.widgets.Widget, "Nsegs")
    assert not _collect_widgets(crawler_card, pn.widgets.Widget, "mode")
    assert not _collect_widgets(memory_card, pn.widgets.Widget, "modality")


def test_intermitter_dict_distribution_parameters_are_not_scalar_editors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    intermitter_card = _find_card(controller.module_sections_box, "intermitter")
    assert not _collect_widgets(intermitter_card, pn.widgets.Widget, "run_dist")
    assert not _collect_widgets(intermitter_card, pn.widgets.Widget, "stridechain_dist")
    assert not _collect_widgets(intermitter_card, pn.widgets.Widget, "pause_dist")


def test_generic_module_editors_only_render_existing_draft_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert not _collect_widgets(
        controller.module_sections_box, pn.widgets.Widget, "closed"
    )


def test_module_cards_do_not_render_debug_metadata_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    text = _collect_text(controller.module_sections_box)
    for label in ("Group:", "State:", "Kind:", "Available modes:", "Parameters:"):
        assert label not in text


def test_controller_comparison_hidden_after_local_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    old_value = controller._draft_model.brain["crawler"]["amp"]
    amp_widget.value = old_value + 0.25
    assert controller.compare_run_button.disabled is True
    controller._on_compare_draw()
    assert "hidden during local edits" in _status_text(controller)


def test_status_and_validation_panes_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    view = controller.view()
    assert (
        controller.status_pane in _collect_widgets(view, pn.viewable.Viewable)
        or controller.status_pane is not None
    )
    assert "Ready" in _status_text(controller)
    assert isinstance(controller.validation_pane.objects, list)


def test_controller_merges_optional_modules_into_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    for model_id in reg.conf.Model.confIDs:
        inspection = inspect_model(model_id)
        if not inspection.optional_modules:
            continue
        _load_registry_model(controller, model_id)
        specs = controller._module_specs_by_id
        assert specs
        optional_ids = {module.module_id for module in inspection.optional_modules}
        assert optional_ids.issubset(set(specs.keys()))
        assert any(spec.subgroup == "Optional" for spec in specs.values())
        return
    pytest.skip("No model with configured optional modules was found.")


def test_controller_builds_grouped_module_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    text = _collect_text(controller.module_sections_box)
    assert "Nervous System" in text
    assert "Locomotion" in text
    assert "Sensation" in text
    assert "Memory" in text
    assert "Body and Metabolism" in text
    assert "Core" in text
    assert "Optional" in text


def test_controller_live_preview_plot_checkbox_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    merged = (
        set(controller.plot_reporters_checkbox_activity.value)
        | set(controller.plot_reporters_checkbox_input.value)
        | set(controller.plot_reporters_checkbox_phase.value)
    )
    assert merged == set(DEFAULT_LIVE_PREVIEW_REPORTER_KEYS)
    activity_keys, input_keys, phase_keys = _reporter_key_columns()
    assert set(controller.plot_reporters_checkbox_activity.options.values()) == set(
        activity_keys
    )
    assert set(controller.plot_reporters_checkbox_input.options.values()) == set(
        input_keys
    )
    assert set(controller.plot_reporters_checkbox_phase.options.values()) == set(
        phase_keys
    )
    assert set(controller.plot_reporters_checkbox_activity.options.values()) | set(
        controller.plot_reporters_checkbox_input.options.values()
    ) | set(controller.plot_reporters_checkbox_phase.options.values()) == set(
        LIVE_PREVIEW_REPORTER_KEYS
    )
    assert (
        "Turner activity (A_T)" in controller.plot_reporters_checkbox_activity.options
    )


def test_controller_shows_all_module_slots_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    titles = _collect_module_card_titles(controller.module_sections_box)
    module_ids = [title.split(" | ", 1)[0] for title in titles]
    assert module_ids == [
        "crawler",
        "turner",
        "interference",
        "intermitter",
        "feeder",
        "olfactor",
        "toucher",
        "windsensor",
        "thermosensor",
        "memory",
        "body",
        "physics",
        "energetics",
        "sensorimotor",
        "Box2D",
    ]
    assert "Box2D" in module_ids


def test_controller_rebuilds_cards_on_primary_model_change_without_replacing_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    container_id = id(controller.module_sections_box)
    current = str(controller._draft_model_id)
    candidates = [
        model_id for model_id in reg.conf.Model.confIDs if model_id != current
    ]
    if not candidates:
        pytest.skip("Need at least two models to validate module section refresh.")
    other_model_id = candidates[0]
    expected_specs = inspect_model_modules(other_model_id)
    _load_registry_model(controller, other_model_id)
    assert id(controller.module_sections_box) == container_id
    assert controller._draft_model_id == other_model_id
    titles = _collect_module_card_titles(controller.module_sections_box)
    module_ids = [title.split(" | ", 1)[0] for title in titles]
    assert set(module_ids) == set(MODEL_MODULE_ORDER)
    expected_title_prefixes = {spec.module_id for spec in expected_specs}
    assert set(module_ids) == expected_title_prefixes


def test_controller_primary_change_replaces_draft_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    current = str(controller._draft_model_id)
    candidates = [
        model_id for model_id in reg.conf.Model.confIDs if model_id != current
    ]
    if not candidates:
        pytest.skip("Need at least two models to validate draft replacement.")
    old_draft_id = id(controller._draft_model)
    _load_registry_model(controller, candidates[0])
    assert id(controller._draft_model) != old_draft_id


def test_controller_refresh_reads_from_draft_not_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    canonical = load_model_draft(str(controller._draft_model_id))
    original_mode = canonical.brain["crawler"]["mode"]
    # Set draft to a different mode than the original
    new_mode = "gaussian" if original_mode != "gaussian" else "square"
    controller._draft_model.brain["crawler"]["mode"] = new_mode
    controller._refresh_inspection()
    titles = _collect_module_card_titles(controller.module_sections_box)
    crawler_title = next(title for title in titles if title.startswith("crawler |"))
    assert crawler_title == f"crawler | {new_mode} | configured"
    canonical_after = load_model_draft(str(controller._draft_model_id))
    # Verify draft changed but canonical didn't
    assert canonical_after.brain["crawler"]["mode"] == original_mode
    assert canonical_after.brain["crawler"]["mode"] != new_mode


def test_core_brain_mode_dropdown_options_match_moduledb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    assert list(mode_select.options) == list(MD.mod_modes("crawler") or ())


def test_changing_brain_mode_updates_draft_with_canonical_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    options = list(mode_select.options)
    if len(options) < 2:
        pytest.skip("Need at least two crawler modes.")
    current = controller._draft_model.brain["crawler"]["mode"]
    target = next((opt for opt in options if opt != current), None)
    if target is None:
        pytest.skip("No different crawler mode available.")
    mode_select.value = target
    assert controller._draft_model.brain["crawler"]["mode"] == target
    canonical = default_brain_module_config("crawler", target)
    assert set(controller._draft_model.brain["crawler"].keys()) == set(canonical.keys())
    assert controller._has_local_edits is True


def test_brain_parameter_edit_writes_draft_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    old_value = controller._draft_model.brain["crawler"]["amp"]
    if isinstance(old_value, (int, float)):
        new_value = old_value + 0.5
    else:
        new_value = old_value
    amp_widget.value = new_value
    assert controller._draft_model.brain["crawler"]["amp"] == new_value
    canonical = load_model_draft(str(controller._draft_model_id))
    assert canonical.brain["crawler"]["amp"] != new_value
    assert controller._has_local_edits is True


def test_mode_change_rebuilds_parameter_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    options = list(mode_select.options)
    if "constant" not in options:
        pytest.skip("Need crawler constant mode for rebuild test.")
    mode_select.value = "constant"
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    constant_keys = set(controller._draft_model.brain["crawler"].keys())
    assert "mode" in constant_keys
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    alternate = next(
        (mode for mode in options if mode not in {"constant", "gaussian"}), None
    )
    if alternate is None:
        pytest.skip(
            "No alternate crawler mode available without scipy gaussian dependency."
        )
    try:
        mode_select.value = alternate
    except Exception as exc:
        pytest.skip(
            f"Skipping mode rebuild check for unavailable runtime dependency: {exc}"
        )
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    assert controller._draft_model.brain["crawler"]["mode"] == alternate
    alternate_keys = set(controller._draft_model.brain["crawler"].keys())
    if alternate_keys == constant_keys:
        pytest.skip("Alternate crawler mode produced identical parameter family.")
    visible_names = {
        widget.name
        for widget in _collect_widgets(crawler_card, pn.widgets.Widget)
        if isinstance(getattr(widget, "name", None), str)
    }
    changed_keys = (alternate_keys - {"mode"}) - (constant_keys - {"mode"})
    assert changed_keys
    assert any(key in visible_names for key in changed_keys)


def test_optional_brain_enable_toggle_disables_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled.value = False
    assert controller._draft_model.brain["olfactor"] is None
    text = _collect_text(_find_card(controller.module_sections_box, "olfactor"))
    assert "Not configured in this draft." in text
    assert not _collect_widgets(
        _find_card(controller.module_sections_box, "olfactor"),
        pn.widgets.Widget,
        "perception",
    )


def test_optional_brain_enable_toggle_enables_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled.value = False
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled.value = True
    assert controller._draft_model.brain["olfactor"] is not None
    assert (
        controller._draft_model.brain["olfactor"]["mode"]
        == (MD.mod_modes("olfactor") or [None])[0]
    )


def test_optional_larva_enable_toggle_disables_and_enables_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    box2d_card = _find_card(controller.module_sections_box, "Box2D")
    enabled = _find_widget_in_card(box2d_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
    box2d_card = _find_card(controller.module_sections_box, "Box2D")
    enabled = _find_widget_in_card(box2d_card, pn.widgets.Checkbox, "Enabled")
    enabled.value = False
    assert controller._draft_model["Box2D"] is None
    box2d_card = _find_card(controller.module_sections_box, "Box2D")
    enabled = _find_widget_in_card(box2d_card, pn.widgets.Checkbox, "Enabled")
    enabled.value = True
    assert "joint_types" in controller._draft_model["Box2D"]


def test_core_modules_do_not_have_enable_checkbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    for module_id in ("crawler", "body", "physics"):
        card = _find_card(controller.module_sections_box, module_id)
        assert not _collect_widgets(card, pn.widgets.Checkbox, "Enabled")


def test_memory_card_has_enable_mode_and_modality_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    assert _collect_widgets(memory_card, pn.widgets.Checkbox, "Enabled")
    assert _collect_widgets(memory_card, pn.widgets.Select, "Memory mode")
    assert _collect_widgets(memory_card, pn.widgets.Select, "Memory modality")


def test_memory_mode_and_modality_controls_write_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
        memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    if "touch" not in modality_select.options:
        pytest.skip("touch modality unavailable in this environment.")
    modality_select.value = "touch"
    assert controller._draft_model.brain["memory"]["mode"] == "RL"
    assert controller._draft_model.brain["memory"]["modality"] == "touch"


def test_memory_parameter_edit_writes_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
    memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    alpha_widget = _find_widget_in_card(memory_card, pn.widgets.Widget, "alpha")
    old_alpha = controller._draft_model.brain["memory"]["alpha"]
    alpha_widget.value = old_alpha + 0.1
    assert controller._draft_model.brain["memory"]["alpha"] == old_alpha + 0.1
    canonical = load_model_draft(str(controller._draft_model_id))
    if canonical.brain["memory"] is not None and "alpha" in canonical.brain["memory"]:
        assert canonical.brain["memory"]["alpha"] != old_alpha + 0.1


def test_memory_missing_sensor_warning_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
        memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    modality_select.value = "olfaction"
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled_olf = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled_olf.value = False
    text = _collect_text(controller.module_sections_box)
    assert "Memory modality requires enabled sensor module" in text


def test_warning_keeps_run_enabled_and_rebuilds_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
        memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    modality_select.value = "olfaction"
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled_olf = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled_olf.value = False
    assert any(
        issue.code == "memory_sensor_missing"
        for issue in controller._draft_validation_issues
    )
    assert controller.run_button.disabled is False
    assert controller.run_button.button_type == "warning"
    assert controller._brain is not None
    assert controller._runtime is not None
    assert "Memory modality requires enabled sensor module" in _status_text(controller)
    controller._on_run()
    assert controller._is_running is True
    controller._on_pause()


def test_validation_error_blocks_preview_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    if controller._draft_model.brain["memory"] is None:
        controller._set_module_enabled("memory", True)
    controller._draft_model.brain["memory"]["mode"] = "INVALID"
    controller._sync_preview_after_draft_change(
        message="Injected invalid draft.",
        clear_trace=True,
        mark_dirty=True,
    )
    assert any(
        issue.severity == "error" for issue in controller._draft_validation_issues
    )
    assert controller._brain is None
    assert controller._runtime is None
    assert controller.run_button.disabled is True
    assert controller.run_button.button_type == "danger"
    controller._on_run()
    assert controller._is_running is False
    assert "Live preview blocked by draft validation errors" in _status_text(controller)


def test_module_edit_hides_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    candidates = [
        model_id
        for model_id in reg.conf.Model.confIDs
        if model_id != controller._draft_model_id
    ]
    if candidates:
        controller.compare_models_select.value = [
            controller._draft_model_id,
            candidates[0],
        ]
        controller._on_compare_draw()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    options = list(mode_select.options)
    target = next((opt for opt in options if opt != mode_select.value), None)
    if target is None:
        pytest.skip("No alternate crawler mode available.")
    mode_select.value = target
    assert controller._has_local_edits is True
    assert controller.compare_run_button.disabled is True
    controller._on_compare_draw()
    assert "hidden during local edits" in _status_text(controller)


def test_card_validation_issue_rendering_is_generic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
        memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    modality_select.value = "olfaction"
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled_olf = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled_olf.value = False
    memory_card = _find_card(controller.module_sections_box, "memory")
    card_text = _collect_text(memory_card)
    assert "Validation warning:" in card_text
    assert "Memory modality requires enabled sensor module" in card_text
    warning_panes = [
        pane
        for pane in memory_card.select(pn.pane.Markdown)
        if "lw-model-inspector-validation-warning" in getattr(pane, "css_classes", [])
    ]
    assert warning_panes


def test_larva_body_parameter_editor_writes_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    body_card = _find_card(controller.module_sections_box, "body")
    nsegs_widget = _find_widget_in_card(body_card, pn.widgets.Widget, "Nsegs")
    old_value = controller._draft_model["body"]["Nsegs"]
    nsegs_widget.value = old_value + 1
    assert controller._draft_model["body"]["Nsegs"] == old_value + 1


def test_energetics_nested_parameter_editor_writes_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    energetics_card = _find_card(controller.module_sections_box, "energetics")
    enabled = _find_widget_in_card(energetics_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
    energetics_card = _find_card(controller.module_sections_box, "energetics")
    mode_widget = _find_widget_in_card(
        energetics_card, pn.widgets.Widget, "DEB.assimilation_mode"
    )
    mode_widget.value = "gut"
    assert controller._draft_model["energetics"]["DEB"]["assimilation_mode"] == "gut"


def test_box2d_joint_types_literal_editor_writes_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    box2d_card = _find_card(controller.module_sections_box, "Box2D")
    enabled = _find_widget_in_card(box2d_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled.value):
        enabled.value = True
    box2d_card = _find_card(controller.module_sections_box, "Box2D")
    joint_types_widget = _find_widget_in_card(
        box2d_card, pn.widgets.LiteralInput, "joint_types"
    )
    new_value = dict(controller._draft_model["Box2D"]["joint_types"])
    new_value["distance"]["N"] = 2
    joint_types_widget.value = new_value
    assert controller._draft_model["Box2D"]["joint_types"]["distance"]["N"] == 2


def test_controller_live_run_updates_trace_and_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run()
    controller._tick_live_preview()
    controller._tick_live_preview()
    assert isinstance(controller.probe_table.object, pd.DataFrame)
    assert not controller.probe_table.object.empty
    assert controller._step >= 2
    controller._on_pause()
    assert controller._is_running is False


def test_controller_run_resume_preserves_transient_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    brain = controller._brain
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.3
    controller._on_run()
    controller._on_pause()
    controller._on_run()
    assert controller._brain is not None
    assert controller._brain is not brain


def test_controller_preview_settings_are_editable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller.max_steps_input.value = 2
    controller.a_in_input.value = 0.25
    controller.trace_window_input.value = 1
    controller._on_run()
    controller._tick_live_preview()
    controller._tick_live_preview()
    assert controller._step == 2
    assert len(controller.probe_table.object) == 1
    controller._tick_live_preview()
    assert controller._is_running is False


def test_controller_reset_required_settings_lock_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert controller.dt_input.disabled is False
    controller._on_run()
    assert controller.dt_input.disabled is True
    controller._on_pause()
    assert controller.dt_input.disabled is False


def test_controller_dt_change_resets_local_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    old_brain = controller._brain
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.2
    controller._on_run()
    controller._tick_live_preview()
    controller._on_pause()
    controller._draft_model["body"]["Nsegs"] = 5
    controller._has_local_edits = True
    controller.dt_input.value = 0.2
    assert controller._brain is not old_brain
    assert controller._has_local_edits is True
    assert controller._draft_model["body"]["Nsegs"] == 5
    assert controller._step == 0
    assert controller.probe_table.object.empty


def test_controller_clear_trace_preserves_local_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.2
    controller._on_run()
    controller._tick_live_preview()
    assert not controller.probe_table.object.empty
    controller._on_clear_trace()
    assert controller._has_local_edits is True
    assert controller.probe_table.object.empty


def test_controller_reset_to_preset_clears_local_edit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.4
    controller._draft_model.brain["crawler"]["mode"] = "constant"
    controller._on_reset_to_preset()
    assert controller._has_local_edits is False
    assert controller.compare_run_button.disabled is False
    canonical = load_model_draft(str(controller._draft_model_id))
    assert (
        controller._draft_model.brain["crawler"]["mode"]
        == canonical.brain["crawler"]["mode"]
    )


def test_edit_while_running_pauses_rebuilds_and_clears_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run()
    controller._tick_live_preview()
    assert not controller.probe_table.object.empty
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.35
    assert controller._is_running is False
    assert controller._brain is not None
    assert controller._runtime is not None
    assert controller.probe_table.object.empty
    assert "Preview rebuilt from current draft" in _status_text(controller)


def test_dt_change_with_validation_error_blocks_without_clearing_dirty_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._has_local_edits = True
    if controller._draft_model.brain["memory"] is None:
        controller._set_module_enabled("memory", True)
    controller._draft_model.brain["memory"]["mode"] = "INVALID"
    controller.dt_input.value = 0.2
    assert controller._has_local_edits is True
    assert controller._brain is None
    assert controller.run_button.disabled is True
    assert controller.probe_table.object.empty


def test_reset_to_preset_clears_validation_block_and_reenables_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    if controller._draft_model.brain["memory"] is None:
        controller._set_module_enabled("memory", True)
    controller._draft_model.brain["memory"]["mode"] = "INVALID"
    controller._sync_preview_after_draft_change(
        message="Injected invalid draft.",
        clear_trace=True,
        mark_dirty=True,
    )
    assert controller.run_button.disabled is True
    controller._on_reset_to_preset()
    assert controller._draft_validation_issues == ()
    assert controller._has_local_edits is False
    assert controller.run_button.disabled is False
    assert controller.run_button.button_type == "success"
    assert controller._brain is not None


def test_controller_runtime_builder_receives_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    captured: dict[str, object] = {}

    def _fake_builder(model_id: str, model_conf, *, dt: float = 0.1):
        captured["model_id"] = model_id
        captured["model_conf"] = model_conf
        captured["dt"] = dt
        return SimpleNamespace(
            locomotor=SimpleNamespace(step=lambda **_kwargs: (0.0, 0.0, False))
        )

    monkeypatch.setattr(
        app_module, "build_inspection_brain_from_config", _fake_builder, raising=True
    )
    controller._ensure_brain_for_selected_model()
    assert captured["model_id"] == str(controller._draft_model_id)
    assert captured["model_conf"] is controller._draft_model


def test_preview_runtime_rebuild_uses_edited_draft_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        app_module,
        "build_inspection_brain_from_config",
        _fake_preview_builder(captured),
        raising=True,
    )
    monkeypatch.setattr(
        controller,
        "_prepare_reporters",
        lambda: setattr(
            controller, "_reporter_available", {"A_T": False, "A_C": False}
        ),
        raising=True,
    )
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    mode_select = _find_widget_in_card(crawler_card, pn.widgets.Select, "Mode")
    options = list(mode_select.options)
    target = next((opt for opt in options if opt != mode_select.value), None)
    if target is None:
        pytest.skip("No alternate crawler mode available.")
    canonical_mode = load_model_draft(str(controller._draft_model_id)).brain["crawler"][
        "mode"
    ]
    mode_select.value = target
    latest = captured["calls"][-1]
    assert latest["model_conf"] is controller._draft_model
    assert latest["crawler_mode"] == controller._draft_model.brain["crawler"]["mode"]
    assert (
        load_model_draft(str(controller._draft_model_id)).brain["crawler"]["mode"]
        == canonical_mode
    )


def test_preview_runtime_rebuild_uses_edited_draft_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        app_module,
        "build_inspection_brain_from_config",
        _fake_preview_builder(captured),
        raising=True,
    )
    monkeypatch.setattr(
        controller,
        "_prepare_reporters",
        lambda: setattr(
            controller, "_reporter_available", {"A_T": False, "A_C": False}
        ),
        raising=True,
    )
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    old_amp = controller._draft_model.brain["crawler"]["amp"]
    amp_widget.value = old_amp + 0.2
    latest = captured["calls"][-1]
    assert latest["model_conf"] is controller._draft_model
    assert latest["crawler_amp"] == controller._draft_model.brain["crawler"]["amp"]
    assert (
        load_model_draft(str(controller._draft_model_id)).brain["crawler"]["amp"]
        == old_amp
    )


def test_run_rebuilds_from_current_draft_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        app_module,
        "build_inspection_brain_from_config",
        _fake_preview_builder(captured),
        raising=True,
    )
    monkeypatch.setattr(
        controller,
        "_prepare_reporters",
        lambda: setattr(
            controller, "_reporter_available", {"A_T": False, "A_C": False}
        ),
        raising=True,
    )
    monkeypatch.setattr(
        controller,
        "_start_callback",
        lambda: controller._set_running(True),
        raising=True,
    )
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.2
    controller._brain = None
    controller._runtime = None
    controller._on_run()
    latest = captured["calls"][-1]
    assert latest["model_conf"] is controller._draft_model
    assert latest["crawler_amp"] == controller._draft_model.brain["crawler"]["amp"]
    assert controller._is_running is True


def test_reset_to_preset_changes_preview_back_to_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        app_module,
        "build_inspection_brain_from_config",
        _fake_preview_builder(captured),
        raising=True,
    )
    monkeypatch.setattr(
        controller,
        "_prepare_reporters",
        lambda: setattr(
            controller, "_reporter_available", {"A_T": False, "A_C": False}
        ),
        raising=True,
    )
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    canonical_amp = load_model_draft(str(controller._draft_model_id)).brain["crawler"][
        "amp"
    ]
    amp_widget.value = canonical_amp + 0.25
    assert controller._draft_model.brain["crawler"]["amp"] != canonical_amp
    controller._on_reset_to_preset()
    assert controller._has_local_edits is False
    assert controller._draft_model.brain["crawler"]["amp"] == canonical_amp
    latest = captured["calls"][-1]
    assert latest["crawler_amp"] == canonical_amp


def test_comparison_remains_canonical_only_and_hidden_during_draft_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    candidates = [m for m in reg.conf.Model.confIDs if m != controller._draft_model_id]
    if not candidates:
        pytest.skip("Need at least two models for comparison behavior test.")
    controller.compare_models_select.value = [controller._draft_model_id, candidates[0]]
    controller._on_compare_draw()
    assert controller.compare_figure_pane.object
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.2
    assert controller._has_local_edits is True
    assert controller.compare_run_button.disabled is True
    controller._on_compare_draw()
    assert "hidden during local edits" in _status_text(controller)
    assert controller.compare_figure_pane.object == ""
    controller._on_reset_to_preset()
    assert controller.compare_run_button.disabled is False


def test_live_preview_probe_table_still_updates_after_draft_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.2
    controller._on_run()
    controller._tick_live_preview()
    controller._tick_live_preview()
    assert isinstance(controller.probe_table.object, pd.DataFrame)
    assert not controller.probe_table.object.empty
    cols = set(controller.probe_table.object.columns)
    assert {"time", "lin", "ang", "feed_motion"}.issubset(cols)
    controller._on_pause()


def test_model_preset_controls_are_visible_and_default_to_full_registry_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    assert controller.model_preset_controls.view is not None
    assert controller.model_preset_controls.policy.can_save_workspace is True
    assert controller.model_preset_controls.policy.can_save_registry is True
    assert controller.model_preset_controls.policy.can_delete_registry is True
    assert controller.model_preset_controls.policy.can_reset_registry is True
    # dual_write=False here (unlike Environment Builder): workspace and
    # registry are independent targets, so save_target stays a real toggle.
    assert controller.model_preset_controls.save_target is not None
    assert controller.model_preset_controls.reset_button is not None
    assert controller.export_model_btn is not None
    assert controller.import_model_input is not None


def test_workspace_model_preset_save_load_delete_roundtrips_current_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    workspace = _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    saved_amp = controller._draft_model.brain["crawler"]["amp"] + 0.41
    amp_widget.value = saved_amp
    controller.model_preset_controls.preset_name.value = "phase6_roundtrip"
    assert controller.model_preset_controls.save_current() is True
    saved_path = workspace / "metadata" / "model_presets" / "phase6_roundtrip.json"
    assert saved_path.is_file()
    amp_widget = _find_widget_in_card(
        _find_card(controller.module_sections_box, "crawler"), pn.widgets.Widget, "amp"
    )
    amp_widget.value = saved_amp + 0.33
    token = _workspace_preset_token(controller, "phase6_roundtrip")
    controller.model_preset_controls.preset_select.value = token
    assert controller.model_preset_controls.load_selected() is True
    assert controller._draft_model.brain["crawler"]["amp"] == saved_amp
    assert controller._has_local_edits is True
    assert controller.compare_run_button.disabled is True
    assert controller.model_preset_controls.delete_selected() is False
    assert controller.model_preset_controls.confirm_pending_action() is True
    assert not saved_path.exists()


def test_json_download_exports_current_draft_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    expected_amp = controller._draft_model.brain["crawler"]["amp"] + 0.19
    amp_widget.value = expected_amp
    payload = json.loads(controller._draft_json_text())
    assert payload["brain"]["crawler"]["amp"] == expected_amp
    assert payload == controller._draft_payload_for_storage()


def test_import_model_file_replaces_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    exported_amp = controller._draft_model.brain["crawler"]["amp"] + 0.17
    amp_widget.value = exported_amp
    payload_text = controller._draft_json_text()

    controller._on_reset_to_preset()
    assert controller._draft_model.brain["crawler"]["amp"] != exported_amp

    controller.import_model_input.filename = "my_export.json"
    controller.import_model_input.value = payload_text.encode("utf-8")

    assert controller._draft_model.brain["crawler"]["amp"] == exported_amp
    assert controller._has_local_edits is True
    assert controller.model_preset_controls.preset_name.value == "my_export"
    refreshed_card = _find_card(controller.module_sections_box, "crawler")
    refreshed_amp = _find_widget_in_card(refreshed_card, pn.widgets.Widget, "amp")
    assert refreshed_amp.value == exported_amp


def test_import_model_file_rejects_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    controller.import_model_input.filename = "bad.json"
    controller.import_model_input.value = b"[1, 2, 3]"
    assert "not a valid model configuration" in controller.status_pane.object


def test_compare_models_select_defaults_to_registry_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert set(controller.compare_models_select.options) == set(controller._model_ids)
    assert controller.compare_models_select.value == [controller._draft_model_id]


def test_loading_workspace_model_preset_replaces_draft_and_refreshes_ui(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    crawler_card = _find_card(controller.module_sections_box, "crawler")
    amp_widget = _find_widget_in_card(crawler_card, pn.widgets.Widget, "amp")
    saved_amp = controller._draft_model.brain["crawler"]["amp"] + 0.27
    amp_widget.value = saved_amp
    controller.model_preset_controls.preset_name.value = "phase6_load_ui"
    assert controller.model_preset_controls.save_current() is True
    controller._on_reset_to_preset()
    assert controller._draft_model.brain["crawler"]["amp"] != saved_amp
    controller.model_preset_controls.preset_select.value = _workspace_preset_token(
        controller, "phase6_load_ui"
    )
    assert controller.model_preset_controls.load_selected() is True
    assert controller._draft_model.brain["crawler"]["amp"] == saved_amp
    refreshed_card = _find_card(controller.module_sections_box, "crawler")
    refreshed_amp = _find_widget_in_card(refreshed_card, pn.widgets.Widget, "amp")
    assert refreshed_amp.value == saved_amp
    assert controller._brain is not None
    assert controller._runtime is not None


def test_registry_model_preset_save_and_delete_round_trip(
    tmp_path: Path,
    isolated_model_conf_dir: Path,
) -> None:
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()

    controller.model_preset_controls.preset_name.value = "portal_test_registry_model"
    controller.model_preset_controls.save_target.value = "Registry"
    assert controller.model_preset_controls.save_current() is True
    assert "portal_test_registry_model" in reg.conf.Model.dict

    controller.model_preset_controls.preset_select.value = (
        controller.model_preset_controls.preset_select.options[
            "Registry / portal_test_registry_model"
        ]
    )
    assert controller.model_preset_controls.delete_selected() is False
    assert controller.model_preset_controls.confirm_pending_action() is True
    assert "portal_test_registry_model" not in reg.conf.Model.dict


def test_loading_registry_model_preset_replaces_draft_without_registry_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _guard_registry_writes(monkeypatch)
    _activate_test_workspace(tmp_path)
    controller = _ModelInspectorController()
    current_primary = str(controller._draft_model_id)
    registry_ref = next(
        (
            ref
            for ref in controller.model_preset_controls.catalog.refs
            if ref.source == PresetSource.REGISTRY and ref.name != current_primary
        ),
        None,
    )
    if registry_ref is None:
        pytest.skip("Need alternate registry preset.")
    expected_mode = load_model_draft(registry_ref.name).brain["crawler"]["mode"]
    controller.model_preset_controls.preset_select.value = registry_ref.token
    assert controller.model_preset_controls.load_selected() is True
    assert controller._draft_model.brain["crawler"]["mode"] == expected_mode
    # Registry-sourced loads switch the canonical primary model (like the
    # old table-selection flow) rather than restoring a possibly-edited
    # draft snapshot -- unlike workspace/file-sourced loads, this is not
    # a "local edit".
    assert controller._has_local_edits is False
    assert controller.compare_run_button.disabled is False


def test_workspace_unavailable_disables_model_preset_write_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    assert controller._model_preset_workspace_available is False
    assert controller.model_preset_controls.save_button.disabled is True
    assert controller.model_preset_controls.load_button.disabled is True
    assert controller.model_preset_controls.delete_button.disabled is True
    assert controller.module_sections_box.objects


def test_safe_parameter_edit_does_not_replace_sections_slots_or_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    sections_before = list(controller.module_sections_box.objects)
    crawler_slot_before = _card_slot(controller, "crawler")
    crawler_card_before = _find_card(controller.module_sections_box, "crawler")
    body_slot_before = _card_slot(controller, "body")
    body_card_before = _find_card(controller.module_sections_box, "body")
    amp_widget = _find_widget_in_card(crawler_card_before, pn.widgets.Widget, "amp")
    amp_widget.value = controller._draft_model.brain["crawler"]["amp"] + 0.37
    assert list(controller.module_sections_box.objects) == sections_before
    assert _card_slot(controller, "crawler") is crawler_slot_before
    assert _find_card(controller.module_sections_box, "crawler") is crawler_card_before
    assert _card_slot(controller, "body") is body_slot_before
    assert _find_card(controller.module_sections_box, "body") is body_card_before


def test_compare_change_does_not_rebuild_module_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    candidates = [
        model_id
        for model_id in reg.conf.Model.confIDs
        if model_id != controller._draft_model_id
    ]
    if not candidates:
        pytest.skip("Need alternate model for comparison change test.")
    sections_before = list(controller.module_sections_box.objects)
    crawler_card_before = _find_card(controller.module_sections_box, "crawler")
    body_card_before = _find_card(controller.module_sections_box, "body")
    controller.compare_models_select.value = [controller._draft_model_id, candidates[0]]
    controller._on_compare_draw()
    assert list(controller.module_sections_box.objects) == sections_before
    assert _find_card(controller.module_sections_box, "crawler") is crawler_card_before
    assert _find_card(controller.module_sections_box, "body") is body_card_before
    assert controller.compare_figure_pane.object


def test_compare_figure_pane_renders_model_table_then_model_diff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "model table"/"model diff" are real graph-registry plot functions
    # (lib/plot/table.py) rendered as an embedded PNG, not a hand-rolled
    # DataFrame -- see _on_compare_draw/_render_compare_figure.
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    candidates = [
        model_id
        for model_id in reg.conf.Model.confIDs
        if model_id != controller._draft_model_id
    ]
    if not candidates:
        pytest.skip("Need alternate model for comparison layout test.")
    assert controller.compare_figure_pane.css_classes == [
        "lw-model-inspector-section-box"
    ]
    assert controller.compare_figure_pane.object == ""

    controller._on_compare_draw()
    assert "<img" in controller.compare_figure_pane.object
    single_model_figure = controller.compare_figure_pane.object

    controller.compare_models_select.value = [controller._draft_model_id, candidates[0]]
    controller._on_compare_draw()
    assert "<img" in controller.compare_figure_pane.object
    assert controller.compare_figure_pane.object != single_model_figure


def test_mode_change_replaces_only_changed_card_slot_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    sections_before = list(controller.module_sections_box.objects)
    crawler_slot_before = _card_slot(controller, "crawler")
    body_slot_before = _card_slot(controller, "body")
    crawler_card_before = _find_card(controller.module_sections_box, "crawler")
    body_card_before = _find_card(controller.module_sections_box, "body")
    mode_select = _find_widget_in_card(crawler_card_before, pn.widgets.Select, "Mode")
    target = next(
        (opt for opt in mode_select.options if opt != mode_select.value), None
    )
    if target is None:
        pytest.skip("No alternate crawler mode available.")
    mode_select.value = target
    assert list(controller.module_sections_box.objects) == sections_before
    assert _card_slot(controller, "crawler") is crawler_slot_before
    assert _card_slot(controller, "body") is body_slot_before
    assert (
        _find_card(controller.module_sections_box, "crawler") is not crawler_card_before
    )
    assert _find_card(controller.module_sections_box, "body") is body_card_before


def test_enable_disable_replaces_only_changed_card_slot_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    sections_before = list(controller.module_sections_box.objects)
    olf_slot_before = _card_slot(controller, "olfactor")
    body_slot_before = _card_slot(controller, "body")
    olf_card_before = _find_card(controller.module_sections_box, "olfactor")
    body_card_before = _find_card(controller.module_sections_box, "body")
    enabled = _find_widget_in_card(olf_card_before, pn.widgets.Checkbox, "Enabled")
    enabled.value = not bool(enabled.value)
    assert list(controller.module_sections_box.objects) == sections_before
    assert _card_slot(controller, "olfactor") is olf_slot_before
    assert _card_slot(controller, "body") is body_slot_before
    assert _find_card(controller.module_sections_box, "olfactor") is not olf_card_before
    assert _find_card(controller.module_sections_box, "body") is body_card_before


def test_sensor_disable_refreshes_memory_validation_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    memory_card = _find_card(controller.module_sections_box, "memory")
    enabled_memory = _find_widget_in_card(memory_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled_memory.value):
        enabled_memory.value = True
        memory_card = _find_card(controller.module_sections_box, "memory")
    mode_select = _find_widget_in_card(memory_card, pn.widgets.Select, "Memory mode")
    mode_select.value = "RL"
    memory_card = _find_card(controller.module_sections_box, "memory")
    modality_select = _find_widget_in_card(
        memory_card, pn.widgets.Select, "Memory modality"
    )
    modality_select.value = "olfaction"
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled_olf = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    if not bool(enabled_olf.value):
        enabled_olf.value = True
    assert "Memory modality requires enabled sensor module" not in _collect_text(
        _find_card(controller.module_sections_box, "memory")
    )
    memory_slot_before = _card_slot(controller, "memory")
    body_slot_before = _card_slot(controller, "body")
    memory_card_before = _find_card(controller.module_sections_box, "memory")
    body_card_before = _find_card(controller.module_sections_box, "body")
    olfactor_card = _find_card(controller.module_sections_box, "olfactor")
    enabled_olf = _find_widget_in_card(olfactor_card, pn.widgets.Checkbox, "Enabled")
    enabled_olf.value = False
    assert _card_slot(controller, "memory") is memory_slot_before
    assert _card_slot(controller, "body") is body_slot_before
    assert (
        _find_card(controller.module_sections_box, "memory") is not memory_card_before
    )
    assert _find_card(controller.module_sections_box, "body") is body_card_before
    assert any(
        issue.code == "memory_sensor_missing"
        for issue in controller._draft_validation_issues
    )
    assert "Memory modality requires enabled sensor module" in _collect_text(
        _find_card(controller.module_sections_box, "memory")
    )


def test_controller_auto_stops_at_max_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _guard_registry_writes(monkeypatch)
    controller = _ModelInspectorController()
    controller._on_run()
    controller._step = controller.max_steps_input.value
    controller._tick_live_preview()
    assert controller._is_running is False


def test_app_source_does_not_import_legacy_dashboard() -> None:
    model_inspector_app_module = importlib.import_module(
        "larvaworld.portal.models_architecture.model_inspector_app"
    )
    source = Path(model_inspector_app_module.__file__).read_text(encoding="utf-8")
    assert "larvaworld.dashboards.model_inspector" not in source
