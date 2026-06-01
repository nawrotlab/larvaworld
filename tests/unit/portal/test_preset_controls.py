from __future__ import annotations

import copy
import json
from pathlib import Path

import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg import config as reg_config
from larvaworld.portal.config_widgets.preset_controls import (
    ADVANCED_PRESET_POLICY,
    USER_PRESET_POLICY,
    PresetControlsController,
    PresetSource,
    RegistryPresetStore,
    WorkspacePresetStore,
    build_advanced_preset_controls,
    build_user_preset_controls,
)


@pytest.fixture()
def isolated_env_conf_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_conf_dir = reg_config.CONF_DIR
    original_env_dict = reg.conf.Env.dict
    tmp_conf_dir = tmp_path / "confDicts"
    tmp_conf_dir.mkdir()
    monkeypatch.setattr(reg_config, "CONF_DIR", str(tmp_conf_dir))
    reg.conf.Env.reset(recreate=True)
    try:
        yield tmp_conf_dir
    finally:
        monkeypatch.setattr(reg_config, "CONF_DIR", original_conf_dir)
        reg.conf.Env.dict = original_env_dict


def _env_payload(name: str = "dish") -> util.AttrDict:
    return util.AttrDict(copy.deepcopy(reg.conf.Env.getID(name)))


def _new_controller(
    workspace_dir: Path,
    *,
    policy=USER_PRESET_POLICY,
    on_load=None,
    on_save=None,
    on_status=None,
    build_workspace_payload=None,
    build_registry_payload=None,
) -> PresetControlsController:
    return PresetControlsController(
        conftype="Env",
        workspace_store=WorkspacePresetStore(
            workspace_dir,
            directory_key="env-presets",
        ),
        policy=policy,
        build_workspace_payload=build_workspace_payload
        or (lambda _name: _env_payload()),
        build_registry_payload=build_registry_payload,
        on_load=on_load,
        on_save=on_save,
        on_status=on_status,
        title="Stored Configurations",
    )


def test_workspace_store_rejects_unsafe_names() -> None:
    store = WorkspacePresetStore(Path("/tmp") / "safe", directory_key="env")

    for unsafe in ["", "../dish", "/tmp/dish", "a/b", r"a\\b"]:
        with pytest.raises(ValueError):
            store.normalize_name(unsafe)


def test_workspace_store_lists_direct_json_only_and_roundtrips(tmp_path: Path) -> None:
    preset_dir = tmp_path / "presets"
    nested = preset_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "hidden.json").write_text("{}\n", encoding="utf-8")

    store = WorkspacePresetStore(preset_dir, directory_key="env")
    target = store.save("Dish Alpha", {"value": 1})

    assert target.name == "Dish_Alpha.json"
    records = store.list_presets()
    assert [record.filename for record in records] == ["Dish_Alpha.json"]
    assert store.load("Dish_Alpha.json") == {"value": 1}


def test_workspace_store_delete_cannot_escape_workspace_dir(tmp_path: Path) -> None:
    store = WorkspacePresetStore(tmp_path / "presets", directory_key="env")

    with pytest.raises(ValueError):
        store.delete("../dish.json")


def test_catalog_lists_registry_and_workspace_same_name_visible(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "dish.json").write_text(
        json.dumps(_env_payload("dish"), indent=2) + "\n",
        encoding="utf-8",
    )

    controller = _new_controller(workspace_dir)

    labels = set(controller.preset_select.options.keys())
    assert "Registry / dish" in labels
    assert "Workspace / dish" in labels


def test_catalog_token_resolution_uses_preset_ref_mapping(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "dish.json").write_text("{}\n", encoding="utf-8")

    controller = _new_controller(workspace_dir)
    token = controller.preset_select.options["Workspace / dish"]
    ref = controller.catalog.resolve(token)

    assert ref is not None
    assert ref.source == PresetSource.WORKSPACE
    assert ref.name == "dish"
    assert ref.workspace_filename == "dish.json"


def test_build_helpers_return_panel_column(tmp_path: Path) -> None:
    user_view = build_user_preset_controls(
        conftype="Env",
        workspace_directory=tmp_path / "envs",
        directory_key="envs",
        build_workspace_payload=lambda _name: _env_payload(),
    )
    advanced_view = build_advanced_preset_controls(
        conftype="Env",
        workspace_directory=tmp_path / "envs2",
        directory_key="envs",
        build_workspace_payload=lambda _name: _env_payload(),
    )

    assert isinstance(user_view, pn.Column)
    assert isinstance(advanced_view, pn.Column)


def test_user_helper_has_no_reset_control_and_no_reset_route(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    controller = _new_controller(tmp_path / "presets")

    assert controller.reset_button is None
    assert controller.request_reset_registry() is False


def test_user_helper_blocks_registry_delete_and_registry_mutations_not_called(
    tmp_path: Path, isolated_env_conf_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _new_controller(tmp_path / "presets")
    calls = {"save": 0, "delete": 0, "reset": 0}

    def _save(*_args, **_kwargs):
        calls["save"] += 1

    def _delete(*_args, **_kwargs):
        calls["delete"] += 1

    def _reset(*_args, **_kwargs):
        calls["reset"] += 1

    monkeypatch.setattr(controller.registry_store, "save", _save)
    monkeypatch.setattr(controller.registry_store, "delete", _delete)
    monkeypatch.setattr(controller.registry_store, "reset_defaults", _reset)

    controller.preset_select.value = controller.preset_select.options["Registry / dish"]
    assert controller.delete_selected() is False

    assert calls == {"save": 0, "delete": 0, "reset": 0}
    assert "read-only" in str(controller.status.object)


def test_user_workspace_overwrite_requires_confirmation(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_dir.mkdir(parents=True)
    payload_a = _env_payload("dish")
    payload_a["marker"] = "old"
    (workspace_dir / "dish.json").write_text(
        json.dumps(payload_a, indent=2) + "\n",
        encoding="utf-8",
    )

    payload_b = _env_payload("dish")
    payload_b["marker"] = "new"
    controller = _new_controller(
        workspace_dir,
        build_workspace_payload=lambda _name: payload_b,
    )
    controller.preset_name.value = "dish"

    assert controller.save_current() is False
    assert controller.confirmation_host.objects

    assert controller.confirm_pending_action() is True
    updated = json.loads((workspace_dir / "dish.json").read_text(encoding="utf-8"))
    assert updated["marker"] == "new"


def test_user_workspace_delete_requires_confirmation(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "dish.json").write_text("{}\n", encoding="utf-8")

    controller = _new_controller(workspace_dir)
    controller.preset_select.value = controller.preset_select.options[
        "Workspace / dish"
    ]

    assert controller.delete_selected() is False
    assert (workspace_dir / "dish.json").exists()

    assert controller.confirm_pending_action() is True
    assert not (workspace_dir / "dish.json").exists()


def test_on_load_callback_receives_ref_and_raw_payload(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    seen: list[tuple[str, str, object]] = []

    def _on_load(ref, payload):
        seen.append((ref.source, ref.name, payload))

    controller = _new_controller(tmp_path / "presets", on_load=_on_load)
    controller.preset_select.value = controller.preset_select.options["Registry / dish"]

    assert controller.load_selected() is True
    assert seen
    assert seen[0][0] == PresetSource.REGISTRY
    assert seen[0][1] == "dish"
    assert isinstance(seen[0][2], dict)


def test_on_load_callback_exception_returns_false_with_status(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    def _on_load(_ref, _payload):
        raise RuntimeError("bad payload")

    controller = _new_controller(tmp_path / "presets", on_load=_on_load)
    controller.preset_select.value = controller.preset_select.options["Registry / dish"]

    assert controller.load_selected() is False
    assert "Load failed: bad payload" in str(controller.status.object)


def test_on_save_callback_receives_workspace_ref_from_refreshed_catalog(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    seen: list[tuple[str, str, object]] = []

    def _on_save(ref, payload):
        seen.append((ref.source, ref.name, payload))

    controller = _new_controller(tmp_path / "presets", on_save=_on_save)
    controller.preset_name.value = "dish_ws"

    assert controller.save_current() is True
    assert seen == [(PresetSource.WORKSPACE, "dish_ws", _env_payload())]


def test_on_save_callback_receives_registry_ref_from_refreshed_catalog(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    seen: list[tuple[str, str, object]] = []

    def _on_save(ref, payload):
        seen.append((ref.source, ref.name, payload))

    controller = _new_controller(
        tmp_path / "presets",
        policy=ADVANCED_PRESET_POLICY,
        on_save=_on_save,
    )
    controller.save_target.value = "Registry"
    controller.preset_name.value = "dish_registry"

    assert controller.save_current() is True
    assert seen == [(PresetSource.REGISTRY, "dish_registry", _env_payload())]


def test_on_save_exception_keeps_successful_write_and_returns_true(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"

    def _on_save(_ref, _payload):
        raise RuntimeError("post-save failure")

    controller = _new_controller(workspace_dir, on_save=_on_save)
    controller.preset_name.value = "dish_ws"

    assert controller.save_current() is True
    assert (workspace_dir / "dish_ws.json").is_file()
    assert "post-save update failed" in str(controller.status.object)


def test_regression_registry_load_then_workspace_save_keeps_both_visible(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    before_registry = copy.deepcopy(reg.conf.Env.getID("dish"))

    controller = _new_controller(workspace_dir)
    controller.preset_select.value = controller.preset_select.options["Registry / dish"]
    assert controller.load_selected() is True

    controller.preset_name.value = "dish"
    assert controller.save_current() is True

    labels = set(controller.preset_select.options.keys())
    assert "Registry / dish" in labels
    assert "Workspace / dish" in labels
    assert copy.deepcopy(reg.conf.Env.getID("dish")) == before_registry


def test_advanced_helper_saves_to_workspace_and_registry_targets(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_payload = _env_payload("dish")
    workspace_payload["marker"] = "workspace"

    registry_payload = _env_payload("dish")
    registry_payload["marker"] = "registry"

    controller = _new_controller(
        workspace_dir,
        policy=ADVANCED_PRESET_POLICY,
        build_workspace_payload=lambda _name: workspace_payload,
        build_registry_payload=lambda _name: registry_payload,
    )

    controller.save_target.value = "Workspace"
    controller.preset_name.value = "dish_ws"
    assert controller.save_current() is True
    assert (workspace_dir / "dish_ws.json").is_file()

    controller.save_target.value = "Registry"
    controller.preset_name.value = "dish_registry"
    assert controller.save_current() is True
    saved_registry = reg.conf.Env.getID("dish_registry")
    assert saved_registry["marker"] == "registry"


def test_advanced_registry_delete_requires_confirmation(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    reg.conf.Env.setID("to_delete", _env_payload("dish"))

    controller = _new_controller(tmp_path / "presets", policy=ADVANCED_PRESET_POLICY)
    controller.refresh_list()
    controller.preset_select.value = controller.preset_select.options[
        "Registry / to_delete"
    ]

    assert controller.delete_selected() is False
    assert "to_delete" in reg.conf.Env.confIDs

    assert controller.confirm_pending_action() is True
    assert "to_delete" not in reg.conf.Env.confIDs


def test_advanced_reset_registry_leaves_workspace_presets_untouched(
    tmp_path: Path, isolated_env_conf_dir: Path
) -> None:
    workspace_dir = tmp_path / "presets"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "dish.json").write_text("{}\n", encoding="utf-8")

    reg.conf.Env.setID("custom_env", _env_payload("dish"))

    controller = _new_controller(workspace_dir, policy=ADVANCED_PRESET_POLICY)
    controller.refresh_list()
    assert "Registry / custom_env" in controller.preset_select.options

    assert controller.request_reset_registry() is False
    assert controller.confirm_pending_action() is True

    assert (workspace_dir / "dish.json").is_file()
    controller.refresh_list()
    assert "Workspace / dish" in controller.preset_select.options
    assert "Registry / custom_env" not in controller.preset_select.options


def test_policy_constants_match_expected_permissions() -> None:
    assert USER_PRESET_POLICY.can_save_registry is False
    assert USER_PRESET_POLICY.can_delete_registry is False
    assert USER_PRESET_POLICY.can_reset_registry is False

    assert ADVANCED_PRESET_POLICY.can_save_registry is True
    assert ADVANCED_PRESET_POLICY.can_delete_registry is True
    assert ADVANCED_PRESET_POLICY.can_reset_registry is True
