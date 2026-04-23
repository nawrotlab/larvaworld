from __future__ import annotations

from pathlib import Path

import panel as pn
import pytest
from larvaworld.lib import reg
from larvaworld.lib.reg import config as reg_config
from larvaworld.lib.reg.generators import EnvConf, ExpConf, LabFormat
from larvaworld.portal.config_widgets.conftype_widget import (
    ConftypeWidgetController,
    build_conftype_widget,
    resolve_conftype,
)


@pytest.fixture()
def isolated_conf_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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


def test_resolve_conftype_for_env_conf() -> None:
    assert resolve_conftype(EnvConf).conftype == "Env"


def test_build_widget_rejects_non_parameterized_class() -> None:
    with pytest.raises(TypeError):
        build_conftype_widget(dict)  # type: ignore[arg-type]


def test_controller_can_save_load_and_delete_env_configs(
    isolated_conf_dir: Path,
) -> None:
    controller = ConftypeWidgetController(EnvConf, conftype="Env", allow_reset=False)
    controller.config_id_input.value = "portal_test_env"
    controller._current.arena.geometry = "circular"
    controller._on_save()

    assert "portal_test_env" in reg.conf.Env.confIDs

    controller.preset_select.value = "portal_test_env"
    controller._on_load()
    assert controller.config_id_input.value == "portal_test_env"
    assert controller._current.arena.geometry == "circular"

    controller._on_delete()
    assert "portal_test_env" not in reg.conf.Env.confIDs


def test_build_conftype_widget_returns_panel_view() -> None:
    view = build_conftype_widget(EnvConf, conftype="Env", allow_reset=False)
    assert isinstance(view, pn.Column)


@pytest.mark.parametrize(
    ("config_cls", "conftype"),
    [
        (EnvConf, "Env"),
        (LabFormat, "LabFormat"),
        (ExpConf, "Exp"),
        (reg.conf.Ga.conf_class, "Ga"),
    ],
)
def test_supported_conftype_widgets_build_bokeh_roots(
    config_cls: type,
    conftype: str,
) -> None:
    view = build_conftype_widget(config_cls, conftype=conftype, allow_reset=False)
    root = view.get_root()
    assert root is not None
