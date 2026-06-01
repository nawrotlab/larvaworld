from __future__ import annotations

from pathlib import Path

import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg import config as reg_config
from larvaworld.lib.reg.generators import EnvConf
from larvaworld.portal.config_widgets.conftype_actions import (
    ConftypeActionsController,
    build_conftype_actions,
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


def test_conftype_actions_save_load_delete_and_reset(
    isolated_env_conf_dir: Path,
) -> None:
    loaded: list[tuple[str, object]] = []
    saved: list[tuple[str, object]] = []
    deleted: list[str] = []
    reset: list[str | None] = []
    statuses: list[tuple[str, str]] = []
    selected_id = ["portal_test_env_actions"]
    save_id = ["portal_test_env_actions"]

    def _build_payload(config_id: str):
        env = EnvConf()
        env.arena.geometry = "circular"
        return env.nestedConf

    controller = ConftypeActionsController(
        EnvConf,
        conftype="Env",
        build_save_payload=_build_payload,
        get_selected_id=lambda: selected_id[0],
        get_save_id=lambda: save_id[0],
        on_load=lambda config_id, config: loaded.append((config_id, config)),
        on_save=lambda config_id, payload: saved.append((config_id, payload)),
        on_delete=lambda config_id: deleted.append(config_id),
        on_reset=lambda config_id: reset.append(config_id),
        on_status=lambda message, *, tone="neutral": statuses.append((message, tone)),
        confirm_reset=False,
    )

    assert controller.save_current() is True
    assert "portal_test_env_actions" in reg.conf.Env.confIDs
    assert saved[0][0] == "portal_test_env_actions"

    assert controller.load_selected() is True
    assert loaded[0][0] == "portal_test_env_actions"

    assert controller.delete_selected() is True
    assert deleted == ["portal_test_env_actions"]
    assert "portal_test_env_actions" not in reg.conf.Env.confIDs

    assert controller.reset_store() is True
    assert reset == [selected_id[0]]
    assert statuses[-1][1] == "success"


def test_build_conftype_actions_returns_panel_view() -> None:
    view = build_conftype_actions(
        EnvConf,
        conftype="Env",
        build_save_payload=lambda _config_id: util.AttrDict(),
        get_selected_id=lambda: None,
    )

    assert isinstance(view, pn.Column)
