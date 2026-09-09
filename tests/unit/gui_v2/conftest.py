from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_portal_config_dir(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "portal-config"))
