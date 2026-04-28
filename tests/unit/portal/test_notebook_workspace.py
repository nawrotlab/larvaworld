from __future__ import annotations

from pathlib import Path

import pytest

from larvaworld.portal import notebook_workspace
from larvaworld.portal.workspace import (
    WorkspaceError,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("LARVAWORLD_PORTAL_NOTEBOOK_WORKSPACE", raising=False)
    monkeypatch.delenv("LARVAWORLD_JUPYTER_ROOT_DIR", raising=False)


def test_jupyter_root_dir_defaults_to_active_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    assert notebook_workspace._jupyter_root_dir() == workspace_root.resolve()


def test_build_jupyter_url_uses_workspace_relative_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    notebook_path = workspace_root / "metadata" / "notebooks" / "demo.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("{}", encoding="utf-8")

    url = notebook_workspace._build_jupyter_url(notebook_path)

    assert url.endswith("/lab/tree/metadata/notebooks/demo.ipynb")
    assert "/lab/tree//" not in url


def test_jupyter_root_dir_uses_explicit_notebook_workspace_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook_root = tmp_path / "custom-notebooks"
    monkeypatch.setenv("LARVAWORLD_PORTAL_NOTEBOOK_WORKSPACE", str(notebook_root))

    assert notebook_workspace._jupyter_root_dir() == notebook_root.resolve()


def test_workspace_dir_requires_configured_workspace() -> None:
    with pytest.raises(
        WorkspaceError, match="No valid active workspace is configured."
    ):
        notebook_workspace._workspace_dir()


def test_launch_notebook_requires_active_workspace() -> None:
    url, error = notebook_workspace.launch_notebook_for_item("wf.dataset_manager")

    assert url is None
    assert error == "Configure an active workspace before opening notebooks."
