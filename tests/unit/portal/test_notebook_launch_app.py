from __future__ import annotations

import panel as pn
import pytest

from larvaworld.portal import notebook_launch_app


@pytest.fixture
def onload_callbacks(monkeypatch: pytest.MonkeyPatch) -> list:
    callbacks: list = []
    monkeypatch.setattr(
        notebook_launch_app.pn, "extension", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(notebook_launch_app.pn.state, "onload", callbacks.append)
    return callbacks


def test_notebook_launch_app_shows_initializing_message_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    onload_callbacks: list,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        notebook_launch_app, "_query_param", lambda _name: "wf.dataset_manager"
    )
    monkeypatch.setattr(
        notebook_launch_app,
        "launch_notebook_for_item",
        lambda item_id: (
            calls.append(item_id) or ("http://127.0.0.1:8888/lab/tree/demo.ipynb", None)
        ),
    )

    app = notebook_launch_app.notebook_launch_app()

    assert isinstance(app, pn.Column)
    assert "Initializing notebooks" in app[0].object
    assert "The first notebook launch can take a little longer." in app[0].object
    assert calls == []
    assert len(onload_callbacks) == 1


def test_notebook_launch_app_redirects_after_onload_launch(
    monkeypatch: pytest.MonkeyPatch,
    onload_callbacks: list,
) -> None:
    monkeypatch.setattr(
        notebook_launch_app, "_query_param", lambda _name: "wf.dataset_manager"
    )
    monkeypatch.setattr(
        notebook_launch_app,
        "launch_notebook_for_item",
        lambda _item_id: ("http://127.0.0.1:8888/lab/tree/demo.ipynb", None),
    )

    app = notebook_launch_app.notebook_launch_app()
    onload_callbacks[0]()

    assert "Opening notebook" in app[0].object
    assert "window.location.replace" in app[0].object


def test_notebook_launch_app_shows_error_after_onload_failure(
    monkeypatch: pytest.MonkeyPatch,
    onload_callbacks: list,
) -> None:
    monkeypatch.setattr(
        notebook_launch_app, "_query_param", lambda _name: "wf.dataset_manager"
    )
    monkeypatch.setattr(
        notebook_launch_app,
        "launch_notebook_for_item",
        lambda _item_id: (
            None,
            "Configure an active workspace before opening notebooks.",
        ),
    )

    app = notebook_launch_app.notebook_launch_app()
    onload_callbacks[0]()

    assert "Notebook launch unavailable" in app[0].object
    assert "Configure an active workspace before opening notebooks." in app[0].object


def test_notebook_launch_app_requires_id_immediately(
    monkeypatch: pytest.MonkeyPatch,
    onload_callbacks: list,
) -> None:
    monkeypatch.setattr(notebook_launch_app, "_query_param", lambda _name: None)

    app = notebook_launch_app.notebook_launch_app()

    assert "Missing notebook id." in app[0].object
    assert onload_callbacks == []
