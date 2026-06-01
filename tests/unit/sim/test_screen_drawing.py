from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from larvaworld.lib.screen.drawing import ScreenManager


def test_screen_manager_render_display_only_skips_array3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"flip": 0}

    def _flip() -> None:
        calls["flip"] += 1

    monkeypatch.setattr("pygame.display.flip", _flip)
    monkeypatch.setattr(
        "pygame.surfarray.array3d",
        lambda *_: (_ for _ in ()).throw(AssertionError("array3d should not run")),
    )

    manager = SimpleNamespace(
        show_display=True,
        vid_writer=None,
        img_writer=None,
        v=object(),
    )

    image = ScreenManager._render(manager)

    assert image is None
    assert calls["flip"] == 1


def test_screen_manager_render_media_path_keeps_array3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"flip": 0, "array3d": 0}
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    frames: list[np.ndarray] = []

    class DummyWriter:
        def append_data(self, frame: np.ndarray) -> None:
            frames.append(frame)

    def _flip() -> None:
        calls["flip"] += 1

    def _array3d(*_: object) -> np.ndarray:
        calls["array3d"] += 1
        return array

    monkeypatch.setattr("pygame.display.flip", _flip)
    monkeypatch.setattr("pygame.surfarray.array3d", _array3d)

    manager = SimpleNamespace(
        show_display=True,
        vid_writer=DummyWriter(),
        img_writer=None,
        v=object(),
    )

    image = ScreenManager._render(manager)

    assert image is array
    assert calls["flip"] == 1
    assert calls["array3d"] == 1
    assert len(frames) == 1


def test_screen_manager_render_decimation_keeps_input_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "check": 0,
        "evaluate_input": 0,
        "evaluate_graphs": 0,
        "draw_arena": 0,
        "draw_agents": 0,
        "draw_overlay": 0,
        "draw_aux": 0,
        "render": 0,
    }

    monkeypatch.setattr("pygame.display.get_init", lambda: True)

    manager = SimpleNamespace(
        active=True,
        closed=False,
        overlap_mode=False,
        show_display=True,
        display_only_mode=True,
        should_draw_live_frame=False,
        tank_color=(255, 255, 255),
        screen_color=(0, 0, 0),
    )
    manager.check = lambda **_: calls.__setitem__("check", calls["check"] + 1)
    manager.evaluate_input = lambda: calls.__setitem__(
        "evaluate_input", calls["evaluate_input"] + 1
    )
    manager.evaluate_graphs = lambda: calls.__setitem__(
        "evaluate_graphs", calls["evaluate_graphs"] + 1
    )
    manager.draw_arena = lambda: calls.__setitem__(
        "draw_arena", calls["draw_arena"] + 1
    )
    manager.draw_agents = lambda: calls.__setitem__(
        "draw_agents", calls["draw_agents"] + 1
    )
    manager._draw_arena = lambda *_: calls.__setitem__(
        "draw_overlay", calls["draw_overlay"] + 1
    )
    manager.draw_aux = lambda: calls.__setitem__("draw_aux", calls["draw_aux"] + 1)
    manager._render = lambda: calls.__setitem__("render", calls["render"] + 1)

    ScreenManager.render(manager)

    assert calls["check"] == 1
    assert calls["evaluate_input"] == 1
    assert calls["evaluate_graphs"] == 1
    assert calls["draw_arena"] == 0
    assert calls["draw_agents"] == 0
    assert calls["draw_overlay"] == 0
    assert calls["draw_aux"] == 0
    assert calls["render"] == 0


def test_screen_manager_render_does_not_decimate_media_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {
        "check": 0,
        "evaluate_input": 0,
        "evaluate_graphs": 0,
        "draw_arena": 0,
        "draw_agents": 0,
        "draw_overlay": 0,
        "draw_aux": 0,
        "render": 0,
    }

    monkeypatch.setattr("pygame.display.get_init", lambda: True)

    manager = SimpleNamespace(
        active=True,
        closed=False,
        overlap_mode=False,
        show_display=True,
        display_only_mode=False,
        should_draw_live_frame=False,
        tank_color=(255, 255, 255),
        screen_color=(0, 0, 0),
    )
    manager.check = lambda **_: calls.__setitem__("check", calls["check"] + 1)
    manager.evaluate_input = lambda: calls.__setitem__(
        "evaluate_input", calls["evaluate_input"] + 1
    )
    manager.evaluate_graphs = lambda: calls.__setitem__(
        "evaluate_graphs", calls["evaluate_graphs"] + 1
    )
    manager.draw_arena = lambda: calls.__setitem__(
        "draw_arena", calls["draw_arena"] + 1
    )
    manager.draw_agents = lambda: calls.__setitem__(
        "draw_agents", calls["draw_agents"] + 1
    )
    manager._draw_arena = lambda *_: calls.__setitem__(
        "draw_overlay", calls["draw_overlay"] + 1
    )
    manager.draw_aux = lambda: calls.__setitem__("draw_aux", calls["draw_aux"] + 1)
    manager._render = lambda: calls.__setitem__("render", calls["render"] + 1)

    ScreenManager.render(manager)

    assert calls["check"] == 1
    assert calls["evaluate_input"] == 1
    assert calls["evaluate_graphs"] == 1
    assert calls["draw_arena"] == 1
    assert calls["draw_agents"] == 1
    assert calls["draw_overlay"] == 1
    assert calls["draw_aux"] == 1
    assert calls["render"] == 1


def test_screen_manager_display_mode_helpers() -> None:
    manager = SimpleNamespace(
        show_display=True,
        save_video=False,
        image_mode=None,
        vid_writer=None,
        img_writer=None,
        display_every_n_steps=3,
        model=SimpleNamespace(Nticks=1),
    )

    assert ScreenManager.display_only_mode.fget(manager) is True
    assert ScreenManager.should_draw_live_frame.fget(manager) is False

    manager.model.Nticks = 3
    assert ScreenManager.should_draw_live_frame.fget(manager) is True

    manager.save_video = True
    assert ScreenManager.display_only_mode.fget(manager) is False


def test_screen_manager_evaluate_input_uses_injected_pygame_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = {"count": 0}
    monkeypatch.setattr("pygame.event.get", lambda: [])
    monkeypatch.setattr(
        "larvaworld.lib.screen.drawing.reg.controls.load",
        lambda: load_calls.__setitem__("count", load_calls["count"] + 1) or {},
    )

    manager = SimpleNamespace(
        pygame_keys={"pause": "K_SPACE"},
        allow_clicks=False,
        focus_mode=False,
        selected_agents=[],
    )

    ScreenManager.evaluate_input(manager)

    assert load_calls["count"] == 0
    assert manager.pygame_keys["pause"] == "K_SPACE"


def test_screen_manager_evaluate_input_falls_back_to_registry_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pygame.event.get", lambda: [])
    monkeypatch.setattr(
        "larvaworld.lib.screen.drawing.reg.controls.load",
        lambda: {"pygame_keys": {"pause": "K_SPACE"}},
    )

    manager = SimpleNamespace(
        pygame_keys=None,
        allow_clicks=False,
        focus_mode=False,
        selected_agents=[],
    )

    ScreenManager.evaluate_input(manager)

    assert manager.pygame_keys["pause"] == "K_SPACE"
