from __future__ import annotations

import base64
import atexit
import os
import signal
import sys
import threading
import time
import traceback
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from larvaworld.portal.workspace import clear_active_workspace_path


# String-only mapping to keep unit tests free of heavy imports.
APP_ID_TO_FACTORY_PATH: dict[str, str] = {
    # Portal apps
    "/": "larvaworld.portal.serve:loading_app",
    "loading": "larvaworld.portal.serve:loading_app",
    "landing": "larvaworld.portal.landing_app:landing_app",
    "notebook": "larvaworld.portal.notebook_launch_app:notebook_launch_app",
    "wf.run_experiment": "larvaworld.portal.simulation.single_experiment_app:single_experiment_app",
    "wf.open_dataset": "larvaworld.portal.datasets.import_datasets_app:import_datasets_app",
    "wf.dataset_manager": "larvaworld.portal.datasets.dataset_manager_app:dataset_manager_app",
    "wf.export_center": "larvaworld.portal.datasets.analysis_app:analysis_app",
    "wf.environment_builder": "larvaworld.portal.models_architecture.environment_builder_app:environment_builder_app",
    "dev.conftypes": "larvaworld.portal.config_widgets.conftypes_demo_app:conftypes_demo_app",
    # Legacy destinations (served as-is)
    "track_viewer": "larvaworld.portal.datasets.dataset_replay_app:dataset_replay_app",
    "experiment_viewer": "larvaworld.dashboards.experiment_viewer:experiment_viewer_app",
    "larva_models": "larvaworld.portal.models_architecture.model_inspector_app:model_inspector_app",
    "locomotory_modules": "larvaworld.portal.models_architecture.module_inspector_app:module_inspector_app",
    "lateral_oscillator": "larvaworld.dashboards.lateral_oscillator_inspector:lateral_oscillator_app",
}

SERVED_APP_IDS: set[str] = set(APP_ID_TO_FACTORY_PATH.keys())


class _BootstrapState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = False
        self.ready = False
        self.error: str | None = None
        self.started_at = time.monotonic()
        self.current_step = "Waiting to start"
        self.completed_steps = 0
        self.total_steps = 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(time.monotonic() - self.started_at, 0.0)
            percent = (
                100
                if self.ready
                else int((self.completed_steps / max(self.total_steps, 1)) * 100)
            )
            remaining = 0.0
            if self.completed_steps > 0 and self.completed_steps < self.total_steps:
                avg = elapsed / self.completed_steps
                remaining = avg * (self.total_steps - self.completed_steps)
            return {
                "ready": self.ready,
                "error": self.error,
                "step": self.current_step,
                "completed_steps": self.completed_steps,
                "total_steps": self.total_steps,
                "percent": max(0, min(percent, 100)),
                "elapsed": elapsed,
                "remaining": remaining,
            }

    def begin(self, total_steps: int) -> None:
        with self.lock:
            self.started = True
            self.ready = False
            self.error = None
            self.started_at = time.monotonic()
            self.current_step = "Starting..."
            self.completed_steps = 0
            self.total_steps = max(total_steps, 1)

    def set_step(self, step: str) -> None:
        with self.lock:
            self.current_step = step

    def complete_step(self) -> None:
        with self.lock:
            self.completed_steps = min(self.completed_steps + 1, self.total_steps)

    def fail(self, error: str) -> None:
        with self.lock:
            self.error = error
            self.current_step = "Initialization failed"

    def finish(self) -> None:
        with self.lock:
            self.ready = True
            self.current_step = "Ready"
            self.completed_steps = self.total_steps


_BOOTSTRAP_STATE = _BootstrapState()
_BOOTSTRAP_THREAD: threading.Thread | None = None


def _loading_gif_data_uris() -> list[str]:
    gifs_dir = Path(__file__).parent / "media" / "gifs"
    if not gifs_dir.exists():
        return []
    uris: list[str] = []
    for gif_path in sorted(gifs_dir.glob("*.gif")):
        try:
            encoded = base64.b64encode(gif_path.read_bytes()).decode("ascii")
        except OSError:
            continue
        uris.append(f"data:image/gif;base64,{encoded}")
    return uris


_LOADING_GIF_URIS = _loading_gif_data_uris()


def _import_attr(path: str) -> object:
    module_name, attr_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


@lru_cache(maxsize=None)
def _resolve_target(path: str) -> Any:
    return _import_attr(path)


def _lazy_factory(path: str) -> Callable[..., Any]:
    def _factory(*args: Any, **kwargs: Any) -> Any:
        target = _resolve_target(path)
        if callable(target):
            return target(*args, **kwargs)
        return target

    return _factory


def _warmup_steps() -> list[tuple[str, str]]:
    skipped = {"/", "loading"}
    seen_paths: set[str] = set()
    steps: list[tuple[str, str]] = []
    for app_id, path in APP_ID_TO_FACTORY_PATH.items():
        if app_id in skipped or path in seen_paths:
            continue
        # Legacy dashboard modules can execute `servable()` at import time.
        # Warming them up in a background thread can mutate Bokeh documents
        # outside the request/session lifecycle and crash initialization.
        if path.startswith("larvaworld.dashboards."):
            continue
        seen_paths.add(path)
        steps.append((app_id, path))
    return steps


def _run_bootstrap() -> None:
    steps = _warmup_steps()
    _BOOTSTRAP_STATE.begin(total_steps=2 + len(steps))
    try:
        _BOOTSTRAP_STATE.set_step("Initializing larvaworld registry")
        import_module("larvaworld.lib.reg")
        _BOOTSTRAP_STATE.complete_step()

        _BOOTSTRAP_STATE.set_step("Validating portal registry")
        from larvaworld.portal.registry_logic import validate_registry

        validate_registry(strict=True)
        _BOOTSTRAP_STATE.complete_step()

        for app_id, path in steps:
            _BOOTSTRAP_STATE.set_step(f"Loading {app_id}")
            _resolve_target(path)
            _BOOTSTRAP_STATE.complete_step()

        _BOOTSTRAP_STATE.finish()
    except Exception as exc:
        traceback.print_exc()
        _BOOTSTRAP_STATE.fail(f"{type(exc).__name__}: {exc}")


def _start_bootstrap_once() -> None:
    global _BOOTSTRAP_THREAD
    if _BOOTSTRAP_THREAD is not None and _BOOTSTRAP_THREAD.is_alive():
        return
    if _BOOTSTRAP_STATE.started and _BOOTSTRAP_STATE.ready:
        return
    _BOOTSTRAP_THREAD = threading.Thread(
        target=_run_bootstrap, name="portal-bootstrap", daemon=True
    )
    _BOOTSTRAP_THREAD.start()


def loading_app() -> Any:
    import panel as pn
    from larvaworld.portal.panel_components import PORTAL_RAW_CSS
    from larvaworld.portal.workspace_ui import WorkspaceUiController

    _start_bootstrap_once()
    clear_active_workspace_path()

    pn.extension(raw_css=[PORTAL_RAW_CSS])

    redirect = pn.pane.HTML("", margin=0)
    workspace_state = {"confirmed": False}
    workspace_ui = WorkspaceUiController(
        theme="dark",
        on_workspace_change=lambda workspace: (
            workspace_state.__setitem__("confirmed", workspace is not None),
            redirect.__setattr__(
                "object",
                (
                    '<script>window.location.replace("/landing");</script>'
                    '<div style="font-size:12px;color:#86efac;">Workspace configured. Redirecting to landing...</div>'
                )
                if workspace is not None
                else "",
            ),
        )[-1],
    )
    workspace_card = pn.Column(
        pn.pane.HTML(
            '<div style="font-size:24px;font-weight:700;color:#f8fafc;">Choose Workspace</div>',
            margin=(0, 0, 8, 0),
        ),
        pn.pane.HTML(
            (
                '<div style="font-size:13px;line-height:1.5;color:#cbd5e1;">'
                "Select or initialize a Larvaworld workspace before entering the portal. "
                "Notebooks and other persistent workflows are disabled until a workspace is configured."
                "</div>"
            ),
            margin=(0, 0, 14, 0),
        ),
        workspace_ui.build_controls(),
        redirect,
        visible=False,
        width=640,
        margin=0,
        css_classes=["lw-portal-loading-workspace-card"],
        styles={
            "padding": "24px",
            "border": "1px solid rgba(255,255,255,0.14)",
            "border-radius": "12px",
            "background": "rgba(15,23,42,0.90)",
            "box-shadow": "0 10px 30px rgba(0,0,0,0.45)",
        },
    )

    title = pn.pane.HTML(
        '<div style="font-size:24px;font-weight:700;color:#f8fafc;">Loading Larvaworld Portal</div>',
        margin=(0, 0, 8, 0),
    )
    step = pn.pane.HTML(
        '<div style="font-size:14px;color:#cbd5e1;">Starting...</div>',
        margin=(0, 0, 10, 0),
    )
    details = pn.pane.HTML(
        '<div style="font-size:12px;color:#94a3b8;">Preparing initialization...</div>',
        margin=(0, 0, 6, 0),
    )
    error = pn.pane.HTML("", visible=False, margin=(8, 0, 0, 0))
    progress = pn.indicators.Progress(
        value=0, max=100, sizing_mode="stretch_width", bar_color="success"
    )
    background_state = {"index": -1}

    def _update() -> None:
        state = _BOOTSTRAP_STATE.snapshot()
        progress.value = state["percent"]
        step.object = (
            f'<div style="font-size:14px;color:#cbd5e1;">{state["step"]}</div>'
        )
        details.object = (
            '<div style="font-size:12px;color:#94a3b8;">'
            f'{state["completed_steps"]}/{state["total_steps"]} steps • '
            f'elapsed {state["elapsed"]:.1f}s • remaining ~{state["remaining"]:.1f}s'
            "</div>"
        )
        if state["error"]:
            error.visible = True
            card.visible = True
            workspace_card.visible = False
            error.object = (
                '<div style="font-size:12px;color:#fecaca;">'
                f"Initialization failed: {state['error']}</div>"
            )
            return
        if state["ready"]:
            if workspace_state["confirmed"]:
                card.visible = True
                workspace_card.visible = False
            else:
                card.visible = False
                redirect.object = ""
                workspace_card.visible = True
                details.object = (
                    '<div style="font-size:12px;color:#94a3b8;">'
                    "Initialization is complete. Workspace setup is required to continue."
                    "</div>"
                )

        if not _LOADING_GIF_URIS:
            return
        index = int(state["elapsed"] // 4) % len(_LOADING_GIF_URIS)
        if index == background_state["index"]:
            return
        background_state["index"] = index
        root.styles = {
            "background-color": "#3a3a3a",
            "background-image": f"url('{_LOADING_GIF_URIS[index]}')",
            "background-position": "center center",
            "background-size": "contain",
            "background-repeat": "no-repeat",
            "background-attachment": "fixed",
            "min-height": "100vh",
            "padding": "0",
        }

    card = pn.Column(
        title,
        step,
        progress,
        details,
        error,
        redirect,
        width=640,
        margin=0,
        visible=True,
        styles={
            "padding": "24px",
            "border": "1px solid rgba(255,255,255,0.14)",
            "border-radius": "12px",
            "background": "rgba(15,23,42,0.86)",
            "box-shadow": "0 10px 30px rgba(0,0,0,0.45)",
        },
    )
    root = pn.Column(
        pn.Spacer(sizing_mode="stretch_width", height=160),
        pn.Row(
            pn.Spacer(sizing_mode="stretch_width"),
            card,
            workspace_card,
            pn.Spacer(sizing_mode="stretch_width"),
        ),
        sizing_mode="stretch_both",
        styles={"background": "#000000", "min-height": "100vh", "padding": "0"},
    )
    _update()
    pn.state.add_periodic_callback(_update, period=150)
    return root


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _default_open_browser() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def main() -> None:
    import asyncio
    import logging
    import warnings

    import panel as pn

    port = int(os.getenv("LARVAWORLD_PORTAL_PORT", "5006"))
    open_browser = _env_flag("LARVAWORLD_PORTAL_OPEN_BROWSER", _default_open_browser())

    _start_bootstrap_once()

    # Suppress harmless Bokeh validation warnings and WebSocket connection messages
    logging.getLogger("bokeh.core.validation").setLevel(logging.ERROR)
    logging.getLogger("bokeh.server.views.ws").setLevel(logging.ERROR)
    logging.getLogger("tornado.iostream").setLevel(logging.CRITICAL)
    logging.getLogger("tornado.websocket").setLevel(logging.CRITICAL)
    logging.getLogger("tornado.access").setLevel(logging.CRITICAL)
    logging.getLogger("panel.io.document").setLevel(logging.ERROR)
    logging.getLogger("bokeh.server.server").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # Suppress WebSocket closed warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", category=Warning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*WebSocketClosedError.*")
    warnings.filterwarnings("ignore", message=".*StreamClosedError.*")

    # Suppress unhandled exception in asyncio event loop
    def _handle_exception(loop: object, context: dict) -> None:
        exc = context.get("exception")
        if exc and "WebSocketClosedError" in str(type(exc).__name__):
            return
        if exc and "StreamClosedError" in str(type(exc).__name__):
            return
        logging.getLogger("asyncio").exception("Unhandled exception in event loop")

    # Custom logging filter to suppress WebSocket closed errors
    class _WebSocketErrorFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if "WebSocketClosedError" in record.getMessage():
                return False
            if "StreamClosedError" in record.getMessage():
                return False
            if "Task exception was never retrieved" in record.getMessage():
                return False
            return True

    # Apply filter to all handlers
    filter_instance = _WebSocketErrorFilter()
    for handler in logging.root.handlers:
        handler.addFilter(filter_instance)

    # Suppress WebSocket exceptions in traceback output
    _original_excepthook = sys.excepthook

    def _suppress_websocket_excepthook(
        exc_type: type, exc_value: BaseException, tb: object
    ) -> None:
        if "WebSocketClosedError" in str(exc_type):
            return
        if "StreamClosedError" in str(exc_type):
            return
        _original_excepthook(exc_type, exc_value, tb)

    sys.excepthook = _suppress_websocket_excepthook

    def _cleanup() -> None:
        clear_active_workspace_path()

    def _signal_handler(sig: int, frame: object) -> None:
        _cleanup()
        sys.exit(0)

    # Set up event loop exception handler
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_handle_exception)
    except RuntimeError:
        pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _signal_handler)

    apps: dict[str, Callable[..., Any]] = {
        app_id: _lazy_factory(factory_path)
        for app_id, factory_path in APP_ID_TO_FACTORY_PATH.items()
    }

    pn.serve(
        apps,
        port=port,
        show=open_browser,
        threaded=False,
        num_procs=1,
        use_xheaders=False,
    )


if __name__ == "__main__":
    main()
