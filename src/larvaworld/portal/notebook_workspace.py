from __future__ import annotations

import atexit
import importlib.util
import json
import os
import socket
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from larvaworld.portal.landing_registry import NOTEBOOK_TUTORIAL_BY_ITEM_ID
from larvaworld.portal.workspace import (
    WorkspaceError,
    get_active_workspace,
    get_notebook_workspace_dir,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_TUTORIALS_DIR = _REPO_ROOT / "docs" / "tutorials"
_NOTEBOOK_BUTTON_URLS_CACHE: dict[str, str] | None = None
_PREPARED_NOTEBOOK_URLS_CACHE: dict[str, str] | None = None
_JUPYTER_PROCESS: subprocess.Popen[bytes] | None = None
_LAST_RUNTIME_ERROR: str | None = None
_RUNTIME_JUPYTER_BASE_URL: str | None = None
_JUPYTER_LOG_HANDLE: TextIO | None = None
_TRUTHY = {"1", "true", "yes", "on"}


def _workspace_dir() -> Path:
    raw = os.getenv("LARVAWORLD_PORTAL_NOTEBOOK_WORKSPACE")
    if raw:
        return Path(raw).expanduser().resolve()
    return get_notebook_workspace_dir()


def _kernel_name() -> str:
    kernel = os.getenv("LARVAWORLD_PORTAL_NOTEBOOK_KERNEL", "python3").strip()
    return kernel or "python3"


def _normalize_base_url(raw_base: str) -> str:
    base = raw_base.strip()
    base = base.rstrip("/") or "http://127.0.0.1:8888"
    parsed = urlparse(base)
    if parsed.hostname == "localhost":
        host = "127.0.0.1"
        netloc = host
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth += f":{parsed.password}"
            netloc = f"{auth}@{netloc}"
        base = parsed._replace(netloc=netloc).geturl()
    return base


def _has_explicit_jupyter_base_url() -> bool:
    raw = os.getenv("LARVAWORLD_JUPYTER_BASE_URL")
    return bool(raw and raw.strip())


def _jupyter_base_url() -> str:
    if _RUNTIME_JUPYTER_BASE_URL:
        return _RUNTIME_JUPYTER_BASE_URL
    raw = os.getenv("LARVAWORLD_JUPYTER_BASE_URL", "http://127.0.0.1:8888")
    return _normalize_base_url(raw)


def _jupyter_root_dir() -> Path:
    raw = os.getenv("LARVAWORLD_JUPYTER_ROOT_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    raw_notebook_workspace = os.getenv("LARVAWORLD_PORTAL_NOTEBOOK_WORKSPACE")
    if raw_notebook_workspace:
        return Path(raw_notebook_workspace).expanduser().resolve()
    active_workspace = get_active_workspace()
    if active_workspace is not None:
        return active_workspace.root
    return _REPO_ROOT


def _jupyter_state_dirs() -> dict[str, Path]:
    base = _workspace_dir() / ".jupyter_state"
    runtime_base = Path(tempfile.gettempdir()) / "larvaworld_jupyter_runtime"
    try:
        runtime_base = runtime_base / str(os.getuid())
    except AttributeError:
        runtime_base = runtime_base / "default"
    return {
        "JUPYTER_CONFIG_DIR": base / "config",
        "JUPYTER_DATA_DIR": base / "data",
        "JUPYTER_RUNTIME_DIR": runtime_base,
    }


def _jupyter_log_path() -> Path:
    return _workspace_dir() / ".jupyter_state" / "jupyter.log"


def _notebook_autostart_enabled() -> bool:
    value = os.getenv("LARVAWORLD_PORTAL_NOTEBOOK_AUTOSTART", "1").strip().lower()
    return value in _TRUTHY


def _jupyter_host_port() -> tuple[str, int]:
    parsed = urlparse(_jupyter_base_url())
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        return host, parsed.port
    if parsed.scheme == "https":
        return host, 443
    return host, 8888


def _jupyter_reachable(*, timeout: float = 1.0) -> bool:
    base = _jupyter_base_url()
    request = Request(f"{base}/api", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 500
    except HTTPError as exc:
        return exc.code in {401, 403}
    except URLError:
        return False


def _startup_timeout_seconds() -> float:
    raw = os.getenv("LARVAWORLD_JUPYTER_STARTUP_TIMEOUT_SEC", "75").strip()
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 75.0
    return max(10.0, timeout)


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _build_base_url_with_port(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _find_free_port(host: str, start_port: int, max_tries: int = 20) -> int | None:
    port = start_port
    for _ in range(max_tries):
        if not _is_port_in_use(host, port):
            return port
        port += 1
    return None


def _available_notebook_item_ids() -> list[str]:
    available: list[str] = []
    for item_id, notebook_name in NOTEBOOK_TUTORIAL_BY_ITEM_ID.items():
        if (_TUTORIALS_DIR / notebook_name).exists():
            available.append(item_id)
    return available


def _terminate_jupyter_process() -> None:
    global _JUPYTER_PROCESS
    global _JUPYTER_LOG_HANDLE
    if _JUPYTER_PROCESS is None:
        if _JUPYTER_LOG_HANDLE is not None:
            _JUPYTER_LOG_HANDLE.close()
            _JUPYTER_LOG_HANDLE = None
        return
    try:
        if _JUPYTER_PROCESS.poll() is None:
            _JUPYTER_PROCESS.terminate()
            try:
                _JUPYTER_PROCESS.wait(timeout=4)
            except subprocess.TimeoutExpired:
                _JUPYTER_PROCESS.kill()
        _JUPYTER_PROCESS = None
    finally:
        if _JUPYTER_LOG_HANDLE is not None:
            _JUPYTER_LOG_HANDLE.close()
            _JUPYTER_LOG_HANDLE = None


def _tail_jupyter_log(max_lines: int = 16) -> str:
    log_path = _jupyter_log_path()
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    if not content:
        return ""
    return "\n".join(content[-max_lines:])


def _start_jupyter_process() -> bool:
    global _JUPYTER_PROCESS
    global _LAST_RUNTIME_ERROR
    global _RUNTIME_JUPYTER_BASE_URL
    global _JUPYTER_LOG_HANDLE
    if importlib.util.find_spec("jupyterlab") is None:
        _LAST_RUNTIME_ERROR = (
            "JupyterLab is not installed in the active environment. "
            "Install with: python -m pip install jupyterlab ipykernel"
        )
        return False

    host, port = _jupyter_host_port()
    root_dir = _jupyter_root_dir()
    if not _jupyter_reachable(timeout=0.4) and _is_port_in_use(host, port):
        if _has_explicit_jupyter_base_url():
            _LAST_RUNTIME_ERROR = (
                f"Port {port} is in use by another process. "
                "Set LARVAWORLD_JUPYTER_BASE_URL to a free port (e.g. http://127.0.0.1:8890)."
            )
            return False
        free_port = _find_free_port(host, start_port=port + 1)
        if free_port is None:
            _LAST_RUNTIME_ERROR = (
                f"Port {port} is in use and no free fallback port was found."
            )
            return False
        port = free_port
        _RUNTIME_JUPYTER_BASE_URL = _build_base_url_with_port(host, port)
    elif not _has_explicit_jupyter_base_url():
        _RUNTIME_JUPYTER_BASE_URL = _build_base_url_with_port(host, port)

    cmd = [
        sys.executable,
        "-m",
        "jupyterlab",
        "--no-browser",
        f"--ServerApp.ip={host}",
        f"--ServerApp.port={port}",
        "--ServerApp.port_retries=0",
        f"--ServerApp.root_dir={root_dir}",
        "--ServerApp.open_browser=False",
        "--ServerApp.token=",
    ]
    env = os.environ.copy()
    for name, path in _jupyter_state_dirs().items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    log_path = _jupyter_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _terminate_jupyter_process()

    try:
        _JUPYTER_LOG_HANDLE = log_path.open("w", encoding="utf-8")
        _JUPYTER_LOG_HANDLE.write(
            "\n"
            + "=" * 80
            + "\n"
            + f"[{datetime.now(timezone.utc).isoformat()}] Starting JupyterLab:\n"
            + " ".join(cmd)
            + "\n"
        )
        _JUPYTER_LOG_HANDLE.flush()
    except OSError:
        _JUPYTER_LOG_HANDLE = None

    try:
        _JUPYTER_PROCESS = subprocess.Popen(
            cmd,
            env=env,
            stdout=_JUPYTER_LOG_HANDLE
            if _JUPYTER_LOG_HANDLE is not None
            else subprocess.DEVNULL,
            stderr=subprocess.STDOUT
            if _JUPYTER_LOG_HANDLE is not None
            else subprocess.DEVNULL,
        )
    except OSError:
        _JUPYTER_PROCESS = None
        _LAST_RUNTIME_ERROR = "Failed to start a JupyterLab process."
        if _JUPYTER_LOG_HANDLE is not None:
            _JUPYTER_LOG_HANDLE.close()
            _JUPYTER_LOG_HANDLE = None
        return False

    atexit.register(_terminate_jupyter_process)
    deadline = time.monotonic() + _startup_timeout_seconds()
    while time.monotonic() < deadline:
        if _jupyter_reachable(timeout=1.2):
            _LAST_RUNTIME_ERROR = None
            return True
        if _JUPYTER_PROCESS.poll() is not None:
            log_tail = _tail_jupyter_log()
            _LAST_RUNTIME_ERROR = (
                "JupyterLab process exited before becoming reachable on "
                f"{_jupyter_base_url()}."
            )
            if log_tail:
                _LAST_RUNTIME_ERROR += f"\nLast Jupyter log lines:\n{log_tail}"
            if not _has_explicit_jupyter_base_url():
                _RUNTIME_JUPYTER_BASE_URL = None
            return False
        time.sleep(0.5)

    reachable = _jupyter_reachable(timeout=2.0)
    if not reachable:
        log_tail = _tail_jupyter_log()
        _LAST_RUNTIME_ERROR = f"JupyterLab is not reachable at {_jupyter_base_url()}."
        if log_tail:
            _LAST_RUNTIME_ERROR += f"\nLast Jupyter log lines:\n{log_tail}"
        if not _has_explicit_jupyter_base_url():
            _RUNTIME_JUPYTER_BASE_URL = None
    else:
        _LAST_RUNTIME_ERROR = None
    return reachable


def ensure_notebook_runtime() -> bool:
    global _LAST_RUNTIME_ERROR
    if _jupyter_reachable():
        _LAST_RUNTIME_ERROR = None
        return True
    if not _notebook_autostart_enabled():
        _LAST_RUNTIME_ERROR = "Notebook autostart is disabled by environment variable."
        return False
    return _start_jupyter_process()


def _normalize_notebook_kernel(notebook_path: Path, *, kernel_name: str) -> None:
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    metadata = notebook.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": kernel_name,
        "language": "python",
        "name": kernel_name,
    }
    language_info = metadata.setdefault("language_info", {})
    if isinstance(language_info, dict):
        language_info["name"] = "python"

    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def _build_jupyter_url(notebook_path: Path) -> str:
    jupyter_root = _jupyter_root_dir()
    try:
        relative = notebook_path.resolve().relative_to(jupyter_root)
        notebook_ref = relative.as_posix()
    except ValueError:
        notebook_ref = notebook_path.resolve().as_posix()
    return f"{_jupyter_base_url()}/lab/tree/{quote(notebook_ref)}"


def _prepare_notebook_urls() -> dict[str, str]:
    workspace_dir = _workspace_dir()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    kernel = _kernel_name()

    urls: dict[str, str] = {}
    for item_id, notebook_name in NOTEBOOK_TUTORIAL_BY_ITEM_ID.items():
        source = _TUTORIALS_DIR / notebook_name
        if not source.exists():
            continue

        target = workspace_dir / f"{item_id.replace('.', '_')}.ipynb"
        try:
            shutil.copy2(source, target)
        except OSError:
            continue
        _normalize_notebook_kernel(target, kernel_name=kernel)
        urls[item_id] = _build_jupyter_url(target)

    return urls


def notebook_urls_by_item() -> dict[str, str]:
    global _NOTEBOOK_BUTTON_URLS_CACHE
    if _NOTEBOOK_BUTTON_URLS_CACHE is None:
        _NOTEBOOK_BUTTON_URLS_CACHE = {
            item_id: f"/notebook?id={quote(item_id)}"
            for item_id in _available_notebook_item_ids()
        }
    return _NOTEBOOK_BUTTON_URLS_CACHE


def notebook_names_by_item() -> dict[str, str]:
    names: dict[str, str] = {}
    for item_id, notebook_name in NOTEBOOK_TUTORIAL_BY_ITEM_ID.items():
        if (_TUTORIALS_DIR / notebook_name).exists():
            names[item_id] = notebook_name
    return names


def launch_notebook_for_item(item_id: str) -> tuple[str | None, str | None]:
    notebook_name = NOTEBOOK_TUTORIAL_BY_ITEM_ID.get(item_id)
    if not notebook_name:
        return None, f'Unknown notebook id "{item_id}".'
    if not (_TUTORIALS_DIR / notebook_name).exists():
        return None, f'Notebook source "{notebook_name}" was not found.'
    if get_active_workspace() is None:
        return (
            None,
            "Configure an active workspace before opening notebooks.",
        )
    if not ensure_notebook_runtime():
        return None, _LAST_RUNTIME_ERROR or "Notebook runtime is unavailable."

    global _PREPARED_NOTEBOOK_URLS_CACHE
    if _PREPARED_NOTEBOOK_URLS_CACHE is None:
        _PREPARED_NOTEBOOK_URLS_CACHE = _prepare_notebook_urls()

    notebook_url = _PREPARED_NOTEBOOK_URLS_CACHE.get(item_id)
    if not notebook_url:
        return None, f'Notebook URL could not be prepared for "{item_id}".'
    return notebook_url, None
