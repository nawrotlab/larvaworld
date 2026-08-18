from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


WorkspaceKind = Literal[
    "environments", "experiments", "datasets", "analysis", "metadata", "parameters"
]

WORKSPACE_DIR_NAMES: dict[WorkspaceKind, str] = {
    "environments": "environments",
    "experiments": "simulations",
    "datasets": "experiments",
    "analysis": "analysis",
    "metadata": "metadata",
    "parameters": "parameters",
}
WORKSPACE_METADATA_FILENAME = "workspace.json"
GLOBAL_CONFIG_FILENAME = "workspace.json"
WORKSPACE_SCHEMA_VERSION = 1

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]


class WorkspaceError(RuntimeError):
    """Raised when the active workspace is missing or invalid."""


@dataclass(frozen=True)
class WorkspaceValidation:
    path: Path
    exists: bool
    is_dir: bool
    writable: bool
    initialized: bool
    missing_dirs: list[str]
    errors: list[str]


@dataclass(frozen=True)
class WorkspaceState:
    root: Path
    workspace_id: str
    name: str
    metadata_path: Path
    environments_dir: Path
    experiments_dir: Path
    datasets_dir: Path
    analysis_dir: Path
    metadata_dir: Path
    parameters_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _portal_config_dir() -> Path:
    raw_override = os.getenv("LARVAWORLD_PORTAL_CONFIG_DIR", "").strip()
    if raw_override:
        return _resolve_path(raw_override)

    xdg = os.getenv("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return _resolve_path(Path(xdg) / "larvaworld" / "portal")

    return _resolve_path(Path.home() / ".config" / "larvaworld" / "portal")


def _global_config_path() -> Path:
    return _portal_config_dir() / GLOBAL_CONFIG_FILENAME


def _workspace_metadata_dir(root: Path) -> Path:
    return root / WORKSPACE_DIR_NAMES["metadata"]


def _workspace_metadata_path(root: Path) -> Path:
    return _workspace_metadata_dir(root) / WORKSPACE_METADATA_FILENAME


def _reserved_workspace_paths() -> set[Path]:
    return {
        _REPO_ROOT,
        _REPO_ROOT / "src",
        _PACKAGE_ROOT,
        Path(__file__).resolve().parent,
    }


def _is_reserved_workspace_path(path: Path) -> bool:
    return path in _reserved_workspace_paths()


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent
    return candidate


def _path_writable(path: Path) -> bool:
    existing_parent = _nearest_existing_parent(path)
    if existing_parent is None or not existing_parent.is_dir():
        return False
    return os.access(existing_parent, os.W_OK)


def read_global_workspace_config() -> dict[str, object]:
    path = _global_config_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_global_workspace_config(data: dict[str, object]) -> None:
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _known_workspace_entry(
    *, root: Path, workspace_id: str, name: str, last_opened_at: str | None = None
) -> dict[str, str]:
    return {
        "workspace_id": workspace_id,
        "name": name,
        "path": str(root),
        "last_opened_at": last_opened_at or _utc_now_iso(),
    }


def _remember_workspace(
    *, root: Path, workspace_id: str, name: str, mark_opened: bool = True
) -> None:
    """Add or refresh a workspace in the global, path-portable catalog."""
    data = read_global_workspace_config()
    raw_known = data.get("known_workspaces", [])
    known = (
        [item for item in raw_known if isinstance(item, dict)]
        if isinstance(raw_known, list)
        else []
    )
    previous = next(
        (item for item in known if item.get("workspace_id") == workspace_id), None
    )
    previous_opened = previous.get("last_opened_at") if previous else None
    entry = _known_workspace_entry(
        root=root,
        workspace_id=workspace_id,
        name=name,
        last_opened_at=None if mark_opened else previous_opened,
    )
    filtered = [
        item
        for item in known
        if item.get("workspace_id") != workspace_id
        and _resolve_path(str(item.get("path", ""))) != root
    ]
    filtered.append(entry)
    data["known_workspaces"] = sorted(
        filtered, key=lambda item: str(item.get("name", "")).casefold()
    )
    write_global_workspace_config(data)


def get_known_workspaces() -> list[dict[str, object]]:
    """Return known workspaces, retaining unavailable paths for inspection."""
    data = read_global_workspace_config()
    raw_known = data.get("known_workspaces", [])
    if not isinstance(raw_known, list):
        return []
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for item in raw_known:
        if not isinstance(item, dict):
            continue
        workspace_id = item.get("workspace_id")
        raw_path = item.get("path")
        if not isinstance(workspace_id, str) or not isinstance(raw_path, str):
            continue
        root = _resolve_path(raw_path)
        # A moved workspace may have been opened again under a new path. The
        # last catalog entry for its stable UUID is the authoritative one.
        if workspace_id in seen_ids:
            continue
        seen_ids.add(workspace_id)
        records.append(
            {
                "workspace_id": workspace_id,
                "name": item.get("name") or _default_workspace_name(root),
                "path": root,
                "last_opened_at": item.get("last_opened_at"),
                "available": root.is_dir(),
            }
        )
    return records


def forget_known_workspace(workspace_id: str) -> None:
    """Forget a catalog reference without deleting any workspace files."""
    data = read_global_workspace_config()
    raw_known = data.get("known_workspaces", [])
    if isinstance(raw_known, list):
        data["known_workspaces"] = [
            item
            for item in raw_known
            if not isinstance(item, dict) or item.get("workspace_id") != workspace_id
        ]
    write_global_workspace_config(data)


def get_active_workspace_path() -> Path | None:
    data = read_global_workspace_config()
    raw = data.get("active_workspace")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _resolve_path(raw)


def set_active_workspace_path(path: str | Path) -> Path:
    resolved = _resolve_path(path)
    data = read_global_workspace_config()
    data["active_workspace"] = str(resolved)
    write_global_workspace_config(data)
    try:
        metadata = read_workspace_metadata(resolved)
        workspace_id = metadata.get("workspace_id")
        name = metadata.get("workspace_name")
        if isinstance(workspace_id, str) and workspace_id and isinstance(name, str):
            _remember_workspace(
                root=resolved, workspace_id=workspace_id, name=name, mark_opened=True
            )
    except (OSError, json.JSONDecodeError):
        pass
    return resolved


def clear_active_workspace_path() -> None:
    data = read_global_workspace_config()
    data.pop("active_workspace", None)
    write_global_workspace_config(data)


def read_workspace_metadata(path: str | Path) -> dict[str, object]:
    root = _resolve_path(path)
    metadata_path = _workspace_metadata_path(root)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_workspace_metadata(path: str | Path, data: dict[str, object]) -> None:
    root = _resolve_path(path)
    metadata_path = _workspace_metadata_path(root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def validate_workspace(path: str | Path) -> WorkspaceValidation:
    resolved = _resolve_path(path)
    exists = resolved.exists()
    is_dir = resolved.is_dir() if exists else False
    writable = _path_writable(resolved)

    errors: list[str] = []
    if _is_reserved_workspace_path(resolved):
        errors.append("Selected path is reserved for the Larvaworld source tree.")
    if exists and not is_dir:
        errors.append("Selected path exists but is not a directory.")
    if not writable:
        errors.append("Selected path is not writable.")

    missing_dirs: list[str] = []
    if exists and is_dir:
        for dirname in WORKSPACE_DIR_NAMES.values():
            if not (resolved / dirname).is_dir():
                missing_dirs.append(dirname)
    else:
        missing_dirs = list(WORKSPACE_DIR_NAMES.values())

    metadata_path = _workspace_metadata_path(resolved)
    initialized = exists and is_dir and not missing_dirs and metadata_path.is_file()

    return WorkspaceValidation(
        path=resolved,
        exists=exists,
        is_dir=is_dir,
        writable=writable,
        initialized=initialized,
        missing_dirs=missing_dirs,
        errors=errors,
    )


def _default_workspace_name(root: Path) -> str:
    return root.name or "Larvaworld Workspace"


def initialize_workspace(
    path: str | Path, *, name: str | None = None
) -> WorkspaceState:
    resolved = _resolve_path(path)
    validation = validate_workspace(resolved)
    if validation.errors:
        raise WorkspaceError("; ".join(validation.errors))

    resolved.mkdir(parents=True, exist_ok=True)
    for dirname in WORKSPACE_DIR_NAMES.values():
        (resolved / dirname).mkdir(parents=True, exist_ok=True)

    created_at = _utc_now_iso()
    workspace_id = str(uuid.uuid4())
    metadata_path = _workspace_metadata_path(resolved)
    if metadata_path.exists():
        try:
            existing = read_workspace_metadata(resolved)
            existing_created = existing.get("created_at")
            if isinstance(existing_created, str) and existing_created.strip():
                created_at = existing_created
            existing_workspace_id = existing.get("workspace_id")
            if isinstance(existing_workspace_id, str) and existing_workspace_id.strip():
                workspace_id = existing_workspace_id
        except (OSError, json.JSONDecodeError):
            pass

    metadata: dict[str, object] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "workspace_name": (name or _default_workspace_name(resolved)).strip()
        or _default_workspace_name(resolved),
        "created_at": created_at,
        "updated_at": _utc_now_iso(),
        "folders": {kind: dirname for kind, dirname in WORKSPACE_DIR_NAMES.items()},
    }
    write_workspace_metadata(resolved, metadata)
    state = load_workspace(resolved)
    _remember_workspace(
        root=state.root,
        workspace_id=state.workspace_id,
        name=state.name,
        mark_opened=True,
    )
    return state


def load_workspace(path: str | Path) -> WorkspaceState:
    resolved = _resolve_path(path)
    validation = validate_workspace(resolved)
    if validation.errors:
        raise WorkspaceError("; ".join(validation.errors))
    if not validation.exists or not validation.is_dir:
        raise WorkspaceError("Workspace path does not exist as a directory.")
    if not validation.initialized:
        raise WorkspaceError("Workspace is not initialized.")

    metadata = read_workspace_metadata(resolved)
    workspace_id = metadata.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id.strip():
        # Schema-v1 workspaces created before manifest support are upgraded
        # in place the first time they are opened.
        workspace_id = str(uuid.uuid4())
        metadata["workspace_id"] = workspace_id
        metadata["updated_at"] = _utc_now_iso()
        write_workspace_metadata(resolved, metadata)
    name = metadata.get("workspace_name")
    if not isinstance(name, str) or not name.strip():
        name = _default_workspace_name(resolved)

    state = WorkspaceState(
        root=resolved,
        workspace_id=workspace_id,
        name=name,
        metadata_path=_workspace_metadata_path(resolved),
        environments_dir=resolved / WORKSPACE_DIR_NAMES["environments"],
        experiments_dir=resolved / WORKSPACE_DIR_NAMES["experiments"],
        datasets_dir=resolved / WORKSPACE_DIR_NAMES["datasets"],
        analysis_dir=resolved / WORKSPACE_DIR_NAMES["analysis"],
        metadata_dir=resolved / WORKSPACE_DIR_NAMES["metadata"],
        parameters_dir=resolved / WORKSPACE_DIR_NAMES["parameters"],
    )
    _remember_workspace(
        root=state.root,
        workspace_id=state.workspace_id,
        name=state.name,
        mark_opened=True,
    )
    return state


def get_active_workspace() -> WorkspaceState | None:
    active_path = get_active_workspace_path()
    if active_path is None:
        return None
    try:
        return load_workspace(active_path)
    except WorkspaceError:
        return None


def require_active_workspace() -> WorkspaceState:
    workspace = get_active_workspace()
    if workspace is None:
        raise WorkspaceError("No valid active workspace is configured.")
    return workspace


def get_workspace_dir(
    kind: WorkspaceKind, *, workspace: WorkspaceState | None = None
) -> Path:
    state = workspace or require_active_workspace()
    mapping: dict[WorkspaceKind, Path] = {
        "environments": state.environments_dir,
        "experiments": state.experiments_dir,
        "datasets": state.datasets_dir,
        "analysis": state.analysis_dir,
        "metadata": state.metadata_dir,
        "parameters": state.parameters_dir,
    }
    return mapping[kind]


def get_notebook_workspace_dir(*, workspace: WorkspaceState | None = None) -> Path:
    state = workspace or require_active_workspace()
    path = state.metadata_dir / "notebooks"
    path.mkdir(parents=True, exist_ok=True)
    return path
