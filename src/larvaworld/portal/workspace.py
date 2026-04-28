from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


WorkspaceKind = Literal[
    "environments", "experiments", "datasets", "analysis", "metadata"
]

WORKSPACE_DIR_NAMES: dict[WorkspaceKind, str] = {
    "environments": "environments",
    "experiments": "experiments",
    "datasets": "datasets",
    "analysis": "analysis",
    "metadata": "metadata",
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
    name: str
    metadata_path: Path
    environments_dir: Path
    experiments_dir: Path
    datasets_dir: Path
    analysis_dir: Path
    metadata_dir: Path


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


def get_active_workspace_path() -> Path | None:
    data = read_global_workspace_config()
    raw = data.get("active_workspace")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _resolve_path(raw)


def set_active_workspace_path(path: str | Path) -> Path:
    resolved = _resolve_path(path)
    write_global_workspace_config({"active_workspace": str(resolved)})
    return resolved


def clear_active_workspace_path() -> None:
    write_global_workspace_config({})


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
    metadata_path = _workspace_metadata_path(resolved)
    if metadata_path.exists():
        try:
            existing = read_workspace_metadata(resolved)
            existing_created = existing.get("created_at")
            if isinstance(existing_created, str) and existing_created.strip():
                created_at = existing_created
        except (OSError, json.JSONDecodeError):
            pass

    metadata: dict[str, object] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "workspace_name": (name or _default_workspace_name(resolved)).strip()
        or _default_workspace_name(resolved),
        "created_at": created_at,
        "updated_at": _utc_now_iso(),
        "folders": {kind: dirname for kind, dirname in WORKSPACE_DIR_NAMES.items()},
    }
    write_workspace_metadata(resolved, metadata)
    return load_workspace(resolved)


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
    name = metadata.get("workspace_name")
    if not isinstance(name, str) or not name.strip():
        name = _default_workspace_name(resolved)

    return WorkspaceState(
        root=resolved,
        name=name,
        metadata_path=_workspace_metadata_path(resolved),
        environments_dir=resolved / WORKSPACE_DIR_NAMES["environments"],
        experiments_dir=resolved / WORKSPACE_DIR_NAMES["experiments"],
        datasets_dir=resolved / WORKSPACE_DIR_NAMES["datasets"],
        analysis_dir=resolved / WORKSPACE_DIR_NAMES["analysis"],
        metadata_dir=resolved / WORKSPACE_DIR_NAMES["metadata"],
    )


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
    }
    return mapping[kind]


def get_notebook_workspace_dir(*, workspace: WorkspaceState | None = None) -> Path:
    state = workspace or require_active_workspace()
    path = state.metadata_dir / "notebooks"
    path.mkdir(parents=True, exist_ok=True)
    return path
