"""Reusable, Panel-free functions for looking up and loading/saving
`LarvaworldParam` configurations: the registry lookup used throughout the
Parameter Database UI, and the file/bytes/workspace round-trip built on
`LarvaworldParam.to_config()`/`from_config()`/`save_config()`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from larvaworld.lib import reg
from larvaworld.lib.reg.data_aux import LarvaworldParam
from larvaworld.lib.reg.parDB import ParamRegistry
from larvaworld.lib.util import AttrDict
from larvaworld.portal.workspace import WorkspaceError, get_workspace_dir

__all__: list[str] = [
    "get_param_instance",
    "register_new_param",
    "remove_param",
    "save_param_config",
    "param_from_file",
    "save_param_config_to_workspace",
]


def get_param_instance(k: str, par: Optional[ParamRegistry] = None) -> LarvaworldParam:
    """
    Return the live LarvaworldParam instance for key k, instantiating it if
    not already realized (cheap: a single instantiation, not the whole
    registry). Thin wrapper around ParamRegistry.get_param.
    """
    par = par if par is not None else reg.par
    return par.get_param(k)


def register_new_param(
    config: dict[str, Any],
    *,
    overwrite: bool = False,
    par: Optional[ParamRegistry] = None,
) -> str:
    """
    Register a new parameter into the live registry (both par.dict and
    par.kdict) from a config dict, as produced by
    `LarvaworldParam.to_config()` (already correctly typed — no string
    coercion needed). Raises ValueError if `p` is missing, or the resulting
    key already exists and overwrite is not set.
    """
    par = par if par is not None else reg.par
    if not config.get("p"):
        raise ValueError("Parameter name (p) is required.")
    return par.add_and_instantiate(
        overwrite=overwrite, category="custom", **dict(config)
    )


def remove_param(k: str, par: Optional[ParamRegistry] = None) -> None:
    """Remove a parameter from the live registry. Raises KeyError if k is
    not a registered parameter."""
    par = par if par is not None else reg.par
    par.remove(k)


def save_param_config(instance: LarvaworldParam) -> bytes:
    """Save `instance`'s config to a real temp file via the built-in
    `LarvaworldParam.save_config`, then read the bytes back — used for
    browser-download callbacks, which need in-memory content."""
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        instance.save_config(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        os.remove(tmp_path)


def param_from_file(data: bytes, suffix: str = ".pkl") -> LarvaworldParam:
    """Write uploaded config bytes to a real temp file, then load and
    reconstruct via the built-in `AttrDict.load` + `LarvaworldParam.from_config`."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        config = AttrDict.load(tmp_path)
        return LarvaworldParam.from_config(config)
    finally:
        os.remove(tmp_path)


def save_param_config_to_workspace(
    instance: LarvaworldParam, *, filename: Optional[str] = None
) -> Optional[Path]:
    """
    Save `instance`'s config into the active workspace's "parameters"
    folder (created at workspace initialization time, see
    `larvaworld.portal.workspace.WORKSPACE_DIR_NAMES`) -- the default
    persistence location for exported parameter configs, alongside the
    browser download.

    Returns the written path, or None if there is no active workspace
    (e.g. the standalone `python -m larvaworld.portal.parameter_database`
    launcher, which isn't workspace-aware) -- exporting must still work
    without a workspace, so this is best-effort, not a hard requirement.
    """
    try:
        parameters_dir = get_workspace_dir("parameters")
    except WorkspaceError:
        return None
    path = parameters_dir / (filename or f"{instance.k}_config.pkl")
    instance.save_config(str(path))
    return path
