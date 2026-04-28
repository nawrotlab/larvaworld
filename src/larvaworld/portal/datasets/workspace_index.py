from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from larvaworld.lib.util.dictsNlists import load_dict
from larvaworld.portal.datasets.models import WorkspaceDatasetRecord
from larvaworld.portal.workspace import WorkspaceState, get_workspace_dir


logger = logging.getLogger(__name__)


def _normalize_group_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().strip("/")
    return normalized or None


def _path_parts_after_datasets(dataset_dir: Path) -> tuple[str, ...] | None:
    parts = dataset_dir.resolve().parts
    try:
        datasets_idx = len(parts) - 1 - parts[::-1].index("datasets")
    except ValueError:
        logger.debug(
            "Ignoring dataset directory outside a datasets root: %s", dataset_dir
        )
        return None
    return parts[datasets_idx + 1 :]


def _record_from_dataset_dir(dataset_dir: Path) -> WorkspaceDatasetRecord | None:
    dataset_dir = dataset_dir.expanduser().resolve()
    data_dir = dataset_dir / "data"
    conf_path = data_dir / "conf.txt"
    h5_path = data_dir / "data.h5"

    if not dataset_dir.is_dir():
        logger.debug("Ignoring dataset path that is not a directory: %s", dataset_dir)
        return None
    if not conf_path.is_file():
        logger.debug("Ignoring dataset without conf.txt: %s", dataset_dir)
        return None
    if not h5_path.is_file():
        logger.debug("Ignoring dataset without data.h5: %s", dataset_dir)
        return None

    relative_parts = _path_parts_after_datasets(dataset_dir)
    if relative_parts is None:
        return None
    if len(relative_parts) < 3 or relative_parts[0] != "imported":
        logger.debug(
            "Ignoring dataset outside current portal-imported layout: %s", dataset_dir
        )
        return None

    config = load_dict(str(conf_path))
    if not config:
        logger.debug("Ignoring dataset with malformed or empty config: %s", conf_path)
        return None

    lab_id = relative_parts[1].strip() or None
    path_group_id = _normalize_group_id("/".join(relative_parts[2:-1]))
    config_group_id = _normalize_group_id(config.get("group_id"))
    larva_group = config.get("larva_group", {})
    larva_group_id = None
    if isinstance(larva_group, dict):
        larva_group_id = _normalize_group_id(larva_group.get("group_id"))

    dataset_id = config.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        dataset_id = dataset_dir.name

    ref_id = config.get("refID")
    if not isinstance(ref_id, str) or not ref_id.strip():
        ref_id = None

    n_agents = config.get("N")
    if not isinstance(n_agents, int):
        agent_ids = config.get("agent_ids")
        if isinstance(agent_ids, list):
            n_agents = len(agent_ids)
        else:
            n_agents = None

    return WorkspaceDatasetRecord(
        dataset_id=dataset_id.strip(),
        dataset_dir=dataset_dir,
        data_dir=data_dir,
        conf_path=conf_path,
        h5_path=h5_path,
        lab_id=lab_id,
        group_id=config_group_id or larva_group_id or path_group_id,
        ref_id=ref_id,
        n_agents=n_agents,
    )


def get_workspace_dataset(dataset_dir: Path) -> WorkspaceDatasetRecord | None:
    return _record_from_dataset_dir(dataset_dir)


def list_workspace_datasets(
    workspace: WorkspaceState | None = None,
) -> list[WorkspaceDatasetRecord]:
    datasets_dir = get_workspace_dir("datasets", workspace=workspace)
    candidates: set[Path] = set()
    for conf_path in datasets_dir.rglob("conf.txt"):
        if conf_path.parent.name != "data":
            continue
        candidates.add(conf_path.parent.parent.resolve())

    records: list[WorkspaceDatasetRecord] = []
    for dataset_dir in sorted(candidates):
        record = _record_from_dataset_dir(dataset_dir)
        if record is not None:
            records.append(record)
    return records
