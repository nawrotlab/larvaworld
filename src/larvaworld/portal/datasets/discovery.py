from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

from larvaworld.lib import reg


@dataclass(frozen=True)
class RawDatasetCandidate:
    candidate_id: str
    parent_dir: str
    display_name: str
    source_path: Path
    warnings: list[str]


def _normalized_root(raw_root: Path) -> Path | None:
    candidate = Path(raw_root).expanduser()
    if not candidate.exists() or not candidate.is_dir():
        return None
    return candidate.resolve()


def _relative_parent_dir(raw_root: Path, source_path: Path) -> str:
    rel = source_path.resolve().relative_to(raw_root.resolve())
    return "." if str(rel) == "." else rel.as_posix()


def _matching_files(source_path: Path, filesystem: object) -> list[Path]:
    names = []
    for name in sorted(source_path.iterdir()):
        if not name.is_file():
            continue
        if filesystem.file_pref and not name.name.startswith(filesystem.file_pref):
            continue
        if filesystem.file_suf and not name.name.endswith(filesystem.file_suf):
            continue
        if filesystem.file_sep and filesystem.file_sep not in name.name:
            continue
        names.append(name)
    return names


def _candidate_warnings(source_path: Path, filesystem: object) -> list[str]:
    if filesystem.file_pref or filesystem.file_suf or filesystem.file_sep:
        if not _matching_files(source_path, filesystem):
            return ["No matching raw files detected in the candidate directory."]
    return []


def _candidate_display_name(
    parent_dir: str, candidate_id: str, source_path: Path
) -> str:
    if parent_dir == ".":
        return candidate_id
    if candidate_id == source_path.name:
        return parent_dir
    return f"{parent_dir} / {candidate_id}"


def _folder_candidates(lab: object, raw_root: Path) -> list[RawDatasetCandidate]:
    candidates: list[RawDatasetCandidate] = []
    filesystem = lab.filesystem
    for source_path in sorted(path for path in raw_root.rglob("*") if path.is_dir()):
        name = source_path.name
        if filesystem.folder_pref and not name.startswith(filesystem.folder_pref):
            continue
        if filesystem.folder_suff and not name.endswith(filesystem.folder_suff):
            continue
        parent_dir = _relative_parent_dir(raw_root, source_path)
        candidate_id = name
        candidates.append(
            RawDatasetCandidate(
                candidate_id=candidate_id,
                parent_dir=parent_dir,
                display_name=_candidate_display_name(
                    parent_dir, candidate_id, source_path
                ),
                source_path=source_path.resolve(),
                warnings=_candidate_warnings(source_path, filesystem),
            )
        )
    return candidates


def _group_files_by_token(files: list[Path], file_sep: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for file in files:
        token = file.name.split(file_sep, 1)[0].strip()
        if not token:
            continue
        groups.setdefault(token, []).append(file)
    return groups


def _file_candidates(lab: object, raw_root: Path) -> list[RawDatasetCandidate]:
    candidates: list[RawDatasetCandidate] = []
    filesystem = lab.filesystem
    import_params = inspect.signature(lab.import_func).parameters
    per_token = "source_id" in import_params or (
        "source_files" in import_params and bool(filesystem.file_sep)
    )
    dirs = [raw_root] + sorted(path for path in raw_root.rglob("*") if path.is_dir())
    for source_path in dirs:
        files = _matching_files(source_path, filesystem)
        if not files:
            continue
        parent_dir = _relative_parent_dir(raw_root, source_path)
        if per_token and filesystem.file_sep:
            for token in sorted(_group_files_by_token(files, filesystem.file_sep)):
                candidates.append(
                    RawDatasetCandidate(
                        candidate_id=token,
                        parent_dir=parent_dir,
                        display_name=_candidate_display_name(
                            parent_dir, token, source_path
                        ),
                        source_path=source_path.resolve(),
                        warnings=[],
                    )
                )
            continue
        candidate_id = source_path.name if parent_dir != "." else raw_root.name
        candidates.append(
            RawDatasetCandidate(
                candidate_id=candidate_id,
                parent_dir=parent_dir,
                display_name=_candidate_display_name(
                    parent_dir, candidate_id, source_path
                ),
                source_path=source_path.resolve(),
                warnings=[],
            )
        )
    return candidates


def _dedupe_candidates(
    candidates: list[RawDatasetCandidate],
) -> list[RawDatasetCandidate]:
    deduped: dict[tuple[str, str, str], RawDatasetCandidate] = {}
    for candidate in candidates:
        key = (
            candidate.parent_dir,
            candidate.candidate_id,
            str(candidate.source_path),
        )
        deduped[key] = candidate
    return sorted(
        deduped.values(),
        key=lambda candidate: (
            candidate.parent_dir,
            candidate.candidate_id,
            str(candidate.source_path),
        ),
    )


def discover_raw_datasets(lab_id: str, raw_root: Path) -> list[RawDatasetCandidate]:
    normalized_root = _normalized_root(raw_root)
    if normalized_root is None or not str(lab_id).strip():
        return []

    lab = reg.conf.LabFormat.get(str(lab_id).strip())
    filesystem = lab.filesystem
    if filesystem.folder_pref or filesystem.folder_suff:
        return _dedupe_candidates(_folder_candidates(lab, normalized_root))
    return _dedupe_candidates(_file_candidates(lab, normalized_root))


def _candidate_import_overrides(
    lab_id: str, raw_root: Path, candidate: RawDatasetCandidate
) -> dict[str, object]:
    normalized_root = _normalized_root(raw_root)
    if normalized_root is None:
        return {}
    lab = reg.conf.LabFormat.get(str(lab_id).strip())
    filesystem = lab.filesystem
    import_params = inspect.signature(lab.import_func).parameters
    overrides: dict[str, object] = {}
    if "source_id" in import_params:
        overrides["source_id"] = candidate.candidate_id
    if "source_files" in import_params:
        files = _matching_files(candidate.source_path, filesystem)
        if filesystem.file_sep:
            files = [
                file
                for file in files
                if file.name.split(filesystem.file_sep, 1)[0].strip()
                == candidate.candidate_id
            ]
        overrides["source_files"] = [str(file) for file in files]
    return overrides


__all__: list[str] = ["RawDatasetCandidate", "discover_raw_datasets"]
