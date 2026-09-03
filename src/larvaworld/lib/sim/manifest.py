"""Reproducible simulation manifests and filesystem-backed discovery.

The module intentionally has no eager imports of simulation launchers.  This
keeps it safe to use from all five run implementations and avoids circular
imports while Larvaworld's registry is being initialized.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import secrets
import traceback
import uuid
from collections.abc import Iterable as IterableABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_MEDIA_SUFFIXES = {
    ".avi",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mov",
    ".mp4",
    ".png",
    ".tif",
    ".tiff",
    ".webm",
}
_MANIFEST_CACHE: dict[Path, tuple[int, int, dict[str, Any] | None, str | None]] = {}


class RunManifestError(RuntimeError):
    """Base error for malformed or unusable run manifests."""


class RunManifestValidationError(RunManifestError):
    """Raised when a manifest cannot satisfy the requested rerun policy."""


class RunManifestResolutionError(RunManifestError):
    """Raised when dataset provenance cannot resolve one unique manifest."""


@dataclass(frozen=True)
class ManifestValidationReport:
    manifest_path: Path
    manifest: dict[str, Any]
    reproducibility: str
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved_inputs: tuple[dict[str, Any], ...]

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise RunManifestValidationError("; ".join(self.errors))


@dataclass(frozen=True)
class ManifestCatalogRecord:
    workspace_id: str | None
    workspace_name: str
    workspace_path: Path
    available: bool
    manifest_path: Path | None
    valid: bool
    error: str | None
    manifest_id: str | None = None
    mode: str | None = None
    status: str | None = None
    experiment: str | None = None
    run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class RerunResult:
    manifest_path: Path
    run: Any
    result: Any
    comparison: dict[str, Any] | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    """Convert nested Param/NumPy/Pandas-style values to deterministic JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__tuple__": [json_ready(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": [json_ready(item) for item in sorted(value, key=repr)]}
    if isinstance(value, list):
        items = list(value)
        return [json_ready(item) for item in items]
    if isinstance(value, IterableABC) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        try:
            return [json_ready(item) for item in value]
        except Exception:
            pass
    if hasattr(value, "nestedConf"):
        try:
            return json_ready(value.nestedConf)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_ready(value.tolist())
        except Exception:
            pass
    if callable(value):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", ""))
        return {"__callable__": f"{module}:{qualname}"}
    return {"__repr__": repr(value), "__class__": type(value).__qualname__}


def _restore_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value == {"__float__": "nan"}:
        return float("nan")
    if value == {"__float__": "inf"}:
        return float("inf")
    if value == {"__float__": "-inf"}:
        return float("-inf")
    if set(value) == {"__tuple__"} and isinstance(value["__tuple__"], list):
        return tuple(_restore_json_value(item) for item in value["__tuple__"])
    if set(value) == {"__set__"} and isinstance(value["__set__"], list):
        return {_restore_json_value(item) for item in value["__set__"]}
    return {key: _restore_json_value(item) for key, item in value.items()}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_schema(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RunManifestError(f"Manifest must contain a JSON object: {path}")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RunManifestError(
            f"Unsupported manifest schema {payload.get('schema_version')!r}: {path}"
        )
    missing = [
        key
        for key in (
            "run",
            "invocation",
            "randomness",
            "registry_snapshots",
            "inputs",
            "software",
            "provenance",
            "result",
        )
        if key not in payload
    ]
    if missing:
        raise RunManifestError(
            f"Manifest is missing required section(s) {', '.join(missing)}: {path}"
        )
    run = payload.get("run")
    if not isinstance(run, dict) or not isinstance(run.get("manifest_id"), str):
        raise RunManifestError(f"Manifest has no valid run.manifest_id: {path}")
    return payload


def load_run_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunManifestError(f"Run manifest does not exist: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(
            f"Cannot read run manifest {manifest_path}: {exc}"
        ) from exc
    return _validate_schema(payload, manifest_path)


def _cached_load_manifest(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        stat = path.stat()
    except OSError as exc:
        return None, str(exc)
    cached = _MANIFEST_CACHE.get(path)
    key = (stat.st_mtime_ns, stat.st_size)
    if cached is not None and cached[:2] == key:
        return copy.deepcopy(cached[2]), cached[3]
    try:
        payload = load_run_manifest(path)
        error = None
    except RunManifestError as exc:
        payload = None
        error = str(exc)
    _MANIFEST_CACHE[path] = (key[0], key[1], copy.deepcopy(payload), error)
    return payload, error


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def path_checksum(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(path).expanduser().resolve()
    files = _path_files(root)
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for candidate in files:
        relative = (
            candidate.name if root.is_file() else candidate.relative_to(root).as_posix()
        )
        checksum = sha256_file(candidate)
        size = candidate.stat().st_size
        entries.append({"path": relative, "sha256": checksum, "size": size})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(checksum.encode("ascii"))
        aggregate.update(b"\0")
    return aggregate.hexdigest(), entries


def software_versions() -> dict[str, str]:
    from larvaworld import __version__

    versions = {
        "larvaworld": str(__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in ("agentpy", "numpy", "pandas", "scipy", "param"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


def capture_registry_snapshots() -> dict[str, Any]:
    """Capture complete registry entries used as an in-memory rerun boundary."""
    from larvaworld import CONFTYPES
    from larvaworld.lib import reg

    entries: dict[str, Any] = {}
    for conftype in CONFTYPES:
        try:
            entries[conftype] = json_ready(reg.conf[conftype].dict)
        except Exception as exc:
            entries[conftype] = {"__snapshot_error__": str(exc)}
    return {
        "scope": "complete_registry",
        "entries": entries,
        "sha256": canonical_sha256(entries),
    }


@contextmanager
def registry_snapshot_context(snapshot: Mapping[str, Any]) -> Iterator[None]:
    """Apply snapshots in process memory and restore registry state afterwards."""
    from larvaworld.lib import reg, util
    from larvaworld.lib.reg.config import _REGISTRY_PERSISTENCE_SUSPENDED

    entries = snapshot.get("entries", {})
    originals: dict[str, Any] = {}
    persistence_token = _REGISTRY_PERSISTENCE_SUSPENDED.set(True)
    try:
        if isinstance(entries, Mapping):
            for conftype, value in entries.items():
                if not isinstance(value, Mapping) or "__snapshot_error__" in value:
                    continue
                try:
                    conf_type = reg.conf[conftype]
                except Exception:
                    continue
                # AttrDict mirrors its mapping through ``__dict__``. A bare
                # deepcopy duplicates those two views independently and
                # breaks later dot access after restoration. Reconstructing
                # through AttrDict keeps both views synchronized recursively.
                originals[conftype] = util.AttrDict(copy.deepcopy(dict(conf_type.dict)))
                conf_type.dict = util.AttrDict(
                    _restore_json_value(copy.deepcopy(dict(value)))
                )
        yield
    finally:
        for conftype, original in originals.items():
            reg.conf[conftype].dict = original
        _REGISTRY_PERSISTENCE_SUSPENDED.reset(persistence_token)


def prepare_master_seed(seed: int | None = None) -> int:
    return int(seed) if seed is not None else secrets.randbits(128)


def derive_seed(master_seed: int, label: Any) -> int:
    digest = hashlib.sha256(f"{master_seed}:{label!r}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


@contextmanager
def deterministic_random_context(seed: int) -> Iterator[None]:
    """Seed legacy global RNG users while preserving the caller's RNG state."""
    import numpy as np

    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _workspace_identity(run_dir: Path) -> tuple[str, Path | None]:
    for parent in (run_dir, *run_dir.parents):
        metadata_path = parent / "metadata" / "workspace.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            workspace_id = metadata.get("workspace_id")
            if isinstance(workspace_id, str) and workspace_id:
                return workspace_id, parent
        except (OSError, json.JSONDecodeError):
            continue
    fallback_root = run_dir.parent.resolve()
    fallback_id = str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"larvaworld-standalone:{fallback_root}")
    )
    return fallback_id, None


def _dataset_input(dataset: Any, role: str) -> dict[str, Any] | None:
    config = getattr(dataset, "config", None)
    raw_dir = getattr(config, "dir", None)
    if not isinstance(raw_dir, str) or not raw_dir:
        return None
    path = Path(raw_dir).expanduser().resolve()
    checksum, files = path_checksum(path)
    return {
        "role": role,
        "dataset_id": getattr(config, "id", None),
        "ref_id": getattr(config, "refID", None),
        "path": str(path),
        "sha256": checksum,
        "files": files,
    }


def collect_run_inputs(run: Any) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    for role, attribute in (
        ("replay_source", "refDataset"),
        ("evaluation_target", "target"),
        ("input_dataset", "dataset"),
    ):
        value = getattr(run, attribute, None)
        if value is not None:
            candidates.append((role, value))
    evaluator = getattr(run, "evaluator", None)
    target = getattr(evaluator, "target", None)
    if target is not None:
        candidates.append(("ga_target", target))
    inputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for role, dataset in candidates:
        entry = _dataset_input(dataset, role)
        if entry is None or entry["path"] in seen:
            continue
        seen.add(entry["path"])
        inputs.append(entry)
    return inputs


def _generic_invocation(
    run: Any, execute_kwargs: Mapping[str, Any] | None
) -> dict[str, Any]:
    parameters = getattr(run, "parameters", None)
    if parameters is None:
        parameters = getattr(run, "p", {})
    screen_kws = getattr(run, "screen_kws", {})
    return {
        "run_class": f"{type(run).__module__}:{type(run).__qualname__}",
        "resolved_parameters": json_ready(parameters),
        "constructor": {},
        "runtime_options": {
            "store_data": bool(getattr(run, "store_data", True)),
            "screen_kws": json_ready(screen_kws),
        },
        "execute": {"kwargs": json_ready(execute_kwargs or {})},
    }


def manifest_reference(
    manifest_path: str | Path, dataset_dir: str | Path
) -> dict[str, str]:
    path = Path(manifest_path).expanduser().resolve()
    manifest = load_run_manifest(path)
    dataset_root = Path(dataset_dir).expanduser().resolve()
    return {
        "manifest_id": manifest["run"]["manifest_id"],
        "workspace_id": manifest["run"]["workspace_id"],
        # Stored in JSON and read back on any platform, so the separator must be
        # POSIX rather than the host's.
        "path": Path(os.path.relpath(path, start=dataset_root)).as_posix(),
    }


def attach_manifest_to_datasets(
    datasets: Iterable[Any] | None,
    manifest: "RunManifestSession | str | Path | Mapping[str, Any]",
) -> None:
    if datasets is None:
        return
    for dataset in datasets:
        config = getattr(dataset, "config", None)
        dataset_dir = getattr(config, "dir", None)
        if config is None or not isinstance(dataset_dir, str) or not dataset_dir:
            continue
        if isinstance(manifest, RunManifestSession):
            reference = manifest.dataset_reference(dataset_dir)
        elif isinstance(manifest, Mapping):
            raw_path = manifest.get("manifest_path") or manifest.get("path")
            if not isinstance(raw_path, str):
                continue
            reference = {
                "manifest_id": str(manifest["manifest_id"]),
                "workspace_id": str(manifest["workspace_id"]),
                "path": Path(
                    os.path.relpath(
                        Path(raw_path).expanduser().resolve(),
                        start=Path(dataset_dir).expanduser().resolve(),
                    )
                ).as_posix(),
            }
        else:
            reference = manifest_reference(manifest, dataset_dir)
        existing = getattr(config, "provenance", None)
        lineage = []
        if isinstance(existing, Mapping) and isinstance(existing.get("lineage"), list):
            lineage = copy.deepcopy(existing["lineage"])
        config.provenance = {
            "origin": "simulation",
            "run_manifest": reference,
            "lineage": lineage,
        }


def _dataframe_fingerprint(frame: Any) -> str | None:
    if frame is None:
        return None
    try:
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            return None
        digest = hashlib.sha256()
        digest.update(_canonical_bytes(frame.shape))
        digest.update(_canonical_bytes([str(column) for column in frame.columns]))
        digest.update(_canonical_bytes([str(dtype) for dtype in frame.dtypes]))
        digest.update(_canonical_bytes([str(name) for name in frame.index.names]))
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        return digest.hexdigest()
    except Exception:
        return canonical_sha256(repr(frame))


def _series_fingerprint(series: Any) -> str | None:
    if series is None:
        return None
    try:
        import pandas as pd

        if not isinstance(series, pd.Series):
            return None
        digest = hashlib.sha256()
        digest.update(_canonical_bytes(series.shape))
        digest.update(_canonical_bytes(str(series.name)))
        digest.update(_canonical_bytes(str(series.dtype)))
        digest.update(_canonical_bytes([str(name) for name in series.index.names]))
        digest.update(pd.util.hash_pandas_object(series, index=True).values.tobytes())
        return digest.hexdigest()
    except Exception:
        return canonical_sha256(repr(series))


_VOLATILE_SCIENTIFIC_KEYS = {
    "completed_at",
    "created_at",
    "manifest_id",
    "manifest_path",
    "output_dir",
    "path",
    "provenance",
    "run_dir",
    "run_id",
    "started_at",
    "timestamp",
    "workspace_id",
}


def _scientific_value(value: Any) -> Any:
    frame_hash = _dataframe_fingerprint(value)
    if frame_hash is not None:
        return {"__dataframe_sha256__": frame_hash}
    series_hash = _series_fingerprint(value)
    if series_hash is not None:
        return {"__series_sha256__": series_hash}
    if is_dataclass(value):
        return _scientific_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _scientific_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold() not in _VOLATILE_SCIENTIFIC_KEYS
        }
    if isinstance(value, tuple):
        return {"__tuple__": [_scientific_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        return {
            "__set__": [_scientific_value(item) for item in sorted(value, key=repr)]
        }
    if isinstance(value, list):
        return [_scientific_value(item) for item in value]
    if hasattr(value, "nestedConf"):
        try:
            return _scientific_value(value.nestedConf)
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _scientific_value(value.tolist())
        except Exception:
            pass
    return json_ready(value)


def scientific_fingerprints(
    datasets: Iterable[Any] | None = None, scientific_result: Any = None
) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {}
    if datasets is not None:
        for index, dataset in enumerate(datasets):
            config = getattr(dataset, "config", None)
            dataset_id = str(getattr(config, "id", None) or f"dataset_{index}")
            key = dataset_id
            suffix = 2
            while key in fingerprints:
                key = f"{dataset_id}_{suffix}"
                suffix += 1
            step = _dataframe_fingerprint(getattr(dataset, "step_data", None))
            end = _dataframe_fingerprint(getattr(dataset, "endpoint_data", None))
            fingerprints[key] = {
                "step": step,
                "end": end,
                "combined": canonical_sha256({"step": step, "end": end}),
            }
    if scientific_result is not None:
        frame_hash = _dataframe_fingerprint(scientific_result)
        fingerprints["result"] = frame_hash or canonical_sha256(
            _scientific_value(scientific_result)
        )
    return fingerprints


def _scan_outputs(run_dir: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for path in sorted(
        candidate for candidate in run_dir.rglob("*") if candidate.is_file()
    ):
        if path.name == MANIFEST_FILENAME or path.name.endswith(".tmp"):
            continue
        outputs.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "media": path.suffix.casefold() in _MEDIA_SUFFIXES,
            }
        )
    return outputs


class RunManifestSession:
    """Atomic lifecycle manager attached to one top-level simulation run."""

    def __init__(
        self,
        *,
        run: Any,
        invocation: Mapping[str, Any] | None = None,
        execute_kwargs: Mapping[str, Any] | None = None,
        seed: int | None = None,
        child_seeds: Mapping[str, int] | None = None,
        source_manifest: str | Path | Mapping[str, Any] | None = None,
        inputs: Sequence[Mapping[str, Any]] | None = None,
        media_requested: bool | None = None,
    ) -> None:
        raw_dir = getattr(run, "dir", None)
        if not isinstance(raw_dir, str) or not raw_dir:
            raise RunManifestError("A simulation run requires a storage directory.")
        self.run = run
        self.run_dir = Path(raw_dir).expanduser().resolve()
        self._preexisting_paths = (
            {path.relative_to(self.run_dir) for path in self.run_dir.rglob("*")}
            if self.run_dir.is_dir()
            else set()
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / MANIFEST_FILENAME
        if self.manifest_path.exists():
            raise RunManifestError(
                f"Refusing to overwrite an existing run manifest: {self.manifest_path}"
            )
        self.master_seed = prepare_master_seed(seed)
        workspace_id, workspace_root = _workspace_identity(self.run_dir)
        source: dict[str, Any] | None = None
        if source_manifest is not None:
            if isinstance(source_manifest, Mapping):
                source_payload = dict(source_manifest)
                source_path = source_payload.get("manifest_path")
            else:
                source_path_obj = Path(source_manifest).expanduser().resolve()
                source_payload = load_run_manifest(source_path_obj)
                source_path = str(source_path_obj)
            source = {
                "manifest_id": source_payload["run"]["manifest_id"],
                "path": source_path,
            }
        effective_invocation = dict(
            invocation or _generic_invocation(run, execute_kwargs=execute_kwargs)
        )
        effective_invocation.setdefault(
            "execute", {"kwargs": json_ready(execute_kwargs or {})}
        )
        screen_kws = getattr(run, "screen_kws", {})
        if media_requested is None:
            media_requested = bool(
                isinstance(screen_kws, Mapping)
                and (
                    screen_kws.get("save_video")
                    or screen_kws.get("vis_mode") == "video"
                    or screen_kws.get("image_mode")
                )
            )
        self._media_output_paths: set[Path] = set()
        if media_requested and isinstance(screen_kws, Mapping):
            raw_media_dir = screen_kws.get("media_dir")
            media_dir = (
                Path(str(raw_media_dir)).expanduser().resolve()
                if raw_media_dir
                else self.run_dir
            )
            if screen_kws.get("save_video") or screen_kws.get("vis_mode") == "video":
                video_file = screen_kws.get("video_file") or getattr(
                    run, "id", self.run_dir.name
                )
                self._media_output_paths.add(
                    (media_dir / f"{video_file}.mp4").resolve()
                )
            if screen_kws.get("image_mode"):
                image_file = screen_kws.get("image_file") or getattr(
                    run, "id", self.run_dir.name
                )
                self._media_output_paths.add(
                    (media_dir / f"{image_file}.png").resolve()
                )
        self.manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run": {
                "manifest_id": str(uuid.uuid4()),
                "workspace_id": workspace_id,
                "workspace_root": str(workspace_root) if workspace_root else None,
                "mode": str(getattr(run, "runtype", type(run).__name__)),
                "experiment": getattr(run, "experiment", None),
                "run_id": str(getattr(run, "id", self.run_dir.name)),
                "run_dir": str(self.run_dir),
                "status": "running",
                "created_at": _utc_now_iso(),
                "started_at": _utc_now_iso(),
                "completed_at": None,
            },
            "invocation": json_ready(effective_invocation),
            "randomness": {
                "master_seed": self.master_seed,
                "child_seeds": json_ready(child_seeds or {}),
                "algorithm": "sha256-derived-128bit-v1",
            },
            "registry_snapshots": capture_registry_snapshots(),
            "inputs": json_ready(
                list(inputs) if inputs is not None else collect_run_inputs(run)
            ),
            "software": software_versions(),
            "provenance": {
                "source_manifest": source,
                "reproducibility": "original" if source is None else "rerun",
            },
            "result": {
                "outputs": [],
                "scientific_fingerprints": {},
                "comparison": None,
                "media_requested": bool(media_requested),
            },
        }
        self.write()
        setattr(run, "_manifest_session", self)

    def write(self) -> None:
        _atomic_write_json(self.manifest_path, self.manifest)
        try:
            self.manifest_path.stat()
            _MANIFEST_CACHE.pop(self.manifest_path, None)
        except OSError:
            pass

    @property
    def reference(self) -> dict[str, str]:
        run = self.manifest["run"]
        return {
            "manifest_id": run["manifest_id"],
            "workspace_id": run["workspace_id"],
            "manifest_path": str(self.manifest_path),
        }

    def dataset_reference(self, dataset_dir: str | Path) -> dict[str, str]:
        reference = self.reference
        return {
            "manifest_id": reference["manifest_id"],
            "workspace_id": reference["workspace_id"],
            "path": Path(
                os.path.relpath(
                    self.manifest_path,
                    start=Path(dataset_dir).expanduser().resolve(),
                )
            ).as_posix(),
        }

    def set_child_seeds(self, child_seeds: Mapping[str, int]) -> None:
        self.manifest["randomness"]["child_seeds"] = json_ready(child_seeds)
        self.write()

    def _cleanup_manifest_only_outputs(self) -> None:
        current_paths = sorted(
            (path for path in self.run_dir.rglob("*") if path != self.manifest_path),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in current_paths:
            relative = path.relative_to(self.run_dir)
            if relative in self._preexisting_paths and not path.is_dir():
                continue
            try:
                if path.is_file() or path.is_symlink():
                    if path.resolve() in self._media_output_paths:
                        continue
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            except OSError:
                continue

    def finish(
        self,
        *,
        datasets: Iterable[Any] | None = None,
        scientific_result: Any = None,
        status: str = "completed",
    ) -> Path:
        if not bool(getattr(self.run, "store_data", True)):
            self._cleanup_manifest_only_outputs()
        fingerprints = scientific_fingerprints(datasets, scientific_result)
        self.manifest["result"]["scientific_fingerprints"] = fingerprints
        self.manifest["result"]["outputs"] = _scan_outputs(self.run_dir)
        source = self.manifest["provenance"].get("source_manifest")
        if isinstance(source, Mapping) and isinstance(source.get("path"), str):
            try:
                source_payload = load_run_manifest(source["path"])
                expected = source_payload["result"].get("scientific_fingerprints", {})
                self.manifest["result"]["comparison"] = {
                    "matches": expected == fingerprints,
                    "source_manifest_id": source_payload["run"]["manifest_id"],
                    "expected": expected,
                    "actual": fingerprints,
                }
            except RunManifestError as exc:
                self.manifest["result"]["comparison"] = {
                    "matches": None,
                    "error": str(exc),
                }
        self.manifest["run"]["status"] = status
        self.manifest["run"]["completed_at"] = _utc_now_iso()
        self.write()
        return self.manifest_path

    def fail(self, exc: BaseException) -> Path:
        if not bool(getattr(self.run, "store_data", True)):
            self._cleanup_manifest_only_outputs()
        self.manifest["run"]["status"] = "failed"
        self.manifest["run"]["completed_at"] = _utc_now_iso()
        self.manifest["result"]["error"] = {
            "type": type(exc).__qualname__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        self.manifest["result"]["outputs"] = _scan_outputs(self.run_dir)
        self.write()
        return self.manifest_path

    def abort(self, message: str = "Simulation aborted") -> Path:
        if not bool(getattr(self.run, "store_data", True)):
            self._cleanup_manifest_only_outputs()
        self.manifest["run"]["status"] = "aborted"
        self.manifest["run"]["completed_at"] = _utc_now_iso()
        self.manifest["result"]["error"] = {"type": "Aborted", "message": message}
        self.manifest["result"]["outputs"] = _scan_outputs(self.run_dir)
        self.write()
        return self.manifest_path


def _workspace_specs(workspaces: Any = None) -> list[dict[str, Any]]:
    if workspaces is None:
        try:
            from larvaworld.portal.workspace import get_known_workspaces

            return [dict(record) for record in get_known_workspaces()]
        except Exception:
            return []
    if isinstance(workspaces, (str, Path)) or hasattr(workspaces, "root"):
        workspaces = [workspaces]
    specs: list[dict[str, Any]] = []
    for item in workspaces:
        if isinstance(item, Mapping):
            raw_path = item.get("path") or item.get("root")
            if raw_path is None:
                continue
            spec = dict(item)
            spec["path"] = Path(raw_path).expanduser().resolve()
            specs.append(spec)
        elif hasattr(item, "root"):
            specs.append(
                {
                    "path": Path(item.root).expanduser().resolve(),
                    "workspace_id": getattr(item, "workspace_id", None),
                    "name": getattr(item, "name", Path(item.root).name),
                }
            )
        else:
            path = Path(item).expanduser().resolve()
            specs.append({"path": path, "name": path.name})
    return specs


def discover_run_manifests(
    workspaces: Any = None,
    modes: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
) -> list[ManifestCatalogRecord]:
    mode_filter = {str(mode).casefold() for mode in modes} if modes else None
    status_filter = (
        {str(status).casefold() for status in statuses} if statuses else None
    )
    records: list[ManifestCatalogRecord] = []
    for spec in _workspace_specs(workspaces):
        root = Path(spec["path"]).expanduser().resolve()
        workspace_id = spec.get("workspace_id")
        workspace_name = str(spec.get("name") or root.name)
        metadata_path = root / "metadata" / "workspace.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                workspace_id = metadata.get("workspace_id") or workspace_id
                workspace_name = str(metadata.get("workspace_name") or workspace_name)
            except (OSError, json.JSONDecodeError):
                pass
        simulations_dir = root / "simulations"
        if not simulations_dir.is_dir():
            records.append(
                ManifestCatalogRecord(
                    workspace_id=str(workspace_id) if workspace_id else None,
                    workspace_name=workspace_name,
                    workspace_path=root,
                    available=False,
                    manifest_path=None,
                    valid=False,
                    error="Workspace is unavailable or has no simulations directory.",
                )
            )
            continue
        for path in sorted(simulations_dir.rglob(MANIFEST_FILENAME)):
            payload, error = _cached_load_manifest(path.resolve())
            if payload is None:
                records.append(
                    ManifestCatalogRecord(
                        workspace_id=str(workspace_id) if workspace_id else None,
                        workspace_name=workspace_name,
                        workspace_path=root,
                        available=True,
                        manifest_path=path.resolve(),
                        valid=False,
                        error=error,
                    )
                )
                continue
            run = payload["run"]
            mode = str(run.get("mode", ""))
            status = str(run.get("status", ""))
            if mode_filter and mode.casefold() not in mode_filter:
                continue
            if status_filter and status.casefold() not in status_filter:
                continue
            records.append(
                ManifestCatalogRecord(
                    workspace_id=str(workspace_id or run.get("workspace_id") or "")
                    or None,
                    workspace_name=workspace_name,
                    workspace_path=root,
                    available=True,
                    manifest_path=path.resolve(),
                    valid=True,
                    error=None,
                    manifest_id=run.get("manifest_id"),
                    mode=mode,
                    status=status,
                    experiment=run.get("experiment"),
                    run_id=run.get("run_id"),
                    started_at=run.get("started_at"),
                    completed_at=run.get("completed_at"),
                    manifest=payload,
                )
            )
    return sorted(
        records,
        key=lambda record: (
            record.started_at or "",
            str(record.manifest_path or record.workspace_path),
        ),
        reverse=True,
    )


def resolve_manifest_id(manifest_id: str, *, workspaces: Any = None) -> Path:
    matches = [
        record.manifest_path
        for record in discover_run_manifests(workspaces=workspaces)
        if record.valid and record.manifest_id == manifest_id and record.manifest_path
    ]
    unique = sorted(set(matches))
    if not unique:
        raise RunManifestResolutionError(
            f"No run manifest with id {manifest_id!r} exists in known workspaces."
        )
    if len(unique) > 1:
        joined = ", ".join(str(path) for path in unique)
        raise RunManifestResolutionError(
            f"Manifest id {manifest_id!r} is ambiguous; matches: {joined}"
        )
    return unique[0]


def _override_for_input(
    entry: Mapping[str, Any], input_overrides: Mapping[str, Any] | None
) -> Path:
    original = str(entry.get("path", ""))
    replacement: Any = None
    if input_overrides:
        for key in (
            original,
            entry.get("dataset_id"),
            entry.get("ref_id"),
            entry.get("role"),
        ):
            if key is not None and key in input_overrides:
                replacement = input_overrides[key]
                break
    return Path(replacement or original).expanduser().resolve()


def _child_seeds_match_derivation(
    mode: str, master_seed: int, child_seeds: Mapping[str, int]
) -> bool:
    try:
        if mode == "Batch":
            return all(
                seed == derive_seed(master_seed, ast.literal_eval(label))
                for label, seed in child_seeds.items()
            )
        if mode == "Ga":
            return all(
                label.startswith("generation_")
                and seed
                == derive_seed(
                    master_seed,
                    ("generation", int(label.removeprefix("generation_"))),
                )
                for label, seed in child_seeds.items()
            )
        if mode == "Eval":
            return child_seeds == {
                "evaluation_exp": derive_seed(master_seed, "evaluation_exp")
            }
    except (SyntaxError, ValueError):
        return False
    return True


def validate_run_manifest(
    path: str | Path,
    reproducibility: str = "strict",
    allow_version_mismatch: bool = False,
    input_overrides: Mapping[str, Any] | None = None,
) -> ManifestValidationReport:
    if reproducibility not in {"strict", "parameters"}:
        raise ValueError("reproducibility must be 'strict' or 'parameters'")
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / MANIFEST_FILENAME
    manifest = load_run_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    run = manifest["run"]
    if run.get("status") not in {"completed", "aborted", "failed"}:
        errors.append(f"Source manifest status is {run.get('status')!r}, not terminal.")
    current_versions = software_versions()
    version_mismatches = {
        name: {"manifest": version, "current": current_versions.get(name)}
        for name, version in manifest["software"].items()
        if name != "platform" and current_versions.get(name) != version
    }
    if version_mismatches:
        message = f"Software version mismatch: {version_mismatches}"
        if reproducibility == "strict" and not allow_version_mismatch:
            errors.append(message)
        else:
            warnings.append(message)
    randomness = manifest.get("randomness", {})
    if not isinstance(randomness.get("master_seed"), int):
        message = "Manifest does not contain an integer master seed."
        (errors if reproducibility == "strict" else warnings).append(message)
    if randomness.get("algorithm") != "sha256-derived-128bit-v1":
        message = "Manifest uses an unknown deterministic seed algorithm."
        (errors if reproducibility == "strict" else warnings).append(message)
    child_seeds = randomness.get("child_seeds")
    child_seed_error = not isinstance(child_seeds, Mapping) or any(
        not isinstance(child_seed, int) for child_seed in (child_seeds or {}).values()
    )
    if run.get("mode") in {"Batch", "Ga", "Eval"} and not child_seeds:
        child_seed_error = True
    if child_seed_error:
        message = "Manifest child seeds are missing or contain non-integer values."
        (errors if reproducibility == "strict" else warnings).append(message)
    elif (
        isinstance(randomness.get("master_seed"), int)
        and isinstance(child_seeds, Mapping)
        and not _child_seeds_match_derivation(
            str(run.get("mode")), randomness["master_seed"], child_seeds
        )
    ):
        message = "Manifest child seeds do not match their deterministic derivation."
        (errors if reproducibility == "strict" else warnings).append(message)
    snapshot = manifest.get("registry_snapshots", {})
    entries = snapshot.get("entries") if isinstance(snapshot, Mapping) else None
    expected_snapshot_hash = (
        snapshot.get("sha256") if isinstance(snapshot, Mapping) else None
    )
    if (
        not isinstance(entries, Mapping)
        or canonical_sha256(entries) != expected_snapshot_hash
    ):
        message = "Registry snapshot is missing or its integrity hash does not match."
        (errors if reproducibility == "strict" else warnings).append(message)
    resolved_inputs: list[dict[str, Any]] = []
    for entry in manifest.get("inputs", []):
        if not isinstance(entry, Mapping):
            continue
        resolved = _override_for_input(entry, input_overrides)
        resolved_entry = dict(entry)
        resolved_entry["resolved_path"] = str(resolved)
        resolved_inputs.append(resolved_entry)
        if not resolved.exists():
            message = f"Input dataset is missing: {resolved}"
            (errors if reproducibility == "strict" else warnings).append(message)
            continue
        checksum, _ = path_checksum(resolved)
        if checksum != entry.get("sha256"):
            message = f"Input checksum mismatch: {resolved}"
            (errors if reproducibility == "strict" else warnings).append(message)
    return ManifestValidationReport(
        manifest_path=manifest_path,
        manifest=manifest,
        reproducibility=reproducibility,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        resolved_inputs=tuple(resolved_inputs),
    )


def _next_rerun_dir(source_dir: Path) -> Path:
    base = source_dir.with_name(f"{source_dir.name}_rerun")
    if not base.exists():
        return base
    index = 2
    while base.with_name(f"{base.name}_{index}").exists():
        index += 1
    return base.with_name(f"{base.name}_{index}")


def _screen_options(
    invocation: Mapping[str, Any], with_media: bool, destination: Path
) -> dict[str, Any]:
    runtime = invocation.get("runtime_options", {})
    screen = (
        copy.deepcopy(runtime.get("screen_kws", {}))
        if isinstance(runtime, Mapping)
        else {}
    )
    if not isinstance(screen, dict):
        screen = {}
    screen["show_display"] = False
    if with_media:
        screen["media_dir"] = str(destination)
        if screen.get("vis_mode") == "video":
            screen["save_video"] = True
        for key, suffix in (("video_file", ".mp4"), ("image_file", ".png")):
            raw_name = screen.get(key)
            if isinstance(raw_name, str) and raw_name:
                name = Path(raw_name).name
                if name.casefold().endswith(suffix):
                    name = name[: -len(suffix)]
                screen[key] = name
    else:
        screen["save_video"] = False
        screen["vis_mode"] = None
        screen["image_mode"] = None
        for key in ("video_file", "image_file", "media_dir"):
            screen.pop(key, None)
    return _restore_json_value(screen)


def _input_dataset_for_role(
    report: ManifestValidationReport, roles: set[str]
) -> Any | None:
    from larvaworld.lib.process import LarvaDataset

    for entry in report.resolved_inputs:
        if entry.get("role") in roles:
            return LarvaDataset(dir=entry["resolved_path"], load_data=True)
    return None


def rerun_from_manifest(
    path: str | Path,
    reproducibility: str = "strict",
    output_dir: str | Path | None = None,
    allow_version_mismatch: bool = False,
    input_overrides: Mapping[str, Any] | None = None,
    with_media: bool = False,
) -> RerunResult:
    report = validate_run_manifest(
        path,
        reproducibility=reproducibility,
        allow_version_mismatch=allow_version_mismatch,
        input_overrides=input_overrides,
    )
    report.raise_for_errors()
    source = report.manifest
    source_dir = report.manifest_path.parent
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _next_rerun_dir(source_dir)
    )
    if destination.exists():
        raise RunManifestError(f"Rerun output already exists: {destination}")
    invocation = _restore_json_value(copy.deepcopy(source["invocation"]))
    mode = str(source["run"]["mode"])
    experiment = source["run"].get("experiment")
    master_seed = int(source["randomness"]["master_seed"])
    constructor = invocation.get("constructor", {})
    parameters = invocation.get("resolved_parameters", {})
    runtime = invocation.get("runtime_options", {})
    execute = invocation.get("execute", {})
    if not isinstance(execute, Mapping):
        execute = {}
    execute_method = str(
        execute.get("method") or ("run" if mode == "Replay" else "simulate")
    )
    execute_kwargs = copy.deepcopy(execute.get("kwargs", {}))
    if not isinstance(execute_kwargs, dict):
        execute_kwargs = {}
    execute_kwargs.pop("seed", None)
    store_data = bool(runtime.get("store_data", True))
    screen_kws = _screen_options(
        invocation, with_media=with_media, destination=destination
    )
    common = {
        "id": destination.name,
        "dir": str(destination),
        "store_data": store_data,
        "_source_manifest": str(report.manifest_path),
    }
    from larvaworld.lib import sim, util

    if isinstance(parameters, dict):
        parameters = util.AttrDict(parameters)

    with registry_snapshot_context(source["registry_snapshots"]):
        if mode == "Exp":
            run = sim.ExpRun(
                experiment=experiment,
                parameters=parameters,
                screen_kws=screen_kws,
                parameter_dict=constructor.get("parameter_dict", {}),
                **common,
            )
            if execute_method == "run":
                result = run.run(seed=master_seed, **execute_kwargs)
            else:
                result = run.simulate(seed=master_seed, **execute_kwargs)
        elif mode == "Replay":
            dataset = _input_dataset_for_role(
                report, {"replay_source", "input_dataset"}
            )
            replay_parameters = constructor.get("replay_parameters", parameters)
            if isinstance(replay_parameters, dict):
                replay_parameters = util.AttrDict(replay_parameters)
            if dataset is not None:
                replay_parameters = copy.deepcopy(replay_parameters)
                replay_parameters["refDir"] = dataset.config.dir
                replay_parameters["refID"] = dataset.config.refID
            run = sim.ReplayRun(
                parameters=replay_parameters,
                dataset=dataset,
                screen_kws=screen_kws,
                **common,
            )
            result = run.run(seed=master_seed, **execute_kwargs)
        elif mode == "Batch":
            batch_kwargs = copy.deepcopy(constructor.get("model_kwargs", {}))
            if not isinstance(batch_kwargs, dict):
                batch_kwargs = {}
            batch_kwargs.update(common)
            batch_kwargs["screen_kws"] = screen_kws
            run = sim.BatchRun(
                experiment=experiment,
                space_search=constructor["space_search"],
                space_kws=constructor.get("space_kws", {}),
                exp=util.AttrDict(constructor.get("exp", parameters)),
                exp_kws=constructor.get("exp_kws", {}),
                iterations=int(constructor.get("iterations", 1)),
                **batch_kwargs,
            )
            if execute_method == "run":
                result = run.run(seed=master_seed, **execute_kwargs)
            else:
                result = run.simulate(seed=master_seed, **execute_kwargs)
        elif mode == "Ga":
            dataset = _input_dataset_for_role(report, {"ga_target", "input_dataset"})
            run = sim.GAlauncher(
                experiment=experiment,
                parameters=parameters,
                dataset=dataset,
                screen_kws=screen_kws,
                **common,
            )
            result = run.simulate(seed=master_seed)
        elif mode == "Eval":
            dataset = _input_dataset_for_role(
                report, {"evaluation_target", "input_dataset"}
            )
            eval_kwargs = copy.deepcopy(constructor.get("eval_kwargs", {}))
            eval_kwargs.update(common)
            eval_kwargs["screen_kws"] = screen_kws
            if dataset is not None:
                eval_kwargs["dataset"] = dataset
                eval_kwargs["refDir"] = dataset.config.dir
                eval_kwargs["refID"] = dataset.config.refID
            run = sim.EvalRun(**eval_kwargs)
            result = run.simulate(seed=master_seed)
        else:
            raise RunManifestError(f"Unsupported simulation mode in manifest: {mode}")
    new_manifest_path = destination / MANIFEST_FILENAME
    new_manifest = load_run_manifest(new_manifest_path)
    return RerunResult(
        manifest_path=new_manifest_path,
        run=run,
        result=result,
        comparison=new_manifest["result"].get("comparison"),
    )


def resolve_dataset_manifest_path(dataset: Any, *, workspaces: Any = None) -> Path:
    config = getattr(dataset, "config", None)
    provenance = getattr(config, "provenance", None)
    if not isinstance(provenance, Mapping):
        raise RunManifestResolutionError("Dataset has no provenance metadata.")
    reference = provenance.get("run_manifest")
    if not isinstance(reference, Mapping):
        raise RunManifestResolutionError("Dataset has no source run manifest.")
    expected_id = reference.get("manifest_id")
    relative = reference.get("path")
    dataset_dir = getattr(config, "dir", None)
    candidate: Path | None = None
    if isinstance(dataset_dir, str) and dataset_dir and isinstance(relative, str):
        candidate = (Path(dataset_dir).expanduser().resolve() / relative).resolve()
        if candidate.is_file():
            manifest = load_run_manifest(candidate)
            actual_id = manifest["run"]["manifest_id"]
            if actual_id != expected_id:
                raise RunManifestResolutionError(
                    f"Manifest id mismatch at {candidate}: expected {expected_id}, got {actual_id}."
                )
            return candidate
    if not isinstance(expected_id, str) or not expected_id:
        raise RunManifestResolutionError(
            f"Dataset manifest path is unavailable and no manifest_id is recorded: {candidate}"
        )
    resolved = resolve_manifest_id(expected_id, workspaces=workspaces)
    manifest = load_run_manifest(resolved)
    if manifest["run"]["manifest_id"] != expected_id:
        raise RunManifestResolutionError(
            f"Resolved manifest id mismatch: expected {expected_id}, got {manifest['run']['manifest_id']}."
        )
    return resolved


def append_dataset_lineage(
    parent: Any,
    derived: Any,
    *,
    operation: str,
    parameters: Mapping[str, Any] | None = None,
) -> None:
    parent_config = getattr(parent, "config", None)
    derived_config = getattr(derived, "config", None)
    if parent_config is None or derived_config is None:
        return
    provenance = copy.deepcopy(getattr(parent_config, "provenance", None))
    if not isinstance(provenance, dict):
        provenance = {"origin": "derived", "run_manifest": None, "lineage": []}
    lineage = provenance.get("lineage")
    if not isinstance(lineage, list):
        lineage = []
    lineage.append(
        {
            "operation": operation,
            "parent_dataset_id": getattr(parent_config, "id", None),
            "created_at": _utc_now_iso(),
            "parameters": json_ready(parameters or {}),
        }
    )
    provenance["lineage"] = lineage
    provenance["origin"] = "derived"
    reference = provenance.get("run_manifest")
    if isinstance(reference, dict):
        try:
            manifest_path = resolve_dataset_manifest_path(parent)
            raw_dir = getattr(derived_config, "dir", None)
            if isinstance(raw_dir, str) and raw_dir:
                reference["path"] = Path(
                    os.path.relpath(
                        manifest_path, start=Path(raw_dir).expanduser().resolve()
                    )
                ).as_posix()
        except RunManifestError:
            pass
    derived_config.provenance = provenance


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestCatalogRecord",
    "ManifestValidationReport",
    "RerunResult",
    "RunManifestError",
    "RunManifestResolutionError",
    "RunManifestSession",
    "RunManifestValidationError",
    "append_dataset_lineage",
    "attach_manifest_to_datasets",
    "canonical_sha256",
    "collect_run_inputs",
    "derive_seed",
    "deterministic_random_context",
    "discover_run_manifests",
    "json_ready",
    "load_run_manifest",
    "manifest_reference",
    "path_checksum",
    "prepare_master_seed",
    "registry_snapshot_context",
    "rerun_from_manifest",
    "resolve_dataset_manifest_path",
    "resolve_manifest_id",
    "scientific_fingerprints",
    "sha256_file",
    "software_versions",
    "validate_run_manifest",
]
