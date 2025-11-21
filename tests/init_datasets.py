"""
Utilities for preparing real-data fixtures used in integration tests.

Run via:
    python larvaworld/tests/init_datasets.py

The script performs three steps:
1. Initializes the registry default reference dataset (`exploration.30controls`).
2. Imports key SchleyerGroup datasets from raw CSV files and persists processed artefacts.
3. Verifies the expected processed files exist, emitting informative messages.

The operations are idempotent; existing processed folders are overwritten safely by
`LabFormat.import_dataset(save_dataset=True)`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PROCESSED_ROOT = SRC_ROOT / "larvaworld" / "data" / "SchleyerGroup" / "processed"

# Ensure larvaworld sources import correctly when package not installed.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if "larvaworld" in sys.modules:
    pkg = sys.modules["larvaworld"]
    pkg_path = str(SRC_ROOT / "larvaworld")
    if hasattr(pkg, "__path__") and pkg_path not in pkg.__path__:
        pkg.__path__.append(pkg_path)


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative configuration for a Schleyer dataset import."""

    dataset_id: str
    parent_dir: str
    ref_id: str
    color: str
    min_duration_in_sec: int
    merged: bool = False

    @property
    def target_dir(self) -> Path:
        return PROCESSED_ROOT / "exploration" / self.dataset_id / "data"

    def exists(self) -> bool:
        data_file = self.target_dir / "data.h5"
        conf_file = self.target_dir / "conf.txt"
        return data_file.exists() and conf_file.exists()


SCHLEYER_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="dish01",
        parent_dir="exploration/dish01",
        ref_id="exploration.dish01",
        color="blue",
        min_duration_in_sec=60,
    ),
    DatasetSpec(
        dataset_id="dish02",
        parent_dir="exploration/dish02",
        ref_id="exploration.dish02",
        color="red",
        min_duration_in_sec=90,
    ),
)


def ensure_default_dataset() -> None:
    """Ensure the default exploration.30controls dataset exists."""
    try:
        from larvaworld.lib.reg import define_default_refID
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import larvaworld dependencies. "
            "Run this script within the project environment (e.g. `poetry run` or the repo venv)."
        ) from exc

    define_default_refID("exploration.30controls")

    target = PROCESSED_ROOT / "exploration" / "30controls" / "data"
    if not (target / "data.h5").exists():
        raise RuntimeError(
            "Default dataset exploration.30controls was not created as expected."
        )


def import_schleyer_datasets(specs: Iterable[DatasetSpec]) -> None:
    """Import Schleyer datasets from raw CSV into processed artefacts."""
    # Import after sys.path adjustment.
    from larvaworld.lib.reg import conf

    lab = conf.LabFormat.get("Schleyer")
    base_kwargs = {"group_id": "exploration"}

    for spec in specs:
        if spec.exists():
            continue

        kwargs = {
            "parent_dir": spec.parent_dir,
            "merged": spec.merged,
            "color": spec.color,
            "min_duration_in_sec": spec.min_duration_in_sec,
            "id": spec.dataset_id,
            "refID": spec.ref_id,
            **base_kwargs,
        }

        dataset = lab.import_dataset(save_dataset=True, **kwargs)
        if dataset is None:
            raise RuntimeError(f"Failed to import dataset {spec.dataset_id}")

        if not spec.exists():
            raise RuntimeError(
                f"Processed artefacts missing after import for {spec.dataset_id}: {spec.target_dir}"
            )


def main() -> None:
    ensure_default_dataset()
    import_schleyer_datasets(SCHLEYER_DATASETS)


if __name__ == "__main__":
    main()
