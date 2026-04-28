from __future__ import annotations

from pathlib import Path

from larvaworld.portal.datasets.discovery import discover_raw_datasets


def test_discover_raw_datasets_returns_empty_for_missing_root(tmp_path: Path) -> None:
    candidates = discover_raw_datasets("Schleyer", tmp_path / "missing")

    assert candidates == []


def test_discover_raw_datasets_detects_recursive_schleyer_folder_candidates(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    box01 = raw_root / "exploration" / "box01"
    box02 = raw_root / "exploration" / "box02"
    box01.mkdir(parents=True)
    box02.mkdir(parents=True)
    (box01 / "larva_001.csv").write_text("time,x,y\n", encoding="utf-8")

    candidates = discover_raw_datasets("Schleyer", raw_root)

    assert [candidate.candidate_id for candidate in candidates] == ["box01", "box02"]
    assert [candidate.parent_dir for candidate in candidates] == [
        "exploration/box01",
        "exploration/box02",
    ]
    assert candidates[0].warnings == []
    assert candidates[1].warnings == [
        "No matching raw files detected in the candidate directory."
    ]


def test_discover_raw_datasets_groups_jovanic_candidates_by_source_id(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    day1 = raw_root / "day1"
    day1.mkdir(parents=True)
    (day1 / "dishA_tracks_larvaid.txt").write_text("x\n", encoding="utf-8")
    (day1 / "dishA_states_larvaid.txt").write_text("x\n", encoding="utf-8")
    (day1 / "dishB_tracks_larvaid.txt").write_text("x\n", encoding="utf-8")
    (day1 / "notes.txt").write_text("ignore\n", encoding="utf-8")

    candidates = discover_raw_datasets("Jovanic", raw_root)

    assert [candidate.candidate_id for candidate in candidates] == ["dishA", "dishB"]
    assert [candidate.parent_dir for candidate in candidates] == ["day1", "day1"]
    assert [candidate.display_name for candidate in candidates] == [
        "day1 / dishA",
        "day1 / dishB",
    ]
    assert all(candidate.source_path == day1.resolve() for candidate in candidates)
    assert all(candidate.warnings == [] for candidate in candidates)
