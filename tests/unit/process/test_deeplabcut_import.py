from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib.process.import_aux import (
    DLCImportError,
    DLCScaleValidationError,
    read_deeplabcut_tracks,
)
from larvaworld.lib.process import discover_deeplabcut_source_directories


def _dlc_dataframe(bodyparts: list[str], span: float = 1.0) -> pd.DataFrame:
    columns = []
    values = {}
    for index, bodypart in enumerate(bodyparts):
        for coord in ("x", "y", "likelihood"):
            column = ("scorer", bodypart, coord)
            columns.append(column)
            if coord == "x":
                values[column] = np.full(4, index * span)
            elif coord == "y":
                values[column] = np.zeros(4)
            else:
                values[column] = np.full(4, 0.2)
    return pd.DataFrame(
        values,
        columns=pd.MultiIndex.from_tuples(
            columns, names=["scorer", "bodyparts", "coords"]
        ),
    )


def _write_csv(path: Path, bodyparts: list[str], span: float = 1.0) -> None:
    _dlc_dataframe(bodyparts, span).to_csv(path)


def test_read_deeplabcut_tracks_converts_ordered_points_and_ignores_likelihood(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "trialDLC_result.csv", ["a", "b", "c"])

    tracks, npoints = read_deeplabcut_tracks(tmp_path)

    assert npoints == 3
    assert list(tracks[0]) == [
        "head_x",
        "head_y",
        "point2_x",
        "point2_y",
        "tail_x",
        "tail_y",
    ]
    assert tracks[0].iloc[0].tolist() == [0.0, 0.0, 1.0, 0.0, 2.0, 0.0]


def test_read_deeplabcut_tracks_accepts_he_thoracic_and_abdominal_headers(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "trialDLC_result.csv", ["he", "T1", "T2", "A1", "A2"])

    tracks, npoints = read_deeplabcut_tracks(tmp_path)

    assert npoints == 5
    assert tracks[0].iloc[0][["head_x", "tail_x"]].tolist() == [0.0, 4.0]


def test_read_deeplabcut_tracks_averages_lateral_pairs(tmp_path: Path) -> None:
    dataframe = _dlc_dataframe(["headL", "headR", "tail_L", "tail_R"])
    dataframe[("scorer", "headR", "x")] = 2.0
    dataframe[("scorer", "tail_L", "x")] = 8.0
    dataframe[("scorer", "tail_R", "x")] = 10.0
    dataframe.to_csv(tmp_path / "trialDLC_result.csv")

    tracks, npoints = read_deeplabcut_tracks(tmp_path)

    assert npoints == 2
    assert tracks[0].iloc[0][["head_x", "tail_x"]].tolist() == [1.0, 9.0]


def test_read_deeplabcut_tracks_prefers_hdf5_over_csv(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    _write_csv(tmp_path / "trialDLC_result.csv", ["a", "b", "c"], span=1.0)
    _dlc_dataframe(["a", "b", "c"], span=2.0).to_hdf(
        tmp_path / "trialDLC_result.h5", key="df"
    )

    tracks, _ = read_deeplabcut_tracks(tmp_path)

    assert tracks[0].iloc[0]["tail_x"] == 4.0


def test_read_deeplabcut_tracks_requires_valid_pixel_scale(tmp_path: Path) -> None:
    _write_csv(tmp_path / "trialDLC_result.csv", ["a", "b"], span=35.0)

    with pytest.raises(DLCScaleValidationError, match="pixel_to_mm"):
        read_deeplabcut_tracks(tmp_path)

    tracks, npoints = read_deeplabcut_tracks(tmp_path, pixel_to_mm=0.02)

    assert npoints == 2
    assert tracks[0].iloc[0]["tail_x"] == pytest.approx(0.7)

    with pytest.raises(DLCScaleValidationError, match="pixel_to_mm"):
        read_deeplabcut_tracks(tmp_path, pixel_to_mm=0.0)


def test_read_deeplabcut_tracks_rejects_incompatible_point_schemas(
    tmp_path: Path,
) -> None:
    _write_csv(tmp_path / "oneDLC_result.csv", ["head", "tail"])
    _write_csv(tmp_path / "twoDLC_result.csv", ["head", "middle", "tail"])

    with pytest.raises(DLCImportError, match="incompatible point schemas"):
        read_deeplabcut_tracks(tmp_path)


def test_read_deeplabcut_tracks_merges_child_source_folders(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_csv(first / "firstDLC_result.csv", ["a", "b", "c"])
    _write_csv(second / "secondDLC_result.csv", ["a", "b", "c"])

    tracks, npoints = read_deeplabcut_tracks([str(first), str(second)], merged=True)

    assert npoints == 3
    assert len(tracks) == 2


def test_read_deeplabcut_tracks_reads_validated_zip_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_csv(source / "trialDLC_result.csv", ["a", "b", "c"])
    archive = tmp_path / "tracks.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.write(source / "trialDLC_result.csv", "recording/trialDLC_result.csv")

    tracks, npoints = read_deeplabcut_tracks(archive, parent_dir="recording")

    assert npoints == 3
    assert len(tracks) == 1
    assert discover_deeplabcut_source_directories(archive) == ["recording"]


def test_read_deeplabcut_tracks_rejects_unsafe_zip_paths(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../trialDLC_result.csv", "not a valid DLC file")

    with pytest.raises(DLCImportError, match="Unsafe ZIP member"):
        discover_deeplabcut_source_directories(archive)
