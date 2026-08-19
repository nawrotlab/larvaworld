import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from larvaworld.lib.process import import_aux


@pytest.mark.fast
def test_read_timeseries_from_raw_files_per_parameter(tmp_path):
    """
    Verify that the legacy Jovanic-format reader assembles columns, applies defaults,
    and computes Step values when tracker metadata is provided.
    """
    prefix = tmp_path / "jovanic" / "ProteinDeprivation" / "Fed" / "sample"
    prefix.parent.mkdir(parents=True)

    # Two larvae, two timesteps each.
    larva_ids = pd.DataFrame([1, 1, 2, 2])
    times = pd.DataFrame([0.0, 0.5, 0.0, 0.5])
    # Npoints = 2 -> head/tail coordinates.
    x_spine = pd.DataFrame(
        [
            [0.0, 1.0],
            [0.1, 1.1],
            [0.2, 1.2],
            [0.3, 1.3],
        ]
    )
    y_spine = pd.DataFrame(
        [
            [0.0, -1.0],
            [0.1, -0.9],
            [0.2, -0.8],
            [0.3, -0.7],
        ]
    )
    states = pd.DataFrame(["rest", "crawl", "rest", "crawl"])

    data_map = {
        "_larvaid.txt": larva_ids,
        "_t.txt": times,
        "_x_spine.txt": x_spine,
        "_y_spine.txt": y_spine,
        "_state.txt": states,
    }

    for suffix, df in data_map.items():
        df.to_csv(prefix.as_posix() + suffix, header=False, index=False, sep="\t")

    tracker = SimpleNamespace(Npoints=2, Ncontour=0, dt=0.5)

    result = import_aux.read_timeseries_from_raw_files_per_parameter(
        prefix.as_posix(), tracker=tracker
    )

    # Columns include AgentID-derived coordinates plus the optional state and Step.
    expected_columns = {
        "t",
        "head_x",
        "tail_x",
        "head_y",
        "tail_y",
        "state",
        "Step",
    }
    assert expected_columns.issubset(result.columns)

    # Verify AgentID index and the derived Step column.
    ordered = result.reset_index()
    assert ordered["AgentID"].tolist() == [1, 1, 2, 2]
    np.testing.assert_allclose(ordered["Step"].values, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(ordered[["head_x", "tail_x"]].iloc[1].values, [0.1, 1.1])


def _write_spine_file(path, tag, rows):
    """
    Write a raw tracker spine file : [tag, track_id, t, x1, y1, ..., xN, yN] per row.
    """
    lines = [
        " ".join([tag] + [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.fast
def test_convert_spine_files_roundtrips_through_the_per_parameter_reader(tmp_path):
    """
    Verify that the spine converter offsets overlapping track IDs across files and
    de-interleaves the midline coordinates into separate x and y files, so that the
    per-parameter reader recovers the original coordinates.
    """
    Npoints = 2
    # Both recordings reuse the track IDs 1 and 2.
    spine_a = tmp_path / "recA.spine"
    spine_b = tmp_path / "recB.spine"
    _write_spine_file(
        spine_a,
        "recA",
        [
            [1, 0.0, 0.0, 10.0, 1.0, 11.0],
            [1, 0.5, 0.1, 10.1, 1.1, 11.1],
            [2, 0.0, 0.2, 10.2, 1.2, 11.2],
        ],
    )
    _write_spine_file(
        spine_b,
        "recB",
        [
            [1, 0.0, 0.3, 10.3, 1.3, 11.3],
            [2, 0.5, 0.4, 10.4, 1.4, 11.4],
        ],
    )

    target = tmp_path / "converted"
    res = import_aux.convert_spine_files_to_per_parameter_txt(
        source_files=[spine_a.as_posix(), spine_b.as_posix()],
        target_dir=target.as_posix(),
        source_id="Fed",
        Npoints=Npoints,
        id_offset=100,
    )

    assert res == {"files": 2, "rows": 5, "tracks": 4}
    for suf in import_aux.PER_PARAMETER_TXT_SUFFIXES:
        assert (target / f"Fed_{suf}.txt").is_file()
    # The state file is not part of the raw tracker output.
    assert not (target / "Fed_state.txt").exists()

    tracker = SimpleNamespace(Npoints=Npoints, Ncontour=0, dt=0.5)
    df = import_aux.read_timeseries_from_raw_files_per_parameter(
        (target / "Fed").as_posix(), tracker=tracker
    )

    # Track IDs of the two recordings stayed distinct after concatenation, and carry the
    # string prefix every other larvaworld agent ID uses.
    assert sorted(df.index.unique().tolist()) == [
        "Larva_101",
        "Larva_102",
        "Larva_201",
        "Larva_202",
    ]
    # A distinct id_base keeps a second dataset's IDs apart from this one's.
    import_aux.convert_spine_files_to_per_parameter_txt(
        source_files=[spine_a.as_posix()],
        target_dir=target.as_posix(),
        source_id="Starved",
        Npoints=Npoints,
        id_offset=100,
        id_base=1000,
    )
    other = import_aux.read_timeseries_from_raw_files_per_parameter(
        (target / "Starved").as_posix(), tracker=tracker
    )
    assert not set(df.index).intersection(other.index)
    # The interleaved xy pairs were split into an x block and a y block.
    np.testing.assert_allclose(
        df[["head_x", "tail_x", "head_y", "tail_y"]].values,
        [
            [0.0, 1.0, 10.0, 11.0],
            [0.1, 1.1, 10.1, 11.1],
            [0.2, 1.2, 10.2, 11.2],
            [0.3, 1.3, 10.3, 11.3],
            [0.4, 1.4, 10.4, 11.4],
        ],
    )


@pytest.mark.fast
def test_convert_spine_files_is_idempotent_unless_overwriting(tmp_path):
    """
    Verify that an existing set of per-parameter files is left untouched by default
    and rewritten when overwrite is requested.
    """
    spine = tmp_path / "rec.spine"
    _write_spine_file(spine, "rec", [[1, 0.0, 0.0, 10.0, 1.0, 11.0]])
    target = tmp_path / "converted"
    kws = {
        "source_files": [spine.as_posix()],
        "target_dir": target.as_posix(),
        "source_id": "Fed",
        "Npoints": 2,
    }

    import_aux.convert_spine_files_to_per_parameter_txt(**kws)
    marker = target / "Fed_t.txt"
    marker.write_text("sentinel\n")

    res = import_aux.convert_spine_files_to_per_parameter_txt(**kws)
    assert res == {"files": 1, "rows": 1, "tracks": 1}
    assert marker.read_text() == "sentinel\n"

    import_aux.convert_spine_files_to_per_parameter_txt(**kws, overwrite=True)
    assert marker.read_text().strip() == "0.0"

    # An empty prefix keeps the IDs numeric, as the raw files hold them.
    import_aux.convert_spine_files_to_per_parameter_txt(
        **kws, id_prefix="", overwrite=True
    )
    assert (target / "Fed_larvaid.txt").read_text().strip() == "100001"


@pytest.mark.fast
def test_convert_spine_files_rejects_invalid_input(tmp_path):
    """
    Verify that a column count inconsistent with Npoints, an ID range exceeding the
    offset, and an empty file list are all reported explicitly.
    """
    spine = tmp_path / "rec.spine"
    _write_spine_file(spine, "rec", [[1, 0.0, 0.0, 10.0, 1.0, 11.0]])
    kws = {
        "source_files": [spine.as_posix()],
        "target_dir": (tmp_path / "converted").as_posix(),
        "source_id": "Fed",
    }

    # The file holds 2 midline points, not the default 11.
    with pytest.raises(ValueError, match="columns"):
        import_aux.convert_spine_files_to_per_parameter_txt(**kws)

    with pytest.raises(ValueError, match="id_offset"):
        import_aux.convert_spine_files_to_per_parameter_txt(
            **kws, Npoints=2, id_offset=1
        )

    with pytest.raises(ValueError, match="No source files"):
        import_aux.convert_spine_files_to_per_parameter_txt(
            source_files=[],
            target_dir=(tmp_path / "converted").as_posix(),
            source_id="Fed",
        )
