import pandas as pd
import pytest
from types import SimpleNamespace

from larvaworld.lib import reg
import larvaworld.lib.process.importing as importing


@pytest.fixture
def tracker():
    return SimpleNamespace(Npoints=3, dt=0.25)


@pytest.fixture
def filesystem():
    return SimpleNamespace(read_sequence=["t", "x", "y"])


def test_import_jovanic_with_match_ids(monkeypatch, tracker, filesystem):
    calls = {}
    raw_df = pd.DataFrame({"Step": [0], "AgentID": [0], "x": [1.0], "y": [2.0]})

    def fake_read(pref, tracker, estimate_dt=False):
        calls["pref"] = pref
        calls["estimate_dt"] = estimate_dt
        return raw_df

    def fake_match(df, **kwargs):
        calls["match_kwargs"] = kwargs
        return df.assign(matched=True)

    def fake_constrain(df, **kwargs):
        calls["constrain_kwargs"] = kwargs
        return df

    def fake_endpoint(df, dt):
        calls["endpoint_dt"] = dt
        return "endpoint"

    def fake_finalize(df, complete_ticks, interpolate_ticks):
        calls["finalize_flags"] = (complete_ticks, interpolate_ticks)
        return "step"

    monkeypatch.setattr(
        importing, "read_timeseries_from_raw_files_per_parameter", fake_read
    )
    monkeypatch.setattr(importing, "match_larva_ids", fake_match)
    monkeypatch.setattr(importing, "constrain_selected_tracks", fake_constrain)
    monkeypatch.setattr(
        importing, "init_endpoint_dataframe_from_timeseries", fake_endpoint
    )
    monkeypatch.setattr(importing, "finalize_timeseries_dataframe", fake_finalize)

    step, end = importing.import_Jovanic(
        source_id="dataset",
        source_dir="/data",
        tracker=tracker,
        filesystem=filesystem,
        match_ids=True,
        matchID_kws={"foo": "bar"},
        interpolate_ticks=True,
        extra="value",
    )

    assert step == "step"
    assert end == "endpoint"
    assert calls["pref"] == "/data/dataset"
    assert calls["match_kwargs"]["dt"] == tracker.dt
    assert calls["match_kwargs"]["Npoints"] == tracker.Npoints
    assert calls["finalize_flags"] == (False, True)
    assert calls["endpoint_dt"] == tracker.dt
    # Timestep estimation is opt-in and off unless the caller asks for it.
    assert calls["estimate_dt"] is False


def test_import_jovanic_without_match_ids(monkeypatch, tracker, filesystem):
    def fake_read(pref, tracker, estimate_dt=False):
        return pd.DataFrame({"Step": [0], "AgentID": [0]})

    def fake_match(*args, **kwargs):  # should not be called
        raise AssertionError("match_larva_ids should be skipped when match_ids=False")

    monkeypatch.setattr(
        importing, "read_timeseries_from_raw_files_per_parameter", fake_read
    )
    monkeypatch.setattr(importing, "match_larva_ids", fake_match)
    monkeypatch.setattr(importing, "constrain_selected_tracks", lambda df, **kw: df)
    monkeypatch.setattr(
        importing, "init_endpoint_dataframe_from_timeseries", lambda df, dt: "endpoint"
    )
    monkeypatch.setattr(
        importing,
        "finalize_timeseries_dataframe",
        lambda df, complete_ticks, interpolate_ticks: "step",
    )

    step, end = importing.import_Jovanic(
        source_id="dataset",
        source_dir="/data",
        tracker=tracker,
        filesystem=filesystem,
        match_ids=False,
    )

    assert step == "step"
    assert end == "endpoint"


def test_import_schleyer_collects_csvs(monkeypatch, tracker, filesystem):
    captured = {"files": []}

    monkeypatch.setattr(
        importing, "get_Schleyer_metadata_inv_x", lambda dir: f"inv:{dir}"
    )

    def fake_read(files, inv_x, read_sequence, save_mode, tracker):
        captured["files"].append((tuple(files), inv_x, save_mode, tuple(read_sequence)))
        return [pd.DataFrame({"file_count": [len(files)]})]

    def fake_generate(dfs, dt, **kwargs):
        captured["generate"] = {"dt": dt, "dfs": dfs, "kwargs": kwargs}
        return ("step", "endpoint")

    monkeypatch.setattr(
        importing, "read_timeseries_from_raw_files_per_larva", fake_read
    )
    monkeypatch.setattr(importing, "generate_dataframes", fake_generate)
    monkeypatch.setattr(
        importing.os, "listdir", lambda d: ["track1.csv", "notes.txt", "track2.csv"]
    )

    step, end = importing.import_Schleyer(
        source_dir="/schleyer",
        tracker=tracker,
        filesystem=filesystem,
        save_mode="full",
        extra="value",
    )

    assert step == "step"
    assert end == "endpoint"
    ((files_tuple, inv_flag, save_mode, read_seq),) = captured["files"]
    # Normalize paths for cross-platform compatibility (Windows uses backslashes)
    normalized_files = tuple(f.replace("\\", "/") for f in files_tuple)
    assert normalized_files == ("/schleyer/track1.csv", "/schleyer/track2.csv")
    assert inv_flag.replace("\\", "/") == "inv:/schleyer"
    assert save_mode == "full"
    assert read_seq == tuple(filesystem.read_sequence)
    assert captured["generate"]["dt"] == tracker.dt
    assert captured["generate"]["kwargs"]["extra"] == "value"
    assert len(captured["generate"]["dfs"]) == 1


@pytest.mark.parametrize("func_name", ["import_Berni", "import_Arguello"])
def test_import_single_track_variants(monkeypatch, tracker, filesystem, func_name):
    captured = {}

    def fake_read(files, read_sequence, tracker, **kwargs):
        captured["files"] = files
        captured["sequence"] = read_sequence
        return [pd.DataFrame({"source": [func_name]})]

    def fake_generate(dfs, dt, **kwargs):
        captured["dt"] = dt
        return ("step", "endpoint")

    monkeypatch.setattr(
        importing, "read_timeseries_from_raw_files_per_larva", fake_read
    )
    monkeypatch.setattr(importing, "generate_dataframes", fake_generate)

    func = getattr(importing, func_name)
    step, end = func(
        source_files=["/path/a.csv", "/path/b.csv"],
        tracker=tracker,
        filesystem=filesystem,
    )

    assert step == "step"
    assert end == "endpoint"
    assert captured["files"] == ["/path/a.csv", "/path/b.csv"]
    assert captured["sequence"] == filesystem.read_sequence
    assert captured["dt"] == tracker.dt


def test_lab_specific_import_functions_mapping():
    expected = {"Jovanic", "Berni", "Schleyer", "Arguello", "DeepLabCut"}
    assert expected == set(importing.lab_specific_import_functions.keys())
    for name in expected:
        assert callable(importing.lab_specific_import_functions[name])


def test_deeplabcut_labformat_is_registered():
    lab = reg.conf.LabFormat.get("DeepLabCut")

    assert lab.tracker.fr == 30.0
    assert lab.filesystem.file_sufs == [".h5", ".hdf5", ".csv"]


def _patch_import_func(monkeypatch, lf, captured):
    def fake_import(tracker, filesystem, estimate_dt=False, **kwargs):
        captured["estimate_dt"] = estimate_dt
        return None, None

    monkeypatch.setattr(type(lf), "import_func", property(lambda self: fake_import))


def test_variable_framerate_estimates_dt_unless_the_caller_opts_out(monkeypatch):
    """
    Verify that a lab format declaring a variable framerate estimates its timestep by
    default, and keeps the nominal value when the caller explicitly opts out.
    """
    lf = reg.conf.LabFormat.get("Jovanic")
    assert lf.tracker.constant_framerate is False
    captured = {}
    _patch_import_func(monkeypatch, lf, captured)

    lf.import_data_to_dfs(parent_dir="whatever")
    assert captured["estimate_dt"] is True

    lf.import_data_to_dfs(parent_dir="whatever", estimate_dt=False)
    assert captured["estimate_dt"] is False


def test_constant_framerate_never_estimates_dt(monkeypatch):
    """
    Verify that a constant-framerate lab format never estimates its timestep, even when
    estimation is explicitly requested, since its nominal value fully describes it.
    """
    lf = reg.conf.LabFormat.get("Schleyer")
    assert lf.tracker.constant_framerate is True
    captured = {}
    _patch_import_func(monkeypatch, lf, captured)

    lf.import_data_to_dfs(parent_dir="whatever")
    assert captured["estimate_dt"] is False

    # An explicit request cannot switch estimation on for a constant-framerate format.
    lf.import_data_to_dfs(parent_dir="whatever", estimate_dt=True)
    assert captured["estimate_dt"] is False


def test_labformat_skips_estimation_for_importers_that_lack_support(monkeypatch):
    """
    Verify that requesting estimation from an importer without an estimate_dt parameter
    is ignored rather than raising, leaving the lab-format timestep in place.
    """
    lf = reg.conf.LabFormat.get("Jovanic")
    captured = {}

    def fake_import(tracker, filesystem, **kwargs):
        captured["kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(type(lf), "import_func", property(lambda self: fake_import))

    lf.import_data_to_dfs(parent_dir="whatever", estimate_dt=True)
    assert "estimate_dt" not in captured["kwargs"]
    assert lf.tracker.dt == pytest.approx(0.07)


def _write_spine(path, tag, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(" ".join([tag] + [str(v) for v in r]) for r in rows) + "\n"
    )


def test_jovanic_converts_raw_spine_files_when_txt_are_missing(tmp_path):
    """
    Verify that a dataset stored only as raw tracker spine files is converted to the
    per-parameter txt files that the Jovanic reader consumes, with the agent IDs carrying
    the dataset ID so that datasets imported together stay distinct.
    """
    source_dir = tmp_path / "exp"
    # Two recording sessions under a folder named after the dataset, reusing track ids.
    for rec, tag in [("rec1", "a"), ("rec2", "b")]:
        _write_spine(
            source_dir / "Fed" / rec / f"{tag}.spine",
            tag,
            [
                [1, 0.0, 0.0, 10.0, 1.0, 11.0],
                [1, 0.1, 0.1, 10.1, 1.1, 11.1],
            ],
        )

    importing._ensure_Jovanic_per_parameter_files("Fed", source_dir.as_posix(), 2)

    for suf in ["larvaid", "t", "x_spine", "y_spine"]:
        assert (source_dir / f"Fed_{suf}.txt").is_file()
    ids = (source_dir / "Fed_larvaid.txt").read_text().split()
    # One agent per recording, offset apart and prefixed with the dataset ID.
    assert sorted(set(ids)) == ["Fed_100001", "Fed_200001"]


def test_jovanic_leaves_existing_per_parameter_files_untouched(tmp_path):
    """
    Verify that a dataset already stored as per-parameter txt files is not regenerated,
    even when raw spine files sit alongside it.
    """
    source_dir = tmp_path / "exp"
    source_dir.mkdir(parents=True)
    for suf in ["larvaid", "t", "x_spine", "y_spine"]:
        (source_dir / f"Fed_{suf}.txt").write_text("sentinel\n")
    _write_spine(
        source_dir / "Fed" / "rec1" / "a.spine", "a", [[1, 0.0, 0.0, 10.0, 1.0, 11.0]]
    )

    importing._ensure_Jovanic_per_parameter_files("Fed", source_dir.as_posix(), 2)

    for suf in ["larvaid", "t", "x_spine", "y_spine"]:
        assert (source_dir / f"Fed_{suf}.txt").read_text() == "sentinel\n"


def test_jovanic_without_spine_files_is_a_noop(tmp_path):
    """
    Verify that a source folder holding neither txt nor spine files is left alone, so the
    reader reports the missing files rather than the conversion masking the cause.
    """
    source_dir = tmp_path / "exp"
    source_dir.mkdir(parents=True)

    importing._ensure_Jovanic_per_parameter_files("Fed", source_dir.as_posix(), 11)

    assert list(source_dir.iterdir()) == []


def test_labformat_midline_points_follow_the_data(monkeypatch, tmp_path):
    """
    Verify that the midline points counted in the raw data replace the lab-format's value
    during import, and that the lab-format's value is kept when counting is switched off.
    """
    lf = reg.conf.LabFormat.get("Jovanic")
    captured = {}

    def fake_import(tracker, filesystem, **kwargs):
        captured["Npoints"] = tracker.Npoints
        return None, None

    monkeypatch.setattr(type(lf), "import_func", property(lambda self: fake_import))
    monkeypatch.setattr(
        type(lf), "raw_folder", property(lambda self: tmp_path.as_posix())
    )
    source = tmp_path / "exp"
    source.mkdir()
    # 4 midline points in the data against the Jovanic value of 11.
    pd.DataFrame([[0.0, 1.0, 2.0, 3.0]]).to_csv(
        source / "Fed_x_spine.txt", header=False, index=False, sep="\t"
    )

    lf.import_data_to_dfs(parent_dir="exp", source_id="Fed")
    assert captured["Npoints"] == 4

    lf2 = reg.conf.LabFormat.get("Jovanic")
    monkeypatch.setattr(type(lf2), "import_func", property(lambda self: fake_import))
    monkeypatch.setattr(
        type(lf2), "raw_folder", property(lambda self: tmp_path.as_posix())
    )
    lf2.import_data_to_dfs(
        parent_dir="exp", source_id="Fed", estimate_midline_points=False
    )
    assert captured["Npoints"] == 11


def test_labformat_keeps_its_midline_points_when_counting_fails(monkeypatch, tmp_path):
    """
    Verify that raw data the counter cannot read leaves the lab-format's value in place
    rather than propagating a fabricated one.
    """
    lf = reg.conf.LabFormat.get("Jovanic")
    captured = {}

    def fake_import(tracker, filesystem, **kwargs):
        captured["Npoints"] = tracker.Npoints
        return None, None

    monkeypatch.setattr(type(lf), "import_func", property(lambda self: fake_import))
    monkeypatch.setattr(
        type(lf), "raw_folder", property(lambda self: tmp_path.as_posix())
    )
    (tmp_path / "empty").mkdir()

    lf.import_data_to_dfs(parent_dir="empty", source_id="Fed")
    assert captured["Npoints"] == 11
