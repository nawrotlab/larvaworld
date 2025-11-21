import pandas as pd
import pytest
from types import SimpleNamespace

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

    def fake_read(pref, tracker):
        calls["pref"] = pref
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


def test_import_jovanic_without_match_ids(monkeypatch, tracker, filesystem):
    def fake_read(pref, tracker):
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
    expected = {"Jovanic", "Berni", "Schleyer", "Arguello"}
    assert expected == set(importing.lab_specific_import_functions.keys())
    for name in expected:
        assert callable(importing.lab_specific_import_functions[name])
