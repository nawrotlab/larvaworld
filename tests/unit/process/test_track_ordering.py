"""Tests for the temporal ordering of imported tracks and for trajectory centering.

Both cover defects that stayed silent in the data : a non-stable sort that reordered the
samples inside each track, and a centering offset that used a half-range instead of a
midpoint. Neither raises, so they are pinned here by their numerical consequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib.process.import_aux import (
    concatenate_larva_tracks,
    constrain_selected_tracks,
    generate_dataframes,
    init_endpoint_dataframe_from_timeseries,
)

DT = 0.05
NTICKS = 12
# Enough agents that the lexicographic order of the IDs ('Larva_10' before 'Larva_2')
# differs from their numeric order, which is what a non-stable sort reshuffles.
NAGENTS = 12


def _single_track(nticks: int = NTICKS) -> pd.DataFrame:
    """One track, in the per-larva layout the DeepLabCut import produces."""
    return pd.DataFrame(
        {
            "head_x": np.arange(nticks, dtype=float),
            "head_y": np.zeros(nticks),
            "tail_x": np.arange(nticks, dtype=float) - 1.0,
            "tail_y": np.zeros(nticks),
        }
    )


def _interleaved_timeseries() -> pd.DataFrame:
    """Rows ordered by time, with the agents mixed within each timepoint.

    This is the layout of a tracker that follows several animals at once, as opposed to
    the one-file-per-animal layout, and it is the case the sort in
    `constrain_selected_tracks` exists for.
    """
    ids = [f"Larva_{i}" for i in range(NAGENTS)]
    rows = [
        {
            "AgentID": aID,
            "t": tick * DT,
            "Step": tick,
            "head_x": float(tick),
            "head_y": 0.0,
        }
        for tick in range(NTICKS)
        for aID in ids
    ]
    return pd.DataFrame(rows).set_index("AgentID")


def _timestamps_by_agent(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        aID: df.loc[aID, "t"].to_numpy(dtype=float) for aID in df.index.unique().values
    }


def test_constrain_selected_tracks_keeps_per_larva_files_in_temporal_order():
    df = concatenate_larva_tracks([_single_track() for _ in range(NAGENTS)], DT)

    constrained = constrain_selected_tracks(df)

    for aID, timestamps in _timestamps_by_agent(constrained).items():
        assert np.all(np.diff(timestamps) > 0), f"{aID} is not in temporal order"


def test_constrain_selected_tracks_keeps_interleaved_tracks_in_temporal_order():
    constrained = constrain_selected_tracks(_interleaved_timeseries())

    for aID, timestamps in _timestamps_by_agent(constrained).items():
        assert np.all(np.diff(timestamps) > 0), f"{aID} is not in temporal order"


def test_constrain_selected_tracks_groups_the_rows_of_each_agent():
    """The point of the sort : an agent's rows end up contiguous."""
    constrained = constrain_selected_tracks(_interleaved_timeseries())

    assert constrained.index.is_monotonic_increasing


def test_imported_endpoint_durations_are_the_recorded_duration():
    _, e = generate_dataframes([_single_track() for _ in range(NAGENTS)], DT)

    duration = (NTICKS - 1) * DT
    assert (e["cum_dur"] > 0).all()
    assert e["cum_dur"].to_numpy() == pytest.approx(duration)
    assert e["initial_t"].to_numpy() == pytest.approx(0.0)
    assert e["final_t"].to_numpy() == pytest.approx(duration)


def test_endpoint_durations_survive_the_track_constraining_step():
    """The endpoint dataframe is built after the sort, so it has to agree with it."""
    df = constrain_selected_tracks(
        concatenate_larva_tracks([_single_track() for _ in range(NAGENTS)], DT)
    )

    e = init_endpoint_dataframe_from_timeseries(df=df, dt=DT)

    assert (e["cum_dur"] > 0).all()
    assert (e["dt"] > 0).all()


def _dataset_with_offset_tracks(tmp_path, offsets):
    """A dataset whose tracks are identical up to a per-agent translation."""
    from larvaworld.lib.process.dataset import LarvaDataset

    agent_ids = [f"Larva_{i}" for i in range(len(offsets))]
    index = pd.MultiIndex.from_product(
        [range(NTICKS), agent_ids], names=["Step", "AgentID"]
    )
    # A track spanning [-1, 1] about its own centre, before the offset is applied.
    base = np.linspace(-1.0, 1.0, NTICKS)
    x = np.concatenate([[base[t] + ox for ox, _ in offsets] for t in range(NTICKS)])
    y = np.concatenate([[base[t] + oy for _, oy in offsets] for t in range(NTICKS)])
    step = pd.DataFrame({"x": x, "y": y}, index=index)
    end = pd.DataFrame(
        {"length": np.ones(len(agent_ids))},
        index=pd.Index(agent_ids, name="AgentID"),
    )
    return LarvaDataset(
        dir=str(tmp_path / "centered"),
        id="centered",
        agent_ids=agent_ids,
        dt=DT,
        Nsteps=NTICKS,
        step=step,
        end=end,
        load_data=False,
    )


def test_align_trajectories_center_puts_each_track_on_the_origin(tmp_path):
    """Each track is centred on its own bounding-box midpoint, whatever its offset."""
    d = _dataset_with_offset_tracks(tmp_path, [(5.0, 6.0), (-3.0, 2.0), (0.0, 0.0)])

    centered = d.align_trajectories(transposition="center", replace=False)

    for aID in d.ids:
        xy = centered.xs(aID, level="AgentID")[["x", "y"]].to_numpy(dtype=float)
        midpoint = (xy.max(axis=0) + xy.min(axis=0)) / 2
        assert midpoint == pytest.approx([0.0, 0.0], abs=1e-9)


def test_align_trajectories_center_is_a_midpoint_not_a_half_range(tmp_path):
    """A track offset by (dx, dy) must shift by exactly that, not by half its span."""
    offset = (5.0, 6.0)
    d = _dataset_with_offset_tracks(tmp_path, [offset])

    centered = d.align_trajectories(transposition="center", replace=False)

    aID = d.ids[0]
    before = d.step_data.xs(aID, level="AgentID")[["x", "y"]].to_numpy(dtype=float)
    after = centered.xs(aID, level="AgentID")[["x", "y"]].to_numpy(dtype=float)
    np.testing.assert_allclose(before - after, np.broadcast_to(offset, before.shape))
