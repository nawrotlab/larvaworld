"""Tests for arena occupancy and for the per-point pathlengths of the midline ends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.lib.process.dataset import LarvaDataset

NTICKS = 400
ARENA = 0.02  # 20 mm across, so a radius of 10 mm
RADIUS = ARENA / 2


def _dataset(tmp_path, xy_by_agent, npoints=1):
    """Build a dataset whose agents follow prescribed trajectories.

    xy_by_agent : mapping of agent id to an (NTICKS, 2) array of positions in metres.
    """
    agent_ids = list(xy_by_agent)
    index = pd.MultiIndex.from_product(
        [range(NTICKS), agent_ids], names=["Step", "AgentID"]
    )
    stacked = np.stack([xy_by_agent[a] for a in agent_ids], axis=1)
    step = pd.DataFrame(
        {"x": stacked[:, :, 0].ravel(), "y": stacked[:, :, 1].ravel()}, index=index
    )
    end = pd.DataFrame(
        {"length": np.full(len(agent_ids), 0.004)},
        index=pd.Index(agent_ids, name="AgentID"),
    )
    d = LarvaDataset(
        dir=str(tmp_path / "occ"),
        id="occ",
        agent_ids=agent_ids,
        dt=0.1,
        Nsteps=NTICKS,
        Npoints=npoints,
        step=step,
        end=end,
        load_data=False,
    )
    d.config.env_params.arena.dims = (ARENA, ARENA)
    d.config.env_params.arena.geometry = "circular"
    return d


def _uniform_disc(n, radius, seed=0):
    """Points spread uniformly over a disc : radius scales with the square root."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _ring(n, radius, frac):
    """Points on a circle at a fixed fraction of the arena radius."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [frac * radius * np.cos(theta), frac * radius * np.sin(theta)]
    )


def test_occupancy_of_a_uniform_agent_matches_the_analytic_expectation(tmp_path):
    """Uniform use of a disc gives a mean normalized radius of 2/3, not 1/2."""
    d = _dataset(tmp_path, {"Larva_0": _uniform_disc(NTICKS, RADIUS)})

    d.comp_occupancy()

    assert d.e[reg.getPar("rr_mu")].iloc[0] == pytest.approx(2 / 3, abs=0.05)
    # The outer 25% of the radius holds 1 - 0.75**2 of the area.
    assert d.e[reg.getPar("bocc25")].iloc[0] == pytest.approx(1 - 0.75**2, abs=0.06)
    assert d.e[reg.getPar("cphi")].iloc[0] == pytest.approx(0.0, abs=0.08)


def test_occupancy_separates_a_central_agent_from_a_border_agent(tmp_path):
    d = _dataset(
        tmp_path,
        {
            "Larva_0": _ring(NTICKS, RADIUS, 0.2),  # stays near the centre
            "Larva_1": _ring(NTICKS, RADIUS, 0.95),  # hugs the border
        },
    )

    d.comp_occupancy()
    e = d.e

    assert e[reg.getPar("rr_mu")]["Larva_0"] == pytest.approx(0.2, abs=1e-6)
    assert e[reg.getPar("rr_mu")]["Larva_1"] == pytest.approx(0.95, abs=1e-6)
    # Centrophily is positive towards the centre and negative towards the border.
    assert e[reg.getPar("cphi")]["Larva_0"] > 0
    assert e[reg.getPar("cphi")]["Larva_1"] < 0
    # The border agent is in the outer bands the whole time; the central one never.
    assert e[reg.getPar("bocc25")]["Larva_1"] == pytest.approx(1.0)
    assert e[reg.getPar("bocc10")]["Larva_1"] == pytest.approx(1.0)
    assert e[reg.getPar("bocc25")]["Larva_0"] == pytest.approx(0.0)


def test_occupancy_is_bounded(tmp_path):
    d = _dataset(tmp_path, {"Larva_0": _uniform_disc(NTICKS, RADIUS, seed=3)})

    d.comp_occupancy()

    assert 0 <= d.e[reg.getPar("rr_mu")].iloc[0] <= 1
    for k in ("bocc10", "bocc25"):
        assert 0 <= d.e[reg.getPar(k)].iloc[0] <= 1
    assert -1 <= d.e[reg.getPar("cphi")].iloc[0] <= 1


def test_occupancy_is_skipped_without_a_usable_arena(tmp_path):
    """A degenerate arena must not raise, and must not leave partial results."""
    d = _dataset(tmp_path, {"Larva_0": _uniform_disc(NTICKS, RADIUS)})
    d.config.env_params.arena.dims = (0.0, 0.0)

    d.comp_occupancy()

    assert reg.getPar("rr_mu") not in d.e.columns


def test_comp_spatial_reports_the_midline_ends_separately(tmp_path):
    """Head and tail pathlengths are computed alongside the tracked point's."""
    from larvaworld.lib.util import nam

    n = 60
    t = np.linspace(0, 4 * np.pi, n)
    # A body crawling along x, with the head sweeping laterally and the tail steady.
    mid = np.column_stack([np.linspace(-0.005, 0.005, n), np.zeros(n)])
    head = mid + np.column_stack([np.zeros(n) + 0.002, 0.001 * np.sin(t)])
    tail = mid - np.column_stack([np.zeros(n) + 0.002, np.zeros(n)])

    agent_ids = ["Larva_0"]
    index = pd.MultiIndex.from_product([range(n), agent_ids], names=["Step", "AgentID"])
    step = pd.DataFrame(
        {
            "head_x": head[:, 0],
            "head_y": head[:, 1],
            "point2_x": mid[:, 0],
            "point2_y": mid[:, 1],
            "tail_x": tail[:, 0],
            "tail_y": tail[:, 1],
        },
        index=index,
    )
    end = pd.DataFrame({"length": [0.004]}, index=pd.Index(agent_ids, name="AgentID"))
    d = LarvaDataset(
        dir=str(tmp_path / "motility"),
        id="motility",
        agent_ids=agent_ids,
        dt=0.1,
        Nsteps=n,
        Npoints=3,
        step=step,
        end=end,
        load_data=False,
    )
    d.config.env_params.arena.dims = (ARENA, ARENA)
    d.config.N = len(agent_ids)

    d.comp_spatial()

    for point in ("", "head", "tail"):
        assert nam.cum(nam.dst(point)) in d.e.columns, point
    # The sweeping head covers more ground than the tail dragged straight behind.
    assert d.e[nam.cum(nam.dst("head"))].iloc[0] > d.e[nam.cum(nam.dst("tail"))].iloc[0]
