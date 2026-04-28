from __future__ import annotations

import numpy as np

from larvaworld.lib.model.envs.arena import Arena


class _DummyAgent:
    def __init__(self, pos, radius: float):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius


class _DummyAgents(list):
    @property
    def pos(self):
        return np.array([agent.pos for agent in self], dtype=float)


class _DummyArena:
    def __init__(self, sources, source_positions):
        self.sources = np.array(sources, dtype=object)
        self.source_positions = np.array(source_positions, dtype=float)
        self.accessible_sources = None
        self.accessible_sources_sorted = None

    def source_positions_in_array(self):
        return None


def test_accessible_sources_multi_handles_empty_source_array_for_closest_lookup() -> (
    None
):
    arena = _DummyArena(sources=[], source_positions=np.empty((0, 2)))
    agents = _DummyAgents([_DummyAgent((0.0, 0.0), radius=0.01)])

    Arena.accessible_sources_multi(arena, agents, return_closest=True)

    assert arena.accessible_sources[agents[0]] is None


def test_accessible_sources_multi_handles_empty_source_array_for_list_lookup() -> None:
    arena = _DummyArena(sources=[], source_positions=np.empty((0, 2)))
    agents = _DummyAgents([_DummyAgent((0.0, 0.0), radius=0.01)])

    Arena.accessible_sources_multi(arena, agents, return_closest=False)

    assert arena.accessible_sources[agents[0]] == []
