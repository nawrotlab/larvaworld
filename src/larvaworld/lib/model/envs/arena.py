from __future__ import annotations
from typing import TYPE_CHECKING, Any
import agentpy
import numpy as np
import param
from shapely.geometry import Point

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ... import util
from ...param import BoundedArea
from .valuegrid import SpatialEntity

__all__: list[str] = [
    "Arena",
]


class ViewableBoundedArea(SpatialEntity, BoundedArea):
    pass


class Arena(ViewableBoundedArea, agentpy.Space):
    """
    Simulation arena providing spatial environment for agents.

    Combines bounded area geometry with agentpy.Space functionality to create
    a simulation environment where agents can be placed, moved, and interact
    with sources. Supports both stable and displaceable source management.

    Attributes:
        boundary_margin: Margin from arena boundaries (default: 0.96)
        edges: List of boundary edge segments as Point pairs
        stable_sources: List of non-movable sources
        displacable_sources: List of movable sources
        accessible_sources: Cached accessible sources for agents

    Example:
        >>> arena = Arena(model=sim_model, dims=(1.0, 1.0))
        >>> arena.place_agent(agent, (0.5, 0.5))
        >>> arena.add_sources([food1, food2], [(0.2, 0.3), (0.8, 0.7)])
    """

    boundary_margin = param.Magnitude(0.96)

    def __init__(self, model: Any | None = None, **kwargs: Any) -> None:
        ViewableBoundedArea.__init__(self, **kwargs)
        self.edges = [
            [Point(x1, y1), Point(x2, y2)]
            for (x1, y1), (x2, y2) in util.SuperList(self.vertices).in_pairs
        ]
        if model is not None:
            agentpy.Space.__init__(self, model=model, torus=self.torus, shape=self.dims)

        self.stable_source_positions = []
        self.displacable_source_positions = []
        self.displacable_sources = []
        self.stable_sources = []
        self.accessible_sources = None
        self.accessible_sources_sorted = None

    def place_agent(self, agent: Any, pos: Any) -> None:
        pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
        self.positions[agent] = pos  # Add pos to agent_dict

    def move_agent(self, agent: Any, pos: Any) -> None:
        self.move_to(agent, pos)

    def add_sources(self, sources: list[Any], positions: list[Any]) -> None:
        for source, pos in zip(sources, positions):
            pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
            if source.can_be_displaced:
                self.displacable_source_positions.append(pos)  # Add pos to agent_dict
                self.displacable_sources.append(source)  # Add pos to agent_dict
            else:
                self.stable_source_positions.append(pos)
                self.stable_sources.append(source)

    def source_positions_in_array(self) -> None:
        if len(self.displacable_sources) > 0:
            for i, source in enumerate(self.displacable_sources):
                self.displacable_source_positions[i] = np.array(source.get_position())
            if len(self.stable_sources) > 0:
                self.source_positions = np.vstack(
                    [
                        np.array(self.displacable_source_positions),
                        np.array(self.stable_source_positions),
                    ]
                )
                self.sources = np.array(self.displacable_sources + self.stable_sources)
            else:
                self.source_positions = np.array(self.displacable_source_positions)
                self.sources = np.array(self.displacable_sources)
        else:
            self.source_positions = np.array(self.stable_source_positions)
            self.sources = np.array(self.stable_sources)

    def accessible_sources(self, pos: Any, radius: float) -> list[Any]:
        return self.sources[
            np.where(util.eudi5x(self.source_positions, pos) <= radius)
        ].tolist()

    def accessible_sources_multi(
        self, agents: Any, positive_amount: bool = True, return_closest: bool = True
    ) -> None:
        self.source_positions_in_array()
        if positive_amount:
            idx = np.array([s.amount > 0 for s in self.sources])
            self.sources = self.sources[idx]
            self.source_positions = self.source_positions[idx]
        ps = np.array(agents.pos)
        ds = util.eudiNxN(self.source_positions, ps)
        self.accessible_sources_sorted = {
            a: {"sources": self.sources[np.argsort(ds[i])], "dsts": np.sort(ds[i])}
            for i, a in enumerate(agents)
        }
        if not return_closest:
            dic = {
                a: dic["sources"][dic["dsts"] <= a.radius].tolist()
                for a, dic in self.accessible_sources_sorted.items()
            }
        else:
            dic = {
                a: dic["sources"][0] if dic["dsts"][0] <= a.radius else None
                for a, dic in self.accessible_sources_sorted.items()
            }
        self.accessible_sources = dic

    def draw(self, v: Any | None = None) -> Figure:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        fig = Figure()
        ax = fig.subplots()
        x, y = np.array(self.dims) * 1000
        if self.geometry == "circular":
            if x != y:
                raise
            arena = plt.Circle(
                (0, 0), x / 2, edgecolor="black", facecolor="lightgrey", lw=3
            )
        elif self.geometry == "rectangular":
            arena = plt.Rectangle(
                (-x / 2, -y / 2), x, y, edgecolor="black", facecolor="lightgrey", lw=3
            )
        else:
            raise ValueError("Not implemented")
        ax.add_patch(arena)
        ax.set_xlim(-x, x)
        ax.set_ylim(-y, y)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.axis("equal")
        return fig
