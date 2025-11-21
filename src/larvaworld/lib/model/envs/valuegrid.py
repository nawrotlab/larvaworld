from __future__ import annotations
from typing import Any
import math

import agentpy
import numpy as np
import param
from scipy.ndimage.filters import gaussian_filter
from shapely import geometry

from ... import util
from ...param import (
    ClassAttr,
    Phase,
    PositiveIntegerRange,
    PositiveNumber,
    Substrate,
    Viewable,
)
from ...screen import ScreenTextBox
from .. import Object

__all__: list[str] = [
    "SpatialEntity",
    "Grid",
    "GridOverSpace",
    "ValueGrid",
    "FoodGrid",
    "OdorScape",
    "GaussianValueLayer",
    "AnalyticalValueLayer",
    "DiffusionValueLayer",
    "WindScape",
    "ThermoScape",
]


class SpatialEntity(Viewable, Object):
    """
    Base class for spatial entities in simulation environment.

    Combines visualization and object properties for entities that exist
    in spatial coordinates and can be rendered on screen.

    Attributes:
        color: Display color (default: white)
        visible: Whether entity is visible on screen

    Example:
        >>> entity = SpatialEntity(color="blue", visible=True)
        >>> entity.record_positions(label="pos")
    """

    color = param.Color(default="white")
    visible = param.Boolean(default=False)

    def record_positions(self, label: str = "p") -> None:
        """
        Records the positions of each agent.

        Arguments:
            label (string, optional):
                Name under which to record each position (default p).
                A number will be added for each coordinate (e.g. p1, p2, ...).

        """
        for agent, pos in self.positions.items():
            for i, p in enumerate(pos):
                agent.record(label + str(i), p)


class Grid(SpatialEntity):
    """
    Base grid class with spatial resolution configuration.

    Provides grid dimensionality and resolution for spatial discretization
    of continuous environments.

    Attributes:
        grid_dims: Grid resolution tuple (X, Y), default (51, 51)
        X: Grid width (number of cells in X dimension)
        Y: Grid height (number of cells in Y dimension)

    Example:
        >>> grid = Grid(grid_dims=(100, 100))
        >>> print(grid.X, grid.Y)  # 100 100
    """

    grid_dims = PositiveIntegerRange(
        (51, 51), softmax=500, doc="The spatial resolution of the food grid."
    )

    @property
    def X(self) -> int:
        return self.grid_dims[0]

    @property
    def Y(self) -> int:
        return self.grid_dims[1]


class GridOverSpace(Grid, agentpy.Grid):
    """
    Grid mapped over agentpy simulation space.

    Extends Grid and agentpy.Grid to create a discretized grid overlay
    on top of continuous simulation space, enabling spatial indexing
    and cell-based computations.

    Attributes:
        unique_id: Identifier for grid instance (default: "GridOverArena")
        space: Reference to model's agentpy.Space
        meshgrid: Numpy meshgrid for grid coordinates
        grid_vertices: Vertex positions for each grid cell

    Example:
        >>> grid = GridOverSpace(model=sim_model, grid_dims=(50, 50))
        >>> cell = grid.get_grid_cell((0.1, 0.2))
    """

    unique_id = param.String("GridOverArena")
    color = param.Color(default="white")
    visible = param.Boolean(default=False)

    # initial_value = param.Number(0.0, doc='initial value over the grid')
    # fixed_max = param.Boolean(False, doc='whether the max is kept constant')
    # grid_dims = PositiveIntegerRange((51, 51), softmax=500, doc='The spatial resolution of the food grid.')

    def __init__(self, model: Any, **kwargs: Any) -> None:
        Grid.__init__(self, **kwargs)
        agentpy.Grid.__init__(self, model=model, shape=self.grid_dims, **kwargs)
        self._torus = self.space._torus
        self.XY = np.array(self.grid_dims)
        self.xy = np.array(self.space.dims)
        x0, x1, y0, y1 = self.space.range

        self.cell_radius = np.sum((self.xy / self.XY / 2) ** 2) ** 0.5
        self.meshgrid = np.meshgrid(
            np.linspace(x0, x1, self.X), np.linspace(y0, y1, self.Y)
        )
        self.grid_vertices = self.generate_grid_vertices()

    @property
    def space(self) -> Any:
        return self.model.space

    def get_grid_cell(self, p: Any) -> tuple[int, int]:
        return tuple(np.floor(self.XY * (p / self.xy + 0.5)).astype(int))

    def generate_grid_vertices(self) -> np.ndarray:
        vertices = np.zeros([self.X, self.Y, 4, 2])
        for i in range(self.X):
            for j in range(self.Y):
                vertices[i, j] = self.cell_vertices(i, j)
        return vertices

    def cell_vertices(self, i: int, j: int) -> np.ndarray:
        x, y = self.xy / self.XY
        X, Y = self.X / 2, self.Y / 2
        return np.array(
            [
                (x * (i - X), y * (j - Y)),
                (x * (i + 1 - X), y * (j - Y)),
                (x * (i + 1 - X), y * (j + 1 - Y)),
                (x * (i - X), y * (j + 1 - Y)),
            ]
        )


class ValueGrid(Grid):
    """
    Grid storing scalar values at each cell position.

    Base class for value-based grids (food, odor, temperature) that maintain
    a 2D array of scalar values with methods for adding, retrieving, and
    visualizing spatial distributions.

    Attributes:
        initial_value: Default value for all grid cells (default: 0.0)
        fixed_max: Whether maximum value is kept constant (default: False)
        grid: 2D numpy array storing cell values
        max_value: Maximum value in grid (for normalization)
        min_value: Minimum value in grid (for clipping)
        sources: List of source objects contributing to grid values

    Example:
        >>> vgrid = ValueGrid(sources=[food1, food2], max_value=100.0)
        >>> value = vgrid.get_value((0.5, 0.5))
        >>> vgrid.add_value((0.1, 0.2), 10.0)
    """

    initial_value = param.Number(0.0, doc="initial value over the grid")

    fixed_max = param.Boolean(False, doc="whether the max is kept constant")

    # grid_dims = PositiveIntegerRange((51, 51),softmax=500, doc='The spatial resolution of the food grid.')

    def __init__(
        self,
        sources: list[Any] | None = None,
        max_value: float | None = None,
        min_value: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if sources is None:
            sources = []
        self.sources = sources
        self.XY0 = np.array(self.grid_dims)
        self.grid = np.ones(self.grid_dims) * self.initial_value

        if max_value is None:
            max_value = np.max(self.grid)
        self.max_value = max_value
        self.min_value = min_value
        if self.model is not None:
            self.match_space(self.model.space)

    def match_space(self, space: Any) -> None:
        self.xy0 = np.array(space.dims)
        self.x, self.y = self.xy0 / self.XY0
        x0, x1, y0, y1 = space.range
        self.cell_radius = np.sum((self.xy0 / self.XY0 / 2) ** 2) ** 0.5
        self.meshgrid = np.meshgrid(
            np.linspace(x0, x1, self.X), np.linspace(y0, y1, self.Y)
        )
        self.grid_vertices = self.generate_grid_vertices()

    def update_values(self) -> None:
        pass

    def add_value(self, p: Any, value: float) -> float:
        return self.add_cell_value(self.get_grid_cell(p), value)

    def get_value(self, p: Any) -> float:
        return self.grid[self.get_grid_cell(p)]
        # return self.get_cell_value(self.get_grid_cell(p))

    def get_grid_cell(self, p: Any) -> tuple[int, int]:
        return tuple(np.floor(self.XY0 * (p / self.xy0 + 0.5)).astype(int))

    def add_cell_value(self, cell: tuple[int, int], value: float) -> float:
        v0 = self.grid[cell]
        v1 = v0 + value
        if not self.fixed_max:
            self.max_value = np.max([self.max_value, v1])
        v2 = np.clip(v1, a_min=self.min_value, a_max=self.max_value)
        self.grid[cell] = v2
        # print(v2/self.initial_value,self.initial_value)
        if v1 < v2:
            return self.min_value - v0
        elif v1 > v2:
            return self.max_value - v0
        else:
            return value

    def generate_grid_vertices(self) -> np.ndarray:
        vertices = np.zeros([self.X, self.Y, 4, 2])
        for i in range(self.X):
            for j in range(self.Y):
                vertices[i, j] = self.cell_vertices(i, j)
        return vertices

    def cell_vertices(self, i: int, j: int) -> np.ndarray:
        x, y = self.x, self.y
        X, Y = self.X / 2, self.Y / 2
        return np.array(
            [
                (x * (i - X), y * (j - Y)),
                (x * (i + 1 - X), y * (j - Y)),
                (x * (i + 1 - X), y * (j + 1 - Y)),
                (x * (i - X), y * (j + 1 - Y)),
            ]
        )

    def cel_pos(self, i: int, j: int) -> tuple[float, float]:
        return self.x * (i - self.X / 2 + 0.5), self.y * (j - self.Y / 2 + 0.5)

    def reset(self) -> None:
        self.grid = np.ones(self.grid_dims) * self.initial_value

    def empty_grid(self) -> None:
        self.grid = np.zeros(self.grid_dims)

    def draw_peak(self, v: Any) -> None:
        idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
        p = self.cel_pos(*idx)

        v.draw_circle(p, self.cell_radius / 2, self.color, filled=True, width=0.0005)
        p_text = (p[0] + self.x, p[1] - self.y)
        text_box = ScreenTextBox(
            text=str(np.round(self.grid.max(), 2)),
            color=self.color,
            visible=True,
            text_centre=tuple(v.space2screen_pos(p_text)),
        )
        text_box.draw(v)

    def draw(self, v: Any, **kwargs: Any) -> None:
        Cgrid = self.get_color_grid().reshape([self.X, self.Y, 3])
        try:
            for i in range(self.X):
                for j in range(self.Y):
                    v.draw_polygon(self.grid_vertices[i, j], Cgrid[i, j], filled=True)
        except:
            pass

        if self.model.screen_manager.odor_aura:
            self.draw_isocontours(v)
        self.draw_peak(v)

    def draw_isocontours(self, v: Any) -> None:
        N = 6
        k = 4
        g = self.grid
        c = self.color
        vmax = np.max(g)
        for i in range(N):
            vv = vmax * k**-i
            if vv <= 0:
                continue
            inds = np.argwhere((vv <= g) & (g < vv * k)).tolist()
            points = [self.cel_pos(i, j) for (i, j) in inds]
            if len(points) > 2:
                try:
                    ps = np.array(points)
                    pxy = ps[np.argmax(ps[:, 0]), :] + np.array([self.x, -self.y])
                    v.draw_convex(points, color=c, filled=False, width=0.0005)
                    text_box = ScreenTextBox(
                        text=str(np.round(vv, 2)),
                        color=c,
                        visible=True,
                        text_centre=tuple(v.space2screen_pos(pxy)),
                    )
                    text_box.draw(v)
                except:
                    pass

    def get_color_grid(self) -> np.ndarray:
        g = self.grid.flatten()
        v0, v1 = self.min_value, self.max_value
        gg = (g - v0) / (v1 - v0)
        k = 10**2
        m = (1 - np.exp(-k)) ** -1
        q = m * (1 - np.exp(-k * gg))
        q = np.clip(q, a_min=0, a_max=1)
        return util.col_range(q, low=(255, 255, 255), high=self.color, mul255=True)


class FoodGrid(ValueGrid):
    """
    Food distribution grid for agent feeding behavior.

    Specialized ValueGrid for food resources with substrate properties,
    green color visualization, and fixed maximum value behavior. Agents
    can consume food from grid cells during simulation.

    Attributes:
        unique_id: Grid identifier (default: "FoodGrid")
        color: Visualization color (default: green)
        fixed_max: Maximum value kept constant (default: True)
        initial_value: Starting food amount per cell (default: 10^-6)
        substrate: Substrate configuration for feeding interactions

    Example:
        >>> food_grid = FoodGrid(initial_value=0.001, max_value=1.0)
        >>> color = food_grid.get_color(0.5)
    """

    unique_id = param.String("FoodGrid")
    color = param.Color(default="green")
    fixed_max = param.Boolean(default=True)
    initial_value = param.Number(10**-6)
    substrate = ClassAttr(
        Substrate,
        default=Substrate(type="standard"),
        doc="The substrate where the agent feeds",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_color(self, v: float) -> np.ndarray:
        v0, v1 = self.min_value, self.max_value
        q = (v - v0) / (v1 - v0)
        return util.col_range(q, low=(255, 255, 255), high=self.color, mul255=True)

    def draw(self, v: Any, **kwargs: Any) -> None:
        v.draw_polygon(
            self.model.space.vertices, self.get_color(v=self.initial_value), filled=True
        )
        for i in range(self.X):
            for j in range(self.Y):
                vv = self.grid[i, j]
                if vv != self.initial_value:
                    v.draw_polygon(
                        self.grid_vertices[i, j], self.get_color(vv), filled=True
                    )


class OdorScape(ValueGrid):
    """
    Odor landscape grid for chemotaxis simulations.

    Factory class that dispatches to specific odor diffusion algorithms
    (Gaussian, Diffusion, or Analytical). Provides base infrastructure
    for modeling chemical gradients from odor sources.

    Attributes:
        unique_id: Grid identifier (default: "Odorscape")
        odorscape: Algorithm type ("Gaussian", "Diffusion", or "Analytical")

    Example:
        >>> odor = OdorScape(odorscape="Gaussian", sources=[source1, source2])
        >>> # Automatically creates GaussianValueLayer instance
    """

    unique_id = param.String("Odorscape")
    odorscape = param.Selector(
        objects=["Gaussian", "Diffusion"], doc="The odorscape algorithm"
    )

    def __init__(self, subclass_initialized: bool = False, **kwargs: Any) -> None:
        if subclass_initialized:
            super().__init__(**kwargs)
        else:
            if "odorscape" not in kwargs:
                raise
            else:
                subclasses = {
                    "Gaussian": GaussianValueLayer,
                    "Diffusion": DiffusionValueLayer,
                    "Analytical": AnalyticalValueLayer,
                }
                odorscape = kwargs["odorscape"]
                kwargs.pop("odorscape")
                subclasses[odorscape](**kwargs)


class AnalyticalValueLayer(OdorScape):
    """
    Analytical odor gradient computation layer.

    Computes odor values analytically using customizable value functions
    for each source. Default uses Gaussian distribution but supports
    arbitrary analytical formulations via value_function callback.

    Attributes:
        odorscape: Algorithm type (default: "Analytical")
        value_function: Callable for computing grid values from sources
                       Signature: (source, value, pos, rel_pos) -> float

    Example:
        >>> layer = AnalyticalValueLayer(sources=[odor_source])
        >>> value = layer.get_value((0.5, 0.5))
        >>> grid = layer.get_grid()
    """

    odorscape = param.Selector(default="Analytical")

    # function reference used to compute a grid value
    # arguments: source object, current value, grid position (pos), relative grid position (rel_pos)
    # the function should add the computed value to the current value to support layering
    # the source parameter refers to a specific layer configuration of a defined food/odor source
    # ie: return value + newly_computed_value
    # default implementation equals the GaussianValueLayer implementation
    value_function = (
        lambda source, value, pos, rel_pos: value + source.odor.gaussian_value(rel_pos)
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(subclass_initialized=True, **kwargs)

    def get_value(self, pos: tuple[float, float]) -> float:
        value = 0
        for s in self.sources:
            p = s.get_position()
            rel_pos = [pos[0] - p[0], pos[1] - p[1]]
            value = AnalyticalValueLayer.value_function(s, value, pos, rel_pos)
        return value

    def get_grid(self) -> np.ndarray:
        X, Y = self.meshgrid

        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        self.max_value = np.max(V.flatten())
        return V

    def draw_isocontours(self, v: Any) -> None:
        for s in self.sources:
            p = s.get_position()

            if p[0] < 0:
                rsign = 1
            else:
                rsign = -1
            w = 0.0005
            for i, r0 in enumerate(np.arange(0, 0.040, 0.005)):
                pX = (p[0] + r0 * rsign, p[1])
                vv = s.odor.gaussian_value((r0 * rsign, 0))
                v.draw_circle(p, r0, self.color, filled=False, width=w)
                text_box = ScreenTextBox(
                    text=str(np.round(vv, 2)),
                    color=self.color,
                    visible=True,
                    text_centre=tuple(
                        v.space2screen_pos((p[0] + r0 * rsign + 5 * w, p[1]))
                    ),
                )
                text_box.draw(v)


class GaussianValueLayer(AnalyticalValueLayer):
    """
    Gaussian odor diffusion model.

    Implements odor gradients using Gaussian distributions centered
    at each source. Standard choice for analytical odor modeling.

    Attributes:
        odorscape: Algorithm type (default: "Gaussian")

    Example:
        >>> gaussian = GaussianValueLayer(sources=[source1, source2])
        >>> odor_value = gaussian.get_value((0.5, 0.5))
    """

    odorscape = param.Selector(default="Gaussian")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def get_value(self, pos: tuple[float, float]) -> float:
        value = 0
        for s in self.sources:
            p = s.get_position()
            rel_pos = [pos[0] - p[0], pos[1] - p[1]]
            value += s.odor.gaussian_value(rel_pos)
        return value


class DiffusionValueLayer(OdorScape):
    """
    Physical odor diffusion model with evaporation and wind.

    Implements realistic odor plume dynamics using Gaussian filtering
    for diffusion, evaporation decay, and optional wind-driven advection.
    Computationally intensive but physically accurate.

    Attributes:
        odorscape: Algorithm type (default: "Diffusion")
        evap_const: Evaporation decay factor per timestep (default: 0.9)
        gaussian_sigma: Gaussian filter sigma for diffusion (default: (0.95, 0.95))

    Note:
        Diffusion coefficients for gas-phase molecules: 10^-6 to 10^-5 m²/s.
        Sigma is automatically scaled based on grid resolution.

    Example:
        >>> diffusion = DiffusionValueLayer(
        ...     sources=[source], evap_const=0.95, gaussian_sigma=(1.0, 1.0)
        ... )
        >>> diffusion.update_values()  # Called each simulation step
    """

    odorscape = param.Selector(default="Diffusion")
    evap_const = param.Magnitude(
        0.9, doc="The evaporation constant of the diffusion algorithm."
    )
    gaussian_sigma = param.NumericTuple(
        (0.95, 0.95), doc="The sigma of the gaussian difusion algorithm."
    )

    """
        A typical diffusion coefficient for a molecule in the gas phase is in the range of 10-6 to 10-5 m2/sigma

        Yes, it does that automatically based on the sigma and truncate parameters.
        Indeed, the function gaussian_filter is implemented by applying multiples 1D gaussian filters (you can see that here).
        This function uses gaussian_filter1d which generate itself the kernel using _gaussian_kernel1d with a radius of
        int(truncate * sigma + 0.5).

        Doing the math, sigma ends up reeeeally small
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(subclass_initialized=True, **kwargs)

    def update_values(self) -> None:
        k = 1000
        if self.model.windscape is not None:
            v, a = self.model.windscape.wind_speed, self.model.windscape.wind_direction
            if v != 0:
                dx = v * np.cos(a) * self.model.dt
                dy = v * np.sin(a) * self.model.dt
                Px, Py = dx / self.x / k, dy / self.y / k

                Pr = np.abs(Px / (Px + Py))
                Px = np.clip(Px, a_min=-Pr, a_max=Pr)
                Py = np.clip(Py, a_min=-1 + Pr, a_max=1 - Pr)
                Gx = self.grid * Px
                Gy = self.grid * Py
                Gx = np.roll(Gx, 1, axis=0)
                Gx[0, :] = 0
                Gy = np.roll(Gy, 1, axis=1)
                Gy[:, 0] = 0
                self.grid *= 1 - Px - Py
                self.grid += Gx + Gy
                np.clip(self.grid, a_min=0, a_max=None)

        for s in self.sources:
            self.add_value(s.get_position(), s.odor.intensity)

        self.grid = (
            gaussian_filter(self.grid, sigma=self.gaussian_sigma) * self.evap_const
        )


class WindScape(SpatialEntity):
    """
    Wind and air-puff simulation layer.

    Models constant wind or pulsed air-puff stimuli affecting agent navigation.
    Supports wind direction, speed, temporal puffs, and obstacle occlusion.
    Visualized with animated arrow lines.

    Attributes:
        unique_id: Grid identifier (default: "WindScape")
        color: Visualization color (default: red)
        wind_direction: Wind direction in radians (default: π)
        wind_speed: Wind velocity magnitude (default soft-max: 100)
        puffs: Dict of scheduled air-puff events with timing parameters
        scapelines: Arrow line positions for visualization

    Example:
        >>> wind = WindScape(wind_direction=0.0, wind_speed=10.0)
        >>> wind.add_puff(duration=5.0, speed=20.0, start_time=10.0)
        >>> value = wind.get_value(agent)
    """

    unique_id = param.String("WindScape")
    color = param.Color(default="red")
    wind_direction = Phase(
        np.pi, doc="The absolute polar direction of the wind/air puff."
    )
    wind_speed = PositiveNumber(softmax=100.0, doc="The speed of the wind/air puff.")
    puffs = param.Parameter(
        {}, label="air-puffs", doc="Repetitive or single air-puff stimuli."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.max_dim = np.max(self.model.space.dims)

        self.N = 40
        self.draw_phi = 0
        self.scapelines = self.generate_scapelines(
            self.max_dim, self.N, self.wind_direction
        )
        self.events = {}
        for idx, puff in self.puffs.items():
            self.add_puff(**puff)

    def get_value(self, agent: Any) -> float:
        if self.obstructed(agent.pos):
            return 0
        else:
            o = np.rad2deg(agent.get_orientation())
            return (
                np.abs(util.angle_dif(o, self.wind_direction)) / 180 * self.wind_speed
            )

    def obstructed(self, pos: tuple[float, float]) -> bool:
        p0 = geometry.Point(pos)
        p1 = geometry.Point(
            p0.x - self.max_dim * math.cos(self.wind_direction),
            p0.y - self.max_dim * math.sin(self.wind_direction),
        )
        ll = geometry.LineString([p0, p1])

        return any([l.intersects(ll) for l in self.model.border_lines])

    def draw(self, v: Any, **kwargs: Any) -> None:
        if self.wind_speed > 0:
            for p0, p1 in self.scapelines:
                l = geometry.LineString([p0, p1])
                ps = [
                    l.intersection(b)
                    for b in self.model.border_lines
                    if l.intersects(b)
                ]
                if len(ps) != 0:
                    p1 = ps[
                        np.argmin([geometry.Point(p0).distance(p2) for p2 in ps])
                    ].coords[0]
                    # print(p1)
                v.draw_arrow_line(
                    p0, p1, self.color, width=0.001, phi=(self.draw_phi % 1000) / 1000
                )
        self.draw_phi += self.wind_speed

    def generate_scapelines(self, D: float, N: int, A: float) -> list[tuple[Any, Any]]:
        ds = D / N * np.sqrt(2)
        p0s = util.rotate_points_around_point(
            [(-D, (i - N / 2) * ds) for i in range(N)], -A
        )
        p1s = util.rotate_points_around_point(
            [(D, (i - N / 2) * ds) for i in range(N)], -A
        )
        return [(p0, p1) for p0, p1 in zip(p0s, p1s)]

    def set_wind_direction(self, A: float) -> None:
        self.wind_direction = A
        self.scapelines = self.generate_scapelines(
            self.max_dim, self.N, self.wind_direction
        )

    def add_puff(
        self,
        duration: float,
        speed: float,
        direction: float | None = None,
        start_time: float | None = None,
        N: int | None = 1,
        interval: float = 10.0,
    ) -> None:
        m = self.model

        Nticks = int(duration / m.dt)
        if start_time is None:
            start = m.Nticks
        else:
            start = int(start_time / m.dt)
        interval_ticks = int(interval / m.dt)
        if N is None:
            N = int(m.Nsteps / interval_ticks)
        for i in range(N):
            t0 = start + i * interval_ticks
            self.events[t0] = {"wind_speed": speed, "wind_direction": direction}
            self.events[t0 + Nticks] = {
                "wind_speed": self.wind_speed,
                "wind_direction": self.wind_direction,
            }

    def update(self) -> None:
        for t, args in self.events.items():
            wd = args["wind_direction"]
            if self.model.Nticks == t:
                if wd is not None and wd != self.wind_direction:
                    self.set_wind_direction(wd)
                self.wind_speed = args["wind_speed"]


class ThermoScape(ValueGrid):
    """
    Temperature gradient grid for thermotaxis simulations.

    Models spatial temperature distributions with multiple hot/cold sources
    using multivariate Gaussian spreads. Returns thermal gains (warming/cooling)
    at each position for agent thermosensation.

    Attributes:
        unique_id: Grid identifier (default: "ThermoScape")
        plate_temp: Baseline plate temperature in °C (default: 22)
        thermo_spread: Gaussian spread parameter for temperature sources
        thermo_sources: List of [x, y] positions for temperature sources
        thermo_source_dTemps: Temperature delta for each source (°C)
        thermoscape_layers: Multivariate normal distributions per source

    Example:
        >>> thermo = ThermoScape(
        ...     plate_temp=22,
        ...     thermo_sources=[[0.5, 0.5], [0.2, 0.8]],
        ...     thermo_source_dTemps=[8, -8]  # Hot and cold sources
        ... )
        >>> gain = thermo.get_value((0.5, 0.5))  # {'warm': ..., 'cool': ...}
    """

    unique_id = param.String("ThermoScape")

    def __init__(
        self,
        plate_temp: float = 22,
        spread: float = 0.1,
        thermo_sources: list[list[float]] | None = None,
        thermo_source_dTemps: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if thermo_source_dTemps is None:
            thermo_source_dTemps = [8, -8, 8, -8]
        if thermo_sources is None:
            thermo_sources = [[0.5, 0.05], [0.05, 0.5], [0.5, 0.95], [0.95, 0.5]]
        self.plate_temp = plate_temp
        self.thermo_sources = {str(i): o for i, o in enumerate(thermo_sources)}
        self.thermo_source_dTemps = {
            str(i): o for i, o in enumerate(thermo_source_dTemps)
        }

        if len(thermo_sources) != len(thermo_source_dTemps):
            raise ValueError  # need to raise a more informative error.

        if spread is None:
            spread = (
                0.1  # just making it so spread is my default of 0.1, if None is given.
            )
        self.thermo_spread = spread
        self.generate_thermoscape()

    def generate_thermoscape(self) -> None:
        """
        size is the length of the square arena in mm.
        rezo is the resolution with 1 being a mm, 0.1 being a 10th of a mm.
        spread is the spread put into the multivariate_normal function.
        pTem is the plate Temp i.e. the standard temperature of the plate - default is 22˚C.
        origins are the coordinate locations on the size x size plate of the heat or cold sources. type: list
        tempDiff needs to be a list the same length as origins, and determines if that source will be cold or hot and by how much.
        In other words a <ptemp> of 22 and a <origins> of [[10,20], [30,40]] and <tempDiff> of [8,-8] would make a temperature source at
        [10,20] that is ~30˚C and a source at [30,40] that is ~14˚C. The mean is taken where the temperatures of multiple sources overlap.
        """
        from scipy.stats import multivariate_normal

        rv_dict = {}
        for k in self.thermo_sources:
            rv_dict[k] = multivariate_normal(
                self.thermo_sources[k],
                [[self.thermo_spread, 0], [0, self.thermo_spread]],
            )

        self.thermoscape_layers = rv_dict

    def get_value(self, pos: tuple[float, float]) -> dict[str, float]:
        size, size2 = [1, 1]
        pos_ad = [size * pos[0], size2 * pos[1]]
        pos_temp = {}
        N = len(self.thermo_source_dTemps)
        thermo_gain = {"cool": 0, "warm": 0}

        for k in self.thermoscape_layers:
            v = self.thermoscape_layers[k]
            pos_temp[k] = (
                v.pdf(pos_ad)
                / v.pdf(self.thermo_sources[k])
                * (self.thermo_source_dTemps[k] * N)
            )  # @todo need to check if this works
            dgain = abs(pos_temp[k] / N)
            if pos_temp[k] < 0:
                thermo_gain["cool"] += dgain
            elif pos_temp[k] > 0:
                thermo_gain["warm"] += dgain
        return thermo_gain

    def get_grid(self) -> np.ndarray:
        X, Y = self.meshgrid

        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        self.max_value = np.max(V.flatten())
        return V

    def draw_isocontours(
        self, v: Any
    ):  # @todo need to make a draw function for thermogrid.
        for k in self.thermo_sources:
            p = self.thermo_sources.k
            for r in np.arange(0, 0.050, 0.01):
                pX = (p[0] + r, p[1])
                vv = self.thermo_source_dTemps[k]
                if vv < 0:
                    color2use = "blue"
                else:
                    color2use = "red"
                v.draw_circle(p, r, color2use, filled=False, width=0.0005)
                text_box = ScreenTextBox(
                    text=str(np.round(vv, 2)), color=self.color, visible=True
                )
                text_box.draw(v)
