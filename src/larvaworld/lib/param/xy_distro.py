from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import param

from .custom import OptionalPositiveInteger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nested_parameter_group import NestedConf
else:
    from .nested_parameter_group import NestedConf

__all__: list[str] = [
    "Spatial_Distro",
    "Larva_Distro",
    "xy_along_circle",
    "xy_along_rect",
    "xy_uniform_circle",
    "xy_grid",
    "generate_xy_distro",
    "generate_xyNor_distro",
]

__displayname__ = "2D spatial distributions"


class Spatial_Distro(NestedConf):
    """
    2D spatial distribution configuration for agent placement.

    Defines how agents are distributed in 2D space using various shapes
    (circle, rectangle, oval) and placement modes (uniform, normal, periphery, grid).

    Attributes:
        shape: Distribution shape ('circle', 'rect', 'oval', 'rectangular')
        mode: Placement mode ('uniform', 'normal', 'periphery', 'grid')
        N: Number of agents to place
        loc: Center coordinates (x, y) of the distribution
        scale: Spread or radius in (x, y) dimensions

    Example:
        >>> distro = Spatial_Distro(shape='circle', mode='uniform', N=50)
        >>> positions = distro()
    """

    shape = param.Selector(
        objects=["circle", "rect", "oval", "rectangular"],
        doc="The shape of the spatial distribution",
    )
    mode = param.Selector(
        objects=["uniform", "normal", "periphery", "grid"],
        doc="The way to place agents in the distribution shape",
    )
    N = OptionalPositiveInteger(
        30, softmax=100, doc="The number of agents in the group"
    )
    loc = param.NumericTuple(
        default=(0.0, 0.0), doc="The xy coordinates of the distribution center"
    )
    # loc = param.Range(default=(0.0, 0.0), softbounds=(-0.1, 0.1),step=0.001, doc='The xy coordinates of the distribution center')
    scale = param.NumericTuple(default=(0.0, 0.0), doc="The spread in x,y")
    # scale = param.Range(default=(0.0, 0.0), softbounds=(-0.1, 0.1),step=0.001, doc='The spread in x,y')

    def __call__(self) -> List[Tuple[float, float]]:
        return generate_xy_distro(
            mode=self.mode, shape=self.shape, N=self.N, loc=self.loc, scale=self.scale
        )

    def draw(self) -> None:
        import matplotlib.pyplot as plt

        ps = generate_xy_distro(
            mode=self.mode, shape=self.shape, N=self.N, loc=self.loc, scale=self.scale
        )
        ps = np.array(ps)
        plt.scatter(ps[:, 0], ps[:, 1])
        # plt.axis('equal')
        plt.xlim(-0.2, 0.2)
        plt.ylim(-0.2, 0.2)
        plt.show()
        # return ps


class Larva_Distro(Spatial_Distro):
    """
    Spatial distribution with orientation for larva agents.

    Extends Spatial_Distro to include random orientation angles for larvae,
    generating both positions and orientations.

    Attributes:
        orientation_range: Range of orientations to sample from, in degrees

    Example:
        >>> distro = Larva_Distro(N=20, orientation_range=(0, 180))
        >>> positions, orientations = distro()
    """

    orientation_range = param.Range(
        default=(0.0, 360.0),
        bounds=(-360.0, 360.0),
        step=1,
        doc="The range of larva body orientations to sample from, in degrees",
    )

    def __call__(self) -> Tuple[List[Tuple[float, float]], List[float]]:
        return generate_xyNor_distro(self)


def single_parametric_interpolate(
    obj_x_loc: Sequence[float], obj_y_loc: Sequence[float], numPts: int = 50
) -> List[Tuple[float, float]]:
    n = len(obj_x_loc)
    vi = [
        [obj_x_loc[(i + 1) % n] - obj_x_loc[i], obj_y_loc[(i + 1) % n] - obj_y_loc[i]]
        for i in range(n)
    ]
    si = [np.linalg.norm(v) for v in vi]
    di = np.linspace(0, sum(si), numPts, endpoint=False)
    new_points = []
    for d in di:
        for i, s in enumerate(si):
            if d > s:
                d -= s
            else:
                break
        l = d / s
        new_points.append((obj_x_loc[i] + l * vi[i][0], obj_y_loc[i] + l * vi[i][1]))
    return new_points


def xy_along_circle(
    N: int, loc: Tuple[float, float], radius: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Generate N points evenly distributed along a circle/ellipse periphery.

    Args:
        N: Number of points to generate
        loc: Center coordinates (x, y)
        radius: Radius in (x, y) dimensions for ellipse

    Returns:
        List of (x, y) coordinate tuples along the periphery
    """
    X, Y = loc
    dX, dY = radius
    angles = np.linspace(0, np.pi * 2, N + 1)[:-1]
    p = [(X + np.cos(a) * dX, Y + np.sin(a) * dY) for a in angles]
    return p


def xy_along_rect(
    N: int, loc: Tuple[float, float], scale: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Generate N points evenly distributed along rectangle periphery.

    Args:
        N: Number of points to generate
        loc: Center coordinates (x, y)
        scale: Half-dimensions (half-width, half-height)

    Returns:
        List of (x, y) coordinate tuples along the rectangle edges
    """
    X, Y = loc
    dX, dY = scale
    rext_x = [X + x for x in [-dX, dX, dX, -dX]]
    rext_y = [Y + y for y in [-dY, -dY, dY, dY]]
    p = single_parametric_interpolate(rext_x, rext_y, numPts=N)
    return p


def xy_uniform_circle(
    N: int, loc: Tuple[float, float], scale: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Generate N points uniformly distributed within a circle/ellipse.

    Args:
        N: Number of points to generate
        loc: Center coordinates (x, y)
        scale: Radius in (x, y) dimensions

    Returns:
        List of (x, y) coordinate tuples within the circle
    """
    X, Y = loc
    dX, dY = scale
    angles = np.random.uniform(0, 2 * np.pi, N).tolist()
    xs = np.random.uniform(0, dX**2, N) ** 0.5 * np.cos(angles)
    ys = np.random.uniform(0, dY**2, N) ** 0.5 * np.sin(angles)
    p = [(X + x, Y + y) for a, x, y in zip(angles, xs, ys)]
    return p


def xy_grid(
    grid_dims: Tuple[int, int],
    area: Tuple[float, float],
    loc: Tuple[float, float] = (0.0, 0.0),
) -> List[Tuple[float, float]]:
    """
    Generate points arranged in a regular grid pattern.

    Args:
        grid_dims: Grid dimensions (Nx, Ny)
        area: Total area dimensions (width, height)
        loc: Center coordinates (x, y)

    Returns:
        List of (x, y) coordinate tuples in grid arrangement
    """
    X, Y = loc
    W, H = area
    Nx, Ny = grid_dims
    dx, dy = W / Nx, H / Ny
    grid = np.meshgrid(
        np.linspace(X - W / 2 + dx / 2, X + W / 2 + dx / 2, Nx),
        np.linspace(Y - H / 2 + dy / 2, Y + H / 2 + dy / 2, Ny),
    )
    cartprod = np.stack(grid, axis=-1).reshape(-1, 2)

    # Convert to list of tuples
    return list(map(tuple, cartprod))


def generate_xy_distro(
    mode: str,
    shape: str,
    N: int | Tuple[int, int],
    loc: Tuple[float, float] = (0.0, 0.0),
    scale: Tuple[float, float] = (0.0, 0.0),
    area: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float]]:
    """
    Generate 2D spatial distribution of N points.

    Main distribution generator supporting multiple shapes and placement modes.

    Args:
        mode: Placement mode ('uniform', 'normal', 'periphery', 'grid')
        shape: Distribution shape ('circle', 'oval', 'rect', 'rectangular')
        N: Number of points (or grid dimensions for grid mode)
        loc: Center coordinates (x, y)
        scale: Spread/radius in (x, y) dimensions
        area: Area dimensions for grid mode (defaults to scale)

    Returns:
        List of (x, y) coordinate tuples

    Example:
        >>> positions = generate_xy_distro('uniform', 'circle', 100, (0,0), (0.05, 0.05))
    """
    loc, scale = np.array(loc), np.array(scale)
    if mode == "uniform":
        if shape in ["circle", "oval"]:
            return xy_uniform_circle(N=N, loc=loc, scale=scale)
        elif shape == "rect":
            return list(
                map(tuple, np.random.uniform(low=-scale, high=scale, size=(N, 2)) + loc)
            )
    elif mode == "normal":
        return np.random.normal(loc=loc, scale=scale / 2, size=(N, 2)).tolist()
    elif mode == "periphery":
        if shape in ["circle", "oval"]:
            return xy_along_circle(N, loc=loc, radius=scale)
        elif shape == "rect":
            return xy_along_rect(N, loc=loc, scale=scale)
    elif mode == "grid":
        if type(N) == tuple:
            grid_dims = N
        else:
            Nx = int(np.sqrt(N))
            Ny = int(N / Nx)
            if Nx * Ny != N:
                raise
            grid_dims = (Nx, Ny)
        if area is None:
            area = scale
        return xy_grid(grid_dims, loc=loc, area=area)
    else:
        raise ValueError(f"XY distribution {mode} not implemented.")


def generate_xyNor_distro(
    d: "Larva_Distro",
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Generate positions and orientations for larva distribution.

    Args:
        d: Larva_Distro configuration object

    Returns:
        Tuple of (positions, orientations) where positions is list of (x,y)
        tuples and orientations is list of angles in radians

    Example:
        >>> distro = Larva_Distro(N=20, orientation_range=(0, 180))
        >>> positions, orientations = generate_xyNor_distro(distro)
    """
    N = d.N
    a1, a2 = np.deg2rad(d.orientation_range)
    ors = (np.random.uniform(low=a1, high=a2, size=N) % (2 * np.pi)).tolist()
    ps = generate_xy_distro(N=N, mode=d.mode, shape=d.shape, loc=d.loc, scale=d.scale)
    return ps, ors
