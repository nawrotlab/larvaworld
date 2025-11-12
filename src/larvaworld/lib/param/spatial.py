from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import param
from shapely import Polygon, geometry

from .. import util
from .custom import (
    IntegerTuple,
    IntegerTuple2DRobust,
    NumericTuple2DRobust,
    PositiveNumber,
    RandomizedPhase,
    XYLine,
)
from .nested_parameter_group import NestedConf

__all__: list[str] = [
    "Pos2D",
    "Pos2DPixel",
    "RadiallyExtended",
    "OrientedPoint",
    "MobilePoint",
    "MobileVector",
    "LineExtended",
    "LineClosed",
    "Area2D",
    "Area2DPixel",
    "Area",
    "BoundedArea",
    "PosPixelRel2Point",
    "PosPixelRel2Area",
]

__displayname__ = "Spatial elements"


# class Pos2D(param.Parameterized):
class Pos2D(NestedConf):
    """
    2D spatial position with tracking of position history.

    Tracks the current position as well as the initial and last positions,
    allowing computation of displacement between positions.

    Attributes:
        pos: The xy spatial position coordinates as (x, y) tuple
        initial_pos: The initial position when object was created
        last_pos: The previous position before the most recent update

    Example:
        >>> pos2d = Pos2D(pos=(1.0, 2.0))
        >>> pos2d.set_position((3.0, 4.0))
        >>> displacement = pos2d.last_delta_pos
    """

    pos = NumericTuple2DRobust(doc="The xy spatial position coordinates")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_pos = self.pos
        self.last_pos = self.pos

    def get_position(self) -> tuple[float, float]:
        return tuple(self.pos)

    def set_position(self, pos) -> None:
        if not isinstance(pos, tuple):
            pos = tuple(pos)
        self.last_pos = self.get_position()
        self.pos = pos

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    @property
    def last_delta_pos(self) -> float:
        x0, y0 = self.last_pos
        x1, y1 = self.get_position()
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)


class Pos2DPixel(Pos2D):
    """
    2D spatial position in pixel coordinates.

    Inherits all position tracking functionality from Pos2D but constrains
    coordinates to integer pixel values instead of floating-point meters.

    Attributes:
        pos: The xy spatial position coordinates as integer (x, y) tuple

    Example:
        >>> pixel_pos = Pos2DPixel(pos=(100, 200))
        >>> pixel_pos.x  # Returns 100
    """

    pos = IntegerTuple2DRobust(doc="The xy spatial position coordinates")


class RadiallyExtended(Pos2D):
    """
    2D position with a circular spatial extent defined by radius.

    Extends a point position with a radius parameter, creating a circular
    region that can test point containment and generate shapely geometries.

    Attributes:
        radius: The spatial radius of the circular region in meters

    Example:
        >>> source = RadiallyExtended(pos=(0.0, 0.0), radius=0.01)
        >>> shape = source.get_shape()
        >>> is_inside = source.contained((0.005, 0.005))
    """

    radius = PositiveNumber(
        0.003, softmax=0.1, step=0.001, doc="The spatial radius of the source in meters"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_shape(self, scale=1) -> geometry.Point | None:
        p = self.get_position()
        return (
            geometry.Point(p).buffer(self.radius * scale)
            if not np.isnan(p).all()
            else None
        )

    def contained(self, point) -> bool:
        return (
            geometry.Point(self.get_position()).distance(geometry.Point(point))
            <= self.radius
        )


class OrientedPoint(Pos2D):
    """
    2D position with orientation and rotation tracking.

    Extends Pos2D with orientation (heading angle) and provides methods for
    rotation transformations, pose updates, and tracking orientation changes.

    Attributes:
        orientation: The absolute orientation angle in radians
        initial_orientation: The initial orientation when object was created
        last_orientation: The previous orientation before most recent update

    Example:
        >>> point = OrientedPoint(pos=(0.0, 0.0), orientation=0.0)
        >>> point.update_pose((1.0, 1.0), np.pi/4)
        >>> rotated = point.translate((1.0, 0.0))
    """

    orientation = RandomizedPhase(
        label="orientation", doc="The absolute orientation in space."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_orientation = self.orientation
        self.last_orientation = self.orientation

    @property
    def rotationMatrix(self) -> np.ndarray:
        a = -self.orientation
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def translate(self, point) -> tuple | list[tuple]:
        p = np.array(self.pos) + np.array(point) @ self.rotationMatrix
        if isinstance(point, tuple):
            return tuple(p)
        else:
            return util.np2Dtotuples(p)

    def set_orientation(self, orientation) -> None:
        self.last_orientation = self.get_orientation()
        self.orientation = orientation

    def get_orientation(self) -> float:
        return self.orientation

    def get_pose(self) -> tuple[np.ndarray, float]:
        return np.array(self.pos), self.orientation

    def update_pose(self, pos, orientation) -> None:
        self.set_position(pos)
        self.set_orientation(orientation % (np.pi * 2))

    @property
    def last_delta_orientation(self) -> float:
        a0 = self.last_orientation
        a1 = self.get_orientation()
        return a1 - a0


class MobilePoint(OrientedPoint):
    """
    Oriented point with velocity and motion tracking.

    Extends OrientedPoint with linear and angular velocities, accelerations,
    and cumulative distance tracking for moving entities.

    Attributes:
        lin_vel: Linear velocity in meters/second
        ang_vel: Angular velocity in radians/second
        ang_acc: Angular acceleration in radians/second²
        cum_dst: Cumulative distance traveled
        dst: Current distance from origin

    Example:
        >>> mobile = MobilePoint(pos=(0.0, 0.0), orientation=0.0)
        >>> mobile.update_all((1.0, 0.0), 0.1, 0.01, 0.1)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.ang_acc = 0.0
        self.cum_dst = 0.0
        self.dst = 0.0

    def get_angularvelocity(self) -> float:
        return self.ang_vel

    def get_linearvelocity(self) -> float:
        return self.lin_vel

    def set_linearvelocity(self, lin_vel) -> None:
        self.lin_vel = lin_vel

    def set_angularvelocity(self, ang_vel) -> None:
        self.ang_vel = ang_vel

    def update_all(self, pos, orientation, lin_vel, ang_vel) -> None:
        self.set_position(pos)
        self.set_orientation(orientation % (np.pi * 2))
        self.set_linearvelocity(lin_vel)
        self.set_angularvelocity(ang_vel)


class MobileVector(MobilePoint):
    """
    Mobile point with length representing a 1D body segment.

    Extends MobilePoint with a length parameter, providing front and rear
    end positions based on orientation and length.

    Attributes:
        length: The length of the body segment in meters

    Example:
        >>> vector = MobileVector(pos=(0.0, 0.0), orientation=0.0, length=0.01)
        >>> front = vector.front_end
        >>> rear = vector.rear_end
    """

    length = PositiveNumber(1, doc="The initial length of the body in meters")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def front_end(self) -> tuple:
        return self.translate((self.length / 2, 0))

    @property
    def rear_end(self) -> tuple:
        return self.translate((-self.length / 2, 0))

    def drag_to_front(self, fp, d_or=0) -> None:
        o = self.get_orientation() + d_or
        k = np.array([np.cos(o), np.sin(o)])
        p = fp - k * self.length / 2
        self.update_pose(p, o)


class LineExtended(NestedConf):
    """
    A line defined by vertices with configurable width and closure.

    Represents a polyline or polygon defined by a sequence of 2D vertices.
    Can be open or closed, and provides edge computation.

    Attributes:
        width: The width of the line for rendering/collision
        vertices: List of (x, y) coordinate tuples defining the line
        closed: Whether the line forms a closed loop

    Example:
        >>> line = LineExtended(vertices=[(0, 0), (1, 0), (1, 1)], width=0.01)
        >>> edges = line._edges
    """

    width = PositiveNumber(0.001, softmax=10.0, doc="The width of the line vertices")
    vertices = XYLine(doc="The list of 2d points")
    closed = param.Boolean(False, doc="Whether the line is closed")

    @property
    def Nvertices(self) -> int:
        return len(self.vertices)

    @property
    def _edges(self) -> list[list[tuple]]:
        vs = self.vertices
        edges = [[vs[i], vs[i + 1]] for i in range(self.Nvertices - 1)]
        if self.closed:
            edges.append([vs[self.Nvertices], vs[0]])
        return edges


class LineClosed(LineExtended):
    """
    A closed line (polygon) defined by vertices.

    Inherits from LineExtended but enforces that the line is always closed,
    forming a polygon where the last vertex connects to the first.

    Example:
        >>> polygon = LineClosed(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> polygon.closed  # Always True
    """

    closed = param.Boolean(True)


class Area2D(NestedConf):
    """
    2D rectangular area with dimension and centering parameters.

    Defines a 2D rectangular region with width and height, optionally
    centered at the origin, with utilities for position transformation.

    Attributes:
        dims: The (width, height) dimensions in meters
        centered: Whether the area is centered at (0, 0)

    Example:
        >>> area = Area2D(dims=(0.2, 0.1), centered=True)
        >>> width = area.w
        >>> height = area.h
    """

    dims = param.NumericTuple((0.1, 0.1), doc="The arena dimensions in meters")
    # dims = PositiveRange(doc='The arena dimensions')
    centered = param.Boolean(True, doc="Whether area is centered to (0,0)")

    @property
    def w(self) -> float:
        return self.dims[0]

    @property
    def h(self) -> float:
        return self.dims[1]

    @property
    def range(self) -> np.ndarray:
        X, Y = self.dims
        return np.array([-X / 2, X / 2, -Y / 2, Y / 2])

    def adjust_pos_to_area(self, pos, area, scaling_factor=1) -> tuple | None:
        if pos is None:
            return None
        if any(np.isnan(pos)):
            return (np.nan, np.nan)
        try:
            return self._transform(pos)
        except:
            s = scaling_factor
            rw, rh = self.w / area.w, self.h / area.h
            pp = (pos[0] / s * rw + self.w / 2, -pos[1] * 2 / s * rh + self.h)
            return pp


class Area2DPixel(Area2D):
    """
    2D rectangular area in pixel coordinates.

    Extends Area2D with pixel-based dimensions and pygame rendering utilities.
    Provides methods for creating pygame Rects and computing relative positions.

    Attributes:
        dims: The (width, height) dimensions in integer pixels

    Example:
        >>> pixel_area = Area2DPixel(dims=(800, 600), centered=True)
        >>> rect = pixel_area.get_rect_at_pos((100, 100))
    """

    dims = IntegerTuple((100, 100), doc="The arena dimensions in pixels")
    # dims = PositiveIntegerRange((100, 100), softmax=10000, step=1, doc='The arena dimensions in pixels')

    def get_rect_at_pos(self, pos=(0, 0), area=None, **kwargs):
        if area is not None:
            pos = self.adjust_pos_to_area(pos=pos, area=area)

        import pygame

        if pos is not None and not any(np.isnan(pos)):
            if self.centered:
                return pygame.Rect(
                    pos[0] - self.w / 2, pos[1] - self.h / 2, self.w, self.h
                )
            else:
                return pygame.Rect(pos[0], pos[1], self.w, self.h)
        else:
            return None

    def get_relative_pos(self, pos_scale, reference=None) -> tuple[int, int]:
        if reference is None:
            reference = (self.w, self.h)
        w, h = pos_scale
        x_pos = int(reference[0] * w)
        y_pos = int(reference[1] * h)
        return x_pos, y_pos

    def get_relative_font_size(self, font_size_scale) -> int:
        return int(self.w * font_size_scale)


class Area(Area2D):
    """
    2D area with configurable geometry and topology.

    Extends Area2D with shape selection (circular/rectangular) and optional
    toroidal topology for wrap-around boundaries.

    Attributes:
        geometry: The arena shape ('circular' or 'rectangular')
        torus: Whether the space has toroidal (wrap-around) boundaries

    Example:
        >>> arena = Area(dims=(0.2, 0.2), geometry='circular', torus=False)
    """

    geometry = param.Selector(
        objects=["circular", "rectangular"], doc="The arena shape"
    )
    torus = param.Boolean(False, doc="Whether to allow a toroidal space")


class PosPixelRel2Point(Pos2DPixel):
    """
    Pixel position defined relative to a reference point.

    Automatically computes position as a scaled fraction of a reference
    point's coordinates, updating when the scale or reference changes.

    Attributes:
        reference_point: The reference Pos2DPixel instance
        pos_scale: Scale factors (w, h) applied to reference coordinates

    Example:
        >>> ref = Pos2DPixel(pos=(100, 200))
        >>> rel_pos = PosPixelRel2Point(reference_point=ref, pos_scale=(0.5, 0.5))
        >>> rel_pos.pos  # (50, 100)
    """

    reference_point = param.ClassSelector(
        class_=Pos2DPixel, doc="The reference position instance", is_instance=False
    )
    pos_scale = param.NumericTuple(
        (0.5, 0.5), doc="The position relative to reference position"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends("pos_scale", "reference_point", watch=True)
    def update_pos(self) -> None:
        w, h = self.pos_scale
        x_pos = int(self.reference_point.x * w)
        y_pos = int(self.reference_point.y * h)
        self.pos = (x_pos, y_pos)


class PosPixelRel2Area(Pos2DPixel):
    """
    Pixel position defined relative to an area's dimensions.

    Automatically computes position as a scaled fraction of a reference
    area's width and height, updating when the scale or area changes.

    Attributes:
        reference_area: The reference Area2DPixel instance
        pos_scale: Scale factors (w, h) applied to area dimensions

    Example:
        >>> area = Area2DPixel(dims=(800, 600))
        >>> rel_pos = PosPixelRel2Area(reference_area=area, pos_scale=(0.5, 0.5))
        >>> rel_pos.pos  # (400, 300) - center of area
    """

    reference_area = param.ClassSelector(
        class_=Area2DPixel, doc="The reference position instance", is_instance=True
    )
    pos_scale = param.NumericTuple(
        (0.5, 0.5), doc="The position relative to reference position"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends("pos_scale", "reference_area", watch=True)
    def update_pos(self) -> None:
        w, h = self.pos_scale
        x_pos = int(self.reference_area.w * w)
        y_pos = int(self.reference_area.h * h)
        self.pos = (x_pos, y_pos)


class BoundedArea(Area, LineClosed):
    """
    Area with explicit boundary vertices forming a closed polygon.

    Combines Area and LineClosed to create an arena with defined geometry
    that can test point containment within its boundary polygon.

    Attributes:
        boundary_margin: Margin width relative to vertices (default 1.0)

    Example:
        >>> arena = BoundedArea(dims=(0.2, 0.2), geometry='circular')
        >>> is_inside = arena.in_area((0.05, 0.05))
        >>> polygon = arena.polygon
    """

    boundary_margin = param.Magnitude(
        1.0, doc="The boundary margin width relative to the area vertices"
    )

    def __init__(self, vertices=None, **kwargs):
        Area.__init__(self, **kwargs)
        X, Y = self.dims
        if vertices is None:
            if self.geometry == "circular":
                # This is a circle_to_polygon shape from the function
                vertices = util.circle_to_polygon(60, X / 2)
            elif self.geometry == "rectangular":
                # This is a rectangular shape
                vertices = [
                    (-X / 2, -Y / 2),
                    (-X / 2, Y / 2),
                    (X / 2, Y / 2),
                    (X / 2, -Y / 2),
                ]
        LineClosed.__init__(self, vertices=vertices, **kwargs)

    @property
    def polygon(self) -> Polygon:
        return Polygon(np.array(self.vertices) * self.boundary_margin)

    def in_area(self, p) -> bool:
        return self.polygon.contains(geometry.Point(p))
