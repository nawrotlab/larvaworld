from __future__ import annotations
from typing import Any
import numpy as np
import param
from shapely import geometry

# import pygame
from shapely.affinity import affine_transform

from ... import util
from ...model import GroupedObject
from ...param import Pos2D, ViewableLine
from .. import Object

__all__: list[str] = [
    "Obstacle",
    "Box",
    "Wall",
    "Border",
]


class Obstacle(GroupedObject, ViewableLine):
    """
    Base class for obstacles in simulation environment.

    Combines grouped object properties with viewable line visualization
    for rendering obstacles (walls, boxes, borders) in the arena.

    Attributes:
        model: Reference to simulation model
        edges: List of edge line segments defining obstacle boundaries

    Example:
        >>> obstacle = Obstacle(model=sim_model, edges=[[p1, p2], [p2, p3]])
    """

    def __init__(
        self, model: Any | None = None, edges: Any | None = None, **kwargs: Any
    ) -> None:
        Object.__init__(self, model=model)
        ViewableLine.__init__(self, **kwargs)

        # self.vertices = vertices
        self.edges = edges

    # def draw(self, viewer):
    #     viewer.draw_polyline(vertices=self.vertices,color=self.color,width=self.width,closed=True)


class Box(Obstacle, Pos2D):
    """
    Rectangular box obstacle.

    Creates a square or rectangular obstacle centered at position (x, y)
    with specified size. Useful for creating containment areas or barriers.

    Attributes:
        x: Center x-coordinate
        y: Center y-coordinate
        size: Box side length
        closed: Whether box forms closed boundary (default: True)

    Example:
        >>> box = Box(x=0.5, y=0.5, size=0.2, color="black")
    """

    closed = param.Boolean(True)

    def __init__(
        self, x: float = 0, y: float = 0, size: float = 1, **kwargs: Any
    ) -> None:
        # self.x = x
        # self.y = y
        self.size = size

        vert1 = geometry.Point(x - size / 2, y - size / 2)
        vert2 = geometry.Point(x + size / 2, y - size / 2)
        vert3 = geometry.Point(x + size / 2, y + size / 2)
        vert4 = geometry.Point(x - size / 2, y + size / 2)

        vertices = [
            (vert1.x, vert1.y),
            (vert2.x, vert2.y),
            (vert3.x, vert3.y),
            (vert4.x, vert4.y),
        ]
        edges = [[vert1, vert2], [vert2, vert3], [vert3, vert4], [vert4, vert1]]
        super().__init__(vertices=vertices, edges=edges, pos=(x, y), **kwargs)


class Wall(Obstacle):
    """
    Linear wall obstacle between two points.

    Creates a straight wall segment from point1 to point2. Does not form
    closed boundary. Useful for maze construction and partial barriers.

    Attributes:
        point1: Starting point of wall segment
        point2: Ending point of wall segment
        closed: Whether wall forms closed boundary (default: False)

    Example:
        >>> wall = Wall(point1=Point(0, 0), point2=Point(1, 0), color="gray")
    """

    closed = param.Boolean(False)

    def __init__(
        self,
        point1: geometry.Point = geometry.Point(0, 0),
        point2: geometry.Point = geometry.Point(-1, 1),
        **kwargs: Any,
    ) -> None:
        self.point1 = point1
        self.point2 = point2

        vertices = [(point1.x, point1.y), (point2.x, point2.y)]
        edges = [[point1, point2]]
        super().__init__(vertices=vertices, edges=edges, **kwargs)


class Border(Obstacle):
    """
    Complex border obstacle from vertex list.

    Creates multi-segment border from sequence of vertices. Each consecutive
    vertex pair forms an edge. Supports scaling and coordinate transformations
    via define_lines method.

    Attributes:
        vertices: List of vertex coordinates defining border path
        points: Alternative point specification (optional)
        border_xy: Transformed vertex coordinates
        border_lines: LineString objects for each border segment
        closed: Whether border forms closed loop (default: False)

    Example:
        >>> border = Border(vertices=[(0, 0), (1, 0), (1, 1), (0, 1)])
        >>> is_inside = border.contained((0.5, 0.5))
    """

    closed = param.Boolean(False)

    def __init__(
        self, vertices: list[Any] = [], points: Any | None = None, **kwargs: Any
    ) -> None:
        self.points = points
        self.border_xy, self.border_lines = self.define_lines(vertices)
        edges = []
        # vertices = self.border_xy
        for l in self.border_lines:
            (x1, y1), (x2, y2) = list(l.coords)
            point1 = geometry.Point(x1, y1)
            point2 = geometry.Point(x2, y2)
            edges.append([point1, point2])

        super().__init__(vertices=vertices, edges=edges, **kwargs)

    def define_lines(
        self, vertices: list[Any], s: float = 1
    ) -> tuple[list[Any], list[geometry.LineString]]:
        # print(points)
        # print(len(points))

        lines = [
            geometry.LineString([tuple(p1), tuple(p2)])
            for (p1, p2) in util.SuperList(vertices).in_pairs
        ]

        T = [s, 0, 0, s, 0, 0]
        ls = [affine_transform(l, T) for l in lines]
        ps = [l.coords.xy for l in ls]
        xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return xy, ls

    def contained(self, p: Any) -> bool:
        return any(
            [l.distance(geometry.Point(p)) < self.width for l in self.border_lines]
        )
