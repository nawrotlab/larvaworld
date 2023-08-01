import numpy as np
import param
#import pygame
from shapely.affinity import affine_transform
from shapely import geometry

from larvaworld.lib import aux
from larvaworld.lib.model import NamedObject
from larvaworld.lib.param import Contour


class Obstacle(NamedObject, Contour):

    def __init__(self,model,edges=None,**kwargs):
        NamedObject.__init__(self,model=model)
        Contour.__init__(self,**kwargs)

        # self.vertices = vertices
        self.edges = edges

    # def draw(self, viewer):
    #     viewer.draw_polyline(vertices=self.vertices,color=self.color,width=self.width,closed=True)



class Box(Obstacle):

    def __init__(self, x, y, size, **kwargs):
        self.x = x
        self.y = y
        self.size = size

        vert1 = geometry.Point(x - size / 2, y - size / 2)
        vert2 = geometry.Point(x + size / 2, y - size / 2)
        vert3 = geometry.Point(x + size / 2, y + size / 2)
        vert4 = geometry.Point(x - size / 2, y + size / 2)

        vertices = [(vert1.x, vert1.y), (vert2.x, vert2.y), (vert3.x, vert3.y), (vert4.x, vert4.y)]
        edges = [[vert1, vert2], [vert2, vert3], [vert3, vert4], [vert4, vert1]]
        super().__init__(vertices=vertices, edges = edges, **kwargs)




class Wall(Obstacle):

    def __init__(self, point1, point2, **kwargs):
        self.point1 = point1
        self.point2 = point2

        vertices = [(point1.x, point1.y), (point2.x, point2.y)]
        edges = [[point1, point2]]
        super().__init__(vertices =vertices, edges =edges,closed=False, **kwargs)



class Border(Obstacle):
    def __init__(self, points=None, **kwargs):
        self.points = points
        self.border_xy, self.border_lines = self.define_lines(points)
        edges = []
        vertices = self.border_xy
        for l in self.border_lines:
            (x1, y1), (x2, y2) = list(l.coords)
            point1 = geometry.Point(x1, y1)
            point2 = geometry.Point(x2, y2)
            edges.append([point1, point2])


        super().__init__(vertices =vertices, edges =edges,closed=False, **kwargs)

    def define_lines(self, points, s=1):
        lines = [geometry.LineString([tuple(p1), tuple(p2)]) for p1, p2 in aux.group_list_by_n(points, 2)]

        T = [s, 0, 0, s, 0, 0]
        ls = [affine_transform(l, T) for l in lines]
        ps = [l.coords.xy for l in ls]
        xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return xy, ls



    def contained(self, p):
        return any([l.distance(geometry.Point(p)) < self.width for l in self.border_lines])




