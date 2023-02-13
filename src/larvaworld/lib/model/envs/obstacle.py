import numpy as np
import pygame
from shapely.affinity import affine_transform
from shapely import geometry

from larvaworld.lib import aux


class Obstacle:

    def __init__(self, vertices, edges, width=0.001, color='black', unique_id=None):
        self.vertices = vertices
        self.edges = edges
        self.width = width
        self.color = color
        self.unique_id = unique_id

        self.selected = False
    def draw(self, viewer):
        viewer.draw_polyline(vertices=self.vertices,color=self.color,width=self.width,closed=True)



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
        super().__init__(vertices, edges, **kwargs)

    # def get_saved_scene_repr(self):
    #     return self.__class__.__name__ + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.size)

    # def draw_label(self, screen):
    #     if pygame.font and self.unique_id is not None:
    #         font = pygame.font.Font(None, 24)
    #         text = font.render(str(self.unique_id), 1, aux.Color.YELLOW, aux.Color.DARK_GRAY)
    #         text_pos = pygame.Rect(self.x, self.y, 50, 50)
    #         screen.blit(text, text_pos)


class Wall(Obstacle):

    def __init__(self, point1, point2, **kwargs):
        self.point1 = point1
        self.point2 = point2

        vertices = [(point1.x, point1.y), (point2.x, point2.y)]
        edges = [[point1, point2]]
        super().__init__(vertices, edges, **kwargs)

    # def get_saved_scene_repr(self):
    #     return self.__class__.__name__ + ' ' + str(self.point1.x) + ' ' + str(self.point1.y) \
    #            + ' ' + str(self.point2.x) + ' ' + str(self.point2.y)

    # def draw_label(self, screen):
    #     if pygame.font and self.unique_id is not None:
    #         font = pygame.font.Font(None, 24)
    #         text = font.render(str(self.unique_id), 1, aux.Color.YELLOW, aux.Color.DARK_GRAY)
    #         rect_x = (self.point1.x + self.point2.x) / 2
    #         rect_y = (self.point1.y + self.point2.y) / 2
    #
    #         text_pos = pygame.Rect(rect_x, rect_y, 50, 50)
    #         screen.blit(text, text_pos)


class Border(Obstacle):
    def __init__(self, points=None, unique_id='Border', width=0.001, default_color='black'):
        self.width = width
        self.points = points
        self.border_xy, self.border_lines = self.define_lines(points)
        edges = []
        vertices = self.border_xy
        for l in self.border_lines:
            (x1, y1), (x2, y2) = list(l.coords)
            point1 = geometry.Point(x1, y1)
            point2 = geometry.Point(x2, y2)
            edges.append([point1, point2])


        self.selected = False
        super().__init__(vertices, edges, color=default_color, unique_id =unique_id)

    def define_lines(self, points, s=1):
        lines = [geometry.LineString([tuple(p1), tuple(p2)]) for p1, p2 in aux.group_list_by_n(points, 2)]

        T = [s, 0, 0, s, 0, 0]
        ls = [affine_transform(l, T) for l in lines]
        ps = [l.coords.xy for l in ls]
        xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return xy, ls

    def draw(self, screen):
        for b in self.border_xy:
            screen.draw_polyline(b, color=self.color, width=self.width, closed=False)
            # if self.selected:
            #     screen.draw_polyline(b, color=self.model.selection_color, width=self.width * 0.5, closed=False)

    def contained(self, p):
        return any([l.distance(geometry.Point(p)) < self.width for l in self.border_lines])

    def set_id(self, id):
        self.unique_id = id


