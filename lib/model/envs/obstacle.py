import numpy as np
import pygame
# from Geometry import Point
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon

from lib.aux import dictsNlists as dNl, ang_aux, sim_aux, shapely_aux
from lib.aux.colsNstr import colorname2tuple, Color


class Obstacle:

    def __init__(self, vertices, edges, default_color, unique_id=None):
        self.vertices = vertices
        self.edges = edges
        self.default_color = default_color
        self.unique_id = unique_id

    def draw(self, viewer):
        viewer.draw_polyline(vertices=self.vertices,color=self.default_color,closed=True)



class Box(Obstacle):

    def __init__(self, x, y, size, default_color):
        self.x = x
        self.y = y
        self.size = size

        vert1 = shapely_aux.Point(x - size / 2, y - size / 2)
        vert2 = shapely_aux.Point(x + size / 2, y - size / 2)
        vert3 = shapely_aux.Point(x + size / 2, y + size / 2)
        vert4 = shapely_aux.Point(x - size / 2, y + size / 2)

        vertices = [(vert1.x, vert1.y), (vert2.x, vert2.y), (vert3.x, vert3.y), (vert4.x, vert4.y)]
        edges = [[vert1, vert2], [vert2, vert3], [vert3, vert4], [vert4, vert1]]
        super().__init__(vertices, edges, default_color)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.size)

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x, self.y, 50, 50)
            screen.blit(text, text_pos)


class Wall(Obstacle):

    def __init__(self, point1, point2, default_color):
        self.point1 = point1
        self.point2 = point2

        vertices = [(point1.x, point1.y), (point2.x, point2.y)]
        edges = [[point1, point2]]
        super().__init__(vertices, edges, default_color)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.point1.x) + ' ' + str(self.point1.y) \
               + ' ' + str(self.point2.x) + ' ' + str(self.point2.y)

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            rect_x = (self.point1.x + self.point2.x) / 2
            rect_y = (self.point1.y + self.point2.y) / 2

            text_pos = pygame.Rect(rect_x, rect_y, 50, 50)
            screen.blit(text, text_pos)


class Border(Obstacle):
    def __init__(self, points=None, unique_id='Border', width=0.001, default_color='black'):


        # from lib.model.space.obstacle import Wall
        # self.model = model
        # if type(default_color) == str:
        #     default_color = colorname2tuple(default_color)
        # self.default_color = default_color
        # self.unique_id = unique_id
        self.width = width
        self.points = points

        self.border_xy, self.border_lines = self.define_lines(points)
        # self.border_bodies = []
        edges = []
        vertices = self.border_xy
        # self.border_walls = []
        for l in self.border_lines:
            # print(list(l.coords))
            (x1, y1), (x2, y2) = list(l.coords)
            point1 = shapely_aux.Point(x1, y1)
            point2 = shapely_aux.Point(x2, y2)
            edges.append([point1, point2])

            # wall = Wall(point1, point2, color=self.default_color)
            # edges = [[point1, point2]]
            # self.border_walls.append(wall)

        self.selected = False
        super().__init__(vertices, edges, default_color, unique_id)

    def define_lines(self, points, s=1):
        lines = [LineString([tuple(p1), tuple(p2)]) for p1, p2 in dNl.group_list_by_n(points, 2)]

        T = [s, 0, 0, s, 0, 0]
        ls = [affine_transform(l, T) for l in lines]
        ps = [l.coords.xy for l in ls]
        xy = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]
        return xy, ls

    def draw(self, screen):
        for b in self.border_xy:
            screen.draw_polyline(b, color=self.default_color, width=self.width, closed=False)
            # if self.selected:
            #     screen.draw_polyline(b, color=self.model.selection_color, width=self.width * 0.5, closed=False)

    def contained(self, p):
        return any([l.distance(shapely_aux.Point(p)) < self.width for l in self.border_lines])

    def set_id(self, id):
        self.unique_id = id


class Arena(Obstacle):
    def __init__(self, arena_dims=(0.1,0.1), arena_shape='circular',vertices=None, unique_id='Arena', default_color='black', k=0.96):

        X, Y = self.dims = arena_dims
        self.range = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        if vertices is None :
            if arena_shape == 'circular':
                # This is a circle_to_polygon shape from the function
                vertices = sim_aux.circle_to_polygon(60, X / 2)
            elif arena_shape == 'rectangular':
                # This is a rectangular shape
                vertices = np.array([(-X / 2, -Y / 2),
                                          (-X / 2, Y / 2),
                                          (X / 2, Y / 2),
                                          (X / 2, -Y / 2)])
            else :
                raise
        self.polygon = Polygon(vertices * k)
        edges = [[shapely_aux.Point(x1,y1), shapely_aux.Point(x2,y2)] for (x1,y1), (x2,y2) in dNl.group_list_by_n(vertices, 2)]
        super().__init__(vertices, edges, default_color, unique_id)