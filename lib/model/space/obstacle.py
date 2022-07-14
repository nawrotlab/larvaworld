import pygame

from lib.aux.color_util import Color
from lib.aux import dictsNlists as dNl, ang_aux, sim_aux, shapely_aux

class Obstacle:

    def __init__(self, vertices, edges, color):
        self.vertices = vertices
        self.edges = edges
        self.color = color

    def draw(self, scene):
        pygame.draw.polygon(scene.screen, self.color, self.vertices, 1)



class Box(Obstacle):

    def __init__(self, x, y, size, color):
        self.x = x
        self.y = y
        self.size = size
        self.label = None

        vert1 = shapely_aux.Point(x - size / 2, y - size / 2)
        vert2 = shapely_aux.Point(x + size / 2, y - size / 2)
        vert3 = shapely_aux.Point(x + size / 2, y + size / 2)
        vert4 = shapely_aux.Point(x - size / 2, y + size / 2)

        vertices = [(vert1.x, vert1.y), (vert2.x, vert2.y), (vert3.x, vert3.y), (vert4.x, vert4.y)]
        edges = [[vert1, vert2], [vert2, vert3], [vert3, vert4], [vert4, vert1]]
        super().__init__(vertices, edges, color)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.size)

    def draw_label(self, screen):
        if pygame.font and self.label is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.label), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x, self.y, 50, 50)
            screen.blit(text, text_pos)


class Wall(Obstacle):

    def __init__(self, point1, point2, color):
        self.point1 = point1
        self.point2 = point2
        self.label = None

        vertices = [(point1.x, point1.y), (point2.x, point2.y)]
        edges = [[point1, point2]]
        super().__init__(vertices, edges, color)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.point1.x) + ' ' + str(self.point1.y) \
               + ' ' + str(self.point2.x) + ' ' + str(self.point2.y)

    def draw_label(self, screen):
        if pygame.font and self.label is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.label), 1, Color.YELLOW, Color.DARK_GRAY)
            rect_x = (self.point1.x + self.point2.x) / 2
            rect_y = (self.point1.y + self.point2.y) / 2

            text_pos = pygame.Rect(rect_x, rect_y, 50, 50)
            screen.blit(text, text_pos)
