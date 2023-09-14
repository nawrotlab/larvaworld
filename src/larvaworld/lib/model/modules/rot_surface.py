import math
import pygame

from larvaworld.lib import aux


__all__ = [
    'RotSurface',
    'RotTriangle',
    'LightSource',
]

class RotSurface:

    def __init__(self, x, y, direction, surf, model=None):
        self.x = x
        self.y = y
        self.direction = direction
        self.surf = surf
        self.speed = 0
        self.model = model

    def step(self):
        dx = self.speed * math.cos(self.direction)
        dy = self.speed * math.sin(self.direction)
        self.x += dx
        self.y += dy

    def draw(self, viewer):
        degrees = math.degrees(self.direction)
        rotated_surf = pygame.transform.rotate(self.surf, degrees)
        rot_rect = rotated_surf.get_rect()
        rot_rect.center = (self.x, self.y)
        viewer._window.blit(rotated_surf, rot_rect)

class RotTriangle(RotSurface):

    def __init__(self, x, y, size, color_fg, color_bg, direction):
        self.size = size
        self.color_fg = color_fg
        self.color_bg = color_bg
        self.surf = pygame.Surface((size, size))
        self.surf.fill(color_bg)
        self.surf.set_colorkey(color_bg)

        # vertices with direction 0
        x1 = 0
        y1 = size / 4
        x2 = 0
        y2 = 0.75 * size
        x3 = size
        y3 = size / 2

        v1 = [x1, y1]
        v2 = [x2, y2]
        v3 = [x3, y3]

        # print([v1, v2, v3])

        pygame.draw.polygon(self.surf, self.color_fg, [v1, v2, v3])

        super().__init__(x, y, direction, self.surf)


class LightSource(RotSurface):

    def __init__(self, x, y, emitting_power, color_fg, color_bg, **kwargs):
        self.emitting_power = emitting_power
        self.color_fg = color_fg
        self.color_bg = color_bg
        self.size = emitting_power
        self.label = None

        self.surf = pygame.Surface((self.size, self.size))
        self.surf.fill(color_bg)
        self.surf.set_colorkey(color_bg)

        pygame.draw.circle(self.surf, color_fg, [int(round(self.size / 2)), int(round(self.size / 2))],
                           int(round(self.size / 2)))

        super().__init__(x, y, 0, self.surf, **kwargs)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.x) + ' ' + str(self.y) + ' ' + str(self.emitting_power)





