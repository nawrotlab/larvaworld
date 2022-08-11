import numpy as np


def get_display_dims():
    import pygame

    pygame.init()
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    return int(W * 2 / 3 / 16) * 16, int(H * 2 / 3 / 16) * 16


def get_window_dims(arena_dims):
    X, Y = np.array(arena_dims)
    W0, H0 = get_display_dims()
    R0, R = W0 / H0, X / Y
    if R0 < R:

        return W0, int(W0 / R / 16) * 16
    else:
        return int(H0 * R / 16) * 16, H0

def get_arena_bounds(arena_dims, s=1):
    X, Y = np.array(arena_dims) * s
    return np.array([-X / 2, X / 2, -Y / 2, Y / 2])

def screen2space_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims
    p = (2 * pos[0] / X0 - 1), -(2 * pos[1] / Y0 - 1)
    pp = p[0] * X / 2, p[1] * Y / 2
    return pp

def space2screen_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims

    p = pos[0] * 2 / X, pos[1] * 2 / Y
    pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
    return pp

