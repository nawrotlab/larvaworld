import pygame


class Obstacle:

    def __init__(self, vertices, edges, color):
        self.vertices = vertices
        self.edges = edges
        self.color = color

    def draw(self, scene):
        pygame.draw.polygon(scene.screen, self.color, self.vertices, 1)
