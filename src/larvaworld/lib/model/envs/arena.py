import agentpy
import numpy as np
from shapely import geometry

from larvaworld.lib import reg, aux


class Arena(agentpy.Space):
    def __init__(self, model, dims, shape= 'rectangular',vertices=None, torus=False):
        X, Y = self.dims = np.array(dims)
        if vertices is None:
            if shape == 'circular':
                # This is a circle_to_polygon shape from the function
                vertices = aux.circle_to_polygon(60, X / 2)
            elif shape == 'rectangular':
                # This is a rectangular shape
                vertices = np.array([(-X / 2, -Y / 2),
                                   (-X / 2, Y / 2),
                                   (X / 2, Y / 2),
                                   (X / 2, -Y / 2)])
            else:
                raise
        self.vertices =vertices
        self.range = np.array([-X / 2, X / 2, -Y / 2, Y / 2])
        k = 0.96
        self.polygon = geometry.Polygon(self.vertices * k)
        self.edges = [[geometry.Point(x1,y1), geometry.Point(x2,y2)] for (x1,y1), (x2,y2) in aux.group_list_by_n(vertices, 2)]
        super().__init__(model=model, torus=torus, shape=dims)

    @staticmethod
    def _border_behavior(position, shape, torus):
        # Border behavior

        # Connected - Jump to other side
        if torus:
            for i in range(len(position)):
                while position[i] > shape[i]/2:
                    position[i] -= shape[i]
                while position[i] < -shape[i]/2:
                    position[i] += shape[i]

        # Not connected - Stop at border
        else:
            for i in range(len(position)):
                if position[i] > shape[i]/2:
                    position[i] = shape[i]/2
                elif position[i] < -shape[i]/2:
                    position[i] = -shape[i]/2

    def place_agent(self, agent, pos):
        pos = pos if isinstance(pos, np.ndarray) else np.array(pos)
        self.positions[agent] = pos  # Add pos to agent_dict

    def move_agent(self, agent, pos):
        self.move_to(agent, pos)
