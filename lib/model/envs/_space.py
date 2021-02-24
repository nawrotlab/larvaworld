import math
import time

import numpy
import numpy as np
import scipy
from mesa.space import ContinuousSpace
from scipy.stats import multivariate_normal


class ValueGrid:
    def __init__(self, space_range, grid_resolution=[100, 100], distribution='uniform', initial_value=1):
        self.initial_value = initial_value
        # print(grid_resolution)
        # print(grid_resolution[0])
        # print(type(grid_resolution[0]))
        # raise
        self.X, self.Y = grid_resolution
        x_range = tuple(space_range[0:2])
        y_range = tuple(space_range[2:])
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        xr, yr = x1 - x0, y1 - y0
        self.x = xr / self.X
        self.y = yr / self.Y
        self.xy=np.array([self.x, self.y])
        self.XY_half=np.array([self.X/2, self.Y/2])

        if distribution == 'uniform':
            self.grid = np.ones([self.X, self.Y]) * initial_value

        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]

    def get_color(self, v=1):
        c = int(v * 255)
        return np.array((0, c, 0))

    def get_grid_cell(self, p):
        c=np.clip(np.array(p/self.xy + self.XY_half).astype(int), a_min=[0,0], a_max=[self.X-1,self.Y-1])
        return tuple(c)

    def get_value(self, cell):
        return self.grid[cell]

    def set_value(self, cell, value):
        self.grid[cell] = value

    def subtract_value(self, cell, value):
        previous_v = self.get_value(cell)
        self.set_value(cell, previous_v - value)
        if self.get_value(cell) < 0:
            self.set_value(cell, 0)
            subtracted_v = previous_v
        else:
            subtracted_v = value
        return subtracted_v

    def generate_grid_vertices(self):
        vertices = []
        for i in range(self.X):
            for j in range(self.Y):
                vertices.append(self.cell_vertices(i, j))
        return vertices

    def draw(self, viewer):
        viewer.draw_polygon(self.grid_edges, self.get_color(), filled=True)
        not_full=np.array([[k, v/self.initial_value] for k,v in enumerate(self.grid.flatten().tolist()) if v!=self.initial_value])
        if not_full.shape[0]!=0 :
            # print(not_full)
            # # raise
            vertices = [self.grid_vertices[int(i)] for i in not_full[:,0]]
            colors = [self.get_color(v) for v in not_full[:,1]]
            for v,c in zip(vertices,colors) :
                viewer.draw_polygon(v, c, filled=True)

    def cell_vertices(self, i, j):
        x, y = self.x, self.y
        X, Y = self.X / 2, self.Y / 2
        v = [[x * int(i - X), y * int(j - Y)],
             [x * int(i + 1 - X), y * int(j - Y)],
             [x * int(i + 1 - X), y * int(j + 1 - Y)],
             [x * int(i - X), y * int(j + 1 - Y)]]
        return v

    def reset(self):
        self.grid = np.ones([self.X, self.Y]) * self.initial_value

    def empty_grid(self):
        self.grid = np.zeros([self.X, self.Y])



class ValueLayer:
    def __init__(self, world, unique_id, sources, **kwargs):
        self.world = world
        self.id = unique_id

        self.sources = sources
        self.num_sources = len(self.sources)

    def get_value(self, pos):
        pass

    def add_value(self, pos, value):
        pass

    def set_value(self, pos, value):
        pass

    def update_values(self):
        pass

    def get_num_sources(self):
        return self.num_sources


class GaussianValueLayer(ValueLayer):
    def __init__(self, world, **kwargs):
        super().__init__(world, **kwargs)

        # self.dist = multivariate_normal([0, 0], [[0.2, 0], [0, 0.2]])

    def update_values(self):
        pass

    def get_value(self, pos):
        value = 0
        for s in self.sources:
            source_pos = s.get_position()
            # print(source_pos)
            # spread = s.get_odor_spread()
            # intensity = s.get_odor_intensity()
            rel_pos = [pos[0] - source_pos[0], pos[1] - source_pos[1]]
            # dist = multivariate_normal([0, 0], [[spread, 0], [0, spread]])
            # v = dist.pdf(rel_pos) * intensity / dist.pdf([0, 0])
            value += s.get_gaussian_odor_value(rel_pos)
        # print(value)
        return value

    # def model(position, width, height):
    #     return  (height / scipy.stats.norm.pdf(position,position,width)) * scipy.stats.norm.pdf(x, position, width)


class DiffusionValueLayer(ValueLayer):
    def __init__(self, world, world_range, grid_resolution, evap_const, diff_const, **kwargs):
        super().__init__(world, **kwargs)

        # The grid parameters measured in number of grid cells. This will be mapped on the space dimensions
        self.grid_res = grid_resolution
        self.grid_width_in_cells = self.grid_res[0]
        self.grid_heigth_in_cells = self.grid_res[1]
        self.world_range = world_range
        self.grid_step_x = (world_range[0][1] - world_range[0][0]) / self.grid_width_in_cells
        self.grid_step_y = (world_range[1][1] - world_range[1][0]) / self.grid_heigth_in_cells

        self.values = np.zeros(shape=(self.grid_width_in_cells, self.grid_heigth_in_cells)) + self.default_val()

        self.x_ticks = np.arange(0, self.grid_width_in_cells, 1)
        self.y_ticks = np.arange(0, self.grid_heigth_in_cells, 1)
        self.X, self.Y = np.meshgrid(self.x_ticks, self.y_ticks)
        # temp = np.array(self.X.flatten(), self.Y.flatten())
        # self.grid_points = np.array([self.X.flatten(), self.Y.flatten()]).T

        # self.grid_kdtree = spatial.KDTree(self.grid_points)

        self.evap_const = evap_const
        self.diff_const = diff_const

        # self.diffuse_layer = np.vectorize(self.aux_diffuse)

    # TODO This is the simplest approach rounding to the cell without taking into account where exactly in the cell.
    #  We will want to do this more elegantly
    def world_pos_to_grid_cell(self, world_pos):
        grid_x = (world_pos[0] - self.world_range[0][0]) / self.grid_step_x
        grid_y = (world_pos[1] - self.world_range[1][0]) / self.grid_step_y
        cell = (int(math.floor(grid_x)), int(math.floor(grid_y)))
        return cell

    @staticmethod
    def default_val():
        """ Default value for new cell elements. """
        return 0

    def out_of_bounds(self, grid_pos):
        x, y = grid_pos
        return x < 0 or x >= self.grid_width_in_cells or y < 0 or y >= self.grid_heigth_in_cells
        # return x < -self.grid_range or x > self.grid_range or y < -self.grid_range or y > self.grid_range

    def iter_neighborhood(self, cell, moore, include_center=False, radius=1):
        """ Return an iterator over cell coordinates that are in the
        neighborhood of a certain point.
        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                        (including diagonals)
                   If False, return Von Neumann neighborhood
                        (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.
        Returns:
            A list of coordinate tuples representing the neighborhood. For
            example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not
            including the center).
        """
        x, y = cell
        coordinates = set()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and not include_center:
                    continue
                # Skip diagonals in Von Neumann neighborhood.
                if not moore and dy != 0 and dx != 0:
                    continue
                # Skip diagonals in Moore neighborhood when distance > radius
                if moore and 1 < radius < (dy ** 2 + dx ** 2) ** .5:
                    continue
                # Skip if not a torus and new coords out of bounds.
                # if not self.torus and (not (0 <= dx + x < self.width) or
                #                        not (0 <= dy + y < self.height)):
                #     continue

                # px = self.torus_adj(x + dx, self.width)
                # py = self.torus_adj(y + dy, self.height)
                px = x + dx
                py = y + dy

                # Skip if new coords out of bounds.
                if self.out_of_bounds((px, py)):
                    continue

                coords = (px, py)
                if coords not in coordinates:
                    coordinates.add(coords)
                    yield coords

    def neighbor_avg(self, cell):
        sum_values = 0
        neighbors = list(self.iter_neighborhood(cell, moore=True, include_center=True, radius=5))
        for cell in neighbors:
            sum_values += self.values[cell]
        return sum_values / len(neighbors)

    def add_value(self, pos, value):
        cell = self.world_pos_to_grid_cell(pos)
        self.values[cell] += value

    def get_value(self, pos):
        cell = self.world_pos_to_grid_cell(pos)
        return self.values[cell]
        # x, y = pos
        # x_index = np.array(np.where(self.x_ticks == x))
        # y_index = np.array(np.where(self.y_ticks == y))
        # # print(pos)
        # # print(x_index.size, len(y_index))
        #
        # if x_index.size == 0 or y_index.size == 0:
        #     # FIXME This produces lots of Nan values
        #     z = griddata(self.grid_points, self.Z.flatten(), pos)
        #     if math.isnan(z):
        #         # print('Nan value using griddata')
        #         x, y = self._get_nearest_grid_point(pos)
        #         x_index = np.where(self.x_ticks == x)
        #         y_index = np.where(self.y_ticks == y)
        #         z = self.Z[x_index, y_index]
        # elif x_index.size == 1 and y_index.size == 1:
        #     z = self.Z[x_index, y_index]
        #     # print('normal', z)
        # else:
        #     raise NotImplementedError('More indexes than expected')
        # return z

    def set_value(self, pos, value):
        cell = self.world_pos_to_grid_cell(pos)
        self.values[cell] = value
        # x, y = pos
        # x_index = np.array(np.where(self.x_ticks == x))
        # y_index = np.array(np.where(self.y_ticks == y))
        # if x_index.size == 0 or y_index.size == 0:
        #     x, y = self._get_nearest_grid_point(pos)
        #     x_index = np.where(self.x_ticks == x)
        #     y_index = np.where(self.y_ticks == y)
        # self.Z[x_index, y_index] = value

    # def _get_nearest_grid_point(self, pos):
    #     """ Get the cell coordinates that a given x,y point falls in. """
    #     if self.out_of_bounds(pos):
    #         raise Exception("Point out of bounds.")
    #
    #     x, y = pos
    #     index = self.grid_kdtree.query([x, y], k=1)[1]
    #     # print([x, y])
    #     # print(index)
    #     nearest_point = self.grid_points[index]
    #     # print(nearest_point)
    #     return nearest_point

    def diffuse_cell(self, cell):
        # pos = [x, y]
        current_value = self.values[cell]
        r = self.evap_const * (current_value + self.diff_const * (self.neighbor_avg(cell) - current_value))
        return r

    # @np.vectorize
    def aux_diffuse(self, x, y):
        v = self.diffuse_cell((x, y))
        return v

    def update_values(self):
        for s in self.sources:
            source_pos = s.get_position()
            intensity = s.get_odor_intensity()
            self.add_value(source_pos, intensity)
        new_values = np.zeros(shape=(self.grid_width_in_cells, self.grid_heigth_in_cells))

        # Iteration approach
        for x in range(self.grid_width_in_cells):
            for y in range(self.grid_heigth_in_cells):
                r = self.diffuse_cell((x, y))
                new_values[x, y] = r

        # Vectorization approach
        # new_values = self.diffuse_layer(self.X,self.Y)

        self.values = new_values


class LimitedSpace(ContinuousSpace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def agents_spatial_query(pos, radius, agent_list):
    if len(agent_list) == 0:
        return []
    # s = time.time()
    agent_positions = np.array([agent.get_position() for agent in agent_list])
    agent_radii = np.array([agent.get_radius() for agent in agent_list])
    dsts = np.linalg.norm(agent_positions - pos, axis=1) - agent_radii
    inds = np.where(dsts <= radius)[0]
    # e = time.time()
    # print(e - s)
    # print(len(inds)>0)
    # print(len(inds))
    return [agent_list[i] for i in inds]
