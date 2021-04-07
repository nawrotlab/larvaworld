import math

import numpy as np
from mesa.space import ContinuousSpace
from scipy.ndimage.filters import gaussian_filter

class ValueGrid:
    def __init__(self, space_range, grid_dims=[50, 50], distribution='uniform',
                 initial_value=0, default_color=(255,255,255)):
        self.initial_value=initial_value
        self.default_color=default_color

        # print(food_grid_dims)
        # print(food_grid_dims[0])
        # print(type(food_grid_dims[0]))
        # raise
        self.X, self.Y = grid_dims
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
            self.grid = np.ones([self.X, self.Y]) * self.initial_value

        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]

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

    def draw(self, viewer):
        color_grid=self.get_color_grid()
        for vertices, col in zip(self.grid_vertices, color_grid) :
            viewer.draw_polygon(vertices, col, filled=True)

    def get_color(self, v):
        v0=self.initial_value
        c = int((v0-v) * 255)
        col=np.clip(np.array(self.default_color)+c, a_min=0, a_max=255)
        return col

    def get_color_grid(self):
        v0 = self.initial_value
        cs = np.array((v0-self.grid.flatten()) * 255).astype(int)
        cs = np.array([cs,cs,cs]).T
        color_grid = np.clip(np.array(self.default_color) + cs, a_min=0, a_max=255)
        return color_grid



class FoodGrid(ValueGrid):
    def __init__(self, space_range, food_grid_dims=[100, 100], distribution='uniform',
                 default_color=(0,255,0), food_grid_amount=1, quality=1):
        if food_grid_amount>0 :
            initial_value=1
        else :
            initial_value=0
        super().__init__(space_range=space_range, distribution=distribution,
                         grid_dims=food_grid_dims, initial_value=initial_value, default_color=default_color)
        self.initial_amount = food_grid_amount
        self.quality = quality

        if distribution == 'uniform':
            self.grid = np.ones([self.X, self.Y]) * self.initial_amount



    def draw(self, viewer):
        viewer.draw_polygon(self.grid_edges, self.get_color(v=self.initial_value), filled=True)
        not_full=np.array([[k, v/self.initial_amount] for k,v in enumerate(self.grid.flatten().tolist()) if v!=self.initial_amount])
        if not_full.shape[0]!=0 :
            vertices = [self.grid_vertices[int(i)] for i in not_full[:,0]]
            colors = [self.get_color(v) for v in not_full[:,1]]
            for v,c in zip(vertices,colors) :
                viewer.draw_polygon(v, c, filled=True)


class ValueLayer:
    def __init__(self, space, unique_id, color, space_range, space2grid, sources=[],
                 visible=False, grid_dims=[50, 50],**kwargs):
        self.space = space
        self.id = unique_id
        self.color = color
        self.sources = sources
        self.value_grid=ValueGrid(space_range=space_range, default_color=color, grid_dims=grid_dims)
        self.space2grid=space2grid
        self.visible=visible

    def get_value(self, pos):
        pass

    def add_value(self, pos, value):
        pass

    def set_value(self, pos, value):
        pass

    def update_values(self):
        pass

    def get_num_sources(self):
        return len(self.sources)

    def get_value_grid(self, X,Y):
        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        return V

    def draw_value_grid(self, viewer):
        X,Y=self.space2grid
        V=self.get_value_grid(X,Y).T
        self.value_grid.grid=V/np.max(V)
        self.value_grid.draw(viewer)


class GaussianValueLayer(ValueLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_values(self):
        pass

    def get_value(self, pos):
        value = 0
        for s in self.sources:
            p = s.get_position()
            rel_pos = [pos[0] - p[0], pos[1] - p[1]]
            value += s.get_gaussian_odor_value(rel_pos)
        return value


class DiffusionValueLayer(ValueLayer):
    def __init__(self, evap_const, diff_const, **kwargs):
        super().__init__(**kwargs)

        self.evap_const = evap_const
        self.diff_const = diff_const

        # self.diffuse_layer = np.vectorize(self.aux_diffuse)


    def out_of_bounds(self, grid_pos):
        x, y = grid_pos
        return x < 0 or x >= self.value_grid.X or y < 0 or y >= self.value_grid.Y

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
            sum_values += self.value_grid.get_value(cell)
        return sum_values / len(neighbors)

    def add_value(self, p, value):
        cell = self.value_grid.get_grid_cell(p)
        v0 = self.value_grid.get_value(cell)
        self.value_grid.set_value(cell, v0+value)

    def get_value(self, p):
        cell = self.value_grid.get_grid_cell(p)
        return self.value_grid.get_value(cell)

    def set_value(self, p, value):
        cell = self.value_grid.get_grid_cell(p)
        self.value_grid.set_value(cell,value)

    def diffuse_cell(self, cell):
        v = self.value_grid.get_value(cell)
        r = self.evap_const * (v + self.diff_const * (self.neighbor_avg(cell) - v))
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
        self.value_grid.grid = gaussian_filter(self.value_grid.grid, sigma=7)*self.evap_const

        # new_values = np.zeros(shape=(self.value_grid.X, self.value_grid.Y))
        #
        # # Iteration approach
        # for x in range(self.value_grid.X):
        #     for y in range(self.value_grid.Y):
        #         r = self.diffuse_cell((x, y))
        #         new_values[x, y] = r
        # self.value_grid.grid = new_values





