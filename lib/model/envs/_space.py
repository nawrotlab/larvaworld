import numpy as np
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import LineString, Point

from lib.aux.colsNstr import colorname2tuple
from lib.aux.dictsNlists import flatten_list
from lib.model.DEB.deb import Substrate


class ValueGrid:
    def __init__(self, unique_id, space_range, grid_dims=[50, 50], distribution='uniform', visible=False,
                 initial_value=0, default_color=(255, 255, 255), max_value=np.inf, min_value=-np.inf):
        self.visible = visible
        self.unique_id = unique_id
        self.initial_value = initial_value
        self.max_value = max_value
        self.min_value = min_value
        if type(default_color) == str:
            default_color = colorname2tuple(default_color)
        self.default_color = default_color
        self.grid_dims = grid_dims
        self.X, self.Y = grid_dims
        x_range = tuple(space_range[0:2])
        y_range = tuple(space_range[2:])
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        xr, yr = x1 - x0, y1 - y0
        self.x = xr / self.X
        self.y = yr / self.Y
        self.xy = np.array([self.x, self.y])
        self.XY_half = np.array([self.X / 2, self.Y / 2])

        x_linspace = np.linspace(x0, x1, self.X)
        y_linspace = np.linspace(y0, y1, self.Y)
        self.meshgrid = np.meshgrid(x_linspace, y_linspace)

        if distribution == 'uniform':
            self.grid = np.ones(self.grid_dims) * self.initial_value

        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]

    def add_value(self, p, value):
        cell = self.get_grid_cell(p)
        v = self.add_cell_value(cell, value)
        return v

    def set_value(self, p, value):
        cell = self.get_grid_cell(p)
        self.set_cell_value(cell, value)

    def get_value(self, p):
        cell = self.get_grid_cell(p)
        return self.get_cell_value(cell)

    def get_grid_cell(self, p):
        c = np.floor(p / self.xy + self.XY_half).astype(int)
        return tuple(c)
        # return tuple(c)

    def get_cell_value(self, cell):
        return self.grid[cell]

    def set_cell_value(self, cell, value):
        self.grid[cell] = value

    def add_cell_value(self, cell, value):
        v0 = self.get_cell_value(cell)
        v1 = v0 + value
        v2 = np.clip(v1, a_min=self.min_value, a_max=self.max_value)
        self.set_cell_value(cell, v2)
        if v1 < v2:
            return self.min_value - v0
        elif v1 > v2:
            return self.max_value - v0
        else:
            return value

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
        self.grid = np.ones(self.grid_dims) * self.initial_value

    def empty_grid(self):
        self.grid = np.zeros(self.grid_dims)

    def draw(self, viewer):
        color_grid = self.get_color_grid()
        for vertices, col in zip(self.grid_vertices, color_grid):
            viewer.draw_polygon(vertices, col, filled=True)

    def get_color_grid(self):
        v0 = self.initial_value
        cs = np.array((v0 - self.grid.flatten()) * 255).astype(int)
        cs = np.array([cs, cs, cs]).T
        color_grid = np.clip(np.array(self.default_color) + cs, a_min=0, a_max=255)
        return color_grid

    def get_grid(self):
        return self.grid


class FoodGrid(ValueGrid):
    def __init__(self, default_color=(0, 255, 0), quality=1, type='standard', **kwargs):
        super().__init__(default_color=default_color, min_value=0.0, **kwargs)
        self.substrate = Substrate(type=type, quality=quality)

    def get_color(self, v):
        v0 = self.initial_value
        c = int((v0 - v) / v0 * 255) if v0 != 0 else 255
        col = np.clip(np.array(self.default_color) + c, a_min=0, a_max=255)
        return col

    def draw(self, viewer):
        viewer.draw_polygon(self.grid_edges, self.get_color(v=self.initial_value), filled=True)
        not_full = np.array([[k, v] for k, v in enumerate(self.grid.flatten().tolist()) if
                             v != self.initial_value])
        if not_full.shape[0] != 0:
            vertices = [self.grid_vertices[int(i)] for i in not_full[:, 0]]
            colors = [self.get_color(v) for v in not_full[:, 1]]
            for v, c in zip(vertices, colors):
                viewer.draw_polygon(v, c, filled=True)


class ValueLayer(ValueGrid):
    def __init__(self, sources=[], **kwargs):
        super().__init__(min_value=0, **kwargs)
        self.sources = sources


class GaussianValueLayer(ValueLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_values(self):
        pass

    def get_value(self, pos):

        value = 0
        for s in self.sources:
            # print(s.unique_id, s.odor_peak_value)
            p = s.get_position()
            rel_pos = [pos[0] - p[0], pos[1] - p[1]]
            value += s.get_gaussian_odor_value(rel_pos)
        return value

    def compute_grid(self):
        X, Y = self.meshgrid

        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        return V

    def draw(self, viewer):
        V = self.compute_grid().T
        self.grid = V / np.max(V)
        color_grid = self.get_color_grid()
        for vertices, col in zip(self.grid_vertices, color_grid):
            viewer.draw_polygon(vertices, col, filled=True)


class DiffusionValueLayer(ValueLayer):

    def __init__(self, dt, scaling_factor, evap_const, gaussian_sigma, **kwargs):
        super().__init__(**kwargs)
        '''
            A typical diffusion coefficient for a molecule in the gas phase is in the range of 10-6 to 10-5 m2/sigma           

            Yes, it does that automatically based on the sigma and truncate parameters.
            Indeed, the function gaussian_filter is implemented by applying multiples 1D gaussian filters (you can see that here). 
            This function uses gaussian_filter1d which generate itself the kernel using _gaussian_kernel1d with a radius of 
            int(truncate * sigma + 0.5).

            Doing the math, sigma ends up reeeeally small
        '''
        D = 10 ** -6
        cell_width, cell_height = self.x / scaling_factor, self.y / scaling_factor
        rad_x, rad_y = D * dt / cell_width, D * dt / cell_height
        temp = 10 ** 5
        sigma = int(rad_x * temp), int(rad_y * temp)
        self.evap_const = evap_const
        self.sigma = gaussian_sigma

    def update_values(self):
        for s in self.sources:
            source_pos = s.get_position()
            intensity = s.odor_intensity
            self.add_value(source_pos, intensity)
        self.grid = gaussian_filter(self.grid, sigma=self.sigma) * self.evap_const


class WindScape:
    def __init__(self, model, wind_direction, wind_speed, default_color='red', visible=False):

        self.model = model
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.max_dim=np.max(self.model.arena_dims)
        self.default_color=default_color
        self.visible=visible

        self.N = 40

        self.scapelines=self.generate_scapelines(self.max_dim, self.N, self.wind_direction)
        # p0s = rotate_around_center_multi([(-self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # p1s = rotate_around_center_multi([(self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # self.scapelines=[(p0,p1) for p0,p1 in zip(p0s,p1s)]

    def get_value(self, agent):
        if self.obstructed(agent.pos):
            return 0
        else:
            from lib.aux.ang_aux import angle_dif
            o = np.rad2deg(agent.head.get_orientation())
            return np.abs(angle_dif(o, self.wind_direction)) / 180 * self.wind_speed

    def obstructed(self, pos):
        from lib.aux.ang_aux import line_through_point
        ll = line_through_point(pos, self.wind_direction, self.max_dim)
        return any([l.intersects(ll) for l in self.model.border_lines])

    def draw(self, viewer):
        if self.wind_speed>0 :
            for p0, p1 in self.scapelines :
                l=LineString([p0, p1])
                ps=[l.intersection(b) for b in self.model.border_lines if l.intersects(b)]
                if len(ps)!=0 :
                    p1=ps[np.argmin([Point(p0).distance(p2) for p2 in ps])].coords[0]
                viewer.draw_arrow_line(p0, p1, self.default_color, width=0.0001*self.wind_speed)

    def generate_scapelines(self,D,N, A):
        from lib.aux.ang_aux import rotate_around_center_multi
        ds = self.max_dim / N * np.sqrt(2)
        p0s = rotate_around_center_multi([(-D, (i - N / 2) * ds) for i in range(N)],-A)
        p1s = rotate_around_center_multi([(D, (i - N / 2) * ds) for i in range(N)],-A)
        return [(p0, p1) for p0, p1 in zip(p0s, p1s)]

    def set_wind_direction(self, A):
        self.wind_direction=A
        self.scapelines = self.generate_scapelines(self.max_dim, self.N, self.wind_direction)
