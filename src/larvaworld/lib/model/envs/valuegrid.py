import numpy as np
from scipy.ndimage.filters import gaussian_filter
from shapely import geometry

from larvaworld.lib import reg, aux
from larvaworld.lib.model.deb.substrate import Substrate
from larvaworld.lib.screen.rendering import InputBox

class ValueGrid:
    def __init__(self, model, unique_id, grid_dims=[51, 51], distribution='uniform', visible=False,
                 initial_value=0.0, default_color=(255, 255, 255), max_value=None, min_value=0.0, fixed_max=False):
        self.model = model
        self.visible = visible
        self.unique_id = unique_id
        self.initial_value = initial_value

        self.min_value = min_value
        self.fixed_max = fixed_max
        if type(default_color) == str:
            default_color = aux.colorname2tuple(default_color)
        self.default_color = default_color
        self.grid_dims = grid_dims
        self.X, self.Y = grid_dims
        x_range = tuple(self.model.space.range[0:2])
        y_range = tuple(self.model.space.range[2:])
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        xr, yr = x1 - x0, y1 - y0
        self.x = xr / self.X
        self.y = yr / self.Y
        self.cell_radius = np.sqrt(np.sum((self.x / 2) ** 2 + (self.y / 2) ** 2))
        self.xy = np.array([self.x, self.y])
        self.XY_half = np.array([self.X / 2, self.Y / 2])
        self.meshgrid = np.meshgrid(np.linspace(x0, x1, self.X), np.linspace(y0, y1, self.Y))
        if distribution == 'uniform':
            self.grid = np.ones(self.grid_dims) * self.initial_value
        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]
        if max_value is None:
            max_value = np.max(self.grid)
        self.max_value = max_value

    def add_value(self, p, value):
        return self.add_cell_value(self.get_grid_cell(p), value)

    def set_value(self, p, value):
        self.set_cell_value(self.get_grid_cell(p), value)

    def get_value(self, p):
        return self.get_cell_value(self.get_grid_cell(p))

    def get_grid_cell(self, p):
        return tuple(np.floor(p / self.xy + self.XY_half).astype(int))

    def get_cell_value(self, cell):
        return self.grid[cell]

    def set_cell_value(self, cell, value):
        self.grid[cell] = value

    def add_cell_value(self, cell, value):
        v0 = self.get_cell_value(cell)
        v1 = v0 + value
        if not self.fixed_max:
            self.max_value = np.max([self.max_value, v1])
        v2 = np.clip(v1, a_min=self.min_value, a_max=self.max_value)
        self.set_cell_value(cell, v2)
        if v1 < v2:
            return self.min_value - v0
        elif v1 > v2:
            return self.max_value - v0
        else:
            return value

    def generate_grid_vertices(self):
        vertices = np.zeros([self.X, self.Y, 4, 2])
        for i in range(self.X):
            for j in range(self.Y):
                vertices[i, j] = self.cell_vertices(i, j)
        return vertices

    def cell_vertices(self, i, j):
        x, y = self.x, self.y
        X, Y = self.X / 2, self.Y / 2
        return np.array([(x * (i - X), y * (j - Y)),
                      (x * (i + 1 - X), y * (j - Y)),
                      (x * (i + 1 - X), y * (j + 1 - Y)),
                      (x * (i - X), y * (j + 1 - Y))])

    def cel_pos(self, i, j):
        return self.x * (i - self.X / 2 + 0.5), self.y * (j - self.Y / 2 + 0.5)

    def reset(self):
        self.grid = np.ones(self.grid_dims) * self.initial_value

    def empty_grid(self):
        self.grid = np.zeros(self.grid_dims)

    def draw_peak(self, viewer):
        idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
        p = self.cel_pos(*idx)
        # vs = self.cell_vertices(*idx)

        viewer.draw_circle(p, self.cell_radius / 2, self.default_color, filled=True, width=0.0005)
        # viewer.draw_polygon(vs, self.default_color, filled=True, width=0.0005)

        p_text = (p[0] + self.x, p[1] - self.y)
        text_box = InputBox(text=str(np.round(self.grid.max(), 2)), color_active=self.default_color, visible=True,
                            screen_pos=viewer._transform(p_text))
        text_box.draw(viewer)

    def draw(self, viewer):
        Cgrid = self.get_color_grid().reshape([self.X, self.Y, 3])
        for i in range(self.X):
            for j in range(self.Y):
                viewer.draw_polygon(self.grid_vertices[i, j], Cgrid[i, j], filled=True)
        self.draw_peak(viewer)
        if self.model.screen_manager.odor_aura:
            self.draw_isocontours(viewer)

    def draw_isocontours(self, viewer):
        N = 8
        k = 4
        g = self.get_grid()
        # c='white'
        c = self.default_color
        vmax = np.max(g)
        for i in range(N):
            v = vmax * k ** -i
            if v <= 0:
                continue
            inds = np.argwhere((v <= g) & (g < v * k)).tolist()
            points = [self.cel_pos(i, j) for (i, j) in inds]
            if len(points) > 2:
                try:
                    ps = np.array(points)
                    pxy = ps[np.argmax(ps[:, 0]), :] + np.array([self.x, -self.y])
                    viewer.draw_convex(points, color=c, filled=False, width=0.0005)
                    text_box = InputBox(text=str(np.round(v, 2)), color_active=c, visible=True,
                                        screen_pos=viewer._transform(pxy))
                    text_box.draw(viewer)
                except:
                    pass

    def get_color_grid(self):
        g = self.get_grid().flatten()
        v0, v1 = self.min_value, self.max_value
        gg = (g - v0) / (v1 - v0)
        k = 10 ** 2
        m = (1 - np.exp(-k)) ** -1
        q = m * (1 - np.exp(-k * gg))
        q = np.clip(q, a_min=0, a_max=1)
        return aux.col_range(q, low=(255, 255, 255), high=self.default_color, mul255=True)

    def get_grid(self):
        return self.grid


class FoodGrid(ValueGrid):
    def __init__(self, default_color=(0, 255, 0), quality=1, type='standard', **kwargs):
        super().__init__(default_color=default_color, fixed_max=True, **kwargs)
        self.substrate = Substrate(type=type, quality=quality)

    def get_color(self, v):
        v0, v1 = self.min_value, self.max_value
        q = (v - v0) / (v1 - v0)
        return aux.col_range(q, low=(255, 255, 255), high=self.default_color, mul255=True)

    def draw(self, viewer):
        viewer.draw_polygon(self.grid_edges, self.get_color(v=self.initial_value), filled=True)
        for i in range(self.X):
            for j in range(self.Y):
                v = self.grid[i, j]
                if v != self.initial_value:
                    viewer.draw_polygon(self.grid_vertices[i, j], self.get_color(v), filled=True)


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

    def get_grid(self):
        X, Y = self.meshgrid

        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        self.max_value = np.max(V.flatten())
        return V

    def draw_isocontours(self, viewer):
        for s in self.sources:
            p = s.get_position()
            for r in np.arange(0, 0.050, 0.01):
                pX = (p[0] + r, p[1])
                v = s.get_gaussian_odor_value(pX)
                viewer.draw_circle(p, r, self.default_color, filled=False, width=0.0005)
                text_box = InputBox(text=str(np.round(v, 2)), color_active=self.default_color, visible=True,
                                    screen_pos=viewer._transform(pX))
                text_box.draw(viewer)


class DiffusionValueLayer(ValueLayer):

    def __init__(self, evap_const, gaussian_sigma, **kwargs):
        super().__init__(**kwargs)
        '''
            A typical diffusion coefficient for a molecule in the gas phase is in the range of 10-6 to 10-5 m2/sigma           

            Yes, it does that automatically based on the sigma and truncate parameters.
            Indeed, the function gaussian_filter is implemented by applying multiples 1D gaussian filters (you can see that here). 
            This function uses gaussian_filter1d which generate itself the kernel using _gaussian_kernel1d with a radius of 
            int(truncate * sigma + 0.5).

            Doing the math, sigma ends up reeeeally small
        '''
        self.evap_const = evap_const
        self.sigma = gaussian_sigma

    def update_values(self):
        k = 1000
        if self.model.windscape is not None:
            v, a = self.model.windscape.wind_speed, self.model.windscape.wind_direction
            if v != 0:
                dx = v * np.cos(a) * self.model.dt
                dy = v * np.sin(a) * self.model.dt
                Px, Py = dx / self.x / k, dy / self.y / k

                Pr = np.abs(Px / (Px + Py))
                Px = np.clip(Px, a_min=-Pr, a_max=Pr)
                Py = np.clip(Py, a_min=-1 + Pr, a_max=1 - Pr)
                Gx = self.grid * Px
                Gy = self.grid * Py
                Gx = np.roll(Gx, 1, axis=0)
                Gx[0, :] = 0
                Gy = np.roll(Gy, 1, axis=1)
                Gy[:, 0] = 0
                self.grid *= (1 - Px - Py)
                self.grid += (Gx + Gy)
                np.clip(self.grid, a_min=0, a_max=None)

        for s in self.sources:
            source_pos = s.get_position()
            intensity = s.odor.odor_intensity
            self.add_value(source_pos, intensity)

        self.grid = gaussian_filter(self.grid, sigma=self.sigma) * self.evap_const


class WindScape:
    def __init__(self, model, wind_direction, wind_speed, puffs={}, default_color='red', visible=False):

        self.model = model
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.max_dim = np.max(self.model.space.dims)
        self.default_color = default_color
        self.visible = visible

        self.N = 40
        self.draw_phi = 0
        self.scapelines = self.generate_scapelines(self.max_dim, self.N, self.wind_direction)
        self.events = {}
        for idx, puff in puffs.items():
            self.add_puff(**puff)

    def get_value(self, agent):
        if self.obstructed(agent.pos):
            return 0
        else:
            o = np.rad2deg(agent.head.get_orientation())
            return np.abs(aux.angle_dif(o, self.wind_direction)) / 180 * self.wind_speed

    def obstructed(self, pos):

        ll = aux.line_through_point(pos, self.wind_direction, self.max_dim)
        return any([l.intersects(ll) for l in self.model.border_lines])

    def draw(self, viewer):
        if self.wind_speed > 0:
            for p0, p1 in self.scapelines:
                l = geometry.LineString([p0, p1])
                ps = [l.intersection(b) for b in self.model.border_lines if l.intersects(b)]
                if len(ps) != 0:
                    p1 = ps[np.argmin([geometry.Point(p0).distance(p2) for p2 in ps])].coords[0]
                viewer.draw_arrow_line(p0, p1, self.default_color, width=0.001,
                                       phi=(self.draw_phi % 1000) / 1000)
        self.draw_phi += self.wind_speed

    def generate_scapelines(self, D, N, A):
        ds = self.max_dim / N * np.sqrt(2)
        p0s = aux.rotate_points_around_point([(-D, (i - N / 2) * ds) for i in range(N)], -A)
        p1s = aux.rotate_points_around_point([(D, (i - N / 2) * ds) for i in range(N)], -A)
        return [(p0, p1) for p0, p1 in zip(p0s, p1s)]

    def set_wind_direction(self, A):
        self.wind_direction = A
        self.scapelines = self.generate_scapelines(self.max_dim, self.N, self.wind_direction)

    def add_puff(self, duration, speed, direction=None, start_time=None, N=1, interval=10.0):
        Nticks = int(duration / self.model.dt)
        if start_time is None:
            start = self.model.Nticks
        else:
            start = int(start_time / self.model.dt)
        interval_ticks = int(interval / self.model.dt)
        if N is None:
            N = int(self.model.Nsteps / interval_ticks)
        for i in range(N):
            t0 = start + i * interval_ticks
            self.events[t0] = {'wind_speed': speed, 'wind_direction': direction}
            self.events[t0 + Nticks] = {'wind_speed': self.wind_speed, 'wind_direction': self.wind_direction}

    def update(self):
        for t, args in self.events.items():
            if self.model.Nticks == t:
                if args['wind_direction'] is not None and args['wind_direction'] != self.wind_direction:
                    self.set_wind_direction(args['wind_direction'])
                self.wind_speed = args['wind_speed']


class ThermoScape(ValueGrid):
    def __init__(self, plate_temp=22, spread=0.1, thermo_sources=[[0.5, 0.05], [0.05, 0.5], [0.5, 0.95], [0.95, 0.5]],
                 thermo_source_dTemps=[8, -8, 8, -8], default_color='green', visible=False):
        self.plate_temp = plate_temp
        self.thermo_sources = {str(i): o for i, o in enumerate(thermo_sources)}
        self.thermo_source_dTemps = {str(i): o for i, o in enumerate(thermo_source_dTemps)}

        if len(thermo_sources) != len(thermo_source_dTemps):
            raise ValueError  # need to raise a more informative error.
        self.default_color = default_color
        self.visible = visible

        if spread is None:
            spread = 0.1  # just making it so spread is my default of 0.1, if None is given.
        self.thermo_spread = spread
        self.generate_thermoscape()

    def update_values(self):
        pass

    def generate_thermoscape(self):
        '''
        size is the length of the square arena in mm.
        rezo is the resolution with 1 being a mm, 0.1 being a 10th of a mm.
        spread is the spread put into the multivariate_normal function.
        pTem is the plate Temp i.e. the standard temperature of the plate - default is 22˚C.
        origins are the coordinate locations on the size x size plate of the heat or cold sources. type: list
        tempDiff needs to be a list the same length as origins, and determines if that source will be cold or hot and by how much.
        In other words a <ptemp> of 22 and a <origins> of [[10,20], [30,40]] and <tempDiff> of [8,-8] would make a temperature source at
        [10,20] that is ~30˚C and a source at [30,40] that is ~14˚C. The mean is taken where the temperatures of multiple sources overlap.
        '''
        from scipy.stats import multivariate_normal

        rv_dict = {}
        for k in self.thermo_sources:
            rv_dict[k] = multivariate_normal(self.thermo_sources[k], [[self.thermo_spread, 0], [0, self.thermo_spread]])

        self.thermoscape_layers = rv_dict


    def get_thermo_value(self, pos):

        size, size2 = [1, 1]
        pos_ad = [size * pos[0], size2 * pos[1]]
        pos_temp = {}
        nSources = len(self.thermo_source_dTemps)
        thermo_gain = {'cool': 0, 'warm': 0}

        for k in self.thermoscape_layers:
            v = self.thermoscape_layers[k]
            pos_temp[k] = v.pdf(pos_ad) / v.pdf(self.thermo_sources[k]) * (
                    self.thermo_source_dTemps[k] * nSources)  # @todo need to check if this works
            if pos_temp[k] < 0:
                thermo_gain['cool'] += abs(pos_temp[k] / nSources)
            elif pos_temp[k] > 0:
                thermo_gain['warm'] += abs(pos_temp[k] / nSources)
        return thermo_gain

    def get_grid(self):
        X, Y = self.meshgrid

        @np.vectorize
        def func(a, b):
            v = self.get_value((a, b))
            return v

        V = func(X, Y)
        self.max_value = np.max(V.flatten())
        return V

    def draw_isocontours(self, viewer):  # @todo need to make a draw function for thermogrid.
        for k in self.thermo_sources:
            p = self.thermo_sources.k
            for r in np.arange(0, 0.050, 0.01):
                pX = (p[0] + r, p[1])
                v = self.thermo_source_dTemps[k]
                if v < 0:
                    color2use = 'blue'
                else:
                    color2use = 'red'
                viewer.draw_circle(p, r, color2use, filled=False, width=0.0005)
                text_box = InputBox(text=str(np.round(v, 2)), color_active=self.default_color, visible=True)
                text_box.draw(viewer)


def create_odor_layers(model, sources, pars=None):
    odor_layers = {}
    ids = aux.unique_list([s.odor_id for s in sources if s.odor_id is not None])
    for id in ids:
        od_sources = [f for f in sources if f.odor_id == id]
        temp = aux.unique_list([s.default_color for s in od_sources])
        if len(temp) == 1:
            c0 = temp[0]
        elif len(temp) == 3 and all([type(k) == float] for k in temp):
            c0 = temp
        else:
            c0 = aux.random_colors(1)[0]
        kwargs = {
            'model': model,
            'unique_id': id,
            'sources': od_sources,
            'default_color': c0,
        }
        if pars.odorscape == 'Diffusion':
            odor_layers[id] = DiffusionValueLayer(grid_dims=pars['grid_dims'],
                                                        evap_const=pars['evap_const'],
                                                        gaussian_sigma=pars['gaussian_sigma'],
                                                        **kwargs)
        elif pars.odorscape == 'Gaussian':
            odor_layers[id] = GaussianValueLayer(**kwargs)
    return odor_layers