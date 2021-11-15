import numpy as np
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import LineString, Point

from lib.anal.rendering import InputBox
from lib.aux.colsNstr import colorname2tuple, col_range
from lib.aux.dictsNlists import flatten_list, unique_list
from lib.model.DEB.deb import Substrate


class ValueGrid:
    def __init__(self,model,  unique_id, space_range, grid_dims=[51, 51], distribution='uniform', visible=False,
                 initial_value=0.0, default_color=(255, 255, 255), max_value=None, min_value=0.0, fixed_max=False):
        self.model = model
        self.visible = visible
        self.unique_id = unique_id
        self.initial_value = initial_value

        self.min_value = min_value
        self.fixed_max = fixed_max
        if type(default_color) == str:
            default_color = colorname2tuple(default_color)
        self.default_color = default_color
        self.grid_dims = grid_dims
        self.X, self.Y = grid_dims
        # print(space_range)
        x_range = tuple(space_range[0:2])
        y_range = tuple(space_range[2:])
        x0, x1 = x_range[0], x_range[1]
        y0, y1 = y_range[0], y_range[1]
        xr, yr = x1 - x0, y1 - y0
        self.x = xr / self.X
        self.y = yr / self.Y
        self.cell_radius=np.sqrt(np.sum((self.x/2)**2+(self.y/2)**2))
        self.xy = np.array([self.x, self.y])
        self.XY_half = np.array([self.X / 2, self.Y / 2])

        x_linspace = np.linspace(x0, x1, self.X+1)
        y_linspace = np.linspace(y0, y1, self.Y+1)
        self.meshgrid = np.meshgrid(x_linspace, y_linspace)
        # print(self.X, self.Y, self.x, self.y)
        if distribution == 'uniform':
            self.grid = np.ones(self.grid_dims) * self.initial_value

        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]
        # self.grid[3,:]=300.0
        # print(self.grid_edges)
        if max_value is None :
            max_value=np.max(self.grid)
        self.max_value=max_value


    def add_value(self, p, value):
        cell = self.get_grid_cell(p)
        # print(p,cell)
        v = self.add_cell_value(cell, value)
        return v

    def set_value(self, p, value):
        cell = self.get_grid_cell(p)
        self.set_cell_value(cell, value)

    def get_value(self, p):
        cell = self.get_grid_cell(p)
        return self.get_cell_value(cell)

    def get_grid_cell(self, p):
        # print(p)
        # print(p/ self.xy)
        # print(p/ self.xy+ self.XY_half)
        c = np.floor(p / self.xy + self.XY_half).astype(int)
        # print(c)
        # raise
        return tuple(c)
        # return tuple(c)

    def get_cell_value(self, cell):
        return self.grid[cell]

    def set_cell_value(self, cell, value):
        self.grid[cell] = value

    def add_cell_value(self, cell, value):
        # print(cell)
        v0 = self.get_cell_value(cell)
        v1 = v0 + value
        if not self.fixed_max :
            self.max_value=np.max([self.max_value, v1])
        v2 = np.clip(v1, a_min=self.min_value, a_max=self.max_value)
        self.set_cell_value(cell, v2)
        if v1 < v2:
            return self.min_value - v0
        elif v1 > v2:
            return self.max_value - v0
        else:
            return value

    def generate_grid_vertices(self):
        vertices = np.zeros([self.X,self.Y,4,2])
        for i in range(self.X):
            for j in range(self.Y):
                vertices[i,j]=self.cell_vertices(i, j)
        return vertices

    def cell_vertices(self, i, j):
        x, y = self.x, self.y
        X, Y = self.X / 2, self.Y / 2
        v = np.array([(x * (i - X), y * (j - Y)),
             (x * (i + 1 - X), y * (j - Y)),
             (x * (i + 1 - X), y * (j + 1 - Y)),
             (x * (i - X), y * (j + 1 - Y))])
        return v

    def cel_pos(self, i, j):
        return self.x * (i - self.X / 2 + 0.5), self.y * (j - self.Y / 2 + 0.5)

    def reset(self):
        self.grid = np.ones(self.grid_dims) * self.initial_value

    def empty_grid(self):
        self.grid = np.zeros(self.grid_dims)

    def draw_peak(self, viewer):
        idx = np.unravel_index(self.grid.argmax(), self.grid.shape)
        p = self.cel_pos(*idx)
        vs = self.cell_vertices(*idx)

        viewer.draw_circle(p, self.cell_radius/2, self.default_color, filled=True, width=0.0005)
        # viewer.draw_polygon(vs, self.default_color, filled=True, width=0.0005)

        p_text = (p[0] + self.x, p[1] - self.y)
        text_box = InputBox(text=str(np.round(self.grid.max(), 2)), color_active=self.default_color, visible=True,screen_pos=viewer._transform(p_text))
        text_box.draw(viewer)

    def draw(self, viewer):
        Cgrid = self.get_color_grid().reshape([self.X,self.Y,3])
        for i in range(self.X):
            for j in range(self.Y):
                viewer.draw_polygon(self.grid_vertices[i,j], Cgrid[i,j], filled=True)
        # print(color_grid.shape)
        # for vertices, col in zip(self.grid_vertices, color_grid):
        #     viewer.draw_polygon(vertices, col, filled=True)
        self.draw_peak(viewer)
        if self.model.odor_aura :

            self.draw_isocontours(viewer)

    def draw_isocontours(self, viewer):
        N = 8
        k=4
        g = self.get_grid()
        # c='white'
        c=self.default_color
        vmax = np.max(g)
        for i in range(N):
            v = vmax *k**-i
            if v<=0 :
                continue
            inds = np.argwhere((v <= g) & (g < v*k)).tolist()
            points = [self.cel_pos(i, j) for (i, j) in inds]
            if len(points) > 2:
                try:
                    ps=np.array(points)
                    pxy=ps[np.argmax(ps[:,0]),:]+np.array([self.x, -self.y])
                    viewer.draw_convex(points, color=c, filled=False, width=0.0005)
                    text_box = InputBox(text=str(np.round(v, 2)), color_active=c, visible=True,
                                        screen_pos=viewer._transform(pxy))
                    text_box.draw(viewer)
                except :
                    pass


    def get_color_grid(self):
        g = self.get_grid().flatten()
        # if not self.fixed_max :
        #     self.max_value=np.max([self.max_value, np.max(g)])
        v0,v1 = self.min_value, self.max_value
        gg=(g-v0)/(v1-v0)
        k=10**2
        m=(1-np.exp(-k))**-1
        q=m*(1-np.exp(-k*gg))
        q=np.clip(q, a_min=0, a_max=1)
        return col_range(q, low=(255, 255, 255), high=self.default_color, mul255=True)

    def get_grid(self):
        return self.grid


class FoodGrid(ValueGrid):
    def __init__(self, default_color=(0, 255, 0), quality=1, type='standard', **kwargs):
        super().__init__(default_color=default_color,fixed_max=True, **kwargs)
        self.substrate = Substrate(type=type, quality=quality)
        # self.max_value=self.initial_value

    def get_color(self, v):
        v0, v1 = self.min_value, self.max_value
        q = (v - v0) / (v1 - v0)
        return col_range(q, low=(255, 255, 255), high=self.default_color, mul255=True)

    def draw(self, viewer):
        viewer.draw_polygon(self.grid_edges, self.get_color(v=self.initial_value), filled=True)
        for i in range(self.X):
            for j in range(self.Y):
                v=self.grid[i,j]
                if v != self.initial_value :
                    viewer.draw_polygon(self.grid_vertices[i,j], self.get_color(v), filled=True)


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
        return V

    # def draw(self, viewer):
    def draw_isocontours(self, viewer):
        # g=self.get_grid()
        # vs=np.linspace(np.min(g), np.max(g), 5)
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
        # sigma = int(rad_x * temp), int(rad_y * temp)
        self.evap_const = evap_const
        self.sigma = gaussian_sigma
        # print(gaussian_sigma)

    def update_values(self):
        k=1000
        if self.model.windscape is not None :
            v,a=self.model.windscape.wind_speed, self.model.windscape.wind_direction
            if v!=0 :
                dx=v*np.cos(a)*self.model.dt
                dy=v*np.sin(a)*self.model.dt
                Px,Py=dx/self.x/k, dy/self.y/k

                Pr=np.abs(Px/(Px+Py))
                # print(Pr, Px, Py, self.max_value)
                Px=np.clip(Px, a_min=-Pr, a_max=Pr)
                Py=np.clip(Py, a_min=-1+Pr, a_max=1-Pr)
                # print(Pr, Px, Py, self.max_value)
                Gx=self.grid*Px
                Gy=self.grid*Py
                Gx=np.roll(Gx,1, axis=0)
                Gx[0,:]=0
                Gy=np.roll(Gy,1, axis=1)
                Gy[:,0] = 0
                self.grid*=(1-Px-Py)
                self.grid+=(Gx+Gy)
                np.clip(self.grid, a_min=0, a_max=None)


        for s in self.sources:
            source_pos = s.get_position()
            intensity = s.odor_intensity
            self.add_value(source_pos, intensity)

        self.grid = gaussian_filter(self.grid, sigma=self.sigma) * self.evap_const


class WindScape:
    def __init__(self, model, wind_direction, wind_speed, puffs={}, default_color='red', visible=False):

        self.model = model
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.max_dim = np.max(self.model.arena_dims)
        self.default_color = default_color
        self.visible = visible

        self.N = 40
        self.draw_phi = 0
        self.scapelines = self.generate_scapelines(self.max_dim, self.N, self.wind_direction)
        # p0s = rotate_around_center_multi([(-self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # p1s = rotate_around_center_multi([(self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # self.scapelines=[(p0,p1) for p0,p1 in zip(p0s,p1s)]
        self.events = {}
        for idx, puff in puffs.items():
            self.add_puff(**puff)

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
        if self.wind_speed > 0:
            for p0, p1 in self.scapelines:
                l = LineString([p0, p1])
                ps = [l.intersection(b) for b in self.model.border_lines if l.intersects(b)]
                if len(ps) != 0:
                    p1 = ps[np.argmin([Point(p0).distance(p2) for p2 in ps])].coords[0]
                viewer.draw_arrow_line(p0, p1, self.default_color, width=0.001,
                                       phi=(self.draw_phi % 1000) / 1000)
        self.draw_phi += self.wind_speed

    def generate_scapelines(self, D, N, A):
        from lib.aux.ang_aux import rotate_around_center_multi
        ds = self.max_dim / N * np.sqrt(2)
        p0s = rotate_around_center_multi([(-D, (i - N / 2) * ds) for i in range(N)], -A)
        p1s = rotate_around_center_multi([(D, (i - N / 2) * ds) for i in range(N)], -A)
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
