import numpy as np
from scipy.ndimage.filters import gaussian_filter
from shapely.geometry import LineString, Point

from lib.anal.rendering import InputBox
from lib.aux.colsNstr import colorname2tuple, col_range
from lib.aux.dictsNlists import flatten_list, unique_list
from lib.model.DEB.substrate import Substrate


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
        self.meshgrid = np.meshgrid(np.linspace(x0, x1, self.X), np.linspace(y0, y1, self.Y))
        if distribution == 'uniform':
            self.grid = np.ones(self.grid_dims) * self.initial_value
        self.grid_vertices = self.generate_grid_vertices()
        self.grid_edges = [[-xr / 2, -yr / 2],
                           [xr / 2, -yr / 2],
                           [xr / 2, yr / 2],
                           [-xr / 2, yr / 2]]
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

        v0,v1 = self.min_value, self.max_value
        gg=(g-v0)/(v1-v0)
        k=10**2
        m=(1-np.exp(-k))**-1
        q=m*(1-np.exp(-k*gg))
        q=np.clip(q, a_min=0, a_max=1)
        # print()
        # print(v0,v1,g.shape, self.grid_dims)
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
        self.max_value = np.max(V.flatten())
        return V

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
        # rad_x, rad_y = D * dt / cell_width, D * dt / cell_height
        # temp = 10 ** 5
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
# @todo adding thermoscape class - need to edit functions within this class mix of gaussian GaussianValueLayer(ValueLayer)

class ThermoScape(ValueGrid):
    def __init__(self, pTemp, spread, origins=[], tempDiff=[], default_color='green', visible=False):

        self.plate_temp = pTemp
        self.thermo_sources = {str(i):o for i,o in enumerate(origins)} 
        self.thermo_source_dTemps = {str(i):o for i,o in enumerate(tempDiff)}
        # self.model = model
        # self.wind_direction = wind_direction
        # self.wind_speed = wind_speed
        # self.max_dim = np.max(self.model.arena_dims)
        self.default_color = default_color
        self.visible = visible

        # p0s = rotate_around_center_multi([(-self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # p1s = rotate_around_center_multi([(self.max_dim, (i - self.N / 2) * ds) for i in range(self.N)], -wind_direction)
        # self.scapelines=[(p0,p1) for p0,p1 in zip(p0s,p1s)]




    def update_values(self):
        pass


#@todo remove rezo, it is actually only important if I want to draw.
#thermo={'temp_spread':None, 'plate_temp':22, 'thermo_sources': None, 'thermo_differences': None}
    def generate_thermoscape(self, spread=0.1, pTemp = 22, origins = [[0.5,0.05], [0.05,0.5], [0.5,0.95], [0.95,0.5]], tempDiff = [8,-8,8,-8]):
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
        
        # size,size2 = [1,1]
        # size, size2 = self.arena_dims * 1000 #model.grid_dims #it is in m and we want it in mm.
        # if spread is None:
        #     spread = size * 10

        self.thermo_spread = spread 
        self.plate_temp = pTemp 
        self.thermo_sources = {str(i):o for i,o in enumerate(origins)} 
        self.thermo_source_dTemps = {str(i):o for i,o in enumerate(tempDiff)}
        if len(origins) != len(tempDiff):
            raise ValueError # need to raise a more informative error.
        # origins =  [[0.5,0.05], [0.05,0.5], [0.5,0.95], [0.95,0.5]] #  [[85,8.5], [8.5,85], [85,161.5], [161.5,85]]
        # origins_ad = [[size*og[0], size2*og[1]] for og in origins] # origins on arena dimensions

        # x, y = np.mgrid[0:size:rezo, 0:size2:rezo] # setting 170 x 170 grid #don't need to do this anymore
        # pos = np.dstack((x, y)) 

        rv_dict = {}
        # thermoDists_Dict = {}
        for k in self.thermo_sources:
            # v_ad = [size*v[0], size2*v[1]]
            rv_dict[k] = multivariate_normal(self.thermo_sources[k], [[spread, 0], [0, spread]])
            # thermoDists_Dict[k] = (rv_dict[k].pdf(pos)/rv_dict[k].pdf(v_ad))*(tempDiff[k] * len(origins)) # don't need this either
        
        self.thermoscape_layers = rv_dict
        # plt.imshow(22 + rv, cmap='hot', interpolation='nearest'); plt.colorbar(); plt.show()
        # plt.hist(pTemp + rv.flatten(), bins=50 ); plt.show()
        # self.thermo_dist = pTemp + sum(thermoDists_Dict.values()) / len(thermoDists_Dict) # I do not need to store this anymore! - so i won't need SIZE anymore. alternatively I could just store thermoDists_Dict and get_thermo_value calculate each time with plateTemp (if this is memory inefificent)
        # return  pTemp + sum(thermoDists_Dict.values()) / len(thermoDists_Dict)


    # def get_thermo_value(self, pos):
    #     size,size2 = [1,1]
    #     # size, size2 = self.arena_dims * 1000  #it is in m and we want it in mm.
    #     pos_ad = [size*pos[0], size2*pos[1]]
    #     pos_temp = {}
    #     if self.thermoscape_layers is None:
    #         return 0 # or np.nan
    #     for k in self.thermoscape_layers:
    #         v=self.thermoscape_layers[k]
    #         pos_temp[k] = v.pdf(pos_ad) / v.pdf(self.thermo_sources[k]) * (self.thermo_source_dTemps[k] * len(self.thermo_source_dTemps)) #@todo need to check if this works
    #     return self.plate_temp + sum(pos_temp.values()) / len(pos_temp)

    def get_thermo_value(self, pos):

        size,size2=[1,1]
        pos_ad = [size*pos[0], size2*pos[1]]
        pos_temp = {}
        nSources = len(self.thermo_source_dTemps)
        thermo_gain = {'cool':0, 'warm':0}

        if self.thermoscape_layers is None:
            print(0) # or np.nan
        for k in self.thermoscape_layers:
            v=self.thermoscape_layers[k]
            pos_temp[k] = v.pdf(pos_ad) / v.pdf(self.thermo_sources[k]) * (self.thermo_source_dTemps[k] * nSources) #@todo need to check if this works
            # print(plate_temp + sum(pos_temp.values()) / len(pos_temp))
            # print(plate_temp + pos_temp[k] / len(pos_temp))
            print(pos_temp[k] / nSources)
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

    def draw_isocontours(self, viewer): #@todo need to make a draw function for thermogrid.
        # g=self.get_grid()
        # vs=np.linspace(np.min(g), np.max(g), 5)
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
