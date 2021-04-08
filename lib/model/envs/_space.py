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

        x_linspace = np.linspace(x0, x1, self.X)
        y_linspace = np.linspace(y0, y1, self.Y)
        self.meshgrid = np.meshgrid(x_linspace, y_linspace)


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
    def __init__(self, space, unique_id, color, space_range, sources=[],
                 visible=False, grid_dims=[50, 50],**kwargs):
        self.space = space
        self.id = unique_id
        self.color = color
        self.sources = sources
        self.value_grid=ValueGrid(space_range=space_range, default_color=color, grid_dims=grid_dims)
        # self.space2grid=space2grid
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
        X,Y=self.value_grid.meshgrid
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


    def __init__(self, dt, scaling_factor, evap_const,gaussian_sigma, **kwargs):
        super().__init__(**kwargs)
        '''
            A typical diffusion coefficient for a molecule in the gas phase is in the range of 10-6 to 10-5 m2/s
            
            

Yes, it does that automatically based on the sigma and truncate parameters.
Indeed, the function gaussian_filter is implemented by applying multiples 1D gaussian filters (you can see that here). 
This function uses gaussian_filter1d which generate itself the kernel using _gaussian_kernel1d with a radius of 
int(truncate * sigma + 0.5).


            Doing the math, sigma ends up reeeeally small
        '''
        D = 10**-6
        cell_width, cell_height = self.value_grid.x/scaling_factor, self.value_grid.y/scaling_factor
        rad_x, rad_y = D*dt/cell_width, D*dt/cell_height
        temp=10**5
        sigma=int(rad_x*temp), int(rad_y*temp)
        self.evap_const = evap_const
        self.sigma = gaussian_sigma


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


    def update_values(self):
        for s in self.sources:
            source_pos = s.get_position()
            intensity = s.get_odor_intensity()
            self.add_value(source_pos, intensity)
        self.value_grid.grid = gaussian_filter(self.value_grid.grid, sigma=self.sigma)*self.evap_const





