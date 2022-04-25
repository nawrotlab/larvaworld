import numpy as np
from Box2D import Box2D, b2ChainShape
from scipy.spatial.distance import euclidean
from shapely import affinity
from shapely.geometry import Point, Polygon

import lib.aux.sim_aux
from lib.aux.xy_aux import eudis5, xy_uniform_circle
from lib.model.DEB.substrate import Substrate
from lib.model.agents._agent import LarvaworldAgent


class Source(LarvaworldAgent):
    def __init__(self, shape_vertices=None, shape='circle', **kwargs):
        super().__init__(**kwargs)
        self.shape_vertices = shape_vertices
        shape = lib.aux.sim_aux.circle_to_polygon(60, self.radius)

        if self.model.Box2D:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=self.pos)
            self.Box2D_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.Box2D_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, self.pos)
        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

    def get_vertices(self):
        v0 = self.shape_vertices
        x0, y0 = self.get_position()
        if v0 is not None and not np.isnan((x0, y0)).all():
            return [(x + x0, y + y0) for x, y in v0]
        else:
            return None

    def get_shape(self, scale=1):
        p = self.get_position()
        if np.isnan(p).all():
            return None
        elif self.get_vertices() is None:
            return Point(p).buffer(self.radius * scale)
        else:
            p0 = Polygon(self.get_vertices())
            p = affinity.scale(p0, xfact=scale, yfact=scale)
            return p

    def step(self):
        if self.can_be_displaced:
            w = self.model.windscape
            dt = self.model.dt
            r = self.radius * 10000
            if w is not None:
                ws, wo = w.wind_speed, w.wind_direction
                if ws != 0.0:
                    self.pos = (self.pos[0] + np.cos(wo) * ws * dt / r, self.pos[1] + np.sin(wo) * ws * dt / r)
                    from lib.aux.sim_aux import inside_polygon
                    in_tank = inside_polygon(points=[self.pos], tank_polygon=self.model.tank_polygon)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else :
                            self.model.delete_agent(self)


class Food(Source):
    def __init__(self, amount=1.0, quality=1.0, default_color=None, type='standard', **kwargs):
        if default_color is None:
            default_color = 'green'
        super().__init__(default_color=default_color, **kwargs)
        self.initial_amount = amount
        self.amount = self.initial_amount
        self.substrate = Substrate(type=type, quality=quality)

    def get_amount(self):
        return self.amount

    def subtract_amount(self, amount):
        prev_amount = self.amount
        self.amount -= amount
        if self.amount <= 0.0:
            self.amount = 0.0
            self.model.delete_agent(self)
        else:
            r = self.amount / self.initial_amount
            self.color = (1 - r) * np.array((255, 255, 255)) + r * np.array(self.default_color)
        return np.min([amount, prev_amount])

    def draw(self, viewer, filled=True):
        # if not self.visible :
        #     return
        # if self.get_shape() is None :
        #     return
        p, c, r = self.get_position(), self.color, self.radius
        # viewer.draw_polygon(self.get_shape().boundary.coords, c, filled, r/5)
        viewer.draw_circle(p, r, c, filled, r / 5)
        # viewer.draw_circle((p[0]-r, p[1]), r/20, 'red', filled, r / 15)
        # viewer.draw_circle((p[0]+r, p[1]), r/20, 'red', filled, r / 15)
        # print(r)

        if self.odor_id is not None:
            if self.odor_intensity > 0:
                if self.model.odor_aura:
                    viewer.draw_circle(p, r * 1.5, c, False, r / 10)
                    viewer.draw_circle(p, r * 2.0, c, False, r / 15)
                    viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            # viewer.draw_polygon(self.get_shape(1.1).boundary.coords, self.model.selection_color, False, r / 5)
            viewer.draw_circle(p, r * 1.1, self.model.selection_color, False, r / 5)

    def contained(self, point):
        return eudis5(self.get_position(), point) <= self.radius
        # return euclidean(self.get_position(), point)<=self.radius
        # return Point(self.get_position()).distance(Point(point))<=self.radius
        # return Circle(self.get_position(), radius=self.radius).contains_point(point)

# @todo adding thermoscape class - need to edit functions within this class mix of gaussian GaussianValueLayer(ValueLayer)

class ThermoScape(ValueGrid):
    def __init__(self, pTemp, origins=[], tempDiff=[], default_color='green', visible=False):

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

# @todo need to make this return a dictionary of cool and warm.
    def get_thermo_value(self, pos):
        size,size2 = [1,1]
        # size, size2 = self.arena_dims * 1000  #it is in m and we want it in mm.
        pos_ad = [size*pos[0], size2*pos[1]]
        pos_temp = {}
        if self.thermoscape_layers is None:
            return 0 # or np.nan
        for k in layers[id].thermoscape_layers:
            v=layers[id].thermoscape_layers[k]
            pos_temp[k] = v.pdf(pos_ad) / v.pdf(layers[id].thermo_sources[k]) * (layers[id].thermo_source_dTemps[k] * len(layers[id].thermo_source_dTemps)) #@todo need to check if this works
        return layers[id].plate_temp + sum(pos_temp.values()) / len(pos_temp)


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


