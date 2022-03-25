from multiprocessing.sharedctypes import Value
import numpy as np
from scipy.stats import multivariate_normal
from shapely.geometry import Point

from lib.aux.colsNstr import colorname2tuple
from lib.anal.rendering import InputBox


class LarvaworldAgent:
    def __init__(self,unique_id: str,model, pos=None, default_color=None, radius=None,visible=True,
                 odor={'odor_id':None, 'odor_intensity':None, 'odor_spread':None},thermo={'temp_spread':None, 'plate_temp':22, 'thermo_origins': None, 'thermo_differences': None},regeneration=False,regeneration_pos=None,
                 group='', can_be_carried=False, can_be_displaced=False, **kwargs):
        self.visible = visible
        self.selected = False
        self.unique_id = unique_id
        self.model = model
        self.group = group
        self.base_odor_id = f'{group}_base_odor'
        self.gain_for_base_odor = 100

        self.initial_pos = pos
        self.pos = self.initial_pos
        if type(default_color) == str:
            default_color = colorname2tuple(default_color)
        self.default_color = default_color
        self.color = self.default_color
        self.radius = radius
        self.id_box = InputBox(text=self.unique_id,color_inactive=self.default_color, color_active=self.default_color,agent=self)
        self.odor_id = odor['odor_id']
        self.set_odor_dist(odor['odor_intensity'], odor['odor_spread'])
        #self.set_thermo_dist()
        self.carried_objects = []
        self.can_be_carried = can_be_carried
        self.can_be_displaced = can_be_displaced
        self.is_carried_by = None
        self.regeneration = regeneration
        self.regeneration_pos = regeneration_pos

    def get_position(self):
        return tuple(self.pos)

    # def get_radius(self):
    #     return self.radius

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text = self.unique_id

    def get_shape(self, scale=1):
        p = self.get_position()
        return Point(p).buffer(self.radius*scale) if not np.isnan(p).all() else None

    def set_color(self, color):
        self.color = color

    def contained(self, point):
        # return Point(self.get_position()).distance(Point(point))<=self.radius
        # return Circle(self.get_position(), radius=self.radius).contains_point(point)
        shape = self.get_shape()
        return shape.covers(Point(point)) if shape else False

    # @abc.abstractmethod
    def step(self):
        pass

    def set_default_color(self, color):
        self.default_color = color
        self.id_box.color = self.default_color
        self.set_color(color)

    def set_odor_dist(self, intensity=None, spread=None):
        self.odor_intensity=intensity
        self.odor_spread=spread
        if intensity is not None and spread is not None:
            self.odor_dist = multivariate_normal([0, 0], [[self.odor_spread, 0], [0, self.odor_spread]])
            self.odor_peak_value = self.odor_intensity / self.odor_dist.pdf([0, 0])

    def get_gaussian_odor_value(self, pos):
        return self.odor_dist.pdf(pos) * self.odor_peak_value

#@todo need to add set_thermo_dist and get_thermo_value
# for get_thermo_value I need to see what resolution we want data at (I guess at 10th of a mm will be enough) - so set_thermo_dist needs to be 1700x1700 so its a 10th of a mm.


    def set_thermo_dist(self, rezo=0.1, spread=1700, pTemp = 22, origins = [[85,10], [10,85], [85,160], [160,85]], tempDiff = [8,-8,8,-8], size):
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
        size, size2 = self.model.arena_dims * 1000 #it is in m and we want it in mm.
        if spread is None:
            spread = size * 10

        self.thermo_spread = spread 
        self.plate_temp = pTemp 
        self.thermo_origins = {str(i):o for i,o in enumerate(origins)} # @todo need to put this in dictionary (with same 1 2 3 4 and rv_dict)
        self.thermo_origins_dTemp = {str(i):o for i,o in enumerate(tempDiff)} # @todo need to put this in dictionary (with same 1 2 3 4 and rv_dict)
        if len(origins) != len(tempDiff):
            raise ValueError # need to raise a more informative error.

        x, y = np.mgrid[0:size:rezo, 0:size2:rezo] # setting 170 x 170 grid
        pos = np.dstack((x, y))

        rv_dict = {}
        thermoDists_Dict = {}
        for k,v in enumerate(origins):
            rv_dict[k] = multivariate_normal(v, [[spread, 0], [0, spread]])
            thermoDists_Dict[k] = (rv_dict[k].pdf(pos)/rv_dict[k].pdf(v))*(tempDiff[k] * len(origins))
        
        self.thermo_dist_raw = rv_dict
        # plt.imshow(22 + rv, cmap='hot', interpolation='nearest'); plt.colorbar(); plt.show()
        # plt.hist(pTemp + rv.flatten(), bins=50 ); plt.show()
        self.thermo_dist = pTemp + sum(thermoDists_Dict.values()) / len(thermoDists_Dict) # alternatively I could just store thermoDists_Dict and get_thermo_value calculate each time with plateTemp (if this is memory inefificent)
        # return  pTemp + sum(thermoDists_Dict.values()) / len(thermoDists_Dict)

    def get_thermo_value(self, pos):
        pos_temp = {}
        for k,v in self.thermo_dist_raw:
            v.pdf[pos * 1000] / v.pdf(self.origins[k]) * (self.thermo_origins_dTemp[k] * len(self.thermo_origins_dTemp)) #@todo need to check if this works
        return self.plate_temp + sum(pos_temp.values()) / len(pos_temp)
        # x,y = pos * 1000 #@todo need to multiply x and y based on what unit they are in i.e. if they are in mm multiple by 10
        # return self.thermo_dist[y,x]

    def draw(self, viewer, filled=True):
        if self.get_shape() is None :
            return
        p, c, r = self.get_position(), self.color, self.radius
        viewer.draw_polygon(self.get_shape().boundary.coords, c, filled, r/5)
        # viewer.draw_circle(p, r, c, filled, r / 5)

        if self.odor_intensity > 0:
            viewer.draw_polygon(self.get_shape(1.5).boundary.coords, c, False, r / 10)
            viewer.draw_polygon(self.get_shape(2.0).boundary.coords, c, False, r / 15)
            viewer.draw_polygon(self.get_shape(3.0).boundary.coords, c, False, r / 20)
            # viewer.draw_circle(p, r * 1.5, c, False, r / 10)
            # viewer.draw_circle(p, r * 2.0, c, False, r / 15)
            # viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            viewer.draw_polygon(self.get_shape(1.1).boundary.coords, self.model.selection_color, False, r / 5)
            # viewer.draw_circle(p, r * 1.2, self.model.selection_color, False, r / 5)


