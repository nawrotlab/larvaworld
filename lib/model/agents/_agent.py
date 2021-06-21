from copy import deepcopy
import numpy as np
from Box2D import Box2D, b2ChainShape
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal
from shapely import affinity
from shapely.geometry import Point, Polygon

import lib.aux.functions as fun
import lib.aux.rendering as ren
from lib.model.DEB.deb import Substrate


class LarvaworldAgent:
    def __init__(self,unique_id: str,model, pos=None, default_color=None, radius=None,
                 odor_id=None, odor_intensity=0.0, odor_spread=0.1, group='', can_be_carried=False):
        self.selected = False
        self.unique_id = unique_id
        self.model = model
        self.group = group
        self.base_odor_id = f'{group}_base_odor'
        self.gain_for_base_odor = 100

        self.initial_pos = pos
        self.pos = self.initial_pos
        if type(default_color) == str:
            default_color = fun.colorname2tuple(default_color)
        self.default_color = default_color
        self.color = self.default_color
        self.radius = radius
        self.id_box = self.init_id_box()
        self.odor_id = odor_id
        self.odor_intensity = odor_intensity
        if odor_spread is None:
            odor_spread = 0.1
        self.odor_spread = odor_spread
        self.set_odor_dist()

        self.carried_objects = []
        self.can_be_carried = can_be_carried
        self.is_carried_by = None

    def get_position(self):
        return tuple(self.pos)

    def get_radius(self):
        return self.radius

    def init_id_box(self):
        id_box = ren.InputBox(visible=False, text=self.unique_id,
                              color_inactive=self.default_color, color_active=self.default_color,
                              screen_pos=None, agent=self)
        return id_box

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

    def set_odor_dist(self):
        self.odor_dist = multivariate_normal([0, 0], [[self.odor_spread, 0], [0, self.odor_spread]])
        self.odor_peak_value = self.odor_intensity / self.odor_dist.pdf([0, 0])

    def set_odor_id(self, odor_id):
        self.odor_id = odor_id

    def get_odor_id(self):
        return self.odor_id

    def get_odor_intensity(self):
        return self.odor_intensity

    def get_odor_spread(self):
        return self.odor_spread

    def set_odor(self, odor_id, intensity=2, spread=0.0002):
        self.set_odor_id(odor_id)
        self.set_scaled_odor_intensity(intensity)
        self.set_scaled_odor_spread(spread)
        self.set_odor_dist()

    def get_gaussian_odor_value(self, pos):
        return self.odor_dist.pdf(pos) * self.odor_peak_value

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


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color=None, **kwargs):
        if default_color is None:
            default_color = model.generate_larva_color()
        super().__init__(unique_id=unique_id, model=model, default_color=default_color, pos=pos, radius=radius,
                         **kwargs)
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))

    def update_color(self, default_color, behavior_dict, mode='lin'):
        color = deepcopy(default_color)
        if mode == 'lin':
            # if beh_dict['stride_stop'] :
            #     color=np.array([0, 255, 0])
            if behavior_dict['stride_id']:
                color = np.array([0, 150, 0])
            elif behavior_dict['pause_id']:
                color = np.array([255, 0, 0])
            elif behavior_dict['feed_id']:
                color = np.array([0, 0, 255])
        elif mode == 'ang':
            if behavior_dict['Lturn_id']:
                color[2] = 150
            elif behavior_dict['Rturn_id']:
                color[2] = 50
        return color

    @property
    def turner_activation(self):
        return self.brain.turner.activation

    @property
    def olfactory_activation(self):
        return self.brain.olfactory_activation

    @property
    def first_odor_concentration(self):
        return list(self.brain.olfactor.Con.values())[0]

    @property
    def second_odor_concentration(self):
        return list(self.brain.olfactor.Con.values())[1]

    @property
    def first_odor_best_gain(self):
        return list(self.brain.memory.best_gain.values())[0]

    @property
    def second_odor_best_gain(self):
        return list(self.brain.memory.best_gain.values())[1]

    @property
    def best_olfactor_decay(self):
        return self.brain.memory.best_decay_coef

    best_olfactor_decay

    @property
    def cum_reward(self):
        return self.brain.memory.rewardSum

    @property
    def first_odor_concentration_change(self):
        return list(self.brain.olfactor.dCon.values())[0]

    # @property
    # def length_in_mm(self):
    #     return self.get_real_length() * 1000

    @property
    def length_in_mm(self):
        return self.get_real_length() * 1000
        # from lib.conf.par import TemporalPar, FractionPar
        # k1 = TemporalPar(name='cum_dur')
        # k2 = TemporalPar(name='cum_dur')
        # k=FractionPar(name='some', exists=False, numerator=k1, denominator=k2)
        # v = k.get_from(self)
        # print(v)
        # return v

    @property
    def mass_in_mg(self):
        return self.get_real_mass() * 1000

    @property
    def scaled_amount_eaten(self):
        return self.amount_eaten / self.get_real_mass()

    @property
    def orientation_to_center_in_deg(self):
        return fun.angle_dif(np.rad2deg(self.get_head().get_normalized_orientation()),
                             fun.angle_to_x_axis(self.get_position(), (0, 0),
                                                 in_deg=True), in_deg=True)

    @property
    def x(self):
        return self.pos[0] * 1000 / self.model.scaling_factor

    @property
    def y(self):
        return self.pos[1] * 1000 / self.model.scaling_factor

    @property
    def dispersion_in_mm(self):
        return euclidean(tuple(self.pos),
                         tuple(self.initial_pos)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dispersion(self):
        return euclidean(tuple(self.pos),
                         tuple(self.initial_pos)) / self.get_sim_length()

    @property
    def cum_dst_in_mm(self):
        return self.cum_dst * 1000 / self.model.scaling_factor

    @property
    def cum_scaled_dst(self):
        return self.cum_dst / self.get_sim_length()

    @property
    def dst_to_center_in_mm(self):
        return np.sqrt(np.sum(np.array(self.pos)**2)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_center(self):
        return np.sqrt(np.sum(np.array(self.pos)**2)) / self.get_sim_length()

    @property
    def dst_to_chemotax_odor_in_mm(self):
        return euclidean(tuple(self.pos),
                         (0.8, 0.0)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_chemotax_odor(self):
        return euclidean(tuple(self.pos),
                         (0.8, 0.0)) / self.get_sim_length()

    @property
    def max_dst_to_center_in_mm(self):
        return np.nanmax(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1))) * 1000/ self.model.scaling_factor

    @property
    def max_scaled_dst_to_center(self):
        d = np.nanmax(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))/ self.get_sim_length()
        return d

    @property
    def mean_dst_to_center_in_mm(self):
        return np.nanmean(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))* 1000 / self.model.scaling_factor

    @property
    def mean_scaled_dst_to_center(self):
        d = np.nanmean(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))/ self.get_sim_length()
        return d

    @property
    def dispersion_max_in_mm(self):
        return np.max([euclidean(tuple(self.trajectory[i]),
                                 tuple(self.initial_pos)) for i in
                       range(len(self.trajectory))]) * 1000 / self.model.scaling_factor

    @property
    def scaled_dispersion_max(self):
        return np.max([euclidean(tuple(self.trajectory[i]),
                                 tuple(self.initial_pos)) for i in
                       range(len(self.trajectory))]) / self.get_sim_length()

    @property
    def stride_dst_mean_in_mm(self):
        return (self.cum_dst / self.brain.crawler.iteration_counter) * 1000 / self.model.scaling_factor

    @property
    def stride_scaled_dst_mean(self):
        return (self.cum_dst / self.get_sim_length()) / self.brain.crawler.iteration_counter

    @property
    def crawler_freq(self):
        return self.brain.crawler.freq

    @property
    def num_strides(self):
        return self.brain.crawler.iteration_counter if self.brain.crawler is not None else self.brain.intermitter.stride_counter

    @property
    def stride_dur_ratio(self):
        return self.brain.crawler.total_t / self.cum_dur

    @property
    def pause_dur_ratio(self):
        return self.brain.intermitter.cum_pause_dur / self.cum_dur

    @property
    def stridechain_dur_ratio(self):
        return self.brain.intermitter.cum_stridechain_dur / self.cum_dur

    @property
    def pause_start(self):
        return self.brain.intermitter.pause_start

    @property
    def pause_stop(self):
        return self.brain.intermitter.pause_stop

    @property
    def pause_dur(self):
        return self.brain.intermitter.pause_dur

    @property
    def pause_id(self):
        return self.brain.intermitter.pause_id

    @property
    def stridechain_start(self):
        return self.brain.intermitter.stridechain_start

    @property
    def stridechain_stop(self):
        return self.brain.intermitter.stridechain_stop

    @property
    def stridechain_dur(self):
        return self.brain.intermitter.stridechain_dur

    @property
    def stridechain_id(self):
        return self.brain.intermitter.stridechain_id

    @property
    def stridechain_length(self):
        return self.brain.intermitter.stridechain_length

    @property
    def num_pauses(self):
        return self.brain.intermitter.pause_counter

    @property
    def cum_pause_dur(self):
        return self.brain.intermitter.cum_pause_dur

    @property
    def num_stridechains(self):
        return self.brain.intermitter.stridechain_counter

    @property
    def cum_stridechain_dur(self):
        return self.brain.intermitter.cum_stridechain_dur

    @property
    def num_feeds(self):
        return self.brain.feeder.iteration_counter if self.brain.feeder is not None else self.brain.intermitter.feed_counter

    @property
    def mean_feed_freq(self):
        return self.num_feeds / self.cum_dur

    @property
    def feed_dur_ratio(self):
        return self.brain.feeder.total_t / self.cum_dur

    @property
    def feed_success_rate(self):
        return self.feed_success_counter / self.brain.feeder.iteration_counter

    @property
    def deb_f(self):
        return self.deb.get_f()

    @property
    def deb_f_mean(self):
        return np.mean(self.deb.dict['f'])

    @property
    def gut_occupancy(self):
        return self.deb.gut.get_gut_occupancy()

    @property
    def ingested_body_mass_ratio(self):
        return self.deb.gut.ingested_mass()/self.deb.Ww*100

    @property
    def ingested_body_volume_ratio(self):
        return self.deb.gut.ingested_volume()/self.deb.V *100

    @property
    def ingested_gut_volume_ratio(self):
        return self.deb.gut.ingested_volume() / (self.deb.V*self.deb.gut.V_gm) * 100

    @property
    def ingested_body_area_ratio(self):
        return (self.deb.gut.ingested_volume()/self.deb.V)**(1/2)*100
        # return (self.deb.gut.ingested_volume()/self.deb.V)**(2/3)*100

    @property
    def amount_absorbed(self):
        return self.deb.gut.absorbed_mass('mg')

    @property
    def amount_faeces(self):
        return self.deb.gut.get_M_faeces()

    @property
    def faeces_ratio(self):
        return self.deb.gut.get_R_faeces()

    @property
    def food_absorption_efficiency(self):
        return self.deb.gut.get_R_absorbed()

    @property
    def deb_f_deviation(self):
        return self.deb.get_f() - 1

    @property
    def deb_f_deviation_mean(self):
        return np.mean(np.array(self.deb.dict['f']) - 1)

    @property
    def reserve(self):
        return self.deb.get_E()

    @property
    def reserve_density(self):
        return self.deb.get_e()

    @property
    def structural_length(self):
        return self.deb.get_L()

    @property
    def maturity(self):
        return self.deb.get_E_H() * 1000 #in mJ

    @property
    def reproduction(self):
        return self.deb.get_E_R() * 1000 #in mJ

    @property
    def puppation_buffer(self):
        return self.deb.get_pupation_buffer()

    @property
    def structure(self):
        return self.deb.get_V() * self.deb.E_V * 1000 #in mJ

    @property
    def age_in_hours(self):
        return self.deb.age * 24

    @property
    def hunger(self):
        return self.deb.hunger

    @property
    def death_time_in_hours(self):
        return self.deb.death_time_in_hours

    @property
    def pupation_time_in_hours(self):
        return self.deb.pupation_time_in_hours

    @property
    def birth_time_in_hours(self):
        return self.deb.birth_time_in_hours

    @property
    def hours_as_larva(self):
        return self.deb.hours_as_larva

    # @property
    # def feeder_reoccurence_rate(self):
    #     return self.brain.intermitter.feeder_reoccurence_rate

    @property
    def explore2exploit_balance(self):
        return self.brain.intermitter.EEB


class Source(LarvaworldAgent):
    def __init__(self, shape_vertices=None, shape='circle', **kwargs):
        super().__init__(**kwargs)
        self.shape_vertices = shape_vertices
        shape = fun.circle_to_polygon(60, self.radius)

        if self.model.physics_engine:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=self.pos)
            # else:
            #     # all parameters in real space units
            #     self._body: Box2D.b2Body = self.space.CreateDynamicBody(pos=self.pos, orientation=None)
            #     # super().__init__(space=self.space, pos=self.pos, orientation=None, radius=self.shape_radius)
            #     self._body.bullet = True
            self.Box2D_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.Box2D_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, self.pos)
        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

    def get_vertices(self):
        v0=self.shape_vertices
        x0, y0 = self.get_position()
        if v0 is not None and not np.isnan((x0,y0)).all():
            return [(x+x0,y+y0) for x,y in v0]
        else :
            return None

    def get_shape(self, scale=1):
        p = self.get_position()
        if np.isnan(p).all() :
            return None
        elif self.get_vertices() is None :
            return Point(p).buffer(self.radius*scale)
        else :
            p0 = Polygon(self.get_vertices())
            p = affinity.scale(p0, xfact=scale, yfact=scale)
            return p

class Food(Source):
    def __init__(self, amount=1.0, quality=1.0,default_color=None,type='standard', **kwargs):
        # print(kwargs)
        if default_color is None :
            default_color = 'green'
        super().__init__(default_color=default_color,**kwargs)
        self.initial_amount = amount
        self.quality = quality
        self.amount = self.initial_amount
        self.type = type
        self.substrate = Substrate(type=type)

    # def get_mol(self, V, **kwargs):
    #     return self.substrate.get_mol(V=V, quality=self.quality, **kwargs)

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
        # if self.get_shape() is None :
        #     return
        p, c, r = self.get_position(), self.color, self.radius
        # viewer.draw_polygon(self.get_shape().boundary.coords, c, filled, r/5)
        viewer.draw_circle(p, r, c, filled, r / 5)

        if self.odor_intensity > 0:
            # viewer.draw_polygon(self.get_shape(1.5).boundary.coords, c, False, r / 10)
            # viewer.draw_polygon(self.get_shape(2.0).boundary.coords, c, False, r / 15)
            # viewer.draw_polygon(self.get_shape(3.0).boundary.coords, c, False, r / 20)
            viewer.draw_circle(p, r * 1.5, c, False, r / 10)
            viewer.draw_circle(p, r * 2.0, c, False, r / 15)
            viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            # viewer.draw_polygon(self.get_shape(1.1).boundary.coords, self.model.selection_color, False, r / 5)
            viewer.draw_circle(p, r * 1.1, self.model.selection_color, False, r / 5)

    def contained(self, point):
        return euclidean(self.get_position(), point)<=self.radius
        # return Point(self.get_position()).distance(Point(point))<=self.radius
        # return Circle(self.get_position(), radius=self.radius).contains_point(point)

