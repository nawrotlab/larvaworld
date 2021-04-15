from copy import deepcopy
import numpy as np
from Box2D import Box2D, b2ChainShape
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_normal

import lib.aux.functions as fun
import lib.aux.rendering as ren


class LarvaworldAgent:
    def __init__(self,
                 unique_id : str,
                 model, pos=None, default_color=None, radius=None,
                 odor_id=None,odor_intensity=0.0, odor_spread=0.1, group='', can_be_carried=False):
        self.selected = False
        self.unique_id = unique_id
        self.model = model
        self.group = group
        self.base_odor_id = f'{group} base odor'
        self.gain_for_base_odor = 100


        # Will be set by the respective subclasses

        self.initial_pos = pos
        self.pos = self.initial_pos
        if type(default_color)==str :
            default_color=fun.colorname2tuple(default_color)
        self.default_color = default_color
        self.radius = radius

        self.id_box = self.init_id_box()

        self.odor_id = odor_id
        self.odor_intensity = odor_intensity
        self.odor_spread = odor_spread
        self.set_odor_dist()

        self.carried_objects=[]
        self.can_be_carried=can_be_carried
        self.is_carried_by=None

    def get_position(self):
        return tuple(self.pos)
        # return np.array(self.pos)

    def init_id_box(self):
        id_box = ren.InputBox(visible=False, text=self.unique_id,
                          color_inactive=self.default_color, color_active=self.default_color,
                          screen_pos=None, agent=self)
        return id_box

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text = self.unique_id

    def contained(self, point):
        return Circle(tuple(self.get_position()), radius=self.radius).contains_point(point)

    # @abc.abstractmethod
    def step(self):
        pass

    # @abc.abstractmethod
    def set_color(self, color):
        pass

    def set_default_color(self, color):
        self.default_color=color
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


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color = None, **kwargs):
        if default_color is None :
            default_color = model.generate_larva_color()
        super().__init__(unique_id=unique_id, model=model, default_color=default_color, pos=pos, radius=radius, **kwargs)
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
    def first_odor_concentration(self):
        return list(self.odor_concentrations.values())[0]

    @property
    def second_odor_concentration(self):
        return list(self.odor_concentrations.values())[1]

    @property
    def first_odor_concentration_change(self):
        return list(self.brain.olfactor.dCon.values())[0]

    @property
    def length_in_mm(self):
        return self.get_real_length() * 1000

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
        return euclidean(tuple(self.pos), (0, 0)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_center(self):
        return euclidean(tuple(self.pos), (0, 0)) / self.get_sim_length()

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
        return np.nanmax([euclidean(tuple(self.trajectory[i]),
                                    (0.0, 0.0)) for i in
                          range(len(self.trajectory))]) * 1000 / self.model.scaling_factor

    @property
    def max_scaled_dst_to_center(self):
        return np.nanmax([euclidean(tuple(self.trajectory[i]),
                                    (0.0, 0.0)) for i in
                          range(len(self.trajectory))]) / self.get_sim_length()

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
        return self.brain.crawler.iteration_counter

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
        return self.brain.feeder.iteration_counter

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
    def deb_f_deviation(self):
        return self.deb.get_f() - 1

    @property
    def reserve(self):
        return self.deb.get_reserve()

    @property
    def reserve_density(self):
        return self.deb.get_reserve_density()

    @property
    def structural_length(self):
        return self.deb.get_L()

    @property
    def maturity(self):
        return self.deb.get_U_H() * 1000

    @property
    def reproduction(self):
        return self.deb.get_U_R() * 1000

    @property
    def puppation_buffer(self):
        return self.deb.get_puppation_buffer()

    @property
    def structure(self):
        return self.deb.get_U_V() * 1000

    @property
    def age_in_hours(self):
        return self.deb.age_day * 24

    @property
    def hunger(self):
        return self.deb.hunger

    @property
    def deb_steps_per_day(self):
        return self.deb.steps_per_day

    @property
    def deb_Nticks(self):
        return self.deb.tick_counter

    @property
    def death_time_in_hours(self):
        return self.deb.death_time_in_hours

    @property
    def puppation_time_in_hours(self):
        return self.deb.puppation_time_in_hours

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


class Food(LarvaworldAgent):
    def __init__(self, unique_id, model, position,
                 radius=0.002, amount=1.0, quality=1.0,
                   default_color=None, **kwargs):
        if default_color is None :
            default_color=np.array((100, 200, 120))
        super().__init__(unique_id=unique_id, model=model, pos=position, default_color=default_color,
                         radius=radius*model.scaling_factor, **kwargs)
        self.initial_amount = amount
        self.quality = quality
        self.amount = self.initial_amount



        shape = fun.circle_to_polygon(60, self.radius)

        if self.model.physics_engine:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=self.pos)
            # else:
            #     # all parameters in real space units
            #     self._body: Box2D.b2Body = self.space.CreateDynamicBody(pos=self.pos, orientation=None)
            #     # super().__init__(space=self.space, pos=self.pos, orientation=None, radius=self.shape_radius)
            #     self._body.bullet = True
            self.food_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.food_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, self.pos)
        self._color = self.default_color


        self.circle = Circle(tuple(self.get_position()), radius=self.radius)
        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1






    def get_radius(self):
        return self.radius

    def get_amount(self):
        return self.amount

    def subtract_amount(self, amount):
        prev_amount = self.amount
        self.amount -= amount
        if self.amount <= 0.0:
            self.amount = 0.0
            self.model.delete_agent(self)
        else:
            r = (self.initial_amount - self.amount) / self.initial_amount
            self._color = r * np.array((255, 255, 255)) + (1 - r) * self.default_color
            # self._color=self.default_color + (self.initial_amount - self.amount)*(np.array((255,255,255))-self.default_color)
        return np.min([amount, prev_amount])



    def set_scaled_odor_intensity(self, intensity):
        self.odor_intensity = intensity * self.model.scaling_factor

    def set_scaled_odor_spread(self, spread):
        self.odor_spread = spread * self.model.scaling_factor



    def set_color(self, color):
        self._color = color

    def set_default_color(self, color):
        self.default_color = color
        self._color = self.default_color
        self.id_box.color = self.default_color



    def draw(self, viewer):
        if self.amount > 0:
            filled = True
        else:
            filled = False
        w = self.radius / 5
        viewer.draw_circle(position=self.get_position(), radius=self.radius, color=self._color, filled=filled, width=w)
        if self.odor_intensity > 0:
            viewer.draw_circle(position=self.get_position(), radius=self.radius * 1.5, color=self._color, filled=False,
                               width=w / 2)
            viewer.draw_circle(position=self.get_position(), radius=self.radius * 2.0, color=self._color, filled=False,
                               width=w / 3)
            viewer.draw_circle(position=self.get_position(), radius=self.radius * 3.0, color=self._color, filled=False,
                               width=w / 4)
        if self.selected:
            viewer.draw_circle(position=self.get_position(), radius=self.radius + w, color=self.model.selection_color,
                               filled=False,
                               width=w)
