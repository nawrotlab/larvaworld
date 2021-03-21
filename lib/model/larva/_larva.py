import random
import mesa
import numpy as np
from copy import deepcopy

from matplotlib.patches import Circle
from nengo import Simulator
from scipy.spatial.distance import euclidean

from lib.aux.rendering import InputBox
from lib.model.envs._space import agents_spatial_query
from lib.model.larva._bodies import LarvaBody
from lib.model.larva._effectors import DefaultBrain
from lib.model.larva._sensorimotor import VelocityAgent
from lib.aux import functions as fun
from lib.aux import naming as nam
from lib.model.larva.deb import DEB
from lib.model.larva.nengo_effectors import NengoBrain


class LarvaworldAgent:
    def __init__(self, unique_id, model):
        # super().__init__(unique_id=unique_id, model=model)
        self.selected = False
        self.unique_id = unique_id
        self.model = model
        self.radius=None

    def get_position(self):
        if self.model.physics_engine:
            if isinstance(self, Larva):
                return self.get_global_midspine_of_body()
            else :
                return np.asarray(self._body.position)
        else:
            return np.array(self.pos)
            # return self.get_global_midspine_of_body()

    def init_id_box(self):
        id_box = InputBox(visible=False, text=self.unique_id,
                          color_inactive=self.default_color, color_active=self.default_color,
                          screen_pos=self.model.space2screen_pos(self.get_position()))
        return id_box

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text=self.unique_id

    def contained(self, point):
        return Circle(self.get_position(), radius=self.radius).contains_point(point)



class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id=unique_id, model=model)
        self.default_color = self.model.generate_larva_color()
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))

    def update_color(self, default_color, behavior_dict, mode='lin'):
        color = deepcopy(default_color)
        if mode == 'lin':
            # if behavior_dict['stride_stop'] :
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
        return self.odor_concentrations[0]

    @property
    def second_odor_concentration(self):
        return self.odor_concentrations[1]

    @property
    def body_bend_in_deg(self):
        return np.rad2deg(self.body_bend)

    @property
    def ang_vel_in_deg(self):
        return np.rad2deg(self.get_head().get_angularvelocity())

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
    def front_orientation_in_deg(self):
        return np.rad2deg(self.get_head().get_normalized_orientation())

    @property
    def rear_orientation_in_deg(self):
        return np.rad2deg(self.get_tail().get_normalized_orientation())

    @property
    def orientation_to_center_in_deg(self):
        return fun.angle_dif(np.rad2deg(self.get_head().get_normalized_orientation()),
                             fun.angle_to_x_axis(self.get_position(), (0, 0),
                                                 in_deg=True), in_deg=True)

    @property
    def x(self):
        return self.current_pos[0] * 1000 / self.model.scaling_factor

    @property
    def y(self):
        return self.current_pos[1] * 1000 / self.model.scaling_factor

    @property
    def dispersion_in_mm(self):
        return euclidean(tuple(self.current_pos),
                         tuple(self.initial_pos)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dispersion(self):
        return euclidean(tuple(self.current_pos),
                         tuple(self.initial_pos)) / self.get_sim_length()

    @property
    def cum_dst_in_mm(self):
        return self.cum_dst * 1000 / self.model.scaling_factor

    @property
    def cum_scaled_dst(self):
        return self.cum_dst / self.get_sim_length()

    @property
    def dst_to_center_in_mm(self):
        return euclidean(tuple(self.current_pos), (0, 0)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_center(self):
        return euclidean(tuple(self.current_pos), (0, 0)) / self.get_sim_length()

    @property
    def dst_to_chemotax_odor_in_mm(self):
        return euclidean(tuple(self.current_pos),
                         (0.8, 0.0)) * 1000 / self.model.scaling_factor,

    @property
    def scaled_dst_to_chemotax_odor(self):
        return euclidean(tuple(self.current_pos),
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
        return self.brain.crawler.total_t / self.sim_time

    @property
    def pause_dur_ratio(self):
        return self.brain.intermitter.cum_pause_dur / self.sim_time

    @property
    def stridechain_dur_ratio(self):
        return self.brain.intermitter.cum_stridechain_dur / self.sim_time

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
        return self.brain.feeder.total_t / self.sim_time

    @property
    def feed_success_rate(self):
        return self.feed_success_counter / self.brain.feeder.iteration_counter

    @property
    def deb_f(self):
        return self.deb.get_f()

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



class LarvaReplay(Larva, LarvaBody):
    def __init__(self, unique_id, model, schedule, length=5, data=None):
        Larva.__init__(self, unique_id=unique_id, model=model)

        self.schedule = schedule
        self.data = data
        self.pars = self.data.columns.values
        self.Nticks = len(self.data.index.unique().values)
        self.t0 = self.data.index.unique().values[0]

        d = self.model.dataset
        self.spinepoint_xy_pars = [p for p in fun.flatten_list(d.points_xy) if p in self.pars]
        self.Npoints = int(len(self.spinepoint_xy_pars) / 2)

        self.contour_xy_pars = [p for p in fun.flatten_list(d.contour_xy) if p in self.pars]
        self.Ncontour = int(len(self.contour_xy_pars) / 2)

        self.centroid_xy_pars = [p for p in d.cent_xy if p in self.pars]

        Nsegs = self.model.draw_Nsegs
        if Nsegs is not None:
            if Nsegs == self.Npoints - 1:
                self.orientation_pars = [p for p in nam.orient(d.segs) if p in self.pars]
                self.Nors = len(self.orientation_pars)
                self.Nangles = 0
                if self.Nors != Nsegs:
                    raise ValueError(
                        f'Orientation values are not present for all body segments : {self.Nors} of {Nsegs}')
            elif Nsegs == 2:
                self.orientation_pars = [p for p in ['front_orientation'] if p in self.pars]
                self.Nors = len(self.orientation_pars)
                self.angle_pars = [p for p in ['bend'] if p in self.pars]
                self.Nangles = len(self.angle_pars)
                if self.Nors != 1 or self.Nangles != 1:
                    raise ValueError(
                        f'{self.Nors} orientation and {Nsegs} angle values are present and 1,1 are needed.')
            else:
                raise ValueError(f'Defined number of segments {Nsegs} must be either 2 or {self.Npoints - 1}')
        else:
            self.Nors, self.Nangles = 0, 0

        # self.angle_pars=[p for p in d.angles + ['bend'] if p in self.pars]
        # self.Nangles=len(self.angle_pars)
        #
        # self.orientation_pars=[p for p in nam.orient(d.segments) + ['front_orientation', 'rear_orientation'] if p in self.pars]
        # self.Nors = len(self.orientation_pars)

        self.chunk_ids = None
        self.trajectory = []
        self.color = deepcopy(self.default_color)
        self.sim_length = length
        self.radius = self.sim_length/2

        if self.Npoints > 0:
            self.spinepoint_positions_ar = self.data[self.spinepoint_xy_pars].values
            self.spinepoint_positions_ar = self.spinepoint_positions_ar.reshape([self.Nticks, self.Npoints, 2])
        else:
            self.spinepoint_positions_ar = np.ones([self.Nticks, self.Npoints, 2]) * np.nan

        if self.Ncontour > 0:
            self.contourpoint_positions_ar = self.data[self.contour_xy_pars].values
            self.contourpoint_positions_ar = self.contourpoint_positions_ar.reshape([self.Nticks, self.Ncontour, 2])
        else:
            self.contourpoint_positions_ar = np.ones([self.Nticks, self.Ncontour, 2]) * np.nan

        if len(self.centroid_xy_pars) == 2:
            self.centroid_position_ar = self.data[self.centroid_xy_pars].values
        else:
            self.centroid_position_ar = np.ones([self.Nticks, 2]) * np.nan

        if len(self.model.pos_xy_pars) == 2:
            self.position_ar = self.data[self.model.pos_xy_pars].values
        else:
            self.position_ar = np.ones([self.Nticks, 2]) * np.nan

        if self.Nangles > 0:
            self.spineangles_ar = self.data[self.angle_pars].values
        else:
            self.spineangles_ar = np.ones([self.Nticks, self.Nangles]) * np.nan

        if self.Nors > 0:
            self.orientations_ar = self.data[self.orientation_pars].values
        else:
            self.orientations_ar = np.ones([self.Nticks, self.Nors]) * np.nan

        vp_behavior = [p for p in self.behavior_pars if p in self.pars]
        self.behavior_ar = np.zeros([self.Nticks, len(self.behavior_pars)], dtype=bool)
        for i, p in enumerate(self.behavior_pars):
            if p in vp_behavior:
                self.behavior_ar[:, i] = np.array([not v for v in np.isnan(self.data[p].values).tolist()])

        if self.model.draw_Nsegs is not None:
            LarvaBody.__init__(self, model, pos=self.position_ar[0], orientation=self.orientations_ar[0][0],
                               initial_length=self.sim_length / 1000, length_std=0, Nsegs=self.model.draw_Nsegs, interval=0)

        self.pos = self.position_ar[0]
        self.id_box=self.init_id_box()


    def step(self):
        step = self.schedule.steps
        self.spinepoint_positions = self.spinepoint_positions_ar[step].tolist()
        self.vertices = self.contourpoint_positions_ar[step]
        self.centroid_position = self.centroid_position_ar[step]
        self.pos = self.position_ar[step]
        if not np.isnan(self.pos).any():
            self.model.space.move_agent(self, self.pos)
        self.trajectory = self.position_ar[:step, :].tolist()
        self.spineangles = self.spineangles_ar[step]
        self.orientations = self.orientations_ar[step]
        if self.model.color_behavior:
            behavior_dict = dict(zip(self.behavior_pars, self.behavior_ar[step, :].tolist()))
            self.color = self.update_color(self.default_color, behavior_dict)
        else:
            self.color = self.default_color
        if self.model.draw_Nsegs is not None:
            segs = self.segs

            if len(self.spinepoint_positions) == len(segs) + 1:
                for i, seg in enumerate(segs):
                    pos = [np.nanmean([self.spinepoint_positions[i][j], self.spinepoint_positions[i + 1][j]]) for j in
                           [0, 1]]
                    o = np.deg2rad(self.orientations[i])
                    seg.set_position(pos)
                    # elif self.Nors == len(segs):
                    #     for i, seg in enumerate(segs):
                    seg.set_orientation(o)
                    seg.update_vertices(pos, o)
            elif len(segs) == 2 and self.Nors == 1 and self.Nangles == 1:
                l1, l2 = [self.sim_length * r for r in self.seg_ratio]
                x, y = self.pos
                h_or = np.deg2rad(self.orientations[0])
                b_or = np.deg2rad(self.orientations[0] - self.spineangles[0])
                p_head = np.array(fun.rotate_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or))
                p_tail = np.array(fun.rotate_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or))
                pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
                pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
                segs[0].set_position(pos1)
                segs[0].set_orientation(h_or)
                segs[0].update_vertices(pos1, h_or)
                segs[1].set_position(pos2)
                segs[1].set_orientation(b_or)
                segs[1].update_vertices(pos2, b_or)
                self.spinepoint_positions = np.array([p_head, self.pos, p_tail])
        self.id_box.set_shape(self.model.space2screen_pos(self.get_position()))

    def draw(self, viewer):
        if self.model.draw_contour:
            if self.model.draw_Nsegs is not None:
                for seg in self.segs:
                    seg.set_color(self.color)
                    seg.draw(viewer)
            elif len(self.vertices) > 0:
                viewer.draw_polygon(self.vertices, filled=True, color=self.color)
        if self.model.draw_centroid:
            if not np.isnan(self.centroid_position).any():
                pos = self.centroid_position
            elif not np.isnan(self.pos).any():
                pos = self.pos
            else:
                pos = None
            if pos is not None:
                viewer.draw_circle(radius=.1, position=pos, filled=True, color=self.color, width=1)
        if self.model.draw_midline and self.Npoints > 1:
            if not np.isnan(self.spinepoint_positions[0]).any():
                viewer.draw_polyline(self.spinepoint_positions, color=(0, 0, 255), closed=False, width=.07)
                for i, seg_pos in enumerate(self.spinepoint_positions):
                    c = 255 * i / (len(self.spinepoint_positions) - 1)
                    color = (c, 255 - c, 0)
                    viewer.draw_circle(radius=.07, position=seg_pos, filled=True, color=color, width=.01)
        if self.selected:
            r=self.sim_length/2
            viewer.draw_circle(radius=r,
                               position=self.get_position(),
                               filled=False, color=self.model.selection_color, width=r/5)


class LarvaSim(VelocityAgent, Larva):
    def __init__(self, unique_id, model, fly_params, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model)

        self.brain = self.build_brain(fly_params['neural_params'])
        self.build_energetics(fly_params['energetics_params'])
        VelocityAgent.__init__(self, **fly_params['sensorimotor_params'], **fly_params['body_params'], **kwargs)
        self.build_gut(self.V)

        self.reset_feeder()
        self.id_box = self.init_id_box()
        self.radius = self.sim_length / 2

    def compute_next_action(self):
        self.sim_time += self.model.dt

        self.odor_concentrations = self.sense_odors(self.model.Nodors, self.model.odor_layers)
        self.food_detected, self.food_source, food_quality = self.detect_food(pos=self.get_olfactor_position(),
                                                                              grid=self.model.food_grid)

        lin, ang, self.feeder_motion, feed_success, self.olfactory_activation = self.brain.run(
            self.odor_concentrations,
            self.get_sim_length(),
            self.food_detected)
        self.set_ang_activity(ang)
        self.set_lin_activity(lin)
        self.current_amount_eaten, self.feed_success = self.feed(success=feed_success, source=self.food_source,
                                                                 max_amount_eaten=self.max_feed_amount,
                                                                 empty_gut_M=self.empty_gut_M)
        self.update_gut(self.current_amount_eaten)
        if self.energetics:
            self.run_energetics(self.food_detected, self.feed_success, self.current_amount_eaten, food_quality)
        self.id_box.set_shape(self.model.space2screen_pos(self.get_position()))
    def sense_odors(self, Nodors, odor_layers):
        if Nodors == 0:
            return []
        else:
            pos = self.get_olfactor_position()
            values = [odor_layers[id].get_value(pos) for id in odor_layers]
            if self.brain.olfactor.noise:
                values = [v + np.random.normal(scale=v * self.brain.olfactor.noise) for v in values]
            return values

    def detect_food(self, pos, grid=None):
        if self.brain.feeder is not None:
            radius = self.brain.feeder.feed_radius * self.sim_length,
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_value(cell) > 0:
                    return True, cell, grid.quality
                else:
                    return False, None, None
            else:
                accessible_food = agents_spatial_query(pos=pos, radius=radius,
                                                       agent_list=self.model.get_food())
                if accessible_food:
                    food = random.choice(accessible_food)
                    return True, food, food.quality
                else:
                    return False, None, None
        else:
            return False, None, None

    def feed(self, success, source, max_amount_eaten, empty_gut_M):

        if success:
            if empty_gut_M < max_amount_eaten:
                return 0, False
            if self.model.food_grid:
                amount = self.model.food_grid.subtract_value(source, max_amount_eaten)
            else:
                amount = source.subtract_amount(max_amount_eaten)
            # # TODO fix the radius so that it works with any feeder, nengo included
            # success, amount = self.detect_food(mouth_position=self.get_olfactor_position(),
            #                                    radius=self.brain.feeder.feed_radius * self.sim_length,
            #                                    grid=self.model.food_grid,
            #                                    max_amount_eaten=self.max_feed_amount)

            self.feed_success_counter += int(success)
            self.amount_eaten += amount

        else:
            amount = 0
        return amount, success

    def reset_feeder(self):
        self.feed_success_counter = 0
        self.amount_eaten = 0
        self.feeder_motion = False
        try:
            self.max_feed_amount = self.compute_max_feed_amount()
        except:
            self.max_feed_amount = None
        try:
            self.brain.feeder.reset()
        except:
            pass

    def compute_max_feed_amount(self):
        return self.brain.feeder.max_feed_amount_ratio * self.V**(2/3)

    def build_energetics(self, energetic_pars):
        self.real_length = None
        self.real_mass = None
        self.V = None

        # p_am=260
        if energetic_pars is not None:
            self.energetics = True
            if energetic_pars['deb_on']:
                self.hunger_affects_balance = energetic_pars['hunger_affects_balance']
                self.absorption_c = energetic_pars['absorption_c']#/60
                self.f_decay_coef = energetic_pars['f_decay_coef']
                self.f_exp_coef = np.exp(-self.f_decay_coef * self.model.dt)
                steps_per_day = 24 * 60
                if self.hunger_affects_balance:
                    self.deb = DEB(steps_per_day=steps_per_day, base_hunger=self.brain.intermitter.base_EEB,
                                   hunger_sensitivity=energetic_pars['hunger_sensitivity'])
                else:
                    self.deb = DEB(steps_per_day=steps_per_day,
                                   hunger_sensitivity=energetic_pars['hunger_sensitivity'])
                # self.deb = DEB(steps_per_day=steps_per_day, hunger_sensitivity = energetic_pars['hunger_sensitivity'])
                self.deb.reach_stage('larva')
                self.deb.advance_larva_age(hours_as_larva=self.model.hours_as_larva, f=self.model.deb_base_f,
                                           starvation_hours=self.model.deb_starvation_hours)
                self.deb.steps_per_day = int(24 * 60 * 60 / self.model.dt)
                self.real_length = self.deb.get_real_L()
                self.real_mass = self.deb.get_W()
                self.V = self.deb.get_V()
                # p_am=self.deb.p_am

            else:
                self.deb = None
                self.food_to_biomass_ratio = 0.3
        else:
            self.energetics = False


    def build_gut(self, V):
        self.gut_M_ratio = 0.11
        self.gut_food_M = 0
        self.empty_gut_M = 0
        self.gut_product_M = 0
        self.amount_absorbed = 0
        self.filled_gut_ratio = 0
        # self.digestion_c = 80
        self.absorption_c = 0.3

        self.gut_M = self.gut_M_ratio * V

    def update_gut(self, amount_eaten):
        self.gut_M = self.gut_M_ratio * self.V
        self.empty_gut_M = self.gut_M - self.gut_food_M - self.gut_product_M
        # FIXME here I need to add the k_x but I don't know it
        # For V1-morphs ingestion rate is proportional to L**3 . Kooijman p.269. Didn't use it.
        # self.digestion_tau_unscaled = 24 * 60 * 60 / self.model.dt * self.gut_M_ratio * 550 / p_am
        # Trying to use μ_Ax=11.5 from p.272
        # self.digestion_tau_unscaled = 24*60*60/self.model.dt*self.gut_M_ratio*11.5/p_am
        # self.digestion_tau = self.digestion_tau_unscaled * self.V ** (1 / 3)

        self.gut_food_M += amount_eaten
        gut_food_dM = 0.001 * self.gut_food_M * self.model.dt
        # gut_food_dM = np.clip(self.gut_M/self.digestion_tau, 0, self.gut_food_M)
        # gut_food_dM = self.gut_food_M/self.digestion_tau
        self.gut_food_M -= gut_food_dM
        # print(gut_food_dM, self.gut_food_M)
        # self.gut_product_M += gut_food_dM
        absorbed_M = self.gut_product_M * self.absorption_c * self.model.dt
        # absorbed_M = 0
        self.gut_product_M += (gut_food_dM - absorbed_M)
        # self.empty_gut_M = self.gut_M - self.gut_food_M
        self.empty_gut_M = self.gut_M - self.gut_food_M - self.gut_product_M
        self.amount_absorbed += absorbed_M
        self.filled_gut_ratio = 1 - self.empty_gut_M / self.gut_M

    def build_brain(self, neural_params):
        modules = neural_params['modules']
        if neural_params['nengo']:
            brain = NengoBrain()
            brain.setup(agent=self, modules=modules, conf=neural_params)
            brain.build(brain.nengo_manager, olfactor=brain.olfactor)
            brain.sim = Simulator(brain, dt=0.01)
            brain.Nsteps = int(self.model.dt / brain.sim.dt)
        else:
            brain = DefaultBrain(agent=self, modules=modules, conf=neural_params)
        return brain

    def run_energetics(self, food_detected, feed_success, amount_eaten, food_quality):
        # if isinstance(self.brain, DefaultBrain):
        #     if not food_detected:
        #         # print('DD')
        #         self.brain.intermitter.EEB *= self.brain.intermitter.EEB_exp_coef
        #     else :
        #         self.brain.intermitter.EEB = self.brain.intermitter.base_EEB


        if self.deb:
            f = self.deb.get_f()

            if feed_success:
                # f += self.f_increment
                f += food_quality * self.absorption_c * amount_eaten / self.max_feed_amount
                # f += self.f_increment/np.clip(f-1, 1, +np.inf)
            # else :
            #     f *= self.f_exp_coef
            f *= self.f_exp_coef
            # f=np.clip(f,0,2)
            self.deb.run(f=f)
            self.real_length = self.deb.get_real_L()
            self.real_mass = self.deb.get_W()
            self.V = self.deb.get_V()

            # if self.hunger_affects_feeder :
            #     self.brain.intermitter.feeder_reoccurence_rate_on_success=self.deb.hunger
            if not food_detected:
                # print('DD')
                self.brain.intermitter.EEB *= self.brain.intermitter.EEB_exp_coef
            else:
                if self.hunger_affects_balance:
                    dh = self.deb.hunger - self.deb.base_hunger
                    # print(dh)
                    if dh == 0:
                        pass
                    elif dh > 0:
                        self.brain.intermitter.EEB = dh / (1 - self.deb.base_hunger) * (
                                    1 - self.brain.intermitter.base_EEB) + self.brain.intermitter.base_EEB
                    else:
                        self.brain.intermitter.EEB = dh / self.deb.base_hunger * self.brain.intermitter.base_EEB + self.brain.intermitter.base_EEB
                else:
                    self.brain.intermitter.EEB = self.brain.intermitter.base_EEB
            # if self.hunger_affects_balance and food_detected:
            #     self.brain.intermitter.EEB = self.deb.hunger
            # if not self.deb.alive :
            #     raise ValueError ('Dead')
            self.adjust_body_vertices()
            self.max_feed_amount = self.compute_max_feed_amount()
        else:
            if feed_success:
                self.real_mass += amount_eaten * food_quality * self.food_to_biomass_ratio
                self.adjust_shape_to_mass()
                self.adjust_body_vertices()
                self.max_feed_amount = self.compute_max_feed_amount()
                self.V = self.get_real_length() ** 3

    # def update_balance(self, feed_motion, feed_success):
