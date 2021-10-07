from copy import deepcopy

import numpy as np
from scipy.spatial.distance import euclidean

import lib.aux.sim_aux
from lib.model.agents._agent import LarvaworldAgent


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color=None, **kwargs):
        # print(unique_id)
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
    def final_dst_to_source(self):
        return euclidean(self.pos, (0.04,0))

    @property
    def final_dst_to_center(self):
        return euclidean(self.pos, (0, 0))

    @property
    def turner_activation(self):
        return self.brain.turner.activation

    @property
    def olfactory_activation(self):
        return self.brain.olfactory_activation

    @property
    def first_odor_concentration(self):
        return list(self.brain.olfactor.Con.values())[0]

    # @property
    # def odor_Con0(self):
    #     return list(self.brain.olfactor.Con.values())[0]
    #
    # @property
    # def odor_Con1(self):
    #     return list(self.brain.olfactor.Con.values())[1]
    #
    # @property
    # def odor_Con2(self):
    #     return list(self.brain.olfactor.Con.values())[2]

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


    @property
    def cum_reward(self):
        return self.brain.memory.rewardSum

    @property
    def dt(self):
        return self.model.dt

    @property
    def first_odor_concentration_change(self):
        return list(self.brain.olfactor.dCon.values())[0]

    @property
    def second_odor_concentration_change(self):
        return list(self.brain.olfactor.dCon.values())[1]

    # @property
    # def length_in_mm(self):
    #     return self.get_real_length() * 1000

    # @property
    # def length_in_mm(self):
    #     return self.real_length * 1000
    #     # from lib.conf.par import TemporalPar, FractionPar
    #     # k1 = TemporalPar(name='cum_dur')
    #     # k2 = TemporalPar(name='cum_dur')
    #     # k=FractionPar(name='some', exists=False, numerator=k1, denominator=k2)
    #     # v = k.get_from(self)
    #     # print(v)
    #     # return v

    # @property
    # def mass_in_mg(self):
    #     return self.get_real_mass() * 1000

    @property
    def scaled_amount_eaten(self):
        return self.amount_eaten / self.get_real_mass()

    # @property
    # def orientation_to_center_in_deg(self):
    #     return fun.angle_dif(np.rad2deg(self.get_head().get_normalized_orientation()),
    #                          fun.angle_to_x_axis(self.get_position(), (0, 0),
    #                                              in_deg=True), in_deg=True)

    @property
    def x(self):
        return self.pos[0] / self.model.scaling_factor

    @property
    def y(self):
        return self.pos[1] / self.model.scaling_factor

    @property
    def x0(self):
        return self.initial_pos[0] / self.model.scaling_factor

    @property
    def y0(self):
        return self.initial_pos[1] / self.model.scaling_factor

    @property
    def dispersion_in_mm(self):
        return euclidean(tuple(self.pos),
                         tuple(self.initial_pos)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dispersion(self):
        return euclidean(tuple(self.pos),
                         tuple(self.initial_pos)) / self.sim_length

    @property
    def cum_dst_in_mm(self):
        return self.cum_dst * 1000 / self.model.scaling_factor

    @property
    def cum_scaled_dst(self):
        return self.cum_dst / self.sim_length

    @property
    def dst_to_center_in_mm(self):
        return np.sqrt(np.sum(np.array(self.pos)**2)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_center(self):
        return np.sqrt(np.sum(np.array(self.pos)**2)) / self.sim_length

    @property
    def dst_to_chemotax_odor_in_mm(self):
        return euclidean(tuple(self.pos),
                         (0.8, 0.0)) * 1000 / self.model.scaling_factor

    @property
    def scaled_dst_to_chemotax_odor(self):
        return euclidean(tuple(self.pos),
                         (0.8, 0.0)) / self.sim_length

    @property
    def max_dst_to_center_in_mm(self):
        return np.nanmax(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1))) * 1000/ self.model.scaling_factor

    @property
    def max_scaled_dst_to_center(self):
        d = np.nanmax(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))/ self.sim_length
        return d

    @property
    def mean_dst_to_center_in_mm(self):
        return np.nanmean(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))* 1000 / self.model.scaling_factor

    @property
    def mean_scaled_dst_to_center(self):
        d = np.nanmean(np.sqrt(np.sum(np.array(self.trajectory)**2, axis=1)))/ self.sim_length
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
                       range(len(self.trajectory))]) / self.sim_length

    @property
    def stride_dst_mean_in_mm(self):
        return (self.cum_dst / self.brain.crawler.iteration_counter) * 1000 / self.model.scaling_factor

    @property
    def stride_scaled_dst_mean(self):
        return (self.cum_dst / self.sim_length) / self.brain.crawler.iteration_counter

    @property
    def crawler_freq(self):
        return lib.aux.sim_aux.freq

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
        return self.deb.f

    @property
    def deb_f_mean(self):
        return np.mean(self.deb.dict['f'])

    @property
    def gut_occupancy(self):
        return self.deb.gut.occupancy

    @property
    def ingested_volume(self):
        return self.deb.gut.ingested_volume

    @property
    def ingested_body_mass_ratio(self):
        return self.deb.gut.ingested_mass()/self.deb.Ww*100

    @property
    def ingested_body_volume_ratio(self):
        return self.deb.gut.ingested_volume/self.deb.V *100

    @property
    def ingested_gut_volume_ratio(self):
        return self.deb.gut.ingested_volume / (self.deb.V*self.deb.gut.V_gm) * 100

    @property
    def ingested_body_area_ratio(self):
        return (self.deb.gut.ingested_volume/self.deb.V)**(1/2)*100
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
        return self.deb.f - 1

    @property
    def deb_f_deviation_mean(self):
        return np.mean(np.array(self.deb.dict['f']) - 1)

    @property
    def reserve(self):
        return self.deb.E

    @property
    def reserve_density(self):
        return self.deb.e

    @property
    def structural_length(self):
        return self.deb.L

    @property
    def maturity(self):
        return self.deb.E_H * 1000 #in mJ

    @property
    def reproduction(self):
        return self.deb.E_R * 1000 #in mJ

    @property
    def puppation_buffer(self):
        return self.deb.get_pupation_buffer()

    @property
    def structure(self):
        return self.deb.V * self.deb.E_V * 1000 #in mJ

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
    def exploitVSexplore_balance(self):
        return self.brain.intermitter.EEB