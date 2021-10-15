"""Here we state all possible collected parameters for the simulations"""
from collections import OrderedDict
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
from operator import attrgetter

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

import lib.aux.naming as nam
from lib.aux.dictsNlists import flatten_list

#
# body_pars = {
#     "length": 'length_in_mm',
#     "mass": 'mass_in_mg',
# }
#
# pos_xy = {
#     "centroid_x": 'x',
#     "centroid_y": 'y',
#     # "final_x": 'x',
#     # "final_y": 'y'
# }
#
# orientation_pars = {
#     "front_orientation": 'front_orientation',
#     "rear_orientation": 'rear_orientation',
#     # "orientation_to_center": 'orientation_to_center_in_deg',
#     # "final_orientation_to_center": 'orientation_to_center_in_deg',
# }
#
# lin_pars = {
#     # "dst_to_center": 'dst_to_center_in_mm',
#     # "scaled_dst_to_center": 'scaled_dst_to_center',
#     # "dst_to_chemotax_odor": 'dst_to_chemotax_odor_in_mm',
#     # "scaled_dst_to_chemotax_odor": 'scaled_dst_to_chemotax_odor',
#
#     # "dispersion": 'dispersion_in_mm',
#     # "scaled_dispersion": 'scaled_dispersion',
#     # "dispersion_max": 'dispersion_max_in_mm',
#     # "scaled_dispersion_max": 'scaled_dispersion_max',
#     #
#     # "cum_dst": 'cum_dst_in_mm',
#     # "cum_scaled_dst": 'cum_scaled_dst',
#     # "final_dispersion": 'dispersion_in_mm',
#     # "final_scaled_dispersion": 'scaled_dispersion',
#     # "final_dst_to_center": 'dst_to_center_in_mm',
#     # "final_scaled_dst_to_center": 'scaled_dst_to_center',
#     #
#     # "max_dst_to_center": 'max_dst_to_center_in_mm',
#     # "max_scaled_dst_to_center": 'max_scaled_dst_to_center',
#     # "mean_scaled_dst_to_center": 'mean_scaled_dst_to_center',
#     # "mean_dst_to_center": 'mean_dst_to_center_in_mm',
#     #
#     # "final_dst_to_chemotax_odor": 'dst_to_chemotax_odor_in_mm',
#     # "final_scaled_dst_to_chemotax_odor": 'scaled_dst_to_chemotax_odor',
# }
#
# ang_pars = {
#     "front_orientation_vel": 'front_orientation_vel',
#     "bend": 'bend',
#     "body_bend_vel": 'body_bend_vel',
#     # "body_bend_acc": 'body_bend_acc',
#     # "torque": lambda a : a.torque,
#     "torque": 'torque',
#     "body_bend_errors": 'body_bend_errors',
# }
#
# effector_pars = {
#     "first_odor_concentration": 'first_odor_concentration',
#     "second_odor_concentration": 'second_odor_concentration',
#     "olfactory_activation": 'olfactory_activation',
#     "first_odor_concentration_change": 'first_odor_concentration_change',
#
#     "first_odor_best_gain": 'first_odor_best_gain',
#     "second_odor_best_gain": 'second_odor_best_gain',
#     "cum_reward": 'cum_reward',
#     "best_olfactor_decay": 'best_olfactor_decay',
#
#     "turner_activation": 'turner_activation',
#     "turner output": 'ang_activity',
#
#     "crawler_activity": 'lin_activity',
#
#     "feeder_motion": 'feeder_motion',
#     "amount_eaten": 'amount_eaten',
#     "scaled_amount_eaten": 'scaled_amount_eaten',
#     "ingested_body_volume_ratio": 'ingested_body_volume_ratio',
#     "ingested_gut_volume_ratio": 'ingested_gut_volume_ratio',
#     "ingested_body_area_ratio": 'ingested_body_area_ratio',
#     "ingested_body_mass_ratio": 'ingested_body_mass_ratio',
#     "amount_absorbed": 'amount_absorbed',
#     "feed_success_rate": 'feed_success_rate',
#     "gut_occupancy": 'gut_occupancy',
#
#     "vel_freq": 'crawler_freq',
#     "stride_dst_mean": 'stride_dst_mean_in_mm',
#
#     "stride_scaled_dst_mean": 'stride_scaled_dst_mean',
# }
#
# intermitter_pars = {
#     "pause_start": 'pause_start',
#     "pause_stop": 'pause_stop',
#     "pause_dur": 'pause_dur',
#     "pause_id": 'pause_id',
#
#     "stridechain_start": 'stridechain_start',
#     "stridechain_stop": 'stridechain_stop',
#     "stridechain_dur": 'stridechain_dur',
#     "stridechain_id": 'stridechain_id',
#     "stridechain_length": 'stridechain_length',
#
#     "stridechain_dur_ratio": 'stridechain_dur_ratio',
#     "pause_dur_ratio": 'pause_dur_ratio',
#
#     "num_pauses": 'num_pauses',
#     "cum_pause_dur": 'cum_pause_dur',
#     "num_stridechains": 'num_stridechains',
#     "cum_stridechain_dur": 'cum_stridechain_dur',
#
#     "num_feeds": 'num_feeds',
#     "feed_dur_ratio": 'feed_dur_ratio',
#     "explore2exploit_balance": 'explore2exploit_balance',
#     'mean_feed_freq': 'mean_feed_freq',
#
#     "num_strides": 'num_strides',
#     "stride_dur_ratio": 'stride_dur_ratio',
# }
#
# deb_pars = {
#     "deb_f": 'deb_f',
#     "deb_f_mean": 'deb_f_mean',
#     "deb_f_deviation": 'deb_f_deviation',
#     "deb_f_deviation_mean": 'deb_f_deviation_mean',
#     "reserve": 'reserve',
#     "reserve_density": 'reserve_density',
#     "structural_length": 'structural_length',
#     "maturity": 'maturity',
#     "reproduction": 'reproduction',
#     "puppation_buffer": 'puppation_buffer',
#     "structure": 'structure',
#     "hunger": 'hunger',
#     "birth_time_in_hours": 'birth_time_in_hours',
#     "pupation_time_in_hours": 'pupation_time_in_hours',
#     "death_time_in_hours": 'death_time_in_hours',
#     "hours_as_larva": 'hours_as_larva',
#     "age": 'age_in_hours',
#     "food_absorption_efficiency": 'food_absorption_efficiency',
#     "amount_faeces": 'amount_faeces',
#     "faeces_ratio": 'faeces_ratio',
#
# }
#
# food_pars = {
#     #     These refer to food agents
#     "final_amount": 'amount',
#     "initial_amount": 'initial_amount',
# }

# step_database = {
#     **pos_xy,
#     **orientation_pars,
#     **lin_pars,
#     **ang_pars,
#     **effector_pars,
#     **intermitter_pars,
#     **body_pars,
#     **deb_pars}

# endpoint_database = {
#     "sim_dur": 'sim_dur',
#     nam.cum('dur'): 'cum_dur',
#     **pos_xy,
#     **orientation_pars,
#     **lin_pars,
#     **ang_pars,
#     **effector_pars,
#     **intermitter_pars,
#     **body_pars,
#     **deb_pars,
#     **food_pars}

collection_dict = {
    'bouts': ['x', 'y', 'b', 'fou', 'rou', 'v', 'sv', 'd', 'fov', 'bv', 'sd', 'o_cent'],
    # 'bouts': ['str_d_mu', 'str_sd_mu'],
    'basic': ['x', 'y', 'b', 'fo'],
    'e_basic': ['l_mu', nam.cum('d'), f'{nam.cum("sd")}', nam.cum('t'), 'x', 'y', 'sv_mu'],
    'e_dispersion': ['dsp', 'sdsp', 'dsp_max', 'sdsp_max', 'dsp_0_40', 'dsp_0_80', 'dsp_20_80', 'sdsp_0_40',
                     'sdsp_0_80', 'sdsp_20_80'],
    'spatial': flatten_list([[k, f's{k}'] for k in
                             ['dsp', 'd', 'v', 'a', 'D_x', 'xv', 'xa', 'D_y', 'yv', 'ya', nam.cum('d'),
                              nam.cum('D_x'), nam.cum('D_y'), ]]),
    'e_spatial': [f'tor{i}_mu' for i in ['', 2, 5, 10, 20]],
    'angular': ['b', 'bv', 'ba', 'fo', 'fov', 'foa', 'ro', 'rov', 'roa'],

    'chemorbit': ['d_cent', 'sd_cent', 'o_cent'],
    'e_chemorbit':
        flatten_list(
            [[k, f'{k}_mu', f'{k}_std', f'{k}_max', f'{k}_fin'] for k in ['d_cent', 'sd_cent']]),
    'chemotax': ['d_chem', 'sd_chem', 'o_chem'],
    'e_chemotax': flatten_list(
        [[k, f'{k}_mu', f'{k}_std', f'{k}_max', f'{k}_fin'] for k in ['d_chem', 'sd_chem']]),

    'olfactor': ['Act_tur', 'A_tur', 'A_olf'],
    'odors': ['c_odor1', 'c_odor2', 'c_odor3', 'dc_odor1', 'dc_odor2', 'dc_odor3'],
    # 'constants': ['dt', 'x0', 'y0'],
}


class NamedRandomActivation(RandomActivation):
    from lib.model.agents._agent import LarvaworldAgent
    def __init__(self, id, model, **kwargs):
        super().__init__(model)
        self.id = id
        self._agents = OrderedDict()  # type: Dict[int, LarvaworldAgent]

    @property
    def agents(self) -> List[LarvaworldAgent]:
        return list(self._agents.values())


# Extension of DataCollector class so that it only collects from a given schedule
class TargetedDataCollector(DataCollector):
    def __init__(self, schedule, pars, par_dict):
        self.schedule = schedule
        super().__init__(agent_reporters=self.valid_reporters(pars, par_dict))
        pref = [f'model.{self.schedule.id}.steps', 'unique_id']
        self.rep_funcs = self.agent_reporters.values()
        if all([hasattr(r, 'attribute_name') for r in self.rep_funcs]):
            # attributes = [f.attribute_name for f in self.rep_funcs]
            self.reports = attrgetter(*pref + [r.attribute_name for r in self.rep_funcs])
        else:
            self.reports = None

    def valid_reporters(self, pars, par_dict):
        # from lib.conf.par import load_ParDict
        dic0 = par_dict
        # dic0 = load_ParDict()
        ks = [k for k in pars if k in dic0.keys()]
        dic = {}
        for k in ks:
            d, p = dic0[k]['d'], dic0[k]['p']
            try:
                temp = [getattr(l, p) for l in self.schedule.agents]
                dic.update({d: p})
            except:
                pass
        return dic

    def _record_agents(self, model, schedule):
        if self.reports is not None:
            return map(self.reports, schedule.agents)
        else:
            def get_reports(agent):
                prefix = (schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in self.rep_funcs)
                return prefix + reports

            return map(get_reports, schedule.agents)

    def collect(self, model):
        """ Collect all the data for the given model object_class. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            agent_records = self._record_agents(model, self.schedule)
            self._agent_records[self.schedule.steps] = list(agent_records)

    # def generate_database(self, mode):
    #     if mode == 'step':
    #         midline_xy = midline_xy_pars()
    #         contour_xy = contour_xy_pars()
    #         full_step_database = {**midline_xy,
    #                               **contour_xy,
    #                               **step_database}
    #         return full_step_database
    #
    #     elif mode == 'endpoint':
    #         return endpoint_database


def midline_xy_pars(N=11):
    midline_xy = {}
    points = ['head'] + [f'seg{i}' for i in np.arange(2, N, 1)] + ['tail']
    # points = ['head'] + [f'spinepoint_{i}' for i in np.arange(2, N, 1)] + ['tail']
    for i, p in enumerate(points):
        midline_xy.update(
            {f'{p}_x': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                0] * 1000 / a.model.scaling_factor,
             f'{p}_y': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                1] * 1000 / a.model.scaling_factor,
             })
    return midline_xy


def contour_xy_pars(N=22):
    contour_xy = {}
    contour = [f'contourpoint_{j}' for j in range(N)]
    for j, c_point in enumerate(contour):
        contour_xy.update(
            {f'{c_point}_x': lambda a, j_bound=j: a.get_contour()[j_bound][0] * 1000 / a.model.scaling_factor,
             f'{c_point}_y': lambda a, j_bound=j: a.get_contour()[j_bound][1] * 1000 / a.model.scaling_factor,
             })
    return contour_xy


output_dict = {
    'olfactor': {
        'step': ['c_odor1', 'dc_odor1', 'c_odor2', 'dc_odor2', 'A_olf', 'Act_tur', 'A_tur', 'Act_cr'],
        'endpoint': []},

    'toucher': {
        'step': ['A_touch', 'A_tur', 'cum_f_det', 'on_food_tr'],
        'endpoint': ['on_food_tr']},

    'feeder': {
        'step': ['l', 'm', 'f_am', 'sf_am', 'EEB'],
        'endpoint': ['l', 'm', 'f_am', 'sf_am', 'on_food_tr']},

    'gut': {'step': ['sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M'],
            'endpoint': ['sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']},
    'pose': {'step': ['x', 'y', 'b', 'fo', 'ro'],
             'endpoint': ['l', 'cum_t', 'x']},
    'memory': {'step': [],
               'endpoint': [],
               'tables': {'best_gains': ['unique_id', 'first_odor_best_gain', 'second_odor_best_gain', 'cum_reward',
                                         'best_olfactor_decay']}},
    'midline': None,
    'contour': None,
    'source_vincinity': {'step': [], 'endpoint': ['d_cent_fin']},
    'source_approach': {'step': [], 'endpoint': ['d_chem_fin']},
}

output_keys = list(output_dict.keys())
