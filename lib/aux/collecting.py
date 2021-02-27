"""Here we state all possible collected parameters for the simulations"""
import numpy as np
from operator import attrgetter

from mesa.datacollection import DataCollector
from scipy.spatial.distance import euclidean

import lib.aux.functions as fun

body_pars = {
    "length": 'length_in_mm',
    "mass": 'mass_in_mg',
}

pos_xy = {
    "centroid_x": 'x',
    "centroid_y": 'y',
    "final_x": 'x',
    "final_y": 'y'
}

orientation_pars = {
    "front_orientation": 'front_orientation_in_deg',
    "rear_orientation": 'rear_orientation_in_deg',
    "orientation_to_center": 'orientation_to_center_in_deg',
    "final_orientation_to_center": 'orientation_to_center_in_deg',
}

lin_pars = {
    "dst_to_center": 'dst_to_center_in_mm',
    "scaled_dst_to_center": 'scaled_dst_to_center',
    "dst_to_chemotax_odor": 'dst_to_chemotax_odor_in_mm',
    "scaled_dst_to_chemotax_odor": 'scaled_dst_to_chemotax_odor',

    "dispersion": 'dispersion_in_mm',
    "scaled_dispersion": 'scaled_dispersion',
    "dispersion_max": 'dispersion_max_in_mm',
    "scaled_dispersion_max": 'scaled_dispersion_max',

    "cum_dst": 'cum_dst_in_mm',
    "cum_scaled_dst": 'cum_scaled_dst',
    "final_dispersion": 'dispersion_in_mm',
    "final_scaled_dispersion": 'scaled_dispersion',
    "final_dst_to_center": 'dst_to_center_in_mm',
    "final_scaled_dst_to_center": 'scaled_dst_to_center',

    "max_dst_to_center": 'max_dst_to_center_in_mm',
    "max_scaled_dst_to_center": 'max_scaled_dst_to_center',

    "final_dst_to_chemotax_odor": 'dst_to_chemotax_odor_in_mm',
    "final_scaled_dst_to_chemotax_odor": 'scaled_dst_to_chemotax_odor',
}

ang_pars = {
    "angular_vel": 'ang_vel_in_deg',
    "bend": 'body_bend_in_deg',
    # "torque": lambda a : a.torque,
    "torque": 'torque',
    "body_bend_errors": 'body_bend_errors',
}

effector_pars = {
    "first_odor_concentration": 'first_odor_concentration',
    "second_odor_concentration": 'second_odor_concentration',
    "olfactory_activation": 'olfactory_activation',

    "turner_activation": 'turner_activation',
    "turner_activity": 'ang_activity',

    "crawler_activity": 'lin_activity',

    "feeder_motion": 'feeder_motion',
    "amount_eaten": 'amount_eaten',
    "feed_success_rate": 'feed_success_rate',

    "vel_freq": 'crawler_freq',
    "stride_dst_mean": 'stride_dst_mean_in_mm',

    "stride_scaled_dst_mean": 'stride_scaled_dst_mean',
}

intermitter_pars = {
    "pause_start": 'pause_start',
    "pause_stop": 'pause_stop',
    "pause_dur": 'pause_dur',
    "pause_id": 'pause_id',

    "stridechain_start": 'stridechain_start',
    "stridechain_stop": 'stridechain_stop',
    "stridechain_dur": 'stridechain_dur',
    "stridechain_id": 'stridechain_id',
    "stridechain_length": 'stridechain_length',

    "stridechain_dur_ratio": 'stridechain_dur_ratio',
    "pause_dur_ratio": 'pause_dur_ratio',

    "num_pauses": 'num_pauses',
    "cum_pause_dur": 'cum_pause_dur',
    "num_stridechains": 'num_stridechains',
    "cum_stridechain_dur": 'cum_stridechain_dur',

    "num_feeds": 'num_feeds',
    "feed_dur_ratio": 'feed_dur_ratio',
    "feeder_reoccurence_rate": 'feeder_reoccurence_rate',
    "explore2exploit_bias": 'explore2exploit_bias',

    "num_strides": 'num_strides',
    "stride_dur_ratio": 'stride_dur_ratio',
}

deb_pars = {
    "deb_f": 'deb_f',
    "reserve": 'reserve',
    "reserve_density": 'reserve_density',
    "structural_length": 'structural_length',
    "maturity": 'maturity',
    "reproduction": 'reproduction',
    "puppation_buffer": 'puppation_buffer',
    "structure": 'structure',
    "hunger": 'hunger',
    "birth_time_in_hours": 'birth_time_in_hours',
    "puppation_time_in_hours": 'puppation_time_in_hours',
    "death_time_in_hours": 'death_time_in_hours',
    "hours_as_larva": 'hours_as_larva',
    "deb_Nticks": 'deb_Nticks',
    "deb_steps_per_day": 'deb_steps_per_day',
    "age": 'age_in_hours',
}

food_pars = {
    #     These refer to food agents
    "final_amount": 'amount',
    "initial_amount": 'initial_amount',
}


# Extension of DataCollector class so that it only collects from a given schedule
class TargetedDataCollector(DataCollector):
    def __init__(self, schedule_id, mode, pars):
        self.database = self.generate_database(mode)
        agent_reporters = dict((k, self.database[k]) for k in pars if k in self.database)
        super().__init__(agent_reporters=agent_reporters)
        self.schedule_id = schedule_id

    def _record_agents(self, model, schedule):
        # if schedule_id is None :
        #     schedule_id=self.schedule_id
        # this_schedule = getattr(model, f'{self.schedule_id}')
        """ Record agents data in a mapping of functions and agents. """
        rep_funcs = self.agent_reporters.values()
        if all([hasattr(rep, 'attribute_name') for rep in rep_funcs]):
            prefix = [f'model.{self.schedule_id}.steps', 'unique_id']
            # prefix = [f'model.t.steps', 'unique_id']
            attributes = [func.attribute_name for func in rep_funcs]
            get_reports = attrgetter(*prefix + attributes)
        else:
            def get_reports(agent):
                prefix = (schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in rep_funcs)
                return prefix + reports
        agent_records = map(get_reports, schedule.agents)
        return agent_records

    def collect(self, model):
        schedule = getattr(model, self.schedule_id)
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            agent_records = self._record_agents(model, schedule)
            self._agent_records[schedule.steps] = list(agent_records)

    def generate_database(self, mode):
        if mode == 'step':
            midline_xy = self.midline_xy_pars()
            contour_xy = self.contour_xy_pars()
            step_database = {**midline_xy,
                             **contour_xy,
                             **pos_xy,
                             **orientation_pars,
                             **lin_pars,
                             **ang_pars,
                             **effector_pars,
                             **intermitter_pars,
                             **body_pars,
                             **deb_pars}
            return step_database

        elif mode == 'endpoint':
            endpoint_database = {
                "sim_duration": 'sim_time',
                **pos_xy,
                **orientation_pars,
                **lin_pars,
                **ang_pars,
                **effector_pars,
                **intermitter_pars,
                **body_pars,
                **deb_pars,
                **food_pars}
            return endpoint_database

    def midline_xy_pars(self, N=12):
        midline_xy = {}
        points = ['head'] + [f'spinepoint_{i}' for i in np.arange(2, N, 1)] + ['tail']
        for i, p in enumerate(points):
            midline_xy.update(
                {f'{p}_x': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                    0] * 1000 / a.model.scaling_factor,
                 f'{p}_y': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                    1] * 1000 / a.model.scaling_factor,
                 })
        return midline_xy

    def contour_xy_pars(self, N=22):
        contour_xy = {}
        contour = [f'contourpoint_{j}' for j in range(N)]
        for j, c_point in enumerate(contour):
            contour_xy.update(
                {f'{c_point}_x': lambda a, j_bound=j: a.get_contour()[j_bound][0] * 1000 / a.model.scaling_factor,
                 f'{c_point}_y': lambda a, j_bound=j: a.get_contour()[j_bound][1] * 1000 / a.model.scaling_factor,
                 })
        return contour_xy


step_db = {
    'f_am': lambda a: a.amount_eaten,
    'fee_N': lambda a: a.brain.feeder.iteration_counter,
    'pau_N': lambda a: a.brain.intermitter.pause_counter,
    'str_N': lambda a: a.brain.crawler.iteration_counter,
    'fee_tr': lambda a: a.brain.feeder.total_t / a.sim_time,
    'pau_tr': lambda a: a.brain.intermitter.cum_pause_dur / a.sim_time,
    'str_tr': lambda a: a.brain.crawler.total_t / a.sim_time,
    'chn0': lambda a: a.brain.intermitter.stridechain_start,
    'chn1': lambda a: a.brain.intermitter.stridechain_stop,
    'pau0': lambda a: a.brain.intermitter.pause_start,
    'pau1': lambda a: a.brain.intermitter.pause_stop,
    # 'fee_t': lambda a: a.feeder.total_t / a.sim_time,
    'pau_t': lambda a: a.brain.intermitter.pause_dur,
    'pau_id': lambda a: a.brain.intermitter.pause_id,
    # 'str_t': lambda a: a.crawler.total_t / a.sim_time,
    'chn_t': lambda a: a.brain.intermitter.stridechain_dur,
    'chn_id': lambda a: a.brain.intermitter.stridechain_id,
    'chn_l': lambda a: a.brain.intermitter.stridechain_length,
    # 'str0': lambda a: a.crawler.total_t / a.sim_time,
    # 'str1': lambda a: a.crawler.total_t / a.sim_time,
}

effector_collection = {
    'intermitter': {
        'step': ['pause_id', 'pause_start', 'pause_stop', 'pause_dur'] + ['stridechain_id', 'stridechain_start',
                                                                          'stridechain_stop', 'stridechain_dur'],
        'endpoint': ['num_pauses', 'cum_pause_dur', 'pause_dur_ratio',
                     'num_stridechains', 'cum_stridechain_dur', 'stridechain_dur_ratio']},
    'olfactor': {'step': ['first_odor_concentration', 'olfactory_activation',
                          'turner_activation', 'turner_activity', 'torque', 'orientation_to_center'],
                 'endpoint': ['final_dispersion', 'final_scaled_dispersion',
                              'final_orientation_to_center']},
    'turner': {'step': ['turner_activation', 'turner_activity', 'torque'],
               'endpoint': []},
    'crawler': {'step': ['crawler_activity'],
                'endpoint': ['stride_scaled_dst_mean', 'stride_dst_mean',
                             'cum_dst', 'cum_scaled_dst',
                             'num_strides', 'stride_dur_ratio', 'vel_freq']},
    'feeder': {'step': ['length', 'mass', 'amount_eaten', 'feeder_reoccurence_rate', 'explore2exploit_bias'],
               'endpoint': ['length', 'mass', 'num_feeds', 'feed_success_rate', 'amount_eaten',
                            'feed_dur_ratio']},
    'deb': {'step': ['deb_f', 'reserve', 'reserve_density',
                     # 'structural_length', 'maturity', 'reproduction','structure','age_in_days',
                     'hunger', 'puppation_buffer', 'cum_dst'],
            'endpoint': [
                'cum_dst', 'cum_scaled_dst', 'pause_dur_ratio',
                'num_strides', 'stride_dur_ratio', 'vel_freq',
                'reserve_density', 'puppation_buffer', 'hunger',
                'age','birth_time_in_hours', 'puppation_time_in_hours','death_time_in_hours','hours_as_larva'
            ]},
    'pose': {'step': ['centroid_x', 'centroid_y', 'bend', 'front_orientation', 'rear_orientation'],
             'endpoint': []},
    'nengo': {'step': ['crawler_activity', 'turner_activity', 'feeder_motion'],
              'endpoint': []}

}
