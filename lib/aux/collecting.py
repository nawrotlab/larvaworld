"""Here we state all possible collected parameters for the simulations"""
import numpy as np
from operator import attrgetter

from mesa.datacollection import DataCollector
from scipy.spatial.distance import euclidean

import lib.aux.functions as fun

# Extension of DataCollector class so that it only collects from a given schedule


class TargetedDataCollector(DataCollector):
    def __init__(self, target_schedule, mode, pars):
        self.database = self.generate_database(mode)
        agent_reporters = dict((k, self.database[k]) for k in pars if k in self.database)
        super().__init__(agent_reporters=agent_reporters)
        self.target_schedule = target_schedule

    def _record_agents(self, model, this_schedule):
        # if target_schedule is None :
        #     target_schedule=self.target_schedule
        # this_schedule = getattr(model, f'{target_schedule}')
        """ Record agents data in a mapping of functions and agents. """
        rep_funcs = self.agent_reporters.values()
        if all([hasattr(rep, 'attribute_name') for rep in rep_funcs]):
            prefix = ['model.t.steps', 'unique_id']
            attributes = [func.attribute_name for func in rep_funcs]
            get_reports = attrgetter(*prefix + attributes)
        else:
            def get_reports(agent):
                prefix = (this_schedule.steps, agent.unique_id)
                reports = tuple(rep(agent) for rep in rep_funcs)
                return prefix + reports
        agent_records = map(get_reports, this_schedule.agents)
        return agent_records

    def collect(self, model):
        this_schedule = getattr(model, self.target_schedule)
        """ Collect all the data for the given model object. """
        if self.model_reporters:
            for var, reporter in self.model_reporters.items():
                self.model_vars[var].append(reporter(model))

        if self.agent_reporters:
            agent_records = self._record_agents(model, this_schedule)
            self._agent_records[this_schedule.steps] = list(agent_records)

    def generate_database(self, mode):
        if mode == 'step':
            spinepoints_xy = {}
            spinepoints = ['head'] + [f'spinepoint_{i}' for i in np.arange(2, 12, 1)] + ['tail']
            for i, point in enumerate(spinepoints):
                # print(i,point)
                spinepoints_xy.update(
                    {f'{point}_x': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                            0] * 1000 / a.model.scaling_factor,
                     f'{point}_y': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[
                                                            1] * 1000 / a.model.scaling_factor,
                     f'{point}_x_in_px': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[0],
                     f'{point}_y_in_px': lambda a, i_bound=i: a.get_segment(i_bound).get_position()[1]
                     })
            contourpoints_xy = {}
            contourpoints = [f'contourpoint_{j}' for j in range(22)]
            for j, c_point in enumerate(contourpoints):
                contourpoints_xy.update(
                    {f'{c_point}_x': lambda a, j_bound=j: a.get_contour()[j_bound][0] * 1000 / a.model.scaling_factor,
                     f'{c_point}_y': lambda a, j_bound=j: a.get_contour()[j_bound][1] * 1000 / a.model.scaling_factor,
                     f'{c_point}_x_in_px': lambda a, j_bound=j: a.get_contour()[j_bound][0],
                     f'{c_point}_y_in_px': lambda a, j_bound=j: a.get_contour()[j_bound][1]
                     })
            # print(spinepoint_xy)

            step_parameter_partial = {
                "head_orientation_in_rad": lambda a: a.get_head().get_normalized_orientation(),
                "head_orientation_unwrapped_in_rad": lambda a: a.get_head().get_orientation(),
                "head_orientation": lambda a: np.rad2deg(
                    a.get_head().get_normalized_orientation()),
                "head_orientation_unwrapped": lambda a: np.rad2deg(a.get_head().get_orientation()),

                # FIXME These are the same two parameters
                "front_orientation": lambda a: np.rad2deg(
                    a.get_head().get_normalized_orientation()),
                "front_orientation_unwrapped": lambda a: np.rad2deg(a.get_head().get_orientation()),

                "orientation_to_center_in_rad": lambda a: fun.angle_dif(a.get_head().get_normalized_orientation(),
                                                                        fun.angle_to_x_axis(a.get_position(), (0, 0),
                                                                                            in_deg=False),
                                                                        in_deg=False),
                "orientation_to_center": lambda a: fun.angle_dif(np.rad2deg(a.get_head().get_normalized_orientation()),
                                                                 fun.angle_to_x_axis(a.get_position(), (0, 0),
                                                                                     in_deg=True),
                                                                 in_deg=True),

                "rear_orientation_in_rad": lambda a: a.get_tail().get_normalized_orientation(),
                "rear_orientation_unwrapped_in_rad": lambda a: a.get_tail().get_orientation(),
                "rear_orientation": lambda a: np.rad2deg(
                    a.get_tail().get_normalized_orientation()),
                "rear_orientation_unwrapped": lambda a: np.rad2deg(a.get_tail().get_orientation()),

                "centroid_x_in_px": lambda a: a.get_position()[0],
                "centroid_y_in_px": lambda a: a.get_position()[1],
                # "head_x_in_px": lambda a: a.get_head().get_position()[0],
                # "head_y_in_px": lambda a: a.get_head().get_position()[1],
                # "tail_x_in_px": lambda a: a.get_tail().get_position()[0],
                # "tail_y_in_px": lambda a: a.get_tail().get_position()[1],

                # "centroid_x": lambda a: a.get_position()[0] * 1000 / a.model.scaling_factor,
                # "centroid_y": lambda a: a.get_position()[1] * 1000 / a.model.scaling_factor,

                "centroid_x": lambda a: a.current_pos[0] * 1000 / a.model.scaling_factor,
                "centroid_y": lambda a: a.current_pos[1] * 1000 / a.model.scaling_factor,

                "dst_to_center": lambda a: euclidean(tuple(a.current_pos), (0, 0)) * 1000 / a.model.scaling_factor,
                "scaled_dst_to_center": lambda a: euclidean(tuple(a.current_pos), (0, 0)) / a.get_sim_length(),
                "dst_to_0.4_0.0": lambda a: euclidean(tuple(a.current_pos), (0.4, 0)) * 1000 / a.model.scaling_factor,
                "scaled_dst_to_0.4_0.0": lambda a: euclidean(tuple(a.current_pos), (0.4, 0)) / a.get_sim_length(),
                "dst_to_0.8_0.0": lambda a: euclidean(tuple(a.current_pos), (0.8, 0.0)) * 1000 / a.model.scaling_factor,
                "scaled_dst_to_0.8_0.0": lambda a: euclidean(tuple(a.current_pos), (0.8, 0.0)) / a.get_sim_length(),

                "dst_to_chemotax_odor": lambda a: euclidean(tuple(a.current_pos),
                                                            (0.8, 0.0)) * 1000 / a.model.scaling_factor,
                "scaled_dst_to_chemotax_odor": lambda a: euclidean(tuple(a.current_pos),
                                                                   (0.8, 0.0)) / a.get_sim_length(),

                "dispersion": lambda a: euclidean(tuple(a.current_pos),
                                                  tuple(a.initial_pos)) * 1000 / a.model.scaling_factor,
                "scaled_dispersion": lambda a: euclidean(tuple(a.current_pos),
                                                         tuple(a.initial_pos)) / a.get_sim_length(),

                "head_vel_in_px": lambda a: a.get_head().get_linearvelocity_amp(),
                "angular_vel_in_rad": lambda a: a.get_head().get_angularvelocity(),
                "angular_vel": lambda a: np.rad2deg(a.get_head().get_angularvelocity()),
                "bend_in_rad": lambda a: a.body_bend,
                "bend": lambda a: np.rad2deg(a.body_bend),
                "torque": lambda a: a.torque,

                "first_odor_concentration": lambda a: a.odor_concentrations[0],
                "second_odor_concentration": lambda a: a.odor_concentrations[1],
                "olfactory_activation": lambda a: a.olfactory_activation,

                "turner_activation": lambda a: a.brain.turner.activation,
                "turner_activity": lambda a: a.ang_activity,

                "crawler_activity": lambda a: a.lin_activity,
                "crawler_phi": lambda a: a.brain.crawler.phi,
                "peristalsis_cycle": lambda a: a.brain.crawler.complete_iteration,

                "feeder_cycle": lambda a: a.brain.feeder.complete_iteration,
                # "feed_success": lambda a: a.feed_success,
                "feeder_motion": lambda a: a.feeder_motion,
                # "amount_eaten": par_db['collect'].loc['f_am'],
                "amount_eaten": lambda a: a.amount_eaten,

                "length": lambda a: a.get_real_length() * 1000,
                "mass": lambda a: a.get_real_mass() * 1000,
                "max_feed_amount": lambda a: a.max_feed_amount * 1000,

                # "intermitter_active": lambda a: a.intermitter.active(),
                "pause_start": lambda a: a.brain.intermitter.pause_start,
                "pause_stop": lambda a: a.brain.intermitter.pause_stop,
                "pause_dur": lambda a: a.brain.intermitter.pause_dur,
                "pause_id": lambda a: a.brain.intermitter.pause_id,

                "stridechain_start": lambda a: a.brain.intermitter.stridechain_start,
                "stridechain_stop": lambda a: a.brain.intermitter.stridechain_stop,
                "stridechain_dur": lambda a: a.brain.intermitter.stridechain_dur,
                "stridechain_id": lambda a: a.brain.intermitter.stridechain_id,
                "stridechain_length": lambda a: a.brain.intermitter.stridechain_length,

                "deb_f": lambda a: a.deb.get_f(),
                "reserve": lambda a: a.deb.get_reserve(),
                "reserve_density": lambda a: a.deb.get_reserve_density(),
                "structural_length": lambda a: a.deb.get_L(),
                "maturity": lambda a: a.deb.get_U_H()* 1000,
                "reproduction": lambda a: a.deb.get_U_R()* 1000,
                "puppation_buffer": lambda a: a.deb.get_puppation_buffer(),
                "structure": lambda a: a.deb.get_U_V()* 1000,
                "age_in_days": lambda a: a.deb.age_day,
                "hunger": lambda a: a.deb.hunger,

            }

            step_parameter_database = {**spinepoints_xy, **contourpoints_xy, **step_parameter_partial}
            # print(step_parameter_database)
            return step_parameter_database
        elif mode == 'endpoint':
            endpoint_parameter_database = {
                "sim_duration": lambda a: a.sim_time,

                "final_x_in_px": lambda a: a.get_position()[0],
                "final_y_in_px": lambda a: a.get_position()[1],
                "final_x": lambda a: a.get_position()[0] * 1000 / a.model.scaling_factor,
                "final_y": lambda a: a.get_position()[1] * 1000 / a.model.scaling_factor,
                "num_strides": lambda a: a.brain.crawler.iteration_counter,
                "stride_dur_ratio": lambda a: a.brain.crawler.total_t / a.sim_time,
                "vel_freq": lambda a: a.brain.crawler.freq,
                "cum_dst": lambda a: a.cum_dst * 1000 / a.model.scaling_factor,
                "cum_scaled_dst": lambda a: a.cum_dst / a.get_sim_length(),
                "final_dispersion": lambda a: euclidean(tuple(a.current_pos),
                                                        tuple(a.initial_pos)) * 1000 / a.model.scaling_factor,
                "final_scaled_dispersion": lambda a: euclidean(tuple(a.current_pos),
                                                               tuple(a.initial_pos)) / a.get_sim_length(),
                "final_dst_to_center": lambda a: euclidean(tuple(a.current_pos),
                                                           (0.0, 0.0)) * 1000 / a.model.scaling_factor,
                "final_scaled_dst_to_center": lambda a: euclidean(tuple(a.current_pos),
                                                                  (0.0, 0.0)) / a.get_sim_length(),

                "max_dst_to_center": lambda a: np.nanmax([euclidean(tuple(a.trajectory[i]),
                                                                    (0.0, 0.0)) for i in
                                                          range(len(a.trajectory))]) * 1000 / a.model.scaling_factor,
                "max_scaled_dst_to_center": lambda a: np.nanmax([euclidean(tuple(a.trajectory[i]),
                                                                           (0.0, 0.0)) for i in
                                                                 range(len(a.trajectory))]) / a.get_sim_length(),

                "final_dst_to_0.4_0.0": lambda a: euclidean(tuple(a.current_pos),
                                                            (0.04, 0.0)) * 1000 / a.model.scaling_factor,
                "final_scaled_dst_to_0.4_0.0": lambda a: euclidean(tuple(a.current_pos),
                                                                   (0.04, 0.0)) / a.get_sim_length(),

                "final_dst_to_0.8_0.0": lambda a: euclidean(tuple(a.current_pos),
                                                            (0.8, 0.0)) * 1000 / a.model.scaling_factor,
                "final_scaled_dst_to_0.8_0.0": lambda a: euclidean(tuple(a.current_pos),
                                                                   (0.8, 0.0)) / a.get_sim_length(),

                "final_dst_to_chemotax_odor": lambda a: euclidean(tuple(a.current_pos),
                                                                  (0.8, 0.0)) * 1000 / a.model.scaling_factor,
                "final_scaled_dst_to_chemotax_odor": lambda a: euclidean(tuple(a.current_pos),
                                                                         (0.8, 0.0)) / a.get_sim_length(),

                "final_orientation_to_center": lambda a: fun.angle_dif(
                    np.rad2deg(a.get_head().get_normalized_orientation()),
                    fun.angle_to_x_axis(a.get_position(), (0, 0), in_deg=True),
                    in_deg=True),
                "dispersion_max": lambda a: np.max([euclidean(tuple(a.trajectory[i]),
                                                              tuple(a.initial_pos)) for i in
                                                    range(len(a.trajectory))]) * 1000 / a.model.scaling_factor,
                "scaled_dispersion_max": lambda a: np.max([euclidean(tuple(a.trajectory[i]),
                                                                     tuple(a.initial_pos)) for i in
                                                           range(len(a.trajectory))]) / a.get_sim_length(),
                "length": lambda a: a.get_real_length() * 1000,

                # Assuming that dispacement happens only because of crawling
                "stride_dst_mean":
                    lambda a: (a.cum_dst / a.brain.crawler.iteration_counter) * 1000 / a.model.scaling_factor,

                "stride_scaled_dst_mean":
                    lambda a: (a.cum_dst / a.get_sim_length()) / a.brain.crawler.iteration_counter,

                "stridechain_dur_ratio": lambda a: a.brain.intermitter.cum_stridechain_dur / a.sim_time,
                "pause_dur_ratio": lambda a: a.brain.intermitter.cum_pause_dur / a.sim_time,

                "mass": lambda a: a.get_real_mass() * 1000,
                "num_pauses": lambda a: a.brain.intermitter.pause_counter,
                "cum_pause_dur": lambda a: a.brain.intermitter.cum_pause_dur,
                "num_stridechains": lambda a: a.brain.intermitter.stridechain_counter,
                "cum_stridechain_dur": lambda a: a.brain.intermitter.cum_stridechain_dur,

                "num_feeds": lambda a: a.brain.feeder.iteration_counter,
                "feed_dur_ratio": lambda a: a.brain.feeder.total_t / a.sim_time,
                "feed_success_rate": lambda a: a.feed_success_counter / a.brain.feeder.iteration_counter,
                "amount_eaten": lambda a: a.amount_eaten,

                "body_bend_errors": lambda a: a.body_bend_errors,

                #     These refer to food agents
                "final_amount": lambda a: a.amount,
                "initial_amount": lambda a: a.initial_amount,

                "birth_time_in_hours": lambda a: a.deb.birth_time_in_hours,
                "puppation_time_in_hours": lambda a: a.deb.puppation_time_in_hours,
                "death_time_in_hours": lambda a: a.deb.death_time_in_hours,
                "deb_Nticks": lambda a: a.deb.tick_counter,
                "deb_steps_per_day": lambda a: a.deb.steps_per_day,
                "age": lambda a: a.deb.age_day * 24,


            }
            return endpoint_parameter_database


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
    'intermitter': {'step': ['pause_id', 'pause_start', 'pause_stop', 'pause_dur'] + ['stridechain_id', 'stridechain_start', 'stridechain_stop', 'stridechain_dur'],
                    'endpoint': ['num_pauses', 'cum_pause_dur', 'pause_dur_ratio',
                                 'num_stridechains', 'cum_stridechain_dur', 'stridechain_dur_ratio']},
    'olfactor': {'step': ['first_odor_concentration', 'olfactory_activation',
                          'turner_activation', 'turner_activity', 'torque', 'orientation_to_center'],
                 'endpoint': ['final_dispersion', 'final_scaled_dispersion',
                              'final_orientation_to_center']},
    'turner': {'step': ['turner_activation', 'turner_activity', 'torque', ],
               'endpoint': []},
    'crawler': {'step': ['crawler_activity'],
                'endpoint': ['stride_scaled_dst_mean', 'stride_dst_mean',
                             'cum_dst', 'cum_scaled_dst',
                             'num_strides', 'stride_dur_ratio', 'vel_freq']},
    'feeder': {'step': ['length', 'mass', 'amount_eaten'],
               'endpoint': ['mass', 'num_feeds', 'feed_success_rate', 'amount_eaten',
                            'feed_dur_ratio']},
    'deb': {'step': ['deb_f', 'reserve', 'reserve_density',
                     # 'structural_length', 'maturity', 'reproduction','structure','age_in_days',
                     'hunger',  'puppation_buffer'],
            'endpoint': ['birth_time_in_hours', 'puppation_time_in_hours',
                         'death_time_in_hours', 'age']},
    'pose': {'step': ['centroid_x', 'centroid_y', 'bend', 'front_orientation', 'rear_orientation'],
             'endpoint': []},
    'nengo': {'step': ['crawler_activity', 'turner_activity', 'feeder_motion'],
             'endpoint': []}

}
