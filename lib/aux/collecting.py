"""Here we state all possible collected parameters for the simulations"""
from collections import OrderedDict
from typing import Dict, List

import numpy as np
from operator import attrgetter

from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

import lib.aux.naming as nam
from lib.aux.colsNstr import rgetattr



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
    def __init__(self, schedule, pars):
        self.schedule = schedule
        super().__init__(agent_reporters=self.valid_reporters(pars))
        pref = [f'model.{self.schedule.id}.steps', 'unique_id']
        self.rep_funcs = self.agent_reporters.values()
        if all([hasattr(r, 'attribute_name') for r in self.rep_funcs]):
            self.reports = attrgetter(*pref + [r.attribute_name for r in self.rep_funcs])
        else:
            self.reports = None

    def valid_reporters(self, pars):
        from lib.registry.pars import preg
        D=preg.dict
        ks = [k for k in pars if k in D.keys()]
        dic = {}
        for k in ks:
            d, p = D[k].d, D[k].codename
            try:
                temp = [rgetattr(l, p) for l in self.schedule.agents]
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

    'thermo': {
        'step': ['temp_W', 'dtemp_W', 'temp_C', 'dtemp_C', 'A_therm'],
        'endpoint': []},

    'toucher': {
        'step': ['A_touch', 'A_tur', 'Act_tur', 'cum_f_det', 'on_food_tr', 'on_food'],
        'endpoint': ['on_food_tr']},

    'wind': {
        'step': ['A_wind'],
        'endpoint': []},

    'feeder': {
        'step': ['l', 'm', 'f_am', 'sf_am', 'EEB'],
        'endpoint': ['l', 'm', 'f_am', 'sf_am', 'on_food_tr']
    },

    'gut': {'step': ['sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M', 'sf_abs_M', 'f_abs_M', 'sf_faeces_M', 'f_faeces_M',
                     'f_am'],
            'endpoint': ['sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M', 'sf_abs_M', 'f_abs_M', 'sf_faeces_M',
                         'f_faeces_M', 'f_am']},
    'pose': {'step': ['x', 'y', 'b', 'fo', 'ro'],
             'endpoint': ['l', 'cum_t', 'x']},
    'memory': {'step': [],
               'endpoint': [],
               'tables': {'best_gains': ['unique_id', 'first_odor_best_gain', 'second_odor_best_gain', 'cum_reward',
                                         'best_olfactor_decay']}},
    'midline': None,
    'contour': None,
    # 'source_vincinity': {'step': [], 'endpoint': ['d_cent_fin']},
    # 'source_approach': {'step': [], 'endpoint': ['d_chem_fin']},
}

output_keys = list(output_dict.keys())
