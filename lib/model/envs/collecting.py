from collections import OrderedDict
from typing import Dict, List

import numpy as np
from operator import attrgetter

from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from lib import reg, aux




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
        D=reg.par.dict
        ks = [k for k in pars if k in D.keys()]
        dic = {}
        self.invalid_keys=aux.AttrDict({'not_in_registry' : [k for k in pars if k not in D.keys()], 'not_in_agent':{}})
        for k in ks:
            d, p = D[k].d, D[k].codename
            try:
                temp = [aux.rgetattr(l, p) for l in self.schedule.agents]
                dic.update({d: p})
            except:
                self.invalid_keys.not_in_agent[d]=p
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
