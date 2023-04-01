import copy
import agentpy
import numpy as np

from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import nam


from larvaworld.lib.model import agents, envs


from larvaworld.lib.screen import ScreenManager
from larvaworld.lib.sim.base_run import BaseRun


class ReplayRun(BaseRun):
    def __init__(self, dataset=None, experiment='replay',**kwargs):
        super().__init__(runtype = 'Replay', experiment=experiment,**kwargs)
        self.dataset = reg.retrieve_dataset(dataset=dataset, refID=self.p.refID, dir=self.p.dir)

    def setup(self, screen_kws={}):
        s,e,c=self.smaller_dataset(self.p)
        fp,fs,dc=self.p.fix_point,self.p.fix_segment,self.p.dynamic_color

        if fp is not None:
            s, bg = reg.funcs.preprocessing['fixation'](s, point=fp, fix_segment=fs, c=c)
        else:
            bg = None
        self.dt = c.dt
        self.Nsteps = c.Nsteps
        self._steps = self.Nsteps
        self.draw_Nsegs = self.p.draw_Nsegs


        self.config = c
        self.step_data=s
        self.endpoint_data=e
        self.build_env(c.env_params)

        self.place_agents(s,e,c)

        screen_kws = {
            'video': not self.p.overlap_mode,
            'background_motion': bg,
            'traj_color':s[dc] if dc is not None and dc in s.columns else None,
        }
        self.screen_manager = ScreenManager(model=self, **screen_kws)



    def place_agents(self, s,e,c):
        try:
            ls = e['length'].values
        except:
            ls = np.ones(c.N) * 5
        ids=c.agent_ids
        agent_list = [agents.LarvaReplay(model=self, unique_id=id, length=ls[i], data=s.xs(id, level='AgentID', drop_level=True)) for i, id in enumerate(ids)]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)



    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        if not self.is_paused:

            self.step()
            self.update()
            self.t += 1
            if self.t >= self._steps :
                self.running = False

    def step(self):
        """ Defines the models' events per simulation step. """
        self.agents.step()
        self.screen_manager.step(self.t)

    def end(self):
        self.screen_manager.finalize(self.t)


    def define_config(self, p):
        c = self.dataset.config.get_copy()
        if type(p.track_point) == int:
            c.point = 'centroid' if p.track_point == -1 else nam.midline(c.Npoints, type='point')[p]
        if p.agent_ids is not None:
            if type(p.agent_ids) == list and all([type(i) == int for i in p.agent_ids]):
                p.agent_ids = [c.agent_ids[i] for i in p.agent_ids]
            c.agent_ids = p.agent_ids
            c.N = len(c.agent_ids)
        if p.env_params is not None:
            c.env_params = p.env_params
        c.env_params.windscape = None
        if p.close_view:
            c.env_params.arena = reg.get_null('arena', dims=(0.01, 0.01))
        return c

    def smaller_dataset(self,p):
        c=self.define_config(p)

        def get_data(d,ids) :
            if not hasattr(d, 'step_data'):
                d.load(h5_ks=['contour', 'midline'])
            s, e = d.step_data, d.endpoint_data
            e0=copy.deepcopy(e.loc[ids])
            s0=copy.deepcopy(s.loc[(slice(None), ids), :])
            return s0,e0

        s0,e0=get_data(self.dataset,c.agent_ids)

        if p.time_range is not None:
            a, b = p.time_range
            a = int(a / c.dt)
            b = int(b / c.dt)
            s0 = s0.loc[(slice(a, b), slice(None)), :]

        if p.transposition is not None:
            try:
                s_tr = self.dataset.load_traj(mode=p.transposition)
                s0.update(s_tr)
            except:
                s0 = reg.funcs.preprocessing["transposition"](s0, c=c, transposition=p.transposition,replace=True)
            xy_max=2*np.max(s0[nam.xy(c.point)].dropna().abs().values.flatten())
            c.env_params.arena = reg.get_null('arena', dims=(xy_max, xy_max))
        c.Nsteps = len(s0.index.unique('Step').values)
        return s0,e0, c