import copy
import numpy as np


from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import nam

from larvaworld.lib.model import agents, envs
# from larvaworld.lib.process.dataset import RefDataset
from larvaworld.lib.screen import ScreenManager
from larvaworld.lib.sim.base_run import BaseRun


class ReplayRun(BaseRun):
    def __init__(self,parameters=None,  dataset=None, screen_kws={},**kwargs):
        '''
        Simulation mode 'Replay' reconstructs a real or simulated experiment from stored data.

        Args:
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            dataset: The stored dataset to replay. If not specified it is retrieved using either the storage path (parameters.dir) or the respective unique reference ID (parameters.RefID)
            experiment: The type of experiment. Defaults to 'replay'
            **kwargs: Arguments passed to parent class
        '''
        # RefDataset.__init__(self, refDataset=dataset, refID=parameters.refID, dataset_dir=parameters.dataset_dir)
        # self.refDataset.load()
        # d=self.refDataset
        d = self.refDataset = reg.conf.Ref.retrieve_dataset(dataset=dataset, refID=parameters.refID,
                                                            dir=parameters.dataset_dir)
        # Configure the dataset to replay
        self.step_data, self.endpoint_data, self.config = self.smaller_dataset(parameters, d)
        kwargs.dt=self.config.dt
        kwargs.Nsteps = self.config.Nsteps
        BaseRun.__init__(self,runtype='Replay', parameters=parameters,
                         dt = self.config.dt,Nsteps = self.config.Nsteps, **kwargs)

    def setup(self):
        s,e,c=self.step_data,self.endpoint_data,self.config
        fp,fs,dc=self.p.fix_point,self.p.fix_segment,self.p.dynamic_color

        if fp is not None:
            s, bg = reg.funcs.preprocessing['fixation'](s, point=fp, fix_segment=fs, c=c)
        else:
            bg = None
        self.draw_Nsegs = self.p.draw_Nsegs
        self.build_env(c.env_params)
        self.build_agents(s,e,c)
        screen_kws = {
            'video': not self.p.overlap_mode,
            'background_motion': bg,
            'traj_color':s[dc] if dc is not None and dc in s.columns else None,
        }
        self.screen_manager = ScreenManager(model=self, **screen_kws)



    def build_agents(self, s,e,c):
        try:
            ls = e['length'].values
        except:
            ls = np.ones(c.N) * 5
        confs=[{'unique_id':id, 'length':ls[i], 'data':s.xs(id, level='AgentID', drop_level=True)} for i, id in enumerate(c.agent_ids)]
        self.place_agents(confs)




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
        self.screen_manager.step()

    def end(self):
        self.screen_manager.finalize()


    def define_config(self, p, c):
        if type(p.track_point) == int:
            c.point = 'centroid' if p.track_point == -1 else nam.midline(c.Npoints, type='point')[p.track_point]
        if p.agent_ids not in [None, []]:
            if type(p.agent_ids) == list and all([type(i) == int for i in p.agent_ids]):
                p.agent_ids = [c.agent_ids[i] for i in p.agent_ids]
            c.agent_ids = p.agent_ids
            c.N = len(c.agent_ids)
        if p.env_params is not None:
            c.env_params = p.env_params
        c.env_params.windscape = None
        if p.close_view:
            c.env_params.arena = reg.gen.Arena(dims=(0.01, 0.01))
        return c

    def smaller_dataset(self,p, d):
        c = d.config.get_copy()
        c=self.define_config(p, c)

        def get_data(dd,ids) :
            if not hasattr(dd, 'step_data'):
                dd.load(h5_ks=['contour', 'midline'])
            s, e = dd.step_data, dd.endpoint_data
            e0=copy.deepcopy(e.loc[ids])
            s0=copy.deepcopy(s.loc[(slice(None), ids), :])
            return s0,e0

        s0,e0=get_data(d,c.agent_ids)

        if p.time_range is not None:
            a, b = p.time_range
            a = int(a / c.dt)
            b = int(b / c.dt)
            s0 = s0.loc[(slice(a, b), slice(None)), :]

        if p.transposition is not None:
            try:
                s_tr = d.load_traj(mode=p.transposition)
                s0.update(s_tr)
            except:
                s0 = reg.funcs.preprocessing["transposition"](s0, c=c, transposition=p.transposition,replace=True)
            xy_max=2*np.max(s0[nam.xy(c.point)].dropna().abs().values.flatten())
            c.env_params.arena = reg.gen.Arena(dims=(xy_max, xy_max))
        c.Nsteps = len(s0.index.unique('Step').values)
        c.duration=c.Nsteps * c.dt
        return s0,e0, c