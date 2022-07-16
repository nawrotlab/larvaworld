import numpy as np

from lib.model.envs._larvaworld import LarvaWorld
from lib.model.agents._larva_replay import LarvaReplay
import lib.aux.naming as nam


class LarvaWorldReplay(LarvaWorld):
    def __init__(self, step_data, endpoint_data, config, draw_Nsegs=None, experiment='replay', **kwargs):
        super().__init__(experiment=experiment, dt=config.dt,
                         Nsteps=config.Nsteps,env_params=config.env_params,
                         **kwargs)
        self.draw_Nsegs = draw_Nsegs
        self.step_data = step_data
        self.endpoint_data = endpoint_data
        self.config=config
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_agents = len(self.agent_ids)
        try:
            self.lengths = self.endpoint_data['length'].values
        except:
            self.lengths = np.ones(self.num_agents) * 5

        self.define_pars()
        self.create_flies()

    def define_pars(self):
        c=self.config
        cols=self.step_data.columns
        self.pos_pars = nam.xy(c.point) if set(nam.xy(c.point)).issubset(cols) else ['x','y']
        self.mid_pars = [xy for xy in nam.xy(nam.midline(c.Npoints, type='point')) if
                         set(xy).issubset(cols)]
        self.Npoints = len(self.mid_pars)
        self.con_pars = [xy for xy in nam.xy(nam.contour(c.Ncontour)) if set(xy).issubset(cols)]
        self.Ncontour = len(self.con_pars)
        self.cen_pars = nam.xy('centroid') if set(nam.xy('centroid')).issubset(cols) else []
        self.chunk_pars = [p for p in ['stride_stop', 'stride_id', 'pause_id', 'feed_id'] if
                           p in cols]
        self.or_pars = [p for p in nam.orient(['front', 'rear', 'head', 'tail']) if p in cols]
        self.Nors = len(self.or_pars)
        self.ang_pars = ['bend'] if 'bend' in cols else []
        self.Nangles = len(self.ang_pars)

    def create_flies(self):
        for i, id in enumerate(self.agent_ids):
            data = self.step_data.xs(id, level='AgentID', drop_level=True)
            f = LarvaReplay(model=self, unique_id=id, length=self.lengths[i], data=data)
            self.active_larva_schedule.add(f)
            self.space.place_agent(f, (0, 0))

    def step(self):
        self.Nticks += 1
        # Tick sim_clock
        self.sim_clock.tick_clock()
        self.active_larva_schedule.step()
        self.active_food_schedule.step()


