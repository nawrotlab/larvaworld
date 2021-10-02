import numpy as np

from lib.aux import colsNstr as fun, naming as nam
from lib.envs._larvaworld import LarvaWorld
from lib.model.agents._larva_replay import LarvaReplay


class LarvaWorldReplay(LarvaWorld):
    def __init__(self, step_data, endpoint_data, config,
                 pos_p, mid_p, cen_p, con_p, ang_p, ors_p, chunk_p, draw_Nsegs=None, experiment='replay', **kwargs):
        super().__init__(experiment=experiment, **kwargs)
        self.draw_Nsegs = draw_Nsegs
        # print(self.env_pars['arena'])
        # print(step_data)

        self.step_data = step_data
        self.endpoint_data = endpoint_data
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_agents = len(self.agent_ids)
        try:
            self.lengths = self.endpoint_data['length'].values
        except:
            self.lengths = np.ones(self.num_agents) * 5

        self.pos_pars=pos_p
        self.mid_pars=mid_p
        self.Npoints=len(mid_p)
        self.con_pars = con_p
        self.Ncontour = len(con_p)
        self.cen_pars = cen_p
        self.chunk_pars = chunk_p
        self.or_pars = ors_p
        self.Nors = len(self.or_pars)
        self.ang_pars = ang_p
        self.Nangles = len(self.ang_pars)
        # Nsegs = self.draw_Nsegs

        # if Nsegs == self.Npoints - 1:
        #     self.or_pars = ors_p
        #     self.Nors = len(self.or_pars)
        #     self.ang_pars = []
        #     self.Nangles = 0
        #     if self.Nors != Nsegs:
        #         raise ValueError(
        #             f'Orientation values are not present for all body segments : {self.Nors} of {Nsegs}')
        # elif Nsegs == 2:
        #     self.or_pars = ['front_orientation'] if 'front_orientation' in ors_p else []
        #     self.Nors = len(self.or_pars)
        #     self.ang_pars = ['bend'] if 'bend' in ang_p else []
        #     self.Nangles = len(self.ang_pars)
        #     if self.ang_pars is None  or self.or_pars  is None :
        #         raise ValueError(
        #             f'Front orientation and bend angle values are not available.')
        # elif Nsegs is None:
        #     self.or_pars = []
        #     self.Nors = 0
        #     self.ang_pars = []
        #     self.Nangles = 0
        # else:
        #     raise ValueError(f'Defined number of segments {Nsegs} must be either 2 or {self.Npoints - 1}')

        self.create_flies()

        # if 'food_params' in self.env_pars.keys():
        #     self._place_food(self.env_pars['place_params']['initial_num_food'],
        #                      self.env_pars['place_params']['initial_food_positions'],
        #                      food_pars=self.env_pars['food_params'])

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