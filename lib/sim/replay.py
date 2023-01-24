import copy

import agentpy
import numpy as np

from lib import reg, aux, util
from lib.model import envs, agents
from lib.aux import naming as nam
from lib.process.spatial import fixate_larva
from lib.screen import ScreenManager


class ReplayRun(agentpy.Model):
    def setup(self, experiment='replay', dataset=None, refID=None, id=None, save_to=None, agent_ids=None,  time_range=None,
                  draw_Nsegs=None, env_params=None,close_view=False, track_point=None, dynamic_color=None,overlap_mode=False,
                  transposition=None, fix_point=None, fix_segment=None):

        if dataset is None and refID is not None :
            dataset=reg.loadRef(refID)
        s,e,c=smaller_dataset(d=dataset,ids=agent_ids, transposition=transposition, time_range=time_range, track_point=track_point,
                              env_params=env_params,close_view=close_view)


        c.env_params.windscape=None

        if save_to is None:
            save_to = reg.datapath('visuals',c.dir)
        self.save_to = save_to

        if id is None:
            if transposition is not None:
                n1 = f'aligned_to_{transposition}'
            elif fix_point is not None:
                n1 = f'fixed_at_{fix_point}'
                if overlap_mode :
                    n1=f'{n1}_overlap'
            else:
                n1 = 'normal'
            id = f'{c.id}_{n1}'
        self.id=id

        if not overlap_mode :
            vis_kwargs = reg.get_null(name='visualization', mode='video', video_speed=60, media_name=self.id)
        else :
            vis_kwargs = reg.get_null(name='visualization', mode='image', image_mode='overlap', media_name=self.id, draw_contour=False)


        if fix_point is not None:
            s, bg = fixate_larva(s, point=fix_point, fix_segment=fix_segment, c=c)
        else:
            bg = None
        self.experiment = experiment
        self.dt = c.dt
        self.Nsteps = c.Nsteps
        self._steps = c.Nsteps
        self.scaling_factor = 1
        self.is_paused = False
        self.draw_Nsegs = draw_Nsegs
        self.step_data = s
        self.endpoint_data = e
        self.config = c
        self.agent_ids = s.index.unique('AgentID').values
        try:
            self.lengths = e['length'].values
        except:
            self.lengths = np.ones(len(self.agent_ids)) * 5

        self.build_env(c.env_params)

        self.define_pars()
        self.place_agents()

        screen_kws = {
            'vis_kwargs': vis_kwargs,
            'background_motion': bg,
            'traj_color':s[dynamic_color] if dynamic_color is not None and dynamic_color in s.columns else None,
        }

        self.screen_manager = ScreenManager(model=self, **screen_kws)



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

    def place_agents(self):
        agent_list = [agents.LarvaReplay(model=self, unique_id=id, length=self.lengths[i], data=self.step_data.xs(id, level='AgentID', drop_level=True)) for i, id in enumerate(self.agent_ids)]
        self.space.add_agents(agent_list, positions=[a.pos for a in agent_list])
        self.agents = agentpy.AgentList(model=self, objs=agent_list)

    def place_obstacles(self, barriers={}):
        self.borders, self.border_lines = [], []
        for id, pars in barriers.items():
            b = envs.Border(unique_id=id, **pars)
            self.borders.append(b)
            self.border_lines += b.border_lines

    def place_food(self, food_grid=None, source_groups={}, source_units={}):
        self.food_grid = envs.FoodGrid(**food_grid, model=self) if food_grid else None
        sourceConfs = util.generate_sourceConfs(source_groups, source_units)
        source_list = [agents.Food(model=self, **conf) for conf in sourceConfs]
        self.space.add_agents(source_list, positions=[a.pos for a in source_list])
        self.sources = agentpy.AgentList(model=self, objs=source_list)
        # self.foodtypes = get_all_foodtypes(food_grid, source_groups, source_units)
        # self.source_xy = get_source_xy(source_groups, source_units)

    def sim_step(self):
        """ Proceeds the simulation by one step, incrementing `Model.t` by 1
        and then calling :func:`Model.step` and :func:`Model.update`."""
        if not self.is_paused:
            self.t += 1
            self.step()
            self.update()
            if self.t >= self._steps :
                self.running = False

    def step(self):
        """ Defines the models' events per simulation step. """
        self.agents.step()
        self.screen_manager.step()

    @property
    def Nticks(self):
        return self.t

    def build_env(self, env_params):
        # Define environment
        # self.env_pars = env_params

        self.space = envs.Arena(self, **env_params.arena)
        self.arena_dims = self.space.dims

        self.place_obstacles(env_params.border_list)
        self.place_food(**env_params.food_params)

        '''
        Sensory landscapes of the simulation environment arranged per modality
        - Olfactory landscapes : odorscape
        - Wind landscape : windscape
        - Temperature landscape : thermoscape
        '''
        # self.odor_ids = get_all_odors(self.p.larva_groups, env_params.food_params)
        self.odor_layers = create_odor_layers(model=self, sources=self.sources, pars=env_params.odorscape)
        self.windscape = envs.WindScape(model=self, **env_params.windscape) if env_params.windscape else None
        self.thermoscape = envs.ThermoScape(**env_params.thermoscape) if env_params.thermoscape else None

    def get_food(self):
        return self.sources

    def get_flies(self, ids=None, group=None):
        ls = self.agents
        if ids is not None:
            ls = [l for l in ls if l.unique_id in ids]
        if group is not None:
            ls = [l for l in ls if l.group == group]
        return ls

    def get_all_objects(self):
        return self.get_food() + self.get_flies() + self.borders

def create_odor_layers(model, sources, pars=None):
    odor_layers = {}
    ids = aux.unique_list([s.odor_id for s in sources if s.odor_id is not None])
    for id in ids:
        od_sources = [f for f in sources if f.odor_id == id]
        temp = aux.unique_list([s.default_color for s in od_sources])
        if len(temp) == 1:
            c0 = temp[0]
        elif len(temp) == 3 and all([type(k) == float] for k in temp):
            c0 = temp
        else:
            c0 = aux.random_colors(1)[0]
        kwargs = {
            'model': model,
            'unique_id': id,
            'sources': od_sources,
            'default_color': c0,
        }
        if pars.odorscape == 'Diffusion':
            odor_layers[id] = envs.DiffusionValueLayer(grid_dims=pars['grid_dims'],
                                                        evap_const=pars['evap_const'],
                                                        gaussian_sigma=pars['gaussian_sigma'],
                                                        **kwargs)
        elif pars.odorscape == 'Gaussian':
            odor_layers[id] = envs.GaussianValueLayer(**kwargs)
    return odor_layers

def smaller_dataset(d, track_point=None, ids=None, transposition=None, time_range=None, pars=None,env_params=None,close_view=False):


    c=d.config
    c0=c.get_copy()


    if track_point is None:
        track_point = c.point
    elif type(track_point) == int:
        track_point = 'centroid' if track_point == -1 else nam.midline(c.Npoints, type='point')[track_point]
    c0.point = track_point
    if ids is not None:
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [c.agent_ids[i] for i in ids]
    else :
        ids = c.agent_ids
    c0.agent_ids = ids
    c0.N = len(ids)

    def get_data(d,ids) :
        if not hasattr(d, 'step_data'):
            d.load(h5_ks=['contour', 'midline'])
        s, e = d.step_data, d.endpoint_data
        e0=copy.deepcopy(e.loc[ids])
        s0=copy.deepcopy(s.loc[(slice(None), ids), :])
        return s0,e0

    s0,e0=get_data(d,ids)

    if pars is not None:
        s0 = s0.loc[(slice(None), slice(None)), pars]

    if env_params is not None:
        c0.env_params = env_params

    if transposition is not None:
        try:
            s_tr = d.load_traj(mode=transposition)
            s0.update(s_tr)

        except:
            from lib.process.spatial import align_trajectories
            s0 = align_trajectories(s0, c=c0, transposition=transposition,replace=True)

        xy_max=2*np.max(s0[nam.xy(c0.point)].dropna().abs().values.flatten())
        c0.env_params.arena = reg.get_null('arena', arena_dims=(xy_max, xy_max))

    if close_view:
        c0.env_params.arena = reg.get_null('arena', arena_dims=(0.01, 0.01))


    if time_range is not None:
        a, b = time_range
        a = int(a / c.dt)
        b = int(b / c.dt)
        s0 = s0.loc[(slice(a, b), slice(None)), :]

    c0.Nsteps = len(s0.index.unique('Step').values)
    return s0,e0, c0