import os.path
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
import warnings
import copy

import lib.aux.functions as fun
import lib.aux.naming as nam
from lib.anal.process.basic import preprocess, process
from lib.anal.process.bouts import annotate
from lib.anal.process.spatial import align_trajectories, fixate_larva
import lib.conf.env_conf as env
from lib.conf.data_conf import SimParConf
import lib.conf.dtype_dicts as dtypes
from lib.envs._larvaworld import LarvaWorldReplay


class LarvaDataset:
    def __init__(self, dir, id='unnamed', fr=16, Npoints=3, Ncontour=0, life_params={}, arena_pars=env.dish(0.1),
                 par_conf=SimParConf, filtered_at=np.nan, rescaled_by=np.nan, save_data_flag=True, load_data=True,
                 sample_dataset='reference'):
        self.par_config = par_conf
        self.save_data_flag = save_data_flag
        self.define_paths(dir)
        if os.path.exists(self.dir_dict['conf']):
            self.config = fun.load_dict(self.dir_dict['conf'], use_pickle=False)
        else:
            self.config = {'id': id,
                           'dir': dir,
                           'fr': fr,
                           'filtered_at': filtered_at,
                           'rescaled_by': rescaled_by,
                           'Npoints': Npoints,
                           'Ncontour': Ncontour,
                           'sample_dataset': sample_dataset,
                           **par_conf,
                           'arena_pars': arena_pars,
                           **life_params
                           }


        self.__dict__.update(self.config)

        self.dt = 1 / self.fr
        self.configure_body()
        self.define_linear_metrics(self.config)
        if load_data:
            try:
                self.load()
            except:
                print('Data not found. Load them manually.')

    def set_data(self, step=None, end=None, food=None):
        if step is not None:
            self.step_data = step
            self.agent_ids = step.index.unique('AgentID').values
            self.num_ticks = step.index.unique('Step').size
        if end is not None:
            self.endpoint_data = end
        if food is not None:
            self.food_endpoint_data = food

    def replace_outliers_with_nan(self, pars, stds=None, thresholds=None, additional_pars=None):
        if self.step_data is None:
            self.load()
        s = self.step_data
        for i, p in enumerate(pars):
            for id in self.agent_ids:
                k = s.loc[(slice(None), id), p]
                l = k.values
                if stds is not None:
                    m = k.mean()
                    s = k.std()
                    ind = np.abs(l - m) > stds * s
                if thresholds is not None:
                    low, high = thresholds[i]
                    if low is not None:
                        ind = l < low
                    if high is not None:
                        ind = l > high
                l[ind] = np.nan
                s.loc[(slice(None), id), p] = l
                if additional_pars is not None:
                    for apar in additional_pars:
                        ak = s.loc[(slice(None), id), apar]
                        al = ak.values
                        al[ind] = np.nan
                        s.loc[(slice(None), id), apar] = al

        self.save()
        print('All outliers replaced')

    def drop_agents(self, agents, is_last=True, show_output=True):
        if self.step_data is None:
            self.load()
        self.step_data.drop(agents, level='AgentID', inplace=True)
        self.endpoint_data.drop(agents, inplace=True)
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_ticks = self.step_data.index.unique('Step').size

        fs = [f'{self.aux_dir}/{f}' for f in os.listdir(self.aux_dir)]
        ns = fun.flatten_list([[f'{f}/{n}' for n in os.listdir(f) if n.endswith('.csv')] for f in fs])
        for n in ns:
            try:
                df = pd.read_csv(n, index_col=0)
                df.loc[~df.index.isin(agents)].to_csv(n, index=True, header=True)
            except:
                pass
        if is_last:
            self.save()
        if show_output:
            print(f'{len(agents)} agents dropped and {len(self.endpoint_data.index)} remaining.')

    def drop_pars(self, pars=[], groups=None, is_last=True, show_output=True):
        if groups is None:
            groups = {n: False for n in
                      ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn',
                       'unused']}
        if self.step_data is None:
            self.load()
        s = self.step_data

        if groups['midline']:
            pars += fun.flatten_list(self.points_xy)
        if groups['contour']:
            pars += fun.flatten_list(self.contour_xy)
        for c in ['stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn']:
            if groups[c]:
                pars += [f'{c}_start', f'{c}_stop', f'{c}_id', f'{c}_dur', f'{c}_length']
        if groups['unused']:
            pars += self.get_unused_pars()
        pars = fun.unique_list(pars)
        s.drop(columns=[p for p in pars if p in s.columns], inplace=True)
        self.set_data(step=s)
        if is_last:
            self.save()
            self.load()
        if show_output:
            print(f'{len(pars)} parameters dropped. {len(s.columns)} remain.')

    def get_unused_pars(self):
        vels = [nam.vel(''), nam.scal(nam.vel(''))]
        lin = [nam.dst(''), nam.vel(''), nam.acc('')]
        lins = lin + nam.scal(lin) + nam.cum([nam.dst(''), nam.scal(nam.dst(''))]) + nam.max(vels) + nam.min(vels)
        beh = ['stride', nam.chain('stride'), 'pause', 'turn', 'Lturn', 'Rturn']
        behs = nam.start(beh) + nam.stop(beh) + nam.id(beh) + nam.dur(beh) + nam.length(beh)
        str = [nam.dst('stride'), nam.straight_dst('stride'), nam.orient('stride'), 'dispersion']
        strs = str + nam.scal(str)
        var = ['spinelength', 'ang_color', 'lin_color', 'state']
        vpars = lins + self.ang_pars + self.xy_pars + behs + strs + var
        pars = [p for p in self.step_data.columns.values if p not in vpars]
        return pars

    def load(self, step=True, end=True, food=False):
        if step:
            self.step_data = pd.read_csv(self.dir_dict['step'], index_col=['Step', 'AgentID'])
            self.step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
        if end:
            self.endpoint_data = pd.read_csv(self.dir_dict['end'], index_col=0)
            self.endpoint_data.sort_index(inplace=True)
        if food:
            self.food_endpoint_data = pd.read_csv(self.dir_dict['food'], index_col=0)
            self.food_endpoint_data.sort_index(inplace=True)

    @property
    def N(self):
        try:
            return len(self.agent_ids)
        except :
            return len(self.endpoint_data.index.values)

    @property
    def t0(self):
        try:
            return int(self.step_data.index.unique('Step')[0])
        except :
            return 0


    def save(self, step=True, end=True, food=False, table_entries=None):
        if self.save_data_flag == True:
            if step:
                self.step_data.to_csv(self.dir_dict['step'], index=True, header=True)
            if end:
                self.endpoint_data.to_csv(self.dir_dict['end'], index=True, header=True)
            if food:
                self.food_endpoint_data.to_csv(self.dir_dict['food'], index=True, header=True)
            if table_entries is not None:
                dir = self.dir_dict['table']
                for name, table in table_entries.items():
                    table.to_csv(f'{dir}/{name}.csv', index=True, header=True)
            self.save_config()
            print(f'Dataset {self.id} stored.')

    def save_tables(self, tables):
        for name, table in tables.items():
            path = os.path.join(self.dir_dict['table'], f'{name}.csv')
            df = pd.DataFrame(table)
            if 'unique_id' in df.columns:
                df.rename(columns={'unique_id': 'AgentID'}, inplace=True)
                N = len(df['AgentID'].unique().tolist())
                if N>0 :
                    Nrows = int(len(df.index) / N)
                    df['Step'] = np.array([[i] * N for i in range(Nrows)]).flatten()
                    df.set_index(['Step', 'AgentID'], inplace=True)
                    df.sort_index(level=['Step', 'AgentID'], inplace=True)
                df.to_csv(path, index=True, header=True)

    def save_config(self):
        for a in ['N', 't0', 'duration', 'quality', 'dt'] :
            try:
                self.config[a] = getattr(self,a)
            except:
                pass
        for k,v in self.config.items() :
            if type(v)==np.ndarray :
                self.config[k]=v.tolist()
        fun.save_dict(self.config, self.dir_dict['conf'], use_pickle=False)

    def save_agent(self, pars=None, header=True):
        if self.step_data is None:
            self.load()
        for i, agent in enumerate(self.agent_ids):
            if pars is not None:
                agent_data = self.step_data[pars].xs(agent, level='AgentID', drop_level=True)
            else:
                agent_data = self.step_data.xs(agent, level='AgentID', drop_level=True)
            dir_path = os.path.join(self.data_dir, 'single_larva_data')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path, f'{agent}.csv')
            agent_data.to_csv(file_path, index=True, header=header)
        print('All agent data saved.')

    def load_aux(self, type, name=None, file=None, as_df=True):
        if file is None:
            if name is not None:
                file = f'{name}.csv'
            else:
                raise ValueError('Neither filename nor parameter provided')
        dir = self.dir_dict[type]
        path = f'{dir}/{file}'
        u_path=f'{dir}/units.csv'
        index_col = 0 if type != 'table' else ['Step', 'AgentID']


        try:
            df= pd.read_csv(path, index_col=index_col)
        except:
            try :
                df= fun.load_dicts([path])[0]
                if as_df :
                    df=pd.DataFrame.from_dict(df)
                    df.index.set_names(index_col, inplace=True)
                # return df
            except :
                raise ValueError(f'No data found at {path}')

        if type != 'table' :
            return df
        else :
            u_dic = fun.load_dicts([u_path])[0]
            return df, u_dic

    @property
    def quality(self):
        df = self.step_data[nam.xy(self.point)[0]].values.flatten()
        valid = np.count_nonzero(~np.isnan(df))
        return np.round(valid / df.shape[0], 2)

    @property
    def duration(self):
        return int(self.endpoint_data['cum_dur'].max())


    def load_deb_dicts(self, ids=None, **kwargs):
        if ids is None:
            ids = self.agent_ids
        files = [f'{id}.txt' for id in ids]
        ds = fun.load_dicts(files=files, folder=self.dir_dict['deb'], **kwargs)
        return ds

    def get_par_list(self, track_point=None):
        angle_p = ['bend']
        or_p = ['front_orientation'] + nam.orient(self.segs)
        chunk_p = ['stride_stop', 'stride_id', 'pause_id', 'feed_id']
        if track_point is None:
            track_point = self.point
        elif type(track_point) == int:
            track_point = 'centroid' if track_point == -1 else self.points[track_point]
        if not set(nam.xy(track_point)).issubset(self.step_data.columns):
            track_point = self.points[int(self.Nsegs / 2)]
        pos_p = nam.xy(track_point) if set(nam.xy(track_point)).issubset(self.step_data.columns) else ['x', 'y']
        point_p = nam.xy(self.points, flat=True) if len(self.points_xy) >= 1 else []
        cent_p = self.cent_xy if len(self.cent_xy) >= 1 else []
        contour_p = nam.xy(self.contour, flat=True) if len(self.contour_xy) >= 1 else []

        pars = np.unique(cent_p + point_p + pos_p + contour_p + angle_p + or_p + chunk_p).tolist()
        return pars, pos_p, track_point

    def get_smaller_dataset(self, ids=None, pars=None, time_range=None, dynamic_color=None):
        if self.step_data is None:
            self.load()
        if ids is None:
            ids = self.agent_ids
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [self.agent_ids[i] for i in ids]
        if pars is None:
            pars = self.step_data.columns.values.tolist()
        elif dynamic_color is not None:
            pars.append(dynamic_color)
        pars = [p for p in pars if p in self.step_data.columns]
        if time_range is None:
            s = copy.deepcopy(self.step_data.loc[(slice(None), ids), pars])
        else:
            a, b = time_range
            a = int(a / self.dt)
            b = int(b / self.dt)
            # tick_range=(np.array(time_range)/self.dt).astype(int)
            s = copy.deepcopy(self.step_data.loc[(slice(a, b), ids), pars])
        e = copy.deepcopy(self.endpoint_data.loc[ids])
        return s, e, ids

    def visualize(self, vis_kwargs=None, agent_ids=None, save_to=None, time_range=None, draw_Nsegs=None,
                  arena_pars=None, env_params=None, space_in_mm=True, track_point=None, dynamic_color=None,
                  transposition=None, fix_point=None, secondary_fix_point=None, **kwargs):
        if vis_kwargs is None :
            vis_kwargs=dtypes.get_dict('visualization', mode='video')

        pars, pos_xy_pars, track_point = self.get_par_list(track_point)
        s, e, ids = self.get_smaller_dataset(ids=agent_ids, pars=pars, time_range=time_range,
                                             dynamic_color=dynamic_color)
        if len(ids) == 1:
            n0 = ids[0]
        elif len(ids) == len(self.agent_ids):
            n0 = 'all'
        else:
            n0 = f'{len(ids)}l'
        traj_color = s[dynamic_color] if dynamic_color is not None else None

        if env_params is None:
            if arena_pars is None:
                arena_pars = self.arena_pars
            env_params = {'arena': arena_pars}
        arena_dims = [k * 1000 for k in env_params['arena']['arena_dims']]
        env_params['arena']['arena_dims'] = arena_dims
        # print(arena_dims)



        if transposition is not None:
            s = align_trajectories(s, self.Npoints, self.Ncontour, track_point=track_point, arena_dims=arena_dims,
                                   mode=transposition, config=self.config)
            # s = self.align_trajectories(s=s, mode=transposition, arena_dims=arena_dims, track_point=track_point)
            bg = None
            n1 = 'transposed'
        elif fix_point is not None:
            s, bg = fixate_larva(s, self.Npoints, self.Ncontour, point=fix_point, secondary_point=secondary_fix_point,
                                 arena_dims=arena_dims)
            # s, bg = self.fix_point(point=fix_point, secondary_point=secondary_fix_point, s=s, arena_dims=arena_dims)
            n1 = 'fixed'
        else:
            bg = None
            n1 = 'normal'
        replay_id = f'{n0}_{n1}'
        if vis_kwargs['render']['media_name'] is None:
            vis_kwargs['render']['media_name'] = replay_id
        if save_to is None:
            save_to = self.vis_dir
        Nsteps = len(s.index.unique('Step').values)

        replay_env = LarvaWorldReplay(id=replay_id, env_params=env_params, space_in_mm=space_in_mm, dt=self.dt,
                                      vis_kwargs=vis_kwargs, save_to=save_to, background_motion=bg,
                                      dataset=self, step_data=s, endpoint_data=e, Nsteps=Nsteps, draw_Nsegs=draw_Nsegs,
                                      pos_xy_pars=pos_xy_pars, traj_color=traj_color, **kwargs)

        replay_env.run()
        print('Visualization complete')

    def process(self, is_last=True, **kwargs):
        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'dt': self.dt,
            'Npoints': self.Npoints,
            'Ncontour': self.Ncontour,
            'point': self.point,
            'config': self.config,
            'distro_dir': self.dir_dict['distro'],
            'dsp_dir': self.dir_dict['dispersion'],
        }
        process(**c, **kwargs)
        if is_last:
            self.save()

    def preprocess(self, is_last=True, **kwargs):
        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'dt': self.dt,
            'Npoints': self.Npoints,
            'config': self.config,
        }
        preprocess(**c, **kwargs)
        if is_last:
            self.save()

    def compute_preference_index(self, arena_diameter_in_mm=None, return_num=False, return_all=False, show_output=True):
        if not hasattr(self, 'endpoint_data'):
            self.load(step=False)
        e=self.endpoint_data
        r = 0.2 * self.arena_pars['arena_dims'][0]
        p='x' if 'x' in e.keys() else nam.final('x')
        d = e[p]
        N = d.count()
        N_l = d[d <= -r / 2].count()
        N_r = d[d >= +r / 2].count()
        N_m = d[(d <= +r / 2) & (d >= -r / 2)].count()
        pI = np.round((N_l - N_r) / N, 3)
        if return_num:
            if return_all:
                return pI, N, N_l, N_r
            else:
                return pI, N
        else:
            return pI

    def load_pause_dataset(self, load_simulated=False):
        try:
            filenames = [f'pause_{n}_dataset.csv' for n in ['bends', 'bendvels']]
            paths = [os.path.join(self.data_dir, name) for name in filenames]
            exp_bends = pd.read_csv(paths[0], index_col=0).values
            exp_bendvels = pd.read_csv(paths[1], index_col=0).values
        except:
            raise ValueError('Experimental pauses not found')

        if load_simulated:
            try:
                filenames = [f'pause_best_{n}_dataset.csv' for n in ['bends', 'bendvels', 'acts']]
                paths = [os.path.join(self.data_dir, name) for name in filenames]
                sim_bends = pd.read_csv(paths[0], index_col=0).values
                sim_bendvels = pd.read_csv(paths[1], index_col=0).values
                sim_acts = pd.read_csv(paths[2], index_col=0).values
                return exp_bends, exp_bendvels, sim_bends, sim_bendvels, sim_acts
            except:
                raise ValueError('Simulated pauses not found')
        else:
            return exp_bends, exp_bendvels

    def annotate(self, is_last=True, **kwargs):
        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'dt': self.dt,
            'Npoints': self.Npoints,
            'point': self.point,
            'config': self.config,
            'distro_dir': self.dir_dict['distro'],
            'stride_p_dir': self.dir_dict['stride'],
        }

        annotate(**c, **kwargs)

        if is_last:
            self.save()

    def load_fits(self, filepath=None, selected_pars=None):
        if filepath is None:
            filepath = self.dir_dict['conf']
        target = pd.read_csv(filepath, index_col='parameter')
        if selected_pars is not None:
            valid_pars = [p for p in selected_pars if p in target.index.values]
            target = target.loc[valid_pars]
        pars = target.index.values.tolist()
        dist_names = target['dist_name'].values.tolist()
        dist_args = target['dist_args'].values.tolist()
        dist_args = [tuple(float(s) for s in v.strip("()").split(",")) for v in dist_args]
        dists = [{k: v} for k, v in zip(dist_names, dist_args)]
        stats = target['statistic'].values.tolist()
        return pars, dists, stats


    def configure_body(self):
        N, Nc = self.Npoints, self.Ncontour
        self.points = nam.midline(N, type='point')

        self.Nangles = np.clip(N - 2, a_min=0, a_max=None)
        self.angles = [f'angle{i}' for i in range(self.Nangles)]
        self.Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        self.segs = nam.midline(self.Nsegs, type='seg')

        self.points_xy = nam.xy(self.points)
        self.points_dst = nam.dst(self.points)
        self.points_vel = nam.vel(self.points)
        self.points_acc = nam.acc(self.points)
        self.point_lin_pars = self.points_dst + self.points_vel + self.points_acc

        self.angles_vel = nam.vel(self.angles)
        self.angles_acc = nam.acc(self.angles)
        self.angle_pars = self.angles + self.angles_vel + self.angles_acc

        self.contour = nam.contour(Nc)
        self.contour_xy = nam.xy(self.contour)

        self.cent_xy = nam.xy('centroid')
        self.cent_dst = nam.dst('centroid')
        self.cent_vel = nam.vel('centroid')
        self.cent_acc = nam.acc('centroid')
        self.cent_lin_pars = [self.cent_dst, self.cent_vel, self.cent_acc]

        ang = ['front_orientation', 'rear_orientation', 'bend']
        self.ang_pars = ang + nam.unwrap(ang) + nam.vel(ang) + nam.acc(ang)
        self.xy_pars = nam.xy(self.points + self.contour + ['centroid'], flat=True) + nam.xy('')

        self.config['point'] = self.points[self.config['point_idx'] - 1] if type(
            self.config['point_idx']) == int else 'centroid'
        self.point=self.config['point']

    def define_paths(self, dir):
        self.dir = dir
        self.data_dir = os.path.join(dir, 'data')
        self.plot_dir = os.path.join(dir, 'plots')
        self.vis_dir = os.path.join(dir, 'visuals')
        self.aux_dir = os.path.join(dir, 'aux')
        self.dir_dict = {
            'parent': self.dir,
            'data': self.data_dir,
            'plot': self.plot_dir,
            'vis': self.vis_dir,
            'comp_plot': os.path.join(self.plot_dir, 'comparative'),
            'deb': os.path.join(self.data_dir, 'deb_dicts'),
            'aux': self.aux_dir,
            'distro': os.path.join(self.aux_dir, 'par_distros'),
            'stride': os.path.join(self.aux_dir, 'par_during_stride'),
            'dispersion': os.path.join(self.aux_dir, 'dispersion'),
            'bouts': os.path.join(self.aux_dir, 'bouts'),
            'table': os.path.join(self.aux_dir, 'tables'),
            'step': os.path.join(self.data_dir, 'step.csv'),
            'end': os.path.join(self.data_dir, 'end.csv'),
            'food': os.path.join(self.data_dir, 'food.csv'),
            'sim': os.path.join(self.data_dir, 'sim_conf.txt'),
            'fit': os.path.join(self.data_dir, 'dataset_fit.csv'),
            'conf': os.path.join(self.data_dir, 'dataset_conf.csv'),
        }
        # self.build_dirs()
        for k, v in self.dir_dict.items():
            if not str.endswith(v, 'csv') and not str.endswith(v, 'txt'):
                os.makedirs(v, exist_ok=True)

    def define_linear_metrics(self, config):
        self.distance = nam.dst(self.point)
        self.velocity = nam.vel(self.point)
        self.acceleration = nam.acc(self.point)
        if config['use_component_vel']:
            # self.distance = nam.lin(self.distance)
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

    def enrich(self,preprocessing={},processing={},annotation={},enrich_aux={},
               to_drop={}, show_output=False,is_last=True, **kwargs):
        print()
        print(f'--- Enriching dataset {self.id} with derived parameters ---')
        self.config['front_body_ratio'] = 0.5
        self.save_config()
        warnings.filterwarnings('ignore')
        c = {'show_output': show_output,
             'is_last': False}
        self.preprocess( **preprocessing,**c, **enrich_aux, **kwargs)
        self.process(**processing, **enrich_aux, **c, **kwargs)
        self.annotate(**annotation, **enrich_aux, **c, **kwargs)
        self.drop_pars(**to_drop, **c)
        if is_last:
            self.save()
        return self

    def drop_immobile_larvae(self, vel_threshold=0.1, is_last=True):
        # self.compute_spatial_metrics(mode='minimal')
        D = self.step_data[nam.scal('velocity')]
        immobile_ids = []
        for id in self.agent_ids:
            d = D.xs(id, level='AgentIDs').dropna().values
            if len(d[d > vel_threshold]) == 0:
                immobile_ids.append(id)
        print(f'{len(immobile_ids)} immobile larvae will be dropped')
        if len(immobile_ids) > 0:
            self.drop_agents(agents=immobile_ids, is_last=is_last)

    def get_par(self, par, endpoint_par=True):
        try:
            p_df = self.load_aux(type='distro', name=par)
        except:
            if endpoint_par:
                if not hasattr(self, 'end'):
                    self.load(step=False)
                p_df = self.endpoint_data[par]
            else:
                if not hasattr(self, 'step'):
                    self.load(end=False)
                p_df = self.step_data[par]
        return p_df

    def get_xy(self):
        if self.step_data is None:
            self.load()
        return self.step_data[['x', 'y']]

    def delete(self, show_output=True):
        shutil.rmtree(self.dir)
        if show_output:
            print(f'Dataset {self.id} deleted')

    def set_id(self, id, save=True):
        self.id = id
        self.config['id'] = id
        if save:
            self.save_config()

    # def build_dirs(self):
    #     for k, v in self.dir_dict.items():
    #         if not str.endswith(v, 'csv') and not str.endswith(v, 'txt'):
    #             os.makedirs(v, exist_ok=True)

    def split_dataset(self, groups=None, is_last=True, show_output=True):
        if groups is None:
            groups = fun.unique_list([id.split('_')[0] for id in self.agent_ids])
        if len(groups) == 1:
            return [self]
        new_dirs = [f'{self.dir}/../{self.id}.{f}' for f in groups]
        if all([os.path.exists(new_dir) for new_dir in new_dirs]):
            new_ds = [LarvaDataset(new_dir) for new_dir in new_dirs]
        else:
            if self.step_data is None:
                self.load()
            new_ds = []
            for f, new_dir in zip(groups, new_dirs):
                invalid_ids = [id for id in self.agent_ids if not str.startswith(id, f)]
                copy_tree(self.dir, new_dir)
                new_d = LarvaDataset(new_dir)
                new_d.drop_agents(invalid_ids, is_last=is_last, show_output=show_output)
                # new_d.load()
                # new_d.set_id(f'{self.id}_{f}', save=is_last)
                new_d.set_id(f)
                new_ds.append(new_d)
            if show_output:
                print(f'Dataset {self.id} splitted in {[d.id for d in new_ds]}')
        return new_ds

    def get_chunks(self, chunk, min_dur=0.0, max_dur=np.inf):
        # t=nam.dur(chunk)
        t, id = nam.dur(chunk), nam.id(chunk)
        s0, s1 = nam.start(chunk), nam.stop(chunk)
        if self.step_data is None:
            self.load()
        s = copy.deepcopy(self.step_data)
        e = self.endpoint_data
        counts = s[t].dropna().groupby('AgentID').count()
        ser1 = s[id].loc[s[t] >= min_dur]
        ser1.reset_index(level='Step', drop=True, inplace=True)
        ser1 = ser1.reset_index(drop=False).values.tolist()
        s = s.loc[s[id]]


