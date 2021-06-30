import json
import os.path
import shutil
from distutils.dir_util import copy_tree

import pandas as pd

from lib.anal.process.basic import preprocess, process
from lib.anal.process.bouts import detect_bouts
from lib.anal.process.spatial import align_trajectories, fixate_larva
from lib.anal.plotting import *
import lib.conf.env_conf as env
from lib.conf.data_conf import SimParConf
from lib.model.modules.intermitter import get_EEB_poly1d

from lib.stor.paths import RefFolder
from lib.envs._larvaworld import LarvaWorldReplay


class LarvaDataset:
    def __init__(self, dir, id='unnamed', fr=16, Npoints=3, Ncontour=0, life_params={}, arena_pars=env.dish(0.1),
                 par_conf=SimParConf, filtered_at=np.nan, rescaled_by=np.nan, save_data_flag=True, load_data=True,
                 sample_dataset='reference'):
        self.par_config = par_conf
        self.save_data_flag = save_data_flag
        self.define_paths(dir)
        if os.path.exists(self.dir_dict['conf']):
            with open(self.dir_dict['conf']) as tfp:
                self.config = json.load(tfp)
        else:
            self.config = {'id': id,
                           'fr': fr,
                           'filtered_at': filtered_at,
                           'rescaled_by': rescaled_by,
                           'Npoints': Npoints,
                           'Ncontour': Ncontour,
                           'sample_dataset': sample_dataset,
                           **par_conf,
                           **arena_pars,
                           **life_params
                           }

            # print(f'Initialized dataset {id} with new configuration')
        self.__dict__.update(self.config)
        self.arena_pars = {'arena_xdim': self.arena_xdim,
                           'arena_ydim': self.arena_ydim,
                           'arena_shape': self.arena_shape}
        self.dt = 1 / self.fr
        self.configure_body()
        self.define_linear_metrics(self.config)
        if load_data:
            try:
                self.load()
                # print('Data loaded successfully from stored csv files.')
            except:
                print('Data not found. Load them manually.')

    def set_data(self, step=None, end=None, food=None):
        if step is not None:
            self.step_data = step
            self.agent_ids = step.index.unique('AgentID').values
            self.num_ticks = step.index.unique('Step').size
            self.starting_tick = step.index.unique('Step')[0] if self.num_ticks>0 else 0
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
        self.starting_tick = int(self.step_data.index.unique('Step')[0])
        self.Nagents = len(self.agent_ids)
        fs = [f'{self.aux_dir}/{f}' for f in os.listdir(self.aux_dir)]
        ns = fun.flatten_list([[f'{f}/{n}' for n in os.listdir(f) if n.endswith('.csv')] for f in fs])
        for n in ns:
            # print(n)
            try:
                df = pd.read_csv(n, index_col=0)
                df.loc[~df.index.isin(agents)].to_csv(n, index=True, header=True)
                # print('ddd')
            except:
                pass
        if is_last:
            self.save()
        if show_output:
            print(f'{len(agents)} agents dropped and {len(self.endpoint_data.index)} remaining.')

    def drop_pars(self, pars=[], groups=[], is_last=True, show_output=True):
        if self.step_data is None:
            self.load()
        s = self.step_data

        if 'midline' in groups:
            pars += fun.flatten_list(self.points_xy)
        if 'contour' in groups:
            pars += fun.flatten_list(self.contour_xy)
        for c in ['stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn']:
            if c in groups:
                pars += [f'{c}_start', f'{c}_stop', f'{c}_id', f'{c}_dur', f'{c}_length']
        if 'unused' in groups:
            pars += self.get_unused_pars()

        pars = fun.unique_list(pars)

        s.drop(columns=[p for p in pars if p in s.columns], inplace=True)
        self.set_data(step=s)
        # self.set_step_data(s)
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
            # print(self.step_data)
            self.step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
            self.starting_tick = int(self.step_data.index.unique('Step')[0])
            self.Nagents = len(self.agent_ids)
        if end:
            self.endpoint_data = pd.read_csv(self.dir_dict['end'], index_col=0)

            self.endpoint_data.sort_index(inplace=True)
            self.Nagents = len(self.endpoint_data.index.values)
        if food:
            self.food_endpoint_data = pd.read_csv(self.dir_dict['food'], index_col=0)
            self.food_endpoint_data.sort_index(inplace=True)

    def save(self, step=True, end=True, food=False, table_entries=None):
        if self.save_data_flag == True:
            # print('Saving data')
            # self.build_dirs()
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

    def save_tables(self, tables):
        for name, table in tables.items():
            path = os.path.join(self.dir_dict['table'], f'{name}.csv')
            df = pd.DataFrame(table)
            if 'unique_id' in df.columns:
                df.rename(columns={'unique_id': 'AgentID'}, inplace=True)
                Nagents = len(df['AgentID'].unique().tolist())
                Nrows = int(len(df.index) / Nagents)
                df['Step'] = np.array([[i] * Nagents for i in range(Nrows)]).flatten()
                df.set_index(['Step', 'AgentID'], inplace=True)
                df.sort_index(level=['Step', 'AgentID'], inplace=True)
            df.to_csv(path, index=True, header=True)

    def save_config(self):
        try:
            self.config['Nagents'] = self.Nagents
        except:
            pass
        with open(self.dir_dict['conf'], "w") as fp:
            json.dump(self.config, fp)

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
        # print(path)
        # raise
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


    def load_deb_dicts(self, ids=None):
        if ids is None:
            ids = self.agent_ids
        files = [f'{id}.txt' for id in ids]
        ds = fun.load_dicts(files=files, folder=self.dir_dict['deb'])
        return ds

    def get_par_list(self, track_point=None):
        angle_p = ['bend']
        or_p = ['front_orientation'] + nam.orient(self.segs)
        chunk_p = ['stride_stop', 'stride_id', 'pause_id', 'feed_id']
        if track_point is None:
            track_point = self.point
        elif type(track_point) == int:
            track_point = 'centroid' if track_point == -1 else self.points[track_point]

        pos_p = nam.xy(track_point) if set(nam.xy(track_point)).issubset(self.step_data.columns) else ['x', 'y']
        # pos_p = nam.xy(track_point) if set(nam.xy(track_point)).issubset(self.step_data.columns) else []
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

    def visualize(self, vis_kwargs, agent_ids=None, save_to=None, time_range=None, draw_Nsegs=None,
                  arena_pars=None, env_params=None, space_in_mm=True, track_point=None, dynamic_color=None,
                  transposition=None, fix_point=None, secondary_fix_point=None, **kwargs):

        pars, pos_xy_pars, track_point = self.get_par_list(track_point)
        s, e, ids = self.get_smaller_dataset(ids=agent_ids, pars=pars, time_range=time_range,
                                             dynamic_color=dynamic_color)
        contour_xy = nam.xy(self.contour, flat=True)
        if (len(contour_xy) == 0 or not set(contour_xy).issubset(s.columns)) and draw_Nsegs is None:
            draw_Nsegs = self.Nsegs
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
        # arena_dims = [env_params['arena'][k] * 1 for k in ['arena_xdim', 'arena_ydim']]
        arena_dims = [env_params['arena'][k] * 100 for k in ['arena_xdim', 'arena_ydim']]
        env_params['arena']['arena_xdim'] = arena_dims[0]
        env_params['arena']['arena_ydim'] = arena_dims[1]

        if transposition is not None:
            s = align_trajectories(s, self.Npoints, self.Ncontour, track_point=track_point, arena_dims=arena_dims,
                                   mode=transposition)
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

    def preprocess(self, dic=None, is_last=True, **kwargs):
        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'dt': self.dt,
            'Npoints': self.Npoints,
            'config': self.config,
        }
        preprocess(**c, dic=dic, **kwargs)
        if is_last:
            self.save()

    def compute_preference_index(self, arena_diameter_in_mm=None, return_num=False, return_all=False, show_output=True):
        if not hasattr(self, 'end'):
            self.load(step=False)
        e=self.endpoint_data
        if arena_diameter_in_mm is None:
            arena_diameter_in_mm = self.arena_xdim * 1000
        r = 0.2 * arena_diameter_in_mm
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

    def detect_bouts(self, is_last=True, **kwargs):
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

        detect_bouts(**c, **kwargs)

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
        if not np.isnan(config['point_idx']):
            self.point = self.points[config['point_idx'] - 1]
        else:
            self.point = 'centroid'
        self.distance = nam.dst(self.point)
        self.velocity = nam.vel(self.point)
        self.acceleration = nam.acc(self.point)
        if config['use_component_vel']:
            # self.distance = nam.lin(self.distance)
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

    def enrich(self,
               preprocessing={
                   'rescale_by': None,
                   'drop_collisions': False,
                   'interpolate_nans': False,
                   'filter_f': None
               },
               processing=['angular', 'spatial'],
               to_drop=[], mode='minimal', dispersion_starts=[0, 20], dispersion_stops=[40, 80],
               bouts=['turn', 'stride', 'pause'],
               source=None, show_output=True, recompute=False,
               is_last=True, **kwargs):
        print()
        print(f'--- Enriching dataset {self.id} with derived parameters ---')
        self.config['front_body_ratio'] = 0.5
        self.save_config()
        warnings.filterwarnings('ignore')
        c = {'show_output': show_output,
             'is_last': False}
        self.preprocess(**c, recompute=recompute, mode=mode, dic=preprocessing, **kwargs)
        self.process(types=processing, recompute=recompute, mode=mode, dsp_starts=dispersion_starts,
                     dsp_stops=dispersion_stops, source=source, **c, **kwargs)
        self.detect_bouts(bouts=bouts, recompute=recompute, source=source, **c, **kwargs)
        self.drop_pars(groups=to_drop, **c)
        if is_last:
            self.save()

    def create_reference_dataset(self, dataset_id='reference', Nstd=3, overwrite=False):
        if self.endpoint_data is None:
            self.load()
        # if not os.path.exists(RefFolder):
        #     os.makedirs(RefFolder)
        path_dir = f'{RefFolder}/{dataset_id}'
        path_data = f'{path_dir}/data/reference.csv'
        path_fits = f'{path_dir}/data/bout_fits.csv'
        if not os.path.exists(path_dir) or overwrite:
            copy_tree(self.dir, path_dir)
        new_d = LarvaDataset(path_dir)
        new_d.set_id(dataset_id)
        pars = ['length', nam.freq(nam.scal(nam.vel(''))),
                'stride_reoccurence_rate',
                nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))),
                nam.std(nam.scal(nam.chunk_track('stride', nam.dst(''))))]
        sample_pars = ['body.initial_length', 'brain.crawler_params.initial_freq',
                       'brain.intermitter_params.crawler_reoccurence_rate',
                       'brain.crawler_params.step_to_length_mu',
                       'brain.crawler_params.step_to_length_std'
                       ]

        v = new_d.endpoint_data[pars]
        v['length'] = v['length'] / 1000
        df = pd.DataFrame(v.values, columns=sample_pars)
        df.to_csv(path_data)

        fit_bouts(new_d, store=True, bouts=['stride', 'pause'])

        dic = {
            nam.freq('crawl'): v[nam.freq(nam.scal(nam.vel('')))].mean(),
            nam.freq('feed'): v[nam.freq('feed')].mean() if nam.freq('feed') in v.columns else 2.0,
            'feeder_reoccurence_rate': None,
            'dt': self.dt,
        }
        saveConf(dic, conf_type='Ref', id=dataset_id, mode='update')
        z = get_EEB_poly1d(dataset_id)
        saveConf({'EEB_poly1d': z.c.tolist()}, conf_type='Ref', id=dataset_id, mode='update')

        print(f'Reference dataset {dataset_id} saved.')

    def drop_immobile_larvae(self, vel_threshold=0.1, is_last=True):
        # self.compute_spatial_metrics(mode='minimal')
        D = self.step_data[nam.scal('velocity')]
        immobile_ids = []
        for id in self.agent_ids:
            d = D.xs(id, level='AgentIDs').dropna().values
            if len(d[d > vel_threshold]) == 0:
                immobile_ids.append(id)
        # e = self.end
        # dsts = e[nam.cum(nam.dst(self.point))]
        # immobile_ids = dsts[dsts < min_dst].index.values
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
