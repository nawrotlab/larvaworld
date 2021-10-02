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
from lib.envs._larvaworld_replay import LarvaWorldReplay


class LarvaDataset:
    def __init__(self, dir, id='unnamed', fr=16, Npoints=3, Ncontour=0,
                 par_conf=SimParConf, save_data_flag=True, load_data=True,env_params={},
                 sample_dataset='reference', group_id='SimGroup'):
        self.par_config = par_conf
        self.save_data_flag = save_data_flag
        self.define_paths(dir)
        if os.path.exists(self.dir_dict['conf']):
            self.config = fun.load_dict(self.dir_dict['conf'], use_pickle=False)
        else:
            self.config = {'id': id,
                           'group_id': group_id,
                           'dir': dir,
                           'fr': fr,
                           'dt': 1/fr,
                           'Npoints': Npoints,
                           'Ncontour': Ncontour,
                           'sample_dataset': sample_dataset,
                           **par_conf,
                           # 'arena_pars': arena_pars,
                           'env_params': env_params,
                           # **life_params
                           }

        self.__dict__.update(self.config)
        # print(self.config.keys())
        # self.dt = 1 / self.fr
        self.configure_body()
        self.define_linear_metrics()
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
        try:
            fs = [f'{self.aux_dir}/{f}' for f in os.listdir(self.aux_dir)]
            ns = fun.flatten_list([[f'{f}/{n}' for n in os.listdir(f) if n.endswith('.csv')] for f in fs])
            for n in ns:
                try:
                    df = pd.read_csv(n, index_col=0)
                    df.loc[~df.index.isin(agents)].to_csv(n, index=True, header=True)
                except:
                    pass
        except:
            pass
        if is_last:
            self.save()
        if show_output:
            print(f'{len(agents)} agents dropped and {len(self.endpoint_data.index)} remaining.')

    def drop_pars(self, pars=[], groups=None, is_last=True, show_output=True, **kwargs):
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
                pars += [nam.start(c), nam.stop(c), nam.id(c), nam.dur(c), nam.length(c), nam.dst(c), nam.straight_dst(c), nam.scal(nam.dst(c)), nam.scal(nam.straight_dst(c)), nam.orient(c)]
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

    def read(self, key='end'):
        return pd.read_hdf(self.dir_dict['data_h5'], key)

    def load(self, step=True, end=True, food=False):
        store = pd.HDFStore(self.dir_dict['data_h5'])
        if step:
            self.step_data = store['step']
            # print(self.step_data)
            self.step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
        if end:
            self.endpoint_data = store['end']
            self.endpoint_data.sort_index(inplace=True)
        if food:
            self.food_endpoint_data = store['food']
            self.food_endpoint_data.sort_index(inplace=True)
        store.close()

    @property
    def N(self):
        try:
            return len(self.agent_ids)
        except:
            return len(self.endpoint_data.index.values)

    @property
    def t0(self):
        try:
            return int(self.step_data.index.unique('Step')[0])
        except:
            return 0

    def save(self, step=True, end=True, food=False, add_reference=False):
        if self.save_data_flag == True:
            store = pd.HDFStore(self.dir_dict['data_h5'])
            if step:
                store['step'] = self.step_data
            if end:
                store['end'] = self.endpoint_data
            if food:
                store['food'] = self.food_endpoint_data
            self.save_config(add_reference=add_reference)
            store.close()
            print(f'Dataset {self.id} stored.')

    def save_dicts(self, env):
        for l in env.get_flies():
            if hasattr(l, 'deb') and l.deb is not None:
                l.deb.finalize_dict(self.dir_dict['deb'])
            if l.brain.intermitter is not None:
                l.brain.intermitter.save_dict(self.dir_dict['bout_dicts'])

    def save_tables(self, tables):
        store = pd.HDFStore(self.dir_dict['tables_h5'])
        for name, table in tables.items():
            df = pd.DataFrame(table)
            if 'unique_id' in df.columns:
                df.rename(columns={'unique_id': 'AgentID'}, inplace=True)
                N = len(df['AgentID'].unique().tolist())
                if N > 0:
                    Nrows = int(len(df.index) / N)
                    df['Step'] = np.array([[i] * N for i in range(Nrows)]).flatten()
                    df.set_index(['Step', 'AgentID'], inplace=True)
                    df.sort_index(level=['Step', 'AgentID'], inplace=True)
                store[name] = df
        store.close()

    def save_ExpFitter(self, dic=None):
        fun.save_dict(dic, self.dir_dict['ExpFitter'], use_pickle=False)

    def load_ExpFitter(self):
        try:
            dic=fun.load_dict(self.dir_dict['ExpFitter'], use_pickle=False)
            return dic
        except:
            return None

    def save_config(self, add_reference=False):
        # print(self.config['ExpFitter'])

        for a in ['N', 't0', 'duration', 'quality']:
            try:
                self.config[a] = getattr(self, a)
            except:
                pass
        self.config['dt']=1/self.fr
        for k, v in self.config.items():
            if type(v) == np.ndarray:
                self.config[k] = v.tolist()
        fun.save_dict(self.config, self.dir_dict['conf'], use_pickle=False)
        if add_reference :
            from lib.conf.conf import saveConf
            saveConf(self.config, 'Ref', f'{self.group_id}.{self.id}')

    def save_agents(self, ids=None, pars=None, header=True):
        if self.step_data is None:
            self.load()
        if ids is None :
            ids=self.agent_ids
        if pars is None :
            pars=self.step_data.columns
        for id in ids:
            f = os.path.join(self.dir_dict['single_tracks'], f'{id}.csv')
            store = pd.HDFStore(f)
            store['step'] = self.step_data[pars].loc[(slice(None), id), :]
            # store['step'] = self.step_data[pars].xs(id, level='AgentID', drop_level=True)
            store['end'] =self.endpoint_data.loc[[id]]
            store.close()
        print('All agent data saved.')

    def load_agent(self, id):
        try:
            f = os.path.join(self.dir_dict['single_tracks'], f'{id}.csv')
            store = pd.HDFStore(f)
            s = store['step']
            e = store['end']
            store.close()
            return s, e
        except :
            return None, None

    def load_bout_dicts(self, ids=None):
        if ids is None:
            ids = self.agent_ids
        dir = self.dir_dict['bout_dicts']
        d = {}
        for id in ids:
            file = f'{dir}/{id}.txt'
            dic = fun.load_dicts([file])[0]
            df = pd.DataFrame.from_dict(dic)
            df.index.set_names(0, inplace=True)
            d[id] = df

        return d

    def load_table(self, name):
        store = pd.HDFStore(self.dir_dict['tables_h5'])
        df = store[name]
        store.close()
        return df

    def load_aux(self, type, pars=None):
        # if type in ['distro', 'dispersion', 'stride'] :
        store = pd.HDFStore(self.dir_dict['aux_h5'])
        df = store[f'{type}.{pars}']
        store.close()
        return df

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

    def get_pars_list(self, p0, s0, draw_Nsegs):
        if p0 is None:
            p0 = self.point
        elif type(p0) == int:
            p0 = 'centroid' if p0 == -1 else self.points[p0]
        dic={}
        dic['pos_p']=pos_p = nam.xy(p0) if set(nam.xy(p0)).issubset(s0.columns) else ['x', 'y']
        dic['mid_p']=mid_p = [xy for xy in self.points_xy if set(xy).issubset(s0.columns)]
        dic['cen_p']=cen_p = self.cent_xy if set(self.cent_xy).issubset(s0.columns) else []
        dic['con_p']=con_p = [xy for xy in self.contour_xy if set(xy).issubset(s0.columns)]
        dic['chunk_p'] = chunk_p = [p for p in ['stride_stop', 'stride_id', 'pause_id', 'feed_id'] if p in s0.columns]
        if draw_Nsegs is None :
            dic['ang_p'] = ang_p = []
            dic['ors_p'] = ors_p = []
        elif draw_Nsegs==2 and {'bend', 'front_orientation'}.issubset(s0.columns) :
            dic['ang_p']=ang_p = ['bend']
            dic['ors_p'] = ors_p=['front_orientation']
        elif draw_Nsegs==len(mid_p)-1 and set(nam.orient(self.segs)).issubset(s0.columns) :
            dic['ang_p'] = ang_p = []
            dic['ors_p']=ors_p = nam.orient(self.segs)
        else :
            raise ValueError (f'The required angular parameters for reconstructing a {draw_Nsegs}-segment body do not exist')

        pars = fun.unique_list(cen_p + pos_p+ ang_p + ors_p + chunk_p + fun.flatten_list(mid_p  + con_p))

        return dic, pars, p0

    def get_smaller_dataset(self, s0, e0, ids=None, pars=None, time_range=None, dynamic_color=None):
        if ids is None:
            ids = e0.index.values.tolist()
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [self.agent_ids[i] for i in ids]
        if dynamic_color is not None and dynamic_color in s0.columns:
            pars.append(dynamic_color)
            # traj_color =s0[dynamic_color]
        else :
            dynamic_color=None
        pars = [p for p in pars if p in s0.columns]
        print(ids)
        if time_range is None:
            s = copy.deepcopy(s0.loc[(slice(None), ids), pars])
        else:
            a, b = time_range
            a = int(a / self.dt)
            b = int(b / self.dt)
            # tick_range=(np.array(time_range)/self.dt).astype(int)
            s = copy.deepcopy(s0.loc[(slice(a, b), ids), pars])
        e = copy.deepcopy(e0.loc[ids])
        traj_color = s[dynamic_color] if dynamic_color is not None else None
        return s, e, ids, traj_color

    def visualize(self, s0=None, e0=None, vis_kwargs=None, agent_ids=None, save_to=None, time_range=None,draw_Nsegs=None,env_params=None,
                  track_point=None, dynamic_color=None,use_background=False,
                  transposition=None, fix_point=None, secondary_fix_point=None, **kwargs):
        if vis_kwargs is None:
            vis_kwargs = dtypes.get_dict('visualization', mode='video')
        if s0 is None and e0 is None :
            if self.step_data is None:
                self.load()
            s0, e0=self.step_data, self.endpoint_data
        dic, pars, track_point = self.get_pars_list(track_point, s0, draw_Nsegs)
        s, e, ids, traj_color = self.get_smaller_dataset(ids=agent_ids, pars=pars, time_range=time_range,
                                         dynamic_color=dynamic_color, s0=s0, e0=e0)
        if len(ids) == 1:
            n0 = ids[0]
        elif len(ids) == len(self.agent_ids):
            n0 = 'all'
        else:
            n0 = f'{len(ids)}l'

        if env_params is None:
            env_params = self.env_params
        arena_dims = env_params['arena']['arena_dims']

        if transposition is not None:
            s = align_trajectories(s, track_point=track_point, arena_dims=arena_dims, mode=transposition,
                                   config=self.config)
            bg = None
            n1 = 'transposed'
        elif fix_point is not None:
            s, bg = fixate_larva(s, point=fix_point, secondary_point=secondary_fix_point,
                                 arena_dims=arena_dims, config=self.config)
            n1 = 'fixed'
        else:
            bg = None
            n1 = 'normal'

        replay_id = f'{n0}_{n1}'
        if vis_kwargs['render']['media_name'] is None:
            vis_kwargs['render']['media_name'] = replay_id
        if save_to is None:
            save_to = self.vis_dir

        base_kws = {
            'vis_kwargs': vis_kwargs,
            'env_params': env_params,
            'id': replay_id,
            'dt': self.dt,
            'Nsteps': len(s.index.unique('Step').values),
            'save_to': save_to,
            'background_motion': bg,
            'use_background': True if bg is not None else False,
            'Box2D': False,
            'traj_color': traj_color,
            # 'space_in_mm': space_in_mm
        }

        replay_env = LarvaWorldReplay(step_data=s, endpoint_data=e, config=self.config,
                                      **dic, draw_Nsegs=draw_Nsegs,
                                      **base_kws, **kwargs)

        replay_env.run()
        print('Visualization complete')

    def visualize_single(self, id, close_view=True, fix_point=-1, secondary_fix_point=None, save_to=None,
                         draw_Nsegs=None, vis_kwargs=None, **kwargs):
        try:
            s0, e0 = self.load_agent(id)
        except :
            self.save_agents(ids=[id])
            s0, e0 = self.load_agent(id)
        if close_view :
            from lib.conf.init_dtypes import null_dict
            env_params = {'arena': null_dict('arena', arena_dims=(0.01, 0.01))}
        else :
            env_params=self.env_params
        dic, pars, track_point = self.get_pars_list(fix_point, s0, draw_Nsegs)
        s, bg = fixate_larva(s0, point=fix_point, secondary_point=secondary_fix_point,
                             arena_dims=env_params['arena']['arena_dims'], config=self.config)
        if save_to is None:
            save_to = self.vis_dir
        replay_id=f'{id}_fixed_at_{fix_point}'
        if vis_kwargs is None:
            vis_kwargs = dtypes.get_dict('visualization', mode='video', video_speed=60, media_name=replay_id)
        base_kws = {
            'vis_kwargs': vis_kwargs,
            'env_params': env_params,
            'id': replay_id,
            'dt': self.dt,
            'Nsteps': len(s.index.unique('Step').values),
            'save_to': save_to,
            'background_motion': bg,
            'use_background': True if bg is not None else False,
            'Box2D': False,
            'traj_color': None,
            # 'space_in_mm': space_in_mm
        }
        replay_env = LarvaWorldReplay(step_data=s, endpoint_data=e0, config=self.config,
                                      **dic,draw_Nsegs=draw_Nsegs,
                                      **base_kws, **kwargs)

        replay_env.run()
        print('Visualization complete')




    def compute_preference_index(self, return_num=False, return_all=False):
        if not hasattr(self, 'endpoint_data'):
            self.load(step=False)
        e = self.endpoint_data
        r = 0.2 * self.env_params['arena']['arena_dims'][0]
        p = 'x' if 'x' in e.keys() else nam.final('x')
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

        try:
            self.config['point'] = self.points[self.config['point_idx'] - 1]
        except:
            self.config['point'] = 'centroid'
        self.point = self.config['point']

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
            'single_tracks': os.path.join(self.data_dir, 'single_tracks'),
            'bout_dicts': os.path.join(self.data_dir, 'bout_dicts'),
            'tables_h5': os.path.join(self.data_dir, 'tables.h5'),
            'sim': os.path.join(self.data_dir, 'sim_conf.txt'),
            'ExpFitter': os.path.join(self.data_dir, 'ExpFitter.txt'),
            'fit': os.path.join(self.data_dir, 'dataset_fit.csv'),
            'conf': os.path.join(self.data_dir, 'dataset_conf.csv'),
            'data_h5': os.path.join(self.data_dir, 'data.h5'),
            'aux_h5': os.path.join(self.data_dir, 'aux.h5'),
        }
        for k, v in self.dir_dict.items():
            if not str.endswith(v, 'csv') and not str.endswith(v, 'txt') and not str.endswith(v, 'h5'):
                os.makedirs(v, exist_ok=True)

    def define_linear_metrics(self):
        self.distance = nam.dst(self.point)
        self.velocity = nam.vel(self.point)
        self.acceleration = nam.acc(self.point)
        if self.config['use_component_vel']:
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

    def enrich(self, preprocessing={}, processing={}, annotation={}, enrich_aux={},
               to_drop={}, show_output=False, is_last=True, **kwargs):
        print()
        print(f'--- Enriching dataset {self.id} with derived parameters ---')
        # self.config['front_body_ratio'] = 0.5
        # self.save_config()
        warnings.filterwarnings('ignore')
        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'config': self.config,
            'show_output': show_output,
             'is_last': False
        }
        preprocess(**preprocessing, **c, **enrich_aux, **kwargs)
        process(**processing, **enrich_aux, **c, **kwargs)
        annotate(**annotation, **enrich_aux, **c, **kwargs)
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
            p_df = self.load_aux(type='distro', pars=par)
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

    def delete(self, show_output=True):
        shutil.rmtree(self.dir)
        if show_output:
            print(f'Dataset {self.id} deleted')

    def set_id(self, id, save=True):
        self.id = id
        self.config['id'] = id
        if save:
            self.save_config()

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
            s,e=self.step_data, self.endpoint_data
            new_ds = []
            for f, new_dir in zip(groups, new_dirs):

                valid_ids = [id for id in self.agent_ids if str.startswith(id, f)]
                # copy_tree(self.dir, new_dir)
                new_d = LarvaDataset(new_dir, id=f, load_data=False)
                new_d.set_data(step=s.loc[(slice(None), valid_ids),:], end=e.loc[valid_ids])
                # new_d.drop_agents(invalid_ids, is_last=is_last, show_output=show_output)
                # new_d.set_id(f)
                new_ds.append(new_d)
            if show_output:
                print(f'Dataset {self.id} splitted in {[d.id for d in new_ds]}')
        return new_ds

    # def split_dataset(self, groups=None, is_last=True, show_output=True):
    #     if groups is None:
    #         groups = fun.unique_list([id.split('_')[0] for id in self.agent_ids])
    #     if len(groups) == 1:
    #         return [self]
    #     new_dirs = [f'{self.dir}/../{self.id}.{f}' for f in groups]
    #     if all([os.path.exists(new_dir) for new_dir in new_dirs]):
    #         new_ds = [LarvaDataset(new_dir) for new_dir in new_dirs]
    #     else:
    #         if self.step_data is None:
    #             self.load()
    #         new_ds = []
    #         for f, new_dir in zip(groups, new_dirs):
    #
    #             invalid_ids = [id for id in self.agent_ids if not str.startswith(id, f)]
    #             copy_tree(self.dir, new_dir)
    #             new_d = LarvaDataset(new_dir)
    #             new_d.drop_agents(invalid_ids, is_last=is_last, show_output=show_output)
    #             new_d.set_id(f)
    #             new_ds.append(new_d)
    #         if show_output:
    #             print(f'Dataset {self.id} splitted in {[d.id for d in new_ds]}')
    #     return new_ds
