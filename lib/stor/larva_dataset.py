import os.path
import shutil
from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
import warnings
import copy

import lib.aux.dictsNlists as dNl
import lib.aux.naming as nam

from lib.conf.base.dtypes import null_dict


class LarvaDataset:
    def __init__(self, dir, load_data=True, **kwargs):
        self.define_paths(dir)
        self.retrieve_conf(**kwargs)
        self.larva_tables = {}
        self.larva_dicts = {}
        self.configure_body()
        self.define_linear_metrics()
        if load_data:
            try:
                self.load()
            except:
                print('Data not found. Load them manually.')

    def retrieve_conf(self, id='unnamed', fr=16, Npoints=None, Ncontour=0, spatial_def=None, env_params={},
                      larva_groups={},
                      source_xy={}, **kwargs):
        # try:
        if os.path.exists(self.dir_dict.conf):
            config = dNl.load_dict(self.dir_dict.conf, use_pickle=False)

        # except:
        else:
            if spatial_def is None:
                from lib.conf.stored.conf import loadConf
                spatial_def = loadConf('SimParConf', 'Par')['spatial']
            group_ids = list(larva_groups.keys())
            samples = dNl.unique_list([larva_groups[k]['sample'] for k in group_ids])
            if len(group_ids) == 1:
                group_id = group_ids[0]
                color = larva_groups[group_id]['default_color']
                sample = larva_groups[group_id]['sample']
                life_history = larva_groups[group_id]['life_history']
            else:
                group_id = None
                color = None
                sample = samples[0] if len(samples) == 1 else None
                life_history = None

            if Npoints is None:
                try:
                    # FIXME This only takes the first configuration into account
                    Npoints = list(larva_groups.values())[0]['model']['body']['Nsegs'] + 1
                except:
                    Npoints = 3

            config = {'id': id,
                      'group_id': group_id,
                      'group_ids': group_ids,
                      'refID': None,
                      'dir': self.dir,
                      'dir_dict': self.dir_dict,
                      'aux_dir': self.dir_dict.aux_h5,
                      'parent_plot_dir': f'{self.dir}/plots',
                      'fr': fr,
                      'dt': 1 / fr,
                      'Npoints': Npoints,
                      'Ncontour': Ncontour,
                      'sample': sample,
                      'color': color,
                      **spatial_def,
                      'env_params': env_params,
                      'larva_groups': larva_groups,
                      'source_xy': source_xy,
                      'life_history': life_history
                      }
        self.config = dNl.AttrDict.from_nested_dicts(config)

        self.__dict__.update(self.config)

    def set_data(self, step=None, end=None, food=None):
        if step is not None:
            step.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.step_data = step
            self.agent_ids = step.index.unique('AgentID').values
            self.num_ticks = step.index.unique('Step').size
        if end is not None:
            self.endpoint_data = end
        if food is not None:
            self.food_endpoint_data = food
        self.update_config()

    def drop_pars(self, groups=None, is_last=True, show_output=True, **kwargs):
        if groups is None:
            groups = {n: False for n in
                      ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn',
                       'unused']}
        if self.step_data is None:
            self.load()
        s = self.step_data
        pars = []
        if groups['midline']:
            pars += dNl.flatten_list(self.points_xy)
        if groups['contour']:
            pars += dNl.flatten_list(self.contour_xy)
        for c in ['stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn', 'turn']:
            if groups[c]:
                pars += [nam.start(c), nam.stop(c), nam.id(c), nam.dur(c), nam.length(c), nam.dst(c),
                         nam.straight_dst(c), nam.scal(nam.dst(c)), nam.scal(nam.straight_dst(c)), nam.orient(c)]
        if groups['unused']:
            pars += self.get_unused_pars()
        pars = dNl.unique_list(pars)
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

    def read(self, key='end', file='data_h5'):
        return pd.read_hdf(self.dir_dict[file], key)

    def load(self, step=True, end=True, food=False, contour=True):
        store = pd.HDFStore(self.dir_dict.data_h5)
        if step:
            self.step_data = store['step']
            if contour:
                try:
                    contour_ps = dNl.flatten_list(self.contour_xy)
                    temp = pd.HDFStore(self.dir_dict.contour_h5)
                    self.step_data[contour_ps] = temp['contour']
                    temp.close()
                except:
                    pass
            self.step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
        if end:
            try:
                endpoint = pd.HDFStore(self.dir_dict.endpoint_h5)
                self.endpoint_data = endpoint['end']
                self.endpoint_data.sort_index(inplace=True)
                endpoint.close()
            except:
                self.endpoint_data = store['end']
        if food:
            self.food_endpoint_data = store['food']
            self.food_endpoint_data.sort_index(inplace=True)
        store.close()

    def save(self, step=True, end=True, food=False, contour=True, add_reference=False):
        store = pd.HDFStore(self.dir_dict.data_h5)
        if step:
            contour_ps = dNl.flatten_list(self.contour_xy)
            if contour:
                temp = pd.HDFStore(self.dir_dict.contour_h5)
                temp['contour'] = self.step_data[contour_ps]
                temp.close()

            store['step'] = self.step_data.drop(contour_ps, axis=1, errors='ignore')
        if end:
            endpoint = pd.HDFStore(self.dir_dict.endpoint_h5)
            endpoint['end'] = self.endpoint_data
            endpoint.close()
        if food:
            store['food'] = self.food_endpoint_data
        self.save_config(add_reference=add_reference)
        store.close()
        print(f'Dataset {self.id} stored.')

    def save_larva_dicts(self):
        for k, vs in self.larva_dicts.items():
            os.makedirs(self.dir_dict[k], exist_ok=True)
            for id, dic in vs.items():
                try:
                    dNl.save_dict(dic, f'{self.dir_dict[k]}/{id}.txt', use_pickle=False)
                except:
                    dNl.save_dict(dic, f'{self.dir_dict[k]}/{id}.txt', use_pickle=True)

    def get_larva_dicts(self, env):
        from lib.model.modules.nengobrain import NengoBrain
        deb_dicts = {}
        nengo_dicts = {}
        bout_dicts = {}
        foraging_dicts = {}
        for l in env.get_flies():
            if l.unique_id in self.agent_ids:
                if hasattr(l, 'deb') and l.deb is not None:
                    deb_dicts[l.unique_id] = l.deb.finalize_dict()
                elif isinstance(l.brain, NengoBrain):
                    if l.brain.dict is not None:
                        nengo_dicts[l.unique_id] = l.brain.dict
                if l.brain.locomotor.intermitter is not None:
                    bout_dicts[l.unique_id] = l.brain.locomotor.intermitter.build_dict()
                if len(env.foodtypes) > 0:
                    foraging_dicts[l.unique_id] = l.finalize_foraging_dict()
                self.config.foodtypes = env.foodtypes
        self.larva_dicts = {'deb': deb_dicts, 'nengo': nengo_dicts, 'bout_dicts': bout_dicts,
                            'foraging': foraging_dicts}

    def get_larva_tables(self, env):
        if env.table_collector is not None:
            for name, table in env.table_collector.tables.items():
                df = pd.DataFrame(table)
                if 'unique_id' in df.columns:
                    df.rename(columns={'unique_id': 'AgentID'}, inplace=True)
                    N = len(df['AgentID'].unique().tolist())
                    if N > 0:
                        Nrows = int(len(df.index) / N)
                        df['Step'] = np.array([[i] * N for i in range(Nrows)]).flatten()
                        df.set_index(['Step', 'AgentID'], inplace=True)
                        df.sort_index(level=['Step', 'AgentID'], inplace=True)
                        self.larva_tables[name] = df

    def save_larva_tables(self):
        store = pd.HDFStore(self.dir_dict.tables_h5)
        for name, df in self.larva_tables.items():
            store[name] = df
        store.close()

    def save_ExpFitter(self, dic=None):
        dNl.save_dict(dic, self.dir_dict.ExpFitter, use_pickle=False)

    def load_ExpFitter(self):
        try:
            dic = dNl.load_dict(self.dir_dict.ExpFitter, use_pickle=False)
            return dic
        except:
            return None

    def update_config(self):
        self.config.dt = 1 / self.fr
        self.config.dir_dict = self.dir_dict
        if 'agent_ids' not in self.config.keys():
            try:
                ids = self.agent_ids
            except:
                try:
                    ids = self.endpoint_data.index.values
                except:
                    ids = self.read('end').index.values
            self.config.agent_ids = ids
            self.config.N = len(ids)
        if 't0' not in self.config.keys():
            try:
                self.config.t0 = int(self.step_data.index.unique('Step')[0])
            except:
                self.config.t0 = 0
        if 'Nticks' not in self.config.keys():
            try:
                self.config.Nticks = self.step_data.index.unique('Step').size
            except:
                try:
                    self.config.Nticks = self.endpoint_data['num_ticks'].max()
                except:
                    pass
        if 'duration' not in self.config.keys():
            try:
                self.config.duration = int(self.endpoint_data['cum_dur'].max())
            except:
                self.config.duration = self.config.dt * self.config.Nticks
        if 'quality' not in self.config.keys():
            try:
                df = self.step_data[nam.xy(self.point)[0]].values.flatten()
                valid = np.count_nonzero(~np.isnan(df))
                self.config.duration = np.round(valid / df.shape[0], 2)
            except:
                pass

    def save_config(self, add_reference=False, refID=None):
        self.update_config()
        for k, v in self.config.items():
            if type(v) == np.ndarray:
                self.config[k] = v.tolist()
        dNl.save_dict(self.config, self.dir_dict.conf, use_pickle=False)
        if add_reference:
            from lib.conf.stored.conf import saveConf
            if refID is None:
                refID = f'{self.group_id}.{self.id}'
            self.config.refID = refID
            saveConf(self.config, 'Ref', refID)

    def save_agents(self, ids=None, pars=None):
        if not hasattr(self, 'step_data'):
            self.load()
        if ids is None:
            ids = self.agent_ids
        if pars is None:
            pars = self.step_data.columns
        path = self.dir_dict.single_tracks
        os.makedirs(path, exist_ok=True)
        for id in ids:
            f = os.path.join(path, f'{id}.csv')
            store = pd.HDFStore(f)
            store['step'] = self.step_data[pars].loc[(slice(None), id), :]
            store['end'] = self.endpoint_data.loc[[id]]
            store.close()
        print('All agent data saved.')

    def load_agent(self, id):
        try:
            f = os.path.join(self.dir_dict.single_tracks, f'{id}.csv')
            store = pd.HDFStore(f)
            s = store['step']
            e = store['end']
            store.close()
            return s, e
        except:
            return None, None

    def load_table(self, name):
        store = pd.HDFStore(self.dir_dict.tables_h5)
        df = store[name]
        store.close()
        return df

    def load_aux(self, type, par=None):
        # print(pd.HDFStore(self.dir_dict['aux_h5']).keys())
        df = self.read(key=f'{type}.{par}', file='aux_h5')
        return df

    def load_dicts(self, type, ids=None):
        if ids is None:
            ids = self.agent_ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type].keys() for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            files = [f'{id}.txt' for id in ids]
            try:
                ds = dNl.load_dicts(files=files, folder=self.dir_dict[type], use_pickle=False)
            except:
                ds = dNl.load_dicts(files=files, folder=self.dir_dict[type], use_pickle=True)
        return ds

    def get_pars_list(self, p0, s0, draw_Nsegs):
        if p0 is None:
            p0 = self.point
        elif type(p0) == int:
            p0 = 'centroid' if p0 == -1 else self.points[p0]
        dic = {}
        dic['pos_p'] = pos_p = nam.xy(p0) if set(nam.xy(p0)).issubset(s0.columns) else ['x', 'y']
        dic['mid_p'] = mid_p = [xy for xy in self.points_xy if set(xy).issubset(s0.columns)]
        dic['cen_p'] = cen_p = nam.xy('centroid') if set(nam.xy('centroid')).issubset(s0.columns) else []
        dic['con_p'] = con_p = [xy for xy in self.contour_xy if set(xy).issubset(s0.columns)]
        dic['chunk_p'] = chunk_p = [p for p in ['stride_stop', 'stride_id', 'pause_id', 'feed_id'] if p in s0.columns]
        if draw_Nsegs is None:
            dic['ang_p'] = ang_p = []
            dic['ors_p'] = ors_p = []
        elif draw_Nsegs == 2 and {'bend', 'front_orientation'}.issubset(s0.columns):
            dic['ang_p'] = ang_p = ['bend']
            dic['ors_p'] = ors_p = ['front_orientation']
        elif draw_Nsegs == len(mid_p) - 1 and set(nam.orient(self.segs)).issubset(s0.columns):
            dic['ang_p'] = ang_p = []
            dic['ors_p'] = ors_p = nam.orient(self.segs)
        else:
            raise ValueError(
                f'The required angular parameters for reconstructing a {draw_Nsegs}-segment body do not exist')

        pars = dNl.unique_list(cen_p + pos_p + ang_p + ors_p + chunk_p + dNl.flatten_list(mid_p + con_p))

        return dic, pars, p0

    def get_smaller_dataset(self, s0, e0, ids=None, pars=None, time_range=None, dynamic_color=None):
        if ids is None:
            ids = e0.index.values.tolist()
        if type(ids) == list and all([type(i) == int for i in ids]):
            ids = [self.agent_ids[i] for i in ids]
        if dynamic_color is not None and dynamic_color in s0.columns:
            pars.append(dynamic_color)
        else:
            dynamic_color = None
        pars = [p for p in pars if p in s0.columns]
        if time_range is None:
            s = copy.deepcopy(s0.loc[(slice(None), ids), pars])
        else:
            a, b = time_range
            a = int(a / self.dt)
            b = int(b / self.dt)
            s = copy.deepcopy(s0.loc[(slice(a, b), ids), pars])
        e = copy.deepcopy(e0.loc[ids])
        traj_color = s[dynamic_color] if dynamic_color is not None else None
        return s, e, ids, traj_color

    def visualize(self, s0=None, e0=None, vis_kwargs=None, agent_ids=None, save_to=None, time_range=None,
                  draw_Nsegs=None, env_params=None, track_point=None, dynamic_color=None, use_background=False,
                  transposition=None, fix_point=None, fix_segment=None, **kwargs):
        from lib.model.envs._larvaworld_replay import LarvaWorldReplay
        if vis_kwargs is None:
            vis_kwargs = null_dict('visualization', mode='video')
        if s0 is None and e0 is None:
            if not hasattr(self, 'step_data'):
                self.load()
            s0, e0 = self.step_data, self.endpoint_data
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
            from lib.process.spatial import align_trajectories
            s = align_trajectories(s, track_point=track_point, arena_dims=arena_dims, mode=transposition,
                                   config=self.config)
            bg = None
            n1 = 'transposed'
        elif fix_point is not None:
            from lib.process.spatial import fixate_larva
            s, bg = fixate_larva(s, point=fix_point, fix_segment=fix_segment, arena_dims=arena_dims, config=self.config)
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
        replay_env = LarvaWorldReplay(step_data=s, endpoint_data=e, config=self.config, draw_Nsegs=draw_Nsegs,
                                      **dic, **base_kws, **kwargs)

        replay_env.run()
        print('Visualization complete')

    def visualize_single(self, id=0, close_view=True, fix_point=-1, fix_segment=None, save_to=None,
                         draw_Nsegs=None, vis_kwargs=None, **kwargs):
        from lib.model.envs._larvaworld_replay import LarvaWorldReplay
        from lib.process.spatial import fixate_larva
        if type(id) == int:
            id = self.config.agent_ids[id]
        s0, e0 = self.load_agent(id)
        if s0 is None:
            self.save_agents(ids=[id])
            s0, e0 = self.load_agent(id)
        if close_view:
            from lib.conf.base.dtypes import null_dict
            env_params = {'arena': null_dict('arena', arena_dims=(0.01, 0.01))}
        else:
            env_params = self.env_params
        dic, pars, track_point = self.get_pars_list(fix_point, s0, draw_Nsegs)
        s, bg = fixate_larva(s0, point=fix_point, fix_segment=fix_segment,
                             arena_dims=env_params['arena']['arena_dims'], config=self.config)
        if save_to is None:
            save_to = self.vis_dir
        replay_id = f'{id}_fixed_at_{fix_point}'
        if vis_kwargs is None:
            from lib.conf.base.dtypes import null_dict
            vis_kwargs = null_dict('visualization', mode='video', video_speed=60, media_name=replay_id)
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
        replay_env = LarvaWorldReplay(step_data=s, endpoint_data=e0, config=self.config, draw_Nsegs=draw_Nsegs,
                                      **dic, **base_kws, **kwargs)

        replay_env.run()
        print('Visualization complete')

    def configure_body(self):
        N, Nc = self.Npoints, self.Ncontour
        self.points = nam.midline(N, type='point')

        self.Nangles = np.clip(N - 2, a_min=0, a_max=None)
        self.angles = [f'angle{i}' for i in range(self.Nangles)]
        self.Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        self.segs = nam.midline(self.Nsegs, type='seg')

        self.points_xy = nam.xy(self.points)
        # self.points_dst = nam.dst(self.points)
        # self.points_vel = nam.vel(self.points)
        # self.points_acc = nam.acc(self.points)
        # self.point_lin_pars = self.points_dst + self.points_vel + self.points_acc

        # self.angles_vel = nam.vel(self.angles)
        # self.angles_acc = nam.acc(self.angles)
        # self.angle_pars = self.angles + self.angles_vel + self.angles_acc

        self.contour = nam.contour(Nc)
        self.contour_xy = nam.xy(self.contour)

        # self.cent_xy = nam.xy('centroid')
        # self.cent_dst = nam.dst('centroid')
        # self.cent_vel = nam.vel('centroid')
        # self.cent_acc = nam.acc('centroid')
        # self.cent_lin_pars = [self.cent_dst, self.cent_vel, self.cent_acc]

        ang = ['front_orientation', 'rear_orientation', 'bend']
        self.ang_pars = ang + nam.unwrap(ang) + nam.vel(ang) + nam.acc(ang)
        self.xy_pars = nam.xy(self.points + self.contour + ['centroid'], flat=True) + nam.xy('')

    def define_paths(self, dir):
        self.dir = dir
        self.data_dir = os.path.join(dir, 'data')
        self.plot_dir = os.path.join(dir, 'plots')
        self.vis_dir = os.path.join(dir, 'visuals')
        self.aux_dir = os.path.join(dir, 'aux')
        dir_dict = {
            'parent': self.dir,
            'data': self.data_dir,
            'plot': self.plot_dir,
            'vis': self.vis_dir,
            # 'comp_plot': os.path.join(self.plot_dir, 'comparative'),
            'foraging': os.path.join(self.data_dir, 'foraging_dicts'),
            'deb': os.path.join(self.data_dir, 'deb_dicts'),
            'nengo': os.path.join(self.data_dir, 'nengo_probes'),
            'single_tracks': os.path.join(self.data_dir, 'single_tracks'),
            'bout_dicts': os.path.join(self.data_dir, 'bout_dicts'),
            'group_bout_dicts': os.path.join(self.data_dir, 'group_bout_dicts'),
            'chunk_dicts': os.path.join(self.data_dir, 'chunk_dicts'),
            'tables_h5': os.path.join(self.data_dir, 'tables.h5'),
            'sim': os.path.join(self.data_dir, 'sim_conf.txt'),
            'ExpFitter': os.path.join(self.data_dir, 'ExpFitter.txt'),
            'fit': os.path.join(self.data_dir, 'dataset_fit.csv'),
            'conf': os.path.join(self.data_dir, 'dataset_conf.csv'),
            'data_h5': os.path.join(self.data_dir, 'data.h5'),
            'endpoint_h5': os.path.join(self.data_dir, 'endpoint.h5'),
            'derived_h5': os.path.join(self.data_dir, 'derived.h5'),
            'contour_h5': os.path.join(self.data_dir, 'contour.h5'),
            'aux_h5': os.path.join(self.data_dir, 'aux.h5'),
        }
        self.dir_dict = dNl.AttrDict.from_nested_dicts(dir_dict)
        for k in ['parent', 'data']:
            os.makedirs(self.dir_dict[k], exist_ok=True)


    def define_linear_metrics(self):
        try:
            self.config.point = self.points[self.config.point_idx - 1]
        except:
            self.config.point = 'centroid'
        self.point = self.config.point
        self.distance = nam.dst(self.point)
        self.velocity = nam.vel(self.point)
        self.acceleration = nam.acc(self.point)
        if self.config.use_component_vel:
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

    def enrich(self, metric_definition, preprocessing={}, processing={}, annotation={},
               to_drop={}, recompute=False, mode='minimal', show_output=True, is_last=True, **kwargs):
        md = metric_definition
        for k,v in md.items():
            if v is None :
                md[k]={}
        # print(md)
        self.config.update(**md['angular'])
        self.config.update(**md['spatial'])
        self.define_linear_metrics()
        from lib.process.basic import preprocess, process
        from lib.process.bouts import annotate
        print()
        print(f'--- Enriching dataset {self.id} with derived parameters ---')
        warnings.filterwarnings('ignore')

        c = {
            's': self.step_data,
            'e': self.endpoint_data,
            'config': self.config,
            'show_output': show_output,
            'recompute': recompute,
            'mode': mode,
            'is_last': False,
            # 'metric_definition' : metric_definition
        }
        preprocess(**preprocessing, **c, **kwargs)
        process(processing=processing, **c, **kwargs, **md['dispersion'], **md['tortuosity'])
        annotate(**annotation, **c, **kwargs, **md['stride'], **md['turn'], **md['pause'])
        self.drop_pars(**to_drop, **c)
        if is_last:
            self.save()
        return self

    def get_par(self, par, key=None):
        def get_end_par(par):
            try:
                return self.read(key='end')[par]
            except:
                try:
                    return self.endpoint_data[par]
                except:
                    return None

        def get_step_par(par):
            try:
                return self.read(key='step')[par]
            except:
                try:
                    return self.step_data[par]
                except:
                    return None

        if key == 'end':
            return get_end_par(par)
        elif key == 'step':
            return get_step_par(par)
        else:
            e = get_end_par(par)
            if e is not None:
                return e
            else:
                s = get_step_par(par)
                if s is not None:
                    return s
                else:
                    try:
                        return self.read(key=f'distro.{par}', file='aux_h5')
                    except:
                        return None

    def delete(self, show_output=True):
        shutil.rmtree(self.dir)
        if show_output:
            print(f'Dataset {self.id} deleted')

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()

    def load_group_bout_dict(self, id=None):
        if id is None:
            id = self.id
        path = os.path.join(self.dir_dict['group_bout_dicts'], f'{id}.txt')
        dic = dNl.load_dict(path, use_pickle=True)
        return dic

    def load_chunk_dicts(self, id=None):
        if id is None:
            id = self.id
        path = os.path.join(self.dir_dict.chunk_dicts, f'{id}.txt')
        return dNl.load_dict(path, use_pickle=True)
