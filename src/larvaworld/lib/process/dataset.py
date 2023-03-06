import json
import os
import shutil
import numpy as np
import warnings

from larvaworld.lib import reg, aux, decorators
from larvaworld.lib.aux import nam

class _LarvaDataset:
    def __init__(self, dir=None, load_data=True,config = None, **kwargs):

        if config is None :
            config = reg.load_config(dir)
            if config is None:
                config = generate_dataset_config(dir=dir, **kwargs)

        self.config = config
        self.retrieve_metrics()
        self.__dict__.update(self.config)
        self.larva_tables = {}
        self.larva_dicts = {}
        if load_data:
            try:
                self.load()
            except:
                print('Data not found. Load them manually.')


    def set_data(self, step=None, end=None, food=None):
        if step is not None:

            self.step_data = step.sort_index(level=['Step', 'AgentID'])
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
        if end is not None:

            self.endpoint_data = end.sort_index()
        if food is not None:
            self.food_endpoint_data = food
        self.update_config()


    def load_step(self, h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']):
        s = self.read(key='step')

        stored_ps = []
        for h5_k in h5_ks:
            ss = self.read(key=h5_k)
            if ss is not None :
                self.load_h5_kdic[h5_k] = "a"
                ps = [p for p in ss.columns.values if p not in s.columns.values]
                if len(ps) > 0:
                    stored_ps += ps
                    s = s.join(ss[ps])
            else:
                self.load_h5_kdic[h5_k] = "w"
        s.sort_index(level=['Step', 'AgentID'], inplace=True)
        self.agent_ids = s.index.unique('AgentID').values
        self.num_ticks = s.index.unique('Step').size
        return s

    @decorators.warn_slow
    def load(self, step=True, end=True, food=False, **kwargs):

        if step:
            self.step_data = self.load_step(**kwargs)

        if end:
            self.endpoint_data = self.read(key='end')
            self.endpoint_data.sort_index(inplace=True)

        if food:
            self.food_endpoint_data = self.read(key='food')
            self.food_endpoint_data.sort_index(inplace=True)



    def save_step(self, s=None, h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']):
        if s is None:
            s = self.step_data
        s = s.loc[:, ~s.columns.duplicated()]
        stored_ps = []
        for h5_k in h5_ks:

            pps = [p for p in self.h5_kdic[h5_k] if p in s.columns]
            pps=aux.unique_list(pps)
            if len(pps) > 0:
                self.storeH5(df=s[pps], key=h5_k, mode=self.load_h5_kdic[h5_k])

                stored_ps += pps

        ss = s.drop(stored_ps, axis=1, errors='ignore')
        self.storeH5(df=ss, key='step')

    @decorators.warn_slow
    def save(self, step=True, end=True, food=False, refID=None, **kwargs):

        if step:
            self.save_step(s=self.step_data, **kwargs)

        if end:
            self.storeH5(df=self.endpoint_data, key='end')
        if food:
            self.storeH5(df=self.food_endpoint_data, key='food')
        self.save_config(refID=refID)

        reg.vprint(f'***** Dataset {self.id} stored.-----', 1)

    def save_vel_definition(self):
        from larvaworld.lib.process.calibration import vel_definition
        warnings.filterwarnings('ignore')
        dic=vel_definition(self)
        self.save_config()
        self.storeH5(df=dic, filepath_key='vel_definition')
        print(f'Velocity definition dataset stored.')

        return dic

    def save_config(self, refID=None):
        self.update_config()
        if refID is not None:
            self.config.refID = refID
            reg.Ref_paths(id=refID, dir=self.config.dir)
        aux.save_dict(self.config,reg.datapath('conf', self.config.dir))





    def centralize_xy_tracks(self, replace=True, arena_dims=None, is_last=True):
        if arena_dims is None:
            arena_dims = self.config.env_params.arena.dims
        x0, y0 = arena_dims

        kws0 = {
            'h5_ks': ['contour', 'midline'],
            # 'end' : False,
            # 'step' : True
        }
        s = self.load_step(**kws0)
        xy_pairs=self.midline_xy + self.contour_xy + aux.nam.xy(['centroid', ''])
        xy_pairs = [xy for xy in xy_pairs if set(xy).issubset(s.columns)]

        for x, y in xy_pairs:
            s[x] -= x0 / 2
            s[y] -= y0 / 2
        if replace:
            self.step_data = s
        if is_last:
            self.save_step(s, **kws0)
        return s

    def storeH5(self, df, key, filepath_key=None, mode=None):
        if filepath_key is None :
            filepath_key=key
        aux.storeH5(df, key=key, path=reg.datapath(filepath_key,self.dir), mode=mode)

    def read(self, key=None,file=None):
        filepath_key = file
        if filepath_key is None :
            filepath_key=key
        return aux.read(reg.datapath(filepath_key,self.dir), key)


    def load_traj(self, mode='default'):
        df=self.read(key=mode, file='traj')
        if df is None :
            if mode=='default':
                s=self.load_step(h5_ks=[])
                df = s[['x', 'y']]
                self.storeH5(df=df, key='default', filepath_key='traj')
            elif mode in ['origin', 'center']:
                s = self.load_step(h5_ks=['contour', 'midline'])
                ss = reg.funcs.preprocessing["transposition"](s, c=self.config, store=True, replace=False, transposition=mode)
                df=ss[['x', 'y']]
        return df



    def load_dicts(self, type, ids=None):
        if ids is None:
            ids = self.agent_ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type].keys() for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            ds= aux.loadSoloDics(agent_ids=ids, path=reg.datapath(type, self.dir))
        return ds

    def visualize(self, **kwargs):
        from larvaworld.lib.sim.dataset_replay import ReplayRun
        rep = ReplayRun(dataset=self, **kwargs)
        rep.run()




    @ property
    def plot_dir(self):
        return reg.datapath('plots', self.dir)




    def preprocess(self, pre_kws={},recompute=False, store=True,is_last=False,**kwargs):
        for k, v in pre_kws.items():
            if v:
                cc = {
                    's': self.step_data,
                    'e': self.endpoint_data,
                    'c': self.config,
                    'recompute': recompute,
                    'store': store,
                    **kwargs,
                    k:v
                }
                reg.funcs.preprocessing[k](**cc)

        if is_last:
            self.save()

    def process(self, keys=[],recompute=False, mode='minimal', store=True,is_last=False,**kwargs):
        cc = {
            'mode': mode,
            'is_last': False,
            's': self.step_data,
            'e': self.endpoint_data,
            'c': self.config,
            'recompute': recompute,
            'store': store,
            **kwargs
        }
        for k in keys:
            func = reg.funcs.processing[k]
            func(**cc)

        if is_last:
            self.save()

    def annotate(self, keys=[], store=True,is_last=False,**kwargs):
        cc = {
            'd': self,
            's': self.step_data,
            'e': self.endpoint_data,
            'c': self.config,
            'store': store,
            **kwargs
        }
        for k in keys:
            func = reg.funcs.annotating[k]
            func(**cc)

        if is_last:
            self.save()


    def _enrich(self,pre_kws={}, proc_keys=[],anot_keys=[], recompute=False, mode='minimal', is_last=True,
                store=False, **kwargs):


        warnings.filterwarnings('ignore')
        cc0 = {
                'recompute': recompute,
                'is_last': False,
                'store': store,
            }

        cc = {
            'mode': mode,
            **cc0,
            **kwargs,
        }
        self.preprocess(pre_kws=pre_kws, **cc0)

        self.process(proc_keys, **cc)
        self.annotate(anot_keys, **cc)



        if is_last:
            self.save()
        return self


    def enrich(self, metric_definition=None, preprocessing={}, processing={},annotation={}, **kwargs):
        proc_keys=[k for k, v in processing.items() if v]
        anot_keys=[k for k, v in annotation.items() if v]
        if metric_definition is not None :
            self.config.metric_definition.update(metric_definition)
        return self._enrich(pre_kws=preprocessing,proc_keys=proc_keys,anot_keys=anot_keys, **kwargs)






    def get_par(self, par, key=None):
        def get_end_par(par):
            try:
                return self.read(key='end', file='end')[par]
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

        if key=='distro':
            try:
                return aux.read(key=par,path=reg.datapath("distro", self.dir))
            except:
                return self.get_par(par, key='step')


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
                    return None

    def delete(self, show_output=True):
        shutil.rmtree(self.dir)
        if show_output:
            reg.vprint(f'Dataset {self.id} deleted',2)

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()




    def get_chunk_par(self, chunk, short=None, par=None, min_dur=0, mode='distro'):
        if par is None:
            par = reg.getPar(short)
        dic0 = self.chunk_dicts
        dics = [dic0[id] for id in self.agent_ids]
        sss = [self.step_data[par].xs(id, level='AgentID') for id in self.agent_ids]

        if mode == 'distro':
            chunk_idx = f'{chunk}_idx'
            chunk_dur = f'{chunk}_dur'
            vs = []
            for ss, dic in zip(sss, dics):
                if min_dur == 0:
                    idx = dic[chunk_idx]
                else:
                    epochs = dic[chunk][dic[chunk_dur] >= min_dur]
                    Nepochs = epochs.shape[0]
                    if Nepochs == 0:
                        idx = []
                    elif Nepochs == 1:
                        idx = np.arange(epochs[0][0], epochs[0][1] + 1, 1)
                    else:
                        slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in epochs]
                        idx = np.concatenate(slices)
                vs.append(ss.loc[idx].values)
            vs = np.concatenate(vs)
            return vs
        elif mode == 'extrema':
            cc0s, cc1s, cc01s = [], [], []
            for ss, dic in zip(sss, dics):
                epochs = dic[chunk]
                if min_dur != 0:
                    chunk_dur = f'{chunk}_dur'
                    epochs = epochs[dic[chunk_dur] >= min_dur]
                Nepochs = epochs.shape[0]
                if Nepochs > 0:
                    c0s = ss.loc[epochs[:, 0]].values
                    c1s = ss.loc[epochs[:, 1]].values
                    cc0s.append(c0s)
                    cc1s.append(c1s)
            cc0s = np.concatenate(cc0s)
            cc1s = np.concatenate(cc1s)
            cc01s = cc1s - cc0s
            return cc0s, cc1s, cc01s

    def existing(self, key='end', return_shorts=False):
        if key == 'end':
            e = self.endpoint_data if hasattr(self, 'endpoint_data') else self.read(key='end')
            pars = e.columns.values.tolist()
        elif key == 'step':
            s = self.step_data if hasattr(self, 'step_data') else self.read(key='step')
            pars = s.columns.values.tolist()

        if not return_shorts:
            return sorted(pars)
        else:
            shorts = reg.getPar(d=pars, to_return='k')
            return sorted(shorts)

    def retrieve_metrics(self):
        c = self.config
        N = c.Npoints
        Nc = c.Ncontour

        sp = c.metric_definition.spatial
        self.points = nam.midline(N, type='point')
        try:
            p = self.points[sp.point_idx - 1]
        except:
            p = 'centroid'
        c.point = p



        self.Nangles = np.clip(N - 2, a_min=0, a_max=None)
        self.angles = [f'angle{i}' for i in range(self.Nangles)]
        self.Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        # self.segs = nam.segs(self.Nsegs)
        self.midline_xy = nam.xy(self.points, flat=False)
        self.contour = nam.contour(Nc)
        self.contour_xy = nam.xy(self.contour, flat=False)


        self.distance = nam.dst(p)
        self.velocity = nam.vel(p)
        self.acceleration = nam.acc(p)
        if sp.use_component_vel:
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

        self.h5_kdic = aux.h5_kdic(p, N, Nc)
        self.load_h5_kdic = aux.AttrDict({h5k: "w" for h5k in self.h5_kdic.keys()})

    def update_config(self):
        s,e=self.step_data, self.endpoint_data
        c=self.config
        c.dt = 1 / self.fr

        if 'agent_ids' not in c.keys():
            try:
                ids = self.agent_ids
            except:
                try:
                    ids = e.index.values
                except:
                    ids = self.read('end').index.values

            c.agent_ids = list(ids)
            c.N = len(ids)
        if 't0' not in c.keys():
            try:
                c.t0 = int(s.index.unique('Step')[0])
            except:
                c.t0 = 0
        if 'Nticks' not in c.keys():
            try:
                c.Nticks = s.index.unique('Step').size
            except:
                try:
                    c.Nticks = e['num_ticks'].max()
                except:
                    pass
        if 'duration' not in c.keys():
            try:
                c.duration = int(e['cum_dur'].max())
            except:
                c.duration = c.dt * c.Nticks
        if 'quality' not in c.keys():
            try:
                df = s[aux.nam.xy(c.point)[0]].values.flatten()
                valid = np.count_nonzero(~np.isnan(df))
                c.quality = np.round(valid / df.shape[0], 2)
            except:
                pass

        for k, v in c.items():
            if isinstance(v, np.ndarray):
                c[k] = v.tolist()



LarvaDataset = type('LarvaDataset', (_LarvaDataset,), decorators.dic_manager_kwargs)


def generate_dataset_config(**kwargs):

    c0=aux.AttrDict({'id': 'unnamed',
                  'group_id': None,
                  'refID': None,
                  'dir': None,
                  'fr': 16,
                  'Npoints': 3,
                  'Ncontour': 0,
                  'sample': None,
                  'color': None,
                  'metric_definition': None,
                  'env_params': {},
                  'larva_groups': {},
                  'source_xy': {},
                  'life_history': None,
                  })

    c0.update(kwargs)
    c0.dt=1/c0.fr
    if c0.metric_definition is None:
        c0.metric_definition = reg.get_null('metric_definition')



    if len(c0.larva_groups) == 1:
        c0.group_id, gConf = list(c0.larva_groups.items())[0]
        c0.color = gConf['default_color']
        c0.sample = gConf['sample']
        c0.life_history = gConf['life_history']
    if c0.dir is not None :
        os.makedirs(c0.dir, exist_ok=True)
        os.makedirs(f'{c0.dir}/data', exist_ok=True)
    reg.vprint(f'Generated new conf {c0.id}', 1)
    return c0











