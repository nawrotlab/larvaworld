import json
import os
import shutil
import numpy as np
import pandas as pd
import warnings

from larvaworld.lib import reg, aux

class LarvaDataset:
    def __init__(self, dir=None, load_data=True,config = None, **kwargs):
        '''
        Dataset class that stores a single experiment, real or simulated.
        Metadata and configuration parameters are stored in the 'config' dictionary.
        This can be provided as an argument, retrieved from a stored experiment or generated for a new experiment.
        Timeseries data are loaded as a pd.Dataframe 'step_data' with a 2-level index : 'Step' for the timestep index and 'AgentID' for the agent unique ID.
        Endpoint measurements are loaded as a pd.Dataframe 'endpoint_data' with 'AgentID' indexing
        Data is stored as HDF5 files or nested dictionaries. The core file is 'data.h5' with keys like 'step' for timeseries and 'end' for endpoint metrics.

        Args:
            dir: Path to stored data. Ignored if 'config' is provided. Defaults to None for no storage to disc
            load_data: Whether to load stored data
            config: The metadata dictionary. Defaults to None for attempting to load it from disc or generate a new.
            **kwargs: Any arguments to store in a novel configuration dictionary
        '''

        if config is None :
            config = reg.load_config(dir)
            if config is None:
                config = generate_dataset_config(dir=dir, **kwargs)

        c=self.config = config
        if c.dir is not None:
            os.makedirs(c.dir, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)

        self.h5_kdic = aux.h5_kdic(c.point, c.Npoints, c.Ncontour)
        self.__dict__.update(c)
        self.larva_dicts = {}
        if load_data:
            try:
                self.load()
            except:
                print('Data not found. Load them manually.')


    def set_data(self, step=None, end=None):
        c=self.config
        if step is not None:
            assert step.index.names == ['Step', 'AgentID']
            s = step.sort_index(level=['Step', 'AgentID'])
            self.Nticks = s.index.unique('Step').size
            c.t0 = int(s.index.unique('Step')[0])
            c.Nticks = self.Nticks
            if 'duration' not in c.keys():
                c.duration = c.dt * c.Nticks
            if 'quality' not in c.keys():
                try:
                    df = s[aux.nam.xy(c.point)[0]].values.flatten()
                    valid = np.count_nonzero(~np.isnan(df))
                    c.quality = np.round(valid / df.shape[0], 2)
                except:
                    pass

            self.step_data = s

        if end is not None:
            self.endpoint_data = end.sort_index()
            self.agent_ids = self.endpoint_data.index.values
            c.agent_ids = self.agent_ids
            c.N = len(self.agent_ids)

    def _load_step(self, h5_ks=None):
        s = self.read('step')
        if h5_ks is None :
            h5_ks=list(self.h5_kdic.keys())
        for h5_k in h5_ks:
            ss = self.read(h5_k)
            if ss is not None :
                ps = aux.nonexisting_cols(ss.columns.values,s)
                if len(ps) > 0:
                    s = s.join(ss[ps])
        return s


    def load(self, step=True, h5_ks=None):
        s = self._load_step(h5_ks=h5_ks) if step else None
        e = self.read('end')
        self.set_data(step=s, end=e)


    def _save_step(self, s):
        s = s.loc[:, ~s.columns.duplicated()]
        stored_ps = []
        for h5_k,ps in self.h5_kdic.items():
            pps = aux.unique_list(aux.existing_cols(ps,s))
            if len(pps) > 0:
                s[pps].to_hdf(self.data_path, h5_k)
                stored_ps += pps

        s.drop(stored_ps, axis=1, errors='ignore').to_hdf(self.data_path, 'step')

    def save(self, refID=None):
        if hasattr(self, 'step_data'):
            self._save_step(s=self.step_data)
        if hasattr(self, 'endpoint_data'):
            self.endpoint_data.to_hdf(self.data_path, 'end')
        self.save_config(refID=refID)
        reg.vprint(f'***** Dataset {self.id} stored.-----', 1)



    def read(self, key, file='data'):
        path=f'{self.data_dir}/{file}.h5'
        try :
            return pd.read_hdf(path, key)
        except:
            return None

    def save_config(self, refID=None):
        c = self.config
        if refID is not None:
            c.refID = refID
        if c.refID is not None:
            reg.Ref_paths(id=c.refID, dir=c.dir)
            reg.vprint(f'Saved reference dataset under : {c.refID}', 1)
        for k, v in c.items():
            if isinstance(v, np.ndarray):
                c[k] = v.tolist()
        aux.save_dict(c, f'{self.data_dir}/conf.txt')


    def load_traj(self, mode='default'):
        key=f'traj.{mode}'
        try :
            df = self.read(key)
        except :
            if mode=='default':
                s=self._load_step(h5_ks=[])
                df = s[['x', 'y']]
            elif mode in ['origin', 'center']:
                s = self._load_step(h5_ks=['contour', 'midline'])
                ss = reg.funcs.preprocessing["transposition"](s, c=self.config, store=False, replace=False, transposition=mode)
                df=ss[['x', 'y']]
            else :
                raise ValueError('Not implemented')
            df.to_hdf(self.data_path, key)
        return df



    def load_dicts(self, type, ids=None):
        if ids is None:
            ids = self.agent_ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type].keys() for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            ds= aux.loadSoloDics(agent_ids=ids, path=f'{self.data_dir}/individuals/{type}.txt')
        return ds

    def visualize(self,parameters={}, **kwargs):
        from larvaworld.lib.sim.dataset_replay import ReplayRun
        kwargs['dataset'] = self
        rep = ReplayRun(parameters=parameters, **kwargs)
        rep.run()




    @ property
    def plot_dir(self):
        return f'{self.config.dir}/plots'

    @property
    def data_dir(self):
        return f'{self.config.dir}/data'

    @property
    def data_path(self):
        return f'{self.data_dir}/data.h5'



    def _enrich(self,pre_kws={}, proc_keys=[],anot_keys=[], is_last=True,**kwargs):
        cc = {
            'd': self,
            's': self.step_data,
            'e': self.endpoint_data,
            'c': self.config,
            **kwargs
        }

        warnings.filterwarnings('ignore')
        for k, v in pre_kws.items():
            if v:
                ccc=cc
                ccc[k]=v
                reg.funcs.preprocessing[k](**ccc)
        for k in proc_keys:
            reg.funcs.processing[k](**cc)
        for k in anot_keys:
            reg.funcs.annotating[k](**cc)

        if is_last:
            self.save()
        return self


    def enrich(self, metric_definition=None, preprocessing={}, processing={},annotation={},**kwargs):
        proc_keys=[k for k, v in processing.items() if v]
        anot_keys=[k for k, v in annotation.items() if v]
        if metric_definition is not None :
            self.config.metric_definition.update(metric_definition)
            for k in proc_keys :
                if k in metric_definition.keys():
                    kwargs.update(metric_definition[k])
        return self._enrich(pre_kws=preprocessing,proc_keys=proc_keys,
                            anot_keys=anot_keys,**kwargs)






    def get_par(self, par=None, k=None, key='step'):
        if par is None and k is not None:
            par=reg.getPar(k)

        # if key=='distro':
        #     try:
        #         return pd.read_hdf(f'{self.data_dir}/distro.h5', key=par)
        #     except:
        #         return self.get_par(par, key='step')


        if key == 'end':
            if not hasattr(self, 'endpoint_data'):
                self.load(step=False)
            df=self.endpoint_data

        elif key == 'step':
            if not hasattr(self, 'step_data'):
                self.load()
            df=self.step_data
        else :
            raise

        if par in df.columns :
            return df[par]
        else :
            return reg.par.get(k=k, d=self, compute=True)

    def delete(self):
        shutil.rmtree(self.config.dir)
        reg.vprint(f'Dataset {self.id} deleted',2)

    def set_id(self, id, save=True):
        self.id = id
        self.config.id = id
        if save:
            self.save_config()




    def get_chunk_par(self, chunk, k=None, par=None, min_dur=0, mode='distro'):
        chunk_idx = f'{chunk}_idx'
        chunk_dur = f'{chunk}_dur'
        if par is None:
            par = reg.getPar(k)
            
        dic0 = aux.AttrDict(self.read('chunk_dicts'))
        # for id in self.agent_ids:
        #     ss=self.step_data[par].xs(id, level='AgentID')
        #     dic=dic0[id]
        #     epochs = dic[chunk]
        #     if min_dur != 0:
        #         epochs = epochs[dic[chunk_dur] >= min_dur]
        #     Nepochs = epochs.shape[0]



        dics = [dic0[id] for id in self.agent_ids]
        sss = [self.step_data[par].xs(id, level='AgentID') for id in self.agent_ids]

        if mode == 'distro':

            vs = []
            for ss, dic in zip(sss, dics):
                if min_dur == 0:
                    idx = dic[chunk_idx]+self.t0
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
                vs.append(ss.loc[idx].dropna().values)
            vs = np.concatenate(vs)
            return vs
        elif mode == 'extrema':
            cc0s, cc1s, cc01s = [], [], []
            for ss, dic in zip(sss, dics):
                epochs = dic[chunk]
                if min_dur != 0:
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


    @ property
    def Nangles(self):
        return np.clip(self.config.Npoints - 2, a_min=0, a_max=None)

    @property
    def points(self):
        return aux.nam.midline(self.config.Npoints, type='point')

    @property
    def contour(self):
        return aux.nam.contour(self.config.Ncontour)

    @property
    def data(self):
        s=self.step_data if hasattr(self, 'step_data') else None
        e=self.endpoint_data if hasattr(self, 'endpoint_data') else None
        return s, e, self.config



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

    c0.points =aux.nam.midline(c0.Npoints, type='point')

    try:
        c0.point = c0.points[c0.metric_definition.spatial.point_idx - 1]
    except:
        c0.point = 'centroid'

    if len(c0.larva_groups) == 1:
        c0.group_id, gConf = list(c0.larva_groups.items())[0]
        c0.color = gConf['default_color']
        c0.sample = gConf['sample']
        c0.model = gConf['model']
        c0.life_history = gConf['life_history']

    reg.vprint(f'Generated new conf {c0.id}', 1)
    return c0











