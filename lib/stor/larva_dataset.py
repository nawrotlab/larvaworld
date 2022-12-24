import shutil
import numpy as np
import warnings

from lib.aux import dictsNlists as dNl, xy_aux, naming as nam, stdout

from lib.process.annotation import annotate
from lib.aux.stor_aux import read, storeH5, loadSoloDics

from lib.decorators.timer import warn_slow

from lib.decorators.property import dic_manager_kwargs
# from lib.registry.function_facade import funcs
from lib.stor.config import retrieve_config, update_config, update_metric_definition
from lib import reg

class _LarvaDataset:
    def __init__(self, dir=None, load_data=True, **kwargs):

        self.config = retrieve_config(dir=dir, **kwargs)
        nam.retrieve_metrics(self, self.config)
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
            step.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.step_data = step
            self.agent_ids = step.index.unique('AgentID').values
            self.num_ticks = step.index.unique('Step').size
        if end is not None:
            end.sort_index(inplace=True)
            self.endpoint_data = end
        if food is not None:
            self.food_endpoint_data = food
        self.config=update_config(self,self.config)










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

    @warn_slow
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
        # s = self.step_data
        for h5_k in h5_ks:

            pps = [p for p in self.h5_kdic[h5_k] if p in s.columns]
            pps=dNl.unique_list(pps)
            if len(pps) > 0:
                self.storeH5(df=s[pps], key=h5_k, mode=self.load_h5_kdic[h5_k])

                stored_ps += pps

        ss = s.drop(stored_ps, axis=1, errors='ignore')
        self.storeH5(df=ss, key='step')

    @warn_slow
    def save(self, step=True, end=True, food=False, add_reference=False,refID=None, **kwargs):

        if step:
            self.save_step(s=self.step_data, **kwargs)

        if end:
            self.storeH5(df=self.endpoint_data, key='end')
        if food:
            self.storeH5(df=self.food_endpoint_data, key='food')
        self.save_config(add_reference=add_reference, refID=refID)

        print(f'***** Dataset {self.id} stored.-----')

    def save_vel_definition(self, component_vels=True):
        from lib.process.calibration import comp_stride_variation, comp_segmentation
        warnings.filterwarnings('ignore')
        res_v = comp_stride_variation(self, component_vels=component_vels)
        res_fov = comp_segmentation(self)
        dic={**res_v,**res_fov}
        nam.retrieve_metrics(self, self.config)
        self.save_config()
        self.storeH5(df=dic, filepath_key='vel_definition')
        print(f'Velocity definition dataset stored.')

        return dic





    def retrieveRefID(self, add_reference=False, refID=None):
        if refID is None:
            if self.config.refID is not None:
                refID = self.config.refID
            else:
            # elif add_reference:
                refID = f'{self.group_id}.{self.id}'
                self.config.refID = refID

        return refID




    def save_config(self, add_reference=False, refID=None):
        refID=self.retrieveRefID(add_reference=add_reference, refID=refID)

        self.config=update_config(self, self.config)
        dNl.save_dict(self.config,reg.datapath('conf', self.config.dir))




        if refID is not None:
            reg.saveRef(conf=self.config, id=refID)





    def centralize_xy_tracks(self, replace=True, arena_dims=None, is_last=True):
        if arena_dims is None:
            arena_dims = self.config.env_params.arena.arena_dims
        x0, y0 = arena_dims

        kws0 = {
            'h5_ks': ['contour', 'midline'],
            # 'end' : False,
            # 'step' : True
        }
        s = self.load_step(**kws0)
        xy_pairs=self.midline_xy + self.contour_xy + nam.xy(['centroid', ''])


        xy_pairs = [xy for xy in xy_pairs if set(xy).issubset(s.columns)]
        # xy_flat = np.unique(dNl.flatten_list(xy_pairs))

        for x, y in xy_pairs:
            s[x] -= x0 / 2
            s[y] -= y0 / 2
        if replace:
            self.step_data = s
            # ss = s[xy_flat]
        if is_last:
            self.save_step(s, **kws0)
        return s

    def comp_dsp(self, s0,s1):
        # dt=
        # ids=self.agent_ids

        xy0 = self.load_traj()
        if xy0 is not None :



            AA,df=xy_aux.dsp_single(xy0, s0, s1, self.dt)
        else :
            df=None

        return df



    def storeH5(self, df, key, filepath_key=None, mode=None):
        if filepath_key is None :
            filepath_key=key
        storeH5(df=df, key=key, path=reg.datapath(filepath_key,self.dir), mode=mode)

    def read(self, key=None,file=None):
        filepath_key = file
        if filepath_key is None :
            filepath_key=key
        return read(reg.datapath(filepath_key,self.dir), key)

    def store_distros(self, ks=None):
        if ks is None :
            ks=['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa', 'rov', 'roa', 'tur_fou']+['cum_d', 'run_d', 'str_c_l', 'v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                     'dsp_0_40_max', 'dsp_0_60_max', 'str_N', 'tor5', 'tor20']+['fsv', 'ffov', 'run_t', 'pau_t', 'run_tr', 'pau_tr']+['tor5', 'tor20']+['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N']
        s=self.load_step()
        ps = reg.getPar(ks)
        ps = [p for p in ps if p in s.columns]
        dic = {}
        for p in ps:
            df = s[p].dropna().reset_index(level=0, drop=True)
            df.sort_index(inplace=True)
            dic[p] = df
        self.storeH5(dic, key=None,filepath_key='distro')



    def storeDic(self, d, filepath_key,**kwargs):
        dNl.save_dict(d,reg.datapath(filepath_key,self.dir),**kwargs)

    def loadDic(self, filepath_key,**kwargs):
        return dNl.load_dict(reg.datapath(filepath_key,self.dir),**kwargs)

    def load_traj(self, mode='default'):
        df=self.read(key=mode, file='traj')
        if df is None :
            if mode=='default':
                s=self.load_step(h5_ks=[])
                df = s[['x', 'y']]
                self.storeH5(df=df, key='default', filepath_key='traj')
            elif mode in ['origin', 'center']:
                s = self.load_step(h5_ks=['contour', 'midline'])
                from lib.process.spatial import align_trajectories
                ss = align_trajectories(s, c=self.config, store=True, replace=False, transposition=mode)
                df=ss[['x', 'y']]
        return df



    def load_dicts(self, type, ids=None):
        if ids is None:
            ids = self.agent_ids
        ds0 = self.larva_dicts
        if type in ds0 and all([id in ds0[type].keys() for id in ids]):
            ds = [ds0[type][id] for id in ids]
        else:
            ds= loadSoloDics(agent_ids=ids, path=reg.datapath(type, self.dir))
        return ds

    def visualize(self, **kwargs):
        from lib.sim.replay.replay import ReplayRun
        rep = ReplayRun(dataset=self, **kwargs)

        rep.run()




    @ property
    def plot_dir(self):

        return reg.datapath('plots', self.dir)




    def preprocess(self, pre_kws={},recompute=False, store=True,is_last=False,add_reference=False, **kwargs):

        cc = {
            's': self.step_data,
            'e': self.endpoint_data,
            'c': self.config,
            'recompute': recompute,
            'store': store,
            **kwargs
        }
        for k, v in pre_kws.items():
            if v:
                func = reg.funcs.preprocessing[k]
                # func = D.preproc[k]
                func(**cc, k=v)

        if is_last:
            self.save(add_reference=add_reference)
        # return self

    def process(self, keys=[],recompute=False, mode='minimal', store=True,is_last=False,add_reference=False, **kwargs):
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
            self.save(add_reference=add_reference)


    def _enrich(self,pre_kws={}, proc_keys=[], recompute=False, mode='minimal', show_output=False, is_last=True, bout_annotation=True,
                add_reference=False, store=False, **kwargs):


        with stdout.suppress_stdout(show_output):
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
               # **md['dispersion'], **md['tortuosity']
            }
            self.preprocess(pre_kws=pre_kws, **cc0)

            self.process(proc_keys, **cc)

            if bout_annotation :
                annotate(d=self, store=store, **kwargs)

            if is_last:
                self.save(add_reference=add_reference)
            return self


    def enrich(self, metric_definition=None, preprocessing={}, processing={}, **kwargs):
        proc_keys=[k for k, v in processing.items() if v]
        self.config.metric_definition=update_metric_definition(md=metric_definition,mdconf=self.config.metric_definition)

        return self._enrich(pre_kws=preprocessing,proc_keys=proc_keys, **kwargs)






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
                return read(key=par,path=reg.datapath("distro", self.dir))
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
            print(f'Dataset {self.id} deleted')

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


LarvaDataset = type('LarvaDataset', (_LarvaDataset,), dic_manager_kwargs)
