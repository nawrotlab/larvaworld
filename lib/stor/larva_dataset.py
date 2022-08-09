import itertools
import os.path
import shutil
import numpy as np
import pandas as pd
import warnings
import copy

# from codetiming import Timer
from lib.anal.fitting import fit_epochs, get_bout_distros
from lib.aux import dictsNlists as dNl, xy_aux,data_aux, naming as nam, stdout
# import lib.aux.naming as nam

# from lib.registry.pars import preg
# from lib.registry.timer import check_time
# import logging
# from codetiming import Timer
from lib.aux.annotation import annotate
from lib.aux.stor_aux import read, storeH5, storeDic, loadDic, loadSoloDics, storeSoloDics, datapath

from lib.decorators.timer3 import timer, warn_slow

from lib.decorators.property import dic_manager_kwargs
from lib.stor.config import retrieve_config, update_config, update_metric_definition


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
        stored_ps = []
        # s = self.step_data
        for h5_k in h5_ks:

            pps = [p for p in self.h5_kdic[h5_k] if p in s.columns]
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
        self.storeH5(df=dic, key=None, filepath_key='vel_definition')
        print(f'Velocity definition dataset stored.')

        return dic





    def retrieveRefID(self, add_reference=False, refID=None):
        if refID is None:
            if self.config.refID is not None:
                refID = self.config.refID
            elif add_reference:
                refID = f'{self.group_id}.{self.id}'
                self.config.refID = refID

        return refID




    def save_config(self, add_reference=False, refID=None):
        refID=self.retrieveRefID(add_reference=add_reference, refID=refID)

        self.config=update_config(self, self.config)
        storeDic(self.config,path=datapath('conf', self.config.dir))




        if refID is not None:
            from lib.registry.pars import preg
            preg.saveRef(conf=self.config, id=refID)





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


    def get_filepath(self, filepath_key):
        from lib.aux.stor_aux import datapath, storeH5
        return datapath(filepath_key, self.dir)

    def storeH5(self, df, key, filepath_key=None, mode=None):
        if filepath_key is None :
            filepath_key=key
        storeH5(df=df, key=key, path=self.get_filepath(filepath_key), mode=mode)

    def read(self, key=None,file=None):
        filepath_key = file
        if filepath_key is None :
            filepath_key=key
        return read(key=key,path=self.get_filepath(filepath_key))




    def storeDic(self, d, filepath_key):
        storeDic(d=d, path=self.get_filepath(filepath_key))

    def loadDic(self, filepath_key,**kwargs):
        return loadDic(path=self.get_filepath(filepath_key),**kwargs)

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
            ds= loadSoloDics(agent_ids=ids, path=self.get_filepath(type))
        return ds

    def visualize(self, **kwargs):
        from lib.sim.replay.replay import ReplayRun
        rep = ReplayRun(dataset=self, **kwargs)

        rep.run()




    @ property
    def plot_dir(self):
        from lib.aux.stor_aux import datapath
        return datapath('plots', self.dir)




    def preprocess(self, pre_kws={},recompute=False, store=True,is_last=False,add_reference=False, **kwargs):
        from lib.registry.pars import preg
        FD = preg.proc_func_dict.dict.preproc

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
                func = FD[k]
                func(**cc, k=v)

        if is_last:
            self.save(add_reference=add_reference)
        # return self

    def process(self, keys=[],recompute=False, mode='minimal', store=True,is_last=False,add_reference=False, **kwargs):
        from lib.registry.pars import preg
        FD=preg.proc_func_dict.dict.proc

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
            func = FD[k]
            func(**cc)

        if is_last:
            self.save(add_reference=add_reference)
        # return self


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
                return read(key=par,path=self.get_filepath('distro'))
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
            par = preg.getPar(short)
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
                # for id in self.agent_ids:
                #     ss = self.step_data[par].xs(id, level='AgentID')
                #     dic = dic0[id]
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
            from lib.registry.pars import preg
            shorts = preg.getPar(d=pars, to_return='k')
            return sorted(shorts)


LarvaDataset = type('LarvaDataset', (_LarvaDataset,), dic_manager_kwargs)

if __name__ == '__main__':
    # d=LarvaDataset(dir=None, load_data=False)
    # print(d.config.N)
    #
    # raise
    # #
    #refID = 'None.40controls'


    from lib.registry.pars import preg
    #
    # md = nam.update_metric_definition()
    # print(md)
    # raise

    ds = preg.loadRefDs(['None.40controls', 'None.150controls'], step=False)
    GD=preg.graph_dict
    GD.eval0({'plotID':'crawl pars','title':'test','args': {'datasets':ds, 'show':True}})

    raise
    d.load()
    d.process(keys=['angular', 'spatial','dispersion','tortuosity'], recompute=True, store=True)

    d.annotate(on_food=False, store=True)
    d.save(refID=refID)


    raise


    import pandas as pd

    M = preg.larva_conf_dict
    refID = 'None.150controls'
    # refID='None.Sims2019_controls'
    h5_ks = ['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']
    h5_ks = ['epochs', 'angular', 'dspNtor']
    # h5_ks = []

    d = preg.loadRef(refID)
    d.load(h5_ks=h5_ks, step=False)



    # entries_3m = d.config.modelConfs['3modules']
    # raise
    # dIDs = ['NEU', 'SIN', 'CON']
    # for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
    #     for Ifmod in ['PHI', 'SQ', 'DEF']:
    #         mIDs = [f'{Cmod}_{Tmod}_{Ifmod}_DEF_fit' for Tmod in dIDs]
    #         id = f'Tmod_variable_Cmod_{Cmod}_Ifmod_{Ifmod}'
    #         d.eval_model_graphs(mIDs=mIDs, dIDs=dIDs, norm_modes=['raw', 'minmax'], id=id, N=5)

    dIDs, mIDs = [], []
    for Tmod in ['NEU', 'SIN']:
        for Ifmod in ['PHI', 'SQ']:
            mID = f'RE_{Tmod}_{Ifmod}_DEF_fit'
            dID = f'{Tmod} {Ifmod}'
            mIDs.append(mID)
            dIDs.append(dID)

    entries_var = M.add_var_mIDs(refID=refID, e=d.endpoint_data, c=d.config, mID0s=mIDs)

    # dIDs_avg=[f'{dID} avg' for dID in dIDs]
    # dIDs_var=[f'{dID} var' for dID in dIDs]
    #
    # mIDs1=[mIDs[0], mIDs_var[0],mIDs[1],mIDs_var[1]]
    # dIDs1=[dIDs_avg[0], dIDs_var[0],dIDs_avg[1],dIDs_var[1]]
    # id1 = f'PHIvsSQ_avgVSvar_NEU_RE_50l'
    # d.eval_model_graphs(mIDs=mIDs1, dIDs=dIDs1, norm_modes=['raw', 'minmax'], id=id1, N=50)
    #
    # mIDs2 = [mIDs[2], mIDs_var[2], mIDs[3], mIDs_var[3]]
    # dIDs2 = [dIDs_avg[2], dIDs_var[2], dIDs_avg[3], dIDs_var[3]]
    # id2 = f'PHIvsSQ_avgVSvar_SIN_RE_50l'
    # d.eval_model_graphs(mIDs=mIDs2, dIDs=dIDs2, norm_modes=['raw', 'minmax'], id=id2, N=50)

    # for mID,m in entries_3m.items():
    #     print(mID, m)
    # d.store_model_graphs(mIDs=mIDs_3m)
    # d.modelConf_analysis(mods3=True)
