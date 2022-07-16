import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas
import param

import lib.aux.dictsNlists as dNl

def store_reference_data_confs():
    from lib.stor.larva_dataset import LarvaDataset
    from lib.registry import paths
    DATA = paths.path_dict["DATA"]
    dds = [
        [f'{DATA}/JovanicGroup/processed/3_conditions/AttP{g}@UAS_TNT/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = dNl.flatten_list(dds)
    dds.append(f'{DATA}/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/exploration')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/200_controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/10_controls')
    for dr in dds:
        try:
            d = LarvaDataset(dr, load_data=False)
            d.save_config(add_reference=True)
        except:
            pass

class ConfTypeDict :
    def __init__(self, load=False, save=False):
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body']


        from lib.registry import paths
        self.dict_path = paths.path_dict['ConfTypeDict']
        if not load:
            self.dict = self.build()
            if save :
                dNl.save_dict(self.dict, self.dict_path)
        else:
            self.dict = dNl.load_dict(self.dict_path)




    def build(self):
        from lib.aux.par_aux import sub
        from lib.registry.par_dict import preparePar
        from lib.registry.par import v_descriptor
        d = {}
        for conftype in self.conftypes:
            low = conftype.lower()
            k = f'{low}ID'
            dic = {'dtype': str, 'vparfunc': self.conf_selector_func(conftype), 'vs': self.storedConf(conftype),
                   'sym': sub('ID', low), 'k': k, 'h': f'The {conftype} configuration ID', 'p': conftype,
                   'disp': f'{conftype} ID'}
            prepar = preparePar(**dic)
            p = v_descriptor(**prepar)
            d[p.p] = p
        return dNl.NestDict(d)

    def saveConf(self, conf, conftype, id=None, mode='overwrite', verbose=1):
        if id is not None:
            try:
                d = self.loadConfDict(conftype)
            except:
                d = {}
            if id is None:
                id = conf['id']

            if id in list(d.keys()):
                for k, v in conf.items():
                    if type(k) == dict and k in list(d[id].keys()) and mode == 'update':
                        d[id][k].update(conf[k])
                    else:
                        d[id][k] = v
            else:
                d[id] = conf
            self.saveConfDict(d, conftype=conftype)
            if verbose >= 1:
                print(f'{conftype} Configuration saved under the id : {id}')

    def loadConf(self, conftype, id=None):
        if id is not None:
            try:
                conf_dict = self.loadConfDict(conftype)
                conf = conf_dict[id]
                return dNl.NestDict(conf)
            except:
                raise ValueError(f'{conftype} Configuration {id} does not exist')

    def saveConfDict(self, ConfDict, conftype, use_pickle=False):
        path = self.path_dict[conftype]
        if conftype == 'Ga':
            use_pickle = True
        if use_pickle:
            with open(path, 'wb') as fp:
                pickle.dump(ConfDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, "w") as f:
                json.dump(ConfDict, f)

    def loadConfDict(self, conftype, use_pickle=False):
        path = self.path_dict[conftype]
        if conftype == 'Ga':
            use_pickle = True
        try:
            if use_pickle:
                with open(path, 'rb') as tfp:
                    d = pickle.load(tfp)
            else:
                with open(path) as f:
                    d = json.load(f)
        except:
            d = {}
        return dNl.NestDict(d)

    def expandConf(self, conftype, id=None):
        if id is not None:
            conf = self.loadConf(id=id, conftype=conftype)
            try:
                if conftype == 'Batch':
                    conf.exp = self.expandConf(id=conf.exp, conftype='Exp')
                elif conftype == 'Exp':
                    conf.experiment = id
                    conf.env_params = self.expandConf(id=conf.env_params, conftype='Env')
                    conf.trials = self.loadConf(id=conf.trials, conftype='Trial')
                    for k, v in conf.larva_groups.items():
                        if type(v.model) == str:
                            v.model = self.loadConf(id=v.model, conftype='Model')
                elif conftype == 'Ga':
                    conf.experiment = id
                    conf.env_params = self.expandConf(id=conf.env_params, conftype='Env')
            except:
                pass
            return conf

    def loadRef(self, id=None):
        if id is not None:
            from lib.stor.larva_dataset import LarvaDataset
            return LarvaDataset(self.loadConf(id=id, conftype='Ref')['dir'], load_data=False)

    def loadRefD(self, id=None, **kwargs):
        if id is not None:
            from lib.stor.larva_dataset import LarvaDataset
            d = LarvaDataset(self.loadConf(id=id, conftype='Ref')['dir'], load_data=False)
            d.load(**kwargs)
            return d

    def loadRefDs(self, ids, **kwargs):
        ds = [self.loadRefD(id, **kwargs) for id in ids]
        return ds

    def deleteConf(self, conftype, id=None):
        if id is not None:
            if conftype == 'Data':
                DataGroup = self.loadConf(id=id, conftype=conftype)
                path = DataGroup['path']
                try:
                    shutil.rmtree(path)
                except:
                    pass
            d = self.loadConfDict(conftype=conftype)
            try:
                d.pop(id, None)
                self.saveConfDict(d, conftype=conftype)
                print(f'Deleted {conftype} configuration under the id : {id}')
            except:
                pass

    def storedConf(self, conftype):
        return list(self.loadConfDict(conftype=conftype).keys())

    def conf_selector_func(self, conftype, default=None, single_choice=True, **kwargs):
        def func():

            kws = {
                'objects': self.storedConf(conftype),
                'default': default,
                'allow_None': True,
                'empty_default': True,
            }
            if single_choice:
                func0 = param.Selector
            else:
                func0 = param.ListSelector
            return func0(**kws, **kwargs)

        return func

    def storeConfs(self, conftypes=None):
        if conftypes is None:
            conftypes = self.conftypes

        for conftype in conftypes:
            if conftype == 'Ref':
                store_reference_data_confs()
                continue
            elif conftype == 'Trial':
                from lib.conf.stored.aux_conf import trial_dict as d
            elif conftype == 'Life':
                from lib.conf.stored.aux_conf import life_dict as d
            elif conftype == 'Body':
                from lib.conf.stored.aux_conf import body_dict as d
            elif conftype == 'Tracker':
                from lib.conf.stored.data_conf import tracker_formats as d
            elif conftype == 'Group':
                from lib.conf.stored.data_conf import importformats as d
            elif conftype == 'Model':
                from lib.registry.parConfs import larva_conf_dict
                larva_conf_dict.baseConfs()
                from lib.conf.stored.larva_conf import mod_dict as d
            elif conftype == 'ModelGroup':
                from lib.conf.stored.larva_conf import mod_group_dict as d
            elif conftype == 'Env':
                from lib.conf.stored.env_conf import env_dict as d
            elif conftype == 'Exp':
                from lib.conf.stored.exp_conf import exp_dict as d
            elif conftype == 'ExpGroup':
                from lib.conf.stored.exp_conf import exp_group_dict as d
            elif conftype == 'Essay':
                from lib.conf.stored.essay_conf import essay_dict as d
            elif conftype == 'Batch':
                from lib.conf.stored.batch_conf import batch_dict as d

            elif conftype == 'Ga':
                from lib.conf.stored.ga_conf import ga_dic as d
            else:
                continue
            for id, conf in d.items():
                self.saveConf(conf=conf, conftype=conftype, id=id)


conftype_dict = ConfTypeDict()