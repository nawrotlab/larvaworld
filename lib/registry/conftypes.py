import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas
import param

import lib.aux.dictsNlists as dNl
from lib.aux.par_aux import sub

from lib.registry.pars import preg


class ConfType:
    def __init__(self, k, subks={}, verbose=None):
        if verbose is None:
            from lib.registry.units import base_verbose
            verbose = base_verbose
        self.verbose = verbose

        self.k = k
        self.path = preg.path_dict[k]
        self.use_pickle = False if self.k != 'Ga' else True
        self.subks = subks
        # self.mdict=self.build_mdict()

    # @property
    def loadDict(self):
        try:

            return dNl.load_dict(self.path, self.use_pickle)
        except:
            return dNl.NestDict()

    def loadConf(self, id):
        d = self.loadDict()
        if id in d.keys():
            return dNl.NestDict(d[id])
        else:
            print(f'{self.k} Configuration {id} does not exist')
            raise ValueError()

    def expandConf(self, id):
        from lib.registry.pars import preg
        conf = self.loadConf(id)
        if len(self.subks) > 0:
            CT = preg.conftype_dict.dict
        for subID, subk in self.subks.items():
            if subID == 'larva_groups' and subk == 'Model':
                for k, v in conf[subID].items():
                    if type(v.model) == str:
                        v.model = CT[subk].loadConf(v.model)
            else:
                conf[subID] = CT[subk].expandConf(conf[subID])

        return conf

    def saveConf(self, id, conf, mode='overwrite'):
        d = self.loadDict()

        if id in d.keys() and mode == 'update':
            d[id] = dNl.update_nestdict(d[id], dNl.flatten_dict(conf))
        else:
            d[id] = dNl.NestDict(conf)
        self.saveDict(d)
        if self.verbose >= 1:
            print(f'{self.k} Configuration saved under the id : {id}')

    def saveDict(self, d):
        dNl.save_dict(d, self.path, self.use_pickle)

    def reset_func(self):
        from lib.registry.confResetFuncs import confReset_funcs
        return confReset_funcs(self.k)()

    def resetDict(self):
        dd = self.reset_func()
        d = self.loadDict()

        N0, N1 = len(d), len(dd)

        d.update(dd)

        Ncur = len(d)
        Nnew = Ncur - N0
        Nup = N1 - Nnew
        self.saveDict(d)

        if self.verbose >= 1:
            print(f'{self.k}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing')

    def deleteConf(self, id=None):
        if id is not None:
            d = self.loadDict()
            if id in d.keys():
                d.pop(id, None)
                self.saveDict(d)
                if self.verbose >= 1:
                    print(f'Deleted {self.k} configuration under the id : {id}')

    @property
    def ConfIDs(self):
        return list(self.loadDict().keys())

    def ConfSelector(self, **kwargs):
        from lib.registry.par import selector_func
        def func():
            return selector_func(objects=self.ConfIDs, **kwargs)

        return func

    def ConfParsarg(self):
        return {'dest': f'{self.k}_experiment', 'choices': self.ConfIDs, 'help': f'The {self.k} mode'}

    def ConfID_entry(self, default=None, k=None, symbol=None, single_choice=True):
        # print(self.k)
        from typing import List
        from lib.aux.par_aux import sub
        low = self.k.lower()
        if single_choice:
            t = str
            IDstr = 'ID'
        else:
            t = List[str]
            IDstr = 'IDs'
        if k is None:
            k = f'{low}{IDstr}'
        if symbol is None:
            symbol = sub(IDstr, low)
        d = {'dtype': t, 'vparfunc': self.ConfSelector(default=default, single_choice=single_choice),
             'vs': self.ConfIDs, 'v': default,
             'symbol': symbol, 'k': k, 'h': f'The {self.k} configuration {IDstr}',
             'disp': f'{self.k} {IDstr}'}
        return dNl.NestDict(d)

    # def build_mdict(self):
    def build_mdict(self, k0, initD):
        from lib.aux.data_aux import init2mdict
        self.k0 = k0
        if k0 is not None and k0 in initD.keys():
            self.dict0 = initD[k0]
            self.mdict = init2mdict(initD[k0])
            self.eval = self.checkDict()

        else:
            self.dict0 = None
            self.mdict = None

    def checkDict(self):
        M = preg.larva_conf_dict
        d = self.loadDict()
        eval = {}
        for id, conf in d.items():
            try:
                eval[id] = M.update_mdict(self.mdict, conf)
                # print(f'{id}  SUCCESS')
            except:
                eval[id] = None
                # print(f'{id}  FAIL')
        return eval


class ConfTypeDict:
    def __init__(self, load=False, save=False, verbose=None):
        if verbose is None:
            from lib.registry.units import base_verbose
            verbose = base_verbose
        self.verbose = verbose
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body']

        self.dict = self.build(self.conftypes)

        from lib.aux.stdout import vprint
        vprint('completed ConfTypes', self.verbose)
        # print('completed ConfTypes')

        # self.dict_path = preg.path_dict['ConfTypeDict']
        # if not load:
        #     self.dict = self.build(self.conftypes)
        #     if save:
        #         dNl.save_dict(self.dict, self.dict_path)
        # else:
        #     self.dict = dNl.load_dict(self.dict_path)

    def build_subk_dict(self, ks):
        d0 = dNl.NestDict({k: {} for k in ks})
        d1 = dNl.NestDict({
            'Batch': {'exp': 'Exp'},
            'Ga': {'env_params': 'Env'},
            'Exp': {'env_params': 'Env',
                    'trials': 'Trial',
                    'larva_groups': 'Model',
                    }
        })
        d0.update(d1)
        return d0

    def build_mDicts(self, initD):
        from lib.registry.initDicts import confInit_ks

        for k, ct in self.dict.items():
            ct.build_mdict(k0=confInit_ks(k), initD=initD)

        print('computed mDicts')

    def build(self, ks):

        self.subk_dict = self.build_subk_dict(ks)

        d = dNl.NestDict({k: ConfType(k=k, subks=subks, verbose=self.verbose) for k, subks in self.subk_dict.items()})
        return d

    def saveConf(self, conf, conftype, id=None, **kwargs):
        self.dict[conftype].saveConf(id=id, conf=conf, **kwargs)

    #
    def loadConf(self, conftype, id=None):
        self.dict[conftype].loadConf(id=id, )

    def loadRef(self, id=None):
        if id is not None:
            conf = self.dict.Ref.loadConf(id)
            from lib.stor.larva_dataset import LarvaDataset
            d = LarvaDataset(conf.dir, load_data=False)
            if self.verbose >= 1:
                print(f'Loaded stored reference dataset : {id}')
            return d

    def loadRefD(self, id=None, **kwargs):
        if id is not None:
            d = self.loadRef(id)
            d.load(**kwargs)
            return d

    def loadRefDs(self, ids, **kwargs):
        ds = [self.loadRefD(id, **kwargs) for id in ids]
        return ds

    # def confDict_funcs(self,k):
    #     from lib.conf.stored import aux_conf, data_conf, batch_conf, exp_conf, env_conf, essay_conf, ga_conf, larva_conf
    #     # raise
    #     d = dNl.NestDict({
    #         'Ref': data_conf.Ref_dict,
    #         'Model': larva_conf.Model_dict,
    #         'ModelGroup': larva_conf.ModelGroup_dict,
    #         'Env': env_conf.Env_dict,
    #         'Exp': exp_conf.Exp_dict,
    #         'ExpGroup': exp_conf.ExpGroup_dict,
    #         'Essay': essay_conf.Essay_dict,
    #         'Batch': batch_conf.Batch_dict,
    #         'Ga': ga_conf.Ga_dict,
    #         'Tracker': data_conf.Tracker_dict,
    #         'Group': data_conf.Group_dict,
    #         'Trial': aux_conf.Trial_dict,
    #         'Life': aux_conf.Life_dict,
    #         'Body': aux_conf.Body_dict
    #     })
    #     return d[k]

    def resetConfs(self, ks=None):
        if ks is None:
            ks = self.conftypes

        for k in ks:
            self.dict[k].resetDict()

    def next_idx(self, id, conftype='Exp'):
        from lib.registry.pars import preg
        F0 = preg.path_dict["SimIdx"]
        try:
            with open(F0) as f:
                d = json.load(f)
        except:
            d = {k: {exp: 0 for exp in self.dict[k].ConfIDs} for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']}
        if not conftype in d.keys():
            d[conftype] = {}
        if not id in d[conftype].keys():
            d[conftype][id] = 0
        d[conftype][id] += 1
        with open(F0, "w") as fp:
            json.dump(d, fp)
        return d[conftype][id]


conftype_dict = ConfTypeDict()
