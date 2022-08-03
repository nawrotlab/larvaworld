import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas
import param

import lib.aux.dictsNlists as dNl
from lib.aux.data_aux import update_mdict, update_existing_mdict
from lib.aux.par_aux import sub

from lib.registry.pars import preg




class ConfType:
    def __init__(self,k, subks={}):
        self.k = k
        self.path = preg.paths[k]
        # self.use_pickle = False
        self.use_pickle = False if self.k != 'Ga' else True
        self.subks = subks

    # @property
    def loadDict(self):
        # print(self.k, self.use_pickle)
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

    def expand_mdict(self):



        if self.mdict is  None:
            return

        if len(self.subks) > 0:
            CT = preg.conftype_dict.dict
        for subID, subk in self.subks.items():
            if CT[subk].mdict is None :
                continue
            # print(subID,subk)
            # print(subID,subk)
            # print(subID,subk)
            if subID == 'larva_groups' and subk == 'Model':
                ct=CT['Model']
                m=copy.deepcopy(ct.mdict)
                lg=self.mdict[subID].v
                for k,dic in lg.items():
                    p=dic.model
                    if p in ct.ConfIDs:
                        # print(v)
                        conf = ct.loadConf(p)
                    elif isinstance(p, param.Parameterized):
                        if p.v in ct.ConfIDs:
                            conf = ct.loadConf(p.v)
                    else:
                        conf = None
                    if conf is not None:
                        # print(m, conf)
                        m = update_existing_mdict(m, conf)
                    dic.model=m

            else:
                ct=CT[subk]
                m=copy.deepcopy(ct.mdict)
                p = self.mdict[subID]
                if p in ct.ConfIDs:
                    conf = ct.loadConf(p)
                elif isinstance(p, param.Parameterized):
                    if p.v in ct.ConfIDs:
                        conf = ct.loadConf(p.v)
                else:
                    conf=None
                if conf is not None:
                    m = update_existing_mdict(m, conf)




                self.mdict[subID]=m






    def expandConf(self, id=None,conf=None):
        if conf is None:

            # from lib.registry.pars import preg
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
        preg.vprint(f'{self.k} Configuration saved under the id : {id}')

    def saveDict(self, d):
        print(self.k, self.use_pickle)
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

        preg.vprint(f'{self.k}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing',2)



    def deleteConf(self, id=None):
        if id is not None:
            d = self.loadDict()
            if id in d.keys():
                d.pop(id, None)
                self.saveDict(d)
                preg.vprint(f'Deleted {self.k} configuration under the id : {id}', 2)

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
    def build_mdict(self, dict0):
        self.dict0 =dict0
        if dict0 is not None :


            from lib.aux.data_aux import init2mdict
            self.mdict = init2mdict(dict0)
            # g1=self.gConf()
            # self.mdict = self.expand_mdict(self.mdict)
            # g2 = self.gConf()
            # dNl.dicsprint([g1, g2])


            self.eval = self.checkDict()
        else:
            self.mdict = None
        if self.k=='Trial':
            print(self.dict0, self.mdict)

            # raise



    def gConf(self,**kwargs):
        from lib.aux.data_aux import gConf
        return gConf(self.mdict,**kwargs)



    def checkDict(self):
        d = self.loadDict()
        eval = {}
        for id, conf in d.items():
            try:
                eval[id] =update_mdict(self.mdict, conf)
            except:
                eval[id] = None
        return eval


class ConfTypeDict:
    def __init__(self,load=False, save=False):

        self.SimIdx_path = preg.paths["SimIdx"]

        preg.vprint('started ConfTypes',2)
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body']

        self.dict = self.build(self.conftypes)

        preg.vprint('completed ConfTypes',2)

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



    def build(self, ks):

        self.subk_dict = self.build_subk_dict(ks)

        d = dNl.NestDict({k: ConfType(k=k, subks=subks) for k, subks in self.subk_dict.items()})

        # aa = d['Ga'].loadDict()
        # print(aa)
        # # # aa=CTs['Ga'].ConfID_entry(default='realism')
        # # # aa=CTs['Ga'].ConfID_entry(default='exploration')
        # raise
        return d

    def saveConf(self, conf, conftype, id=None, **kwargs):
        self.dict[conftype].saveConf(id=id, conf=conf, **kwargs)

    #
    def loadConf(self, conftype, id=None):
        return self.dict[conftype].loadConf(id=id)

    def loadRef(self, id=None):
        if id is not None:
            conf = self.dict.Ref.loadConf(id)
            from lib.stor.larva_dataset import LarvaDataset
            d = LarvaDataset(conf.dir, load_data=False)
            preg.vprint(f'Loaded stored reference configuration : {id}')
            return d

    def loadRefD(self, id=None, **kwargs):
        if id is not None:
            d = self.loadRef(id)
            d.load(**kwargs)
            preg.vprint(f'Loaded stored reference dataset : {id}',2)
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
        F0 = self.SimIdx_path
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


# conftype_dict = ConfTypeDict()
