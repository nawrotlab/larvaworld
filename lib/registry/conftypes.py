import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas
import param


from lib.registry.pars import preg
import lib.aux.dictsNlists as dNl


class ConfType:
    def __init__(self, k, subks={},verbose=1):
        self.verbose = verbose
        self.k = k
        self.path = preg.path_dict[k]
        self.use_pickle = False if self.k != 'Ga' else True
        self.subks = subks

    # @property
    def loadDict(self):
        try:

            return dNl.load_dict(self.path,self.use_pickle)
        except:
            return dNl.NestDict()

    def loadConf(self, id):
        d=self.loadDict()
        if id in d.keys() :
            return dNl.NestDict(d[id])
        else :
            print(f'{self.k} Configuration {id} does not exist')
            raise ValueError()

    def expandConf(self, id):
        conf=self.loadConf(id)
        if len(self.subks)>0:
            CT=preg.conftype_dict.dict
        for subID, subk in self.subks.items():
            if subID=='larva_groups' and subk == 'Model':
                for k, v in conf[subID].items():
                    if type(v.model) == str:
                        v.model = CT[subk].loadConf(v.model)
            else :
                conf[subID]=CT[subk].expandConf(conf[subID])

        return conf


    def saveConf(self,id,conf, mode='overwrite'):
        d=self.loadDict()

        if id in d.keys() and mode == 'update':
            d[id] = dNl.update_nestdict(d[id], dNl.flatten_dict(conf))
        else:
            d[id] = dNl.NestDict(conf)
        self.saveDict(d)
        if self.verbose >= 1:
            print(f'{self.k} Configuration saved under the id : {id}')


    def saveDict(self, d):
        dNl.save_dict(d, self.path,self.use_pickle)

    def resetDict(self, dd):
        if self.k=='Model' :
            dnew=preg.larva_conf_dict.baseConfs()

            dd.update(dnew)
        d = self.loadDict()

        N0,N1=len(d), len(dd)

        d.update(dd)

        Ncur=len(d)
        Nnew=Ncur-N0
        Nup=N1-Nnew
        self.saveDict(d)

        if self.verbose >= 1:
            print(f'{self.k}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing')

        # for k,v in d.items():
        #     print(k)
        #     print(v.enrichment.preprocessing)


    def deleteConf(self, id=None):
        if id is not None:
            d = self.loadDict()
            if id in d.keys() :
                d.pop(id, None)
                self.saveDict(d)
                if self.verbose >= 1:
                    print(f'Deleted {self.k} configuration under the id : {id}')

    @ property
    def ConfIDs(self):
        return list(self.loadDict().keys())


    def ConfSelector(self, **kwargs):
        from lib.registry.par import selector_func
        return selector_func(objects=self.ConfIDs, **kwargs)

    def ConfParsarg(self):
        return {'dest': f'{self.k}_experiment', 'choices': self.ConfIDs, 'help': f'The {self.k} mode'}










class ConfTypeDict:
    def __init__(self, load=False, save=False, verbose=1):
        self.verbose = verbose
        self.conftypes = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body']


        self.dict = self.build(self.conftypes)

        self.dict_path = preg.path_dict['ConfTypeDict']
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




    def build(self, ks) :

        self.subk_dict = self.build_subk_dict(ks)

        d = dNl.NestDict({k : ConfType(k=k, subks = subks, verbose=self.verbose) for k,subks in self.subk_dict.items()})
        return d




    def saveConf(self, conf, conftype, id=None,**kwargs):
        self.dict[conftype].saveConf(id=id, conf=conf, **kwargs)

    #
    def loadConf(self, conftype, id=None):
        self.dict[conftype].loadConf(id=id,)


    def loadRef(self, id=None):
        if id is not None:
            conf=self.dict.Ref.loadConf(id)
            from lib.stor.larva_dataset import LarvaDataset
            return LarvaDataset(conf.dir, load_data=False)

    def loadRefD(self, id=None, **kwargs):
        if id is not None:
            d = self.loadRef(id)
            d.load(**kwargs)
            return d

    def loadRefDs(self, ids, **kwargs):
        ds = [self.loadRefD(id, **kwargs) for id in ids]
        return ds

    def confDict_funcs(self,k):
        from lib.conf.stored import aux_conf, data_conf, batch_conf, exp_conf, env_conf, essay_conf, ga_conf, larva_conf
        # raise
        d = dNl.NestDict({
            'Ref': data_conf.Ref_dict,
            'Model': larva_conf.Model_dict,
            'ModelGroup': larva_conf.ModelGroup_dict,
            'Env': env_conf.Env_dict,
            'Exp': exp_conf.Exp_dict,
            'ExpGroup': exp_conf.ExpGroup_dict,
            'Essay': essay_conf.Essay_dict,
            'Batch': batch_conf.Batch_dict,
            'Ga': ga_conf.Ga_dict,
            'Tracker': data_conf.Tracker_dict,
            'Group': data_conf.Group_dict,
            'Trial': aux_conf.Trial_dict,
            'Life': aux_conf.Life_dict,
            'Body': aux_conf.Body_dict
        })
        return d[k]

    def resetConfs(self, ks=None):
        if ks is None:
            ks = self.conftypes


        for k in ks:
            func=self.confDict_funcs(k)
            dic=func()
            self.dict[k].resetDict(dic)

            # if k == 'Ref':
            #     store_reference_data_confs()
            #     continue
            # elif k == 'Trial':
            #     from lib.conf.stored.aux_conf import trial_dict as d
            # elif k == 'Life':
            #     from lib.conf.stored.aux_conf import life_dict as d
            # elif k == 'Body':
            #     from lib.conf.stored.aux_conf import body_dict as d
            # elif k == 'Tracker':
            #     from lib.conf.stored.data_conf import tracker_formats as d
            # elif k == 'Group':
            #     from lib.conf.stored.data_conf import importformats as d
            # elif k == 'Model':
            #     from lib.registry.parConfs import larva_conf_dict
            #     larva_conf_dict.baseConfs()
            #     from lib.conf.stored.larva_conf import mod_dict as d
            # elif k == 'ModelGroup':
            #     from lib.conf.stored.larva_conf import mod_group_dict as d
            # elif k == 'Env':
            #     from lib.conf.stored.env_conf import env_dict as d
            # elif k == 'Exp':
            #     from lib.conf.stored.exp_conf import exp_dict as d
            # elif k == 'ExpGroup':
            #     from lib.conf.stored.exp_conf import exp_group_dict as d
            # elif k == 'Essay':
            #     from lib.conf.stored.essay_conf import essay_dict as d
            # elif k == 'Batch':
            #     from lib.conf.stored.batch_conf import batch_dict as d
            #
            # elif k == 'Ga':
            #     from lib.conf.stored.ga_conf import ga_dic as d
            # else:
            #     continue

            # for id, conf in d.items():
            #     self.saveConf(conf=conf, conftype=conftype, id=id)

    def next_idx(self, id, conftype='Exp'):
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
