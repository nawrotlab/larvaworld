import os

import numpy as np
import param

import larvaworld
from larvaworld.lib import reg, aux, util

SIMTYPES=['Exp', 'Batch', 'Ga', 'Eval','Replay']

CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']

GROUPTYPES = ['LarvaGroup', 'SourceGroup', 'epoch']

Path = aux.AttrDict({k : f'{reg.CONF_DIR}/{k}.txt' for k in CONFTYPES})





def build_ConfTypeSubkeys():
    d0 = {k: {} for k in CONFTYPES}
    d1 = {
        'Batch': {'exp': 'Exp'},
        'Ga': {'env_params': 'Env'},
        'Exp': {'env_params': 'Env',
                'trials': 'Trial',
                'larva_groups': 'Model',
                }
    }
    d0.update(d1)
    return aux.AttrDict(d0)

CONFTYPE_SUBKEYS = build_ConfTypeSubkeys()


def build_GroupTypeSubkeys():
    d0 = {k: {} for k in GROUPTYPES}
    d1 = {
        'LarvaGroup': {'Model'},
        # 'Ga': {'env_params': 'Env'},
        # 'Exp': {'env_params': 'Env',
        #         'trials': 'Trial',
        #         'larva_groups': 'Model',
        #         }
    }
    d0.update(d1)
    return aux.AttrDict(d0)

GROUPTYPE_SUBKEYS = build_GroupTypeSubkeys()


CONFTREE = aux.AttrDict({k : aux.load_dict(Path[k]) for k in CONFTYPES})

def build_conf_tree_expanded():
    c0 = aux.AttrDict({k : aux.load_dict(Path[k]) for k in CONFTYPES})
    sk = CONFTYPE_SUBKEYS
    for confType0 in c0.keys():
        if confType0 in sk.keys():
            pairs = sk[confType0]
            for id, conf in c0[confType0].items():
                for subID, confType in pairs.items():


                    if subID in conf.keys():
                        if isinstance(conf[subID], str) and conf[subID] in c0[confType].keys():
                            conf[subID]=c0[confType][conf[subID]]
                        elif (subID, confType) == ('larva_groups', 'Model'):
                            for gID, gConf in conf[subID].items():
                                mID=gConf.model
                                if mID in c0['Model'].keys():
                                    gConf.model=c0['Model'][mID]
                                else:
                                    # print(f'{mID} not found')
                                    pass
                                    # raise
    return c0

CONFTREE_EXPANDED = build_conf_tree_expanded()


class BaseType:
    def __init__(self, k):
        self.k = k
        if k in reg.par.PI.keys():
            self.dict0 = reg.par.PI[k]
            self.mdict = util.init2mdict(self.dict0)
            self.ks = util.get_ks(self.mdict)
        else:
            self.dict0 = None
            self.mdict = None
            self.ks = None


    def gConf_kws(self, dic):
        kws0 = {}
        for k, kws in dic.items():
            m0 = self.mdict[k]
            kws0[k] = self.gConf(m0, **kws)
        return aux.AttrDict(kws0)

    def gConf(self, m0=None, kwdic=None, **kwargs):
        if m0 is None:
            if self.mdict is None:
                return None
            else:
                m0 = self.mdict
        if kwdic is not None:
            kws0 = self.gConf_kws(kwdic)
            kwargs.update(kws0)

        return aux.AttrDict(util.gConf(m0, **kwargs))

    def entry(self, id, **kwargs):
        return aux.AttrDict({id: self.gConf(**kwargs)})



def lgs(mIDs, ids=None, cs=None,**kwargs):

    if ids is None:
        ids = mIDs
    N = len(mIDs)
    if cs is None :
        cs = aux.N_colors(N)
    return aux.AttrDict(aux.merge_dicts([lg(id, c=c, mID=mID, **kwargs) for mID, c, id in zip(mIDs, cs, ids)]))


def lg(id=None, c='black', N=1, mode='uniform', sh='circle', loc=(0.0, 0.0), ors=(0.0, 360.0),
       s=(0.0, 0.0), mID='explorer',age=0.0, epochs={},  o=None,sample = None, expand=False, **kwargs):

    if mID is not None :
        m = mID if not expand else stored.getModel(mID)
        if id is None:
            id=mID
    else :
        m=None
    if id is None:
        id='LarvaGroup'

    if type(s) == float:
        s = (s, s)
    kws = {'kwdic': {
        'distribution': {'N': N, 'scale': s, 'orientation_range': ors, 'loc': loc, 'shape': sh, 'mode': mode},
        'life_history': {'age': age,'epochs': epochs}
    },
           'default_color': c, 'model': m,'sample':sample,  **kwargs}
    if o is not None:
        kws['odor'] = o

    return stored.group.LarvaGroup.entry(id=id, **kws)

# def GTRvsS2(N=1, age=72.0, q=1.0, h_starved=0.0, sample='exploration.150controls', substrate_type='standard',pref='',
#                 navigator=False, expand=False, **kwargs):
#     if age==0.0 :
#         epochs={}
#     else :
#         if h_starved==0:
#             eps={
#                 0 : {'start': 0.0, 'stop' : age, 'substate':{'type': substrate_type, 'quality': q}}
#             }
#         else :
#             eps = {
#                 0: {'start': 0.0, 'stop': age-h_starved, 'substate': {'type': substrate_type, 'quality': q}},
#                 1: {'start': age-h_starved, 'stop': age, 'substate': {'type': substrate_type, 'quality': 0}},
#             }
#         epochs={}
#         for id,kws in eps.items():
#             epochs.update(stored.group.epoch.entry(id=id, **kws))
#
#
#
#
#     kws0 = {
#         'kwdic': {
#             'distribution': {'N': N, 'scale': (0.005, 0.005)},
#             'life_history': {'age': age,
#                              'epochs': epochs,
#                              },
#         'odor':{}
#         },
#         'sample': sample,
#     }
#
#     mcols = ['blue', 'red']
#     mID0s = ['rover', 'sitter']
#     lgs = {}
#     for mID0, mcol in zip(mID0s, mcols):
#         id=f'{pref}{mID0.capitalize()}'
#
#
#
#         if navigator :
#             mID0=f'navigator_{mID0}'
#         if expand:
#             mID0=stored.getModel(mID0)
#
#
#
#         kws = {
#             'default_color': mcol,
#             'model': mID0,
#             **kws0
#         }
#
#         lgs.update(stored.group.LarvaGroup.entry(id, **kws))
#     return aux.AttrDict(lgs)





def next_idx(id, conftype='Exp'):
    f = f'{reg.CONF_DIR}/SimIdx.txt'
    if not os.path.isfile(f):
        d = aux.AttrDict({k: {} for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']})
    else:
        d = aux.load_dict(f)

    if not conftype in d.keys():
        d[conftype] = {}
    if not id in d[conftype].keys():
        d[conftype][id] = 0
    d[conftype][id] += 1
    aux.save_dict(d, f)
    return d[conftype][id]



class StoredConfRegistry :
    def __init__(self):
        self.group=aux.AttrDict({k: BaseType(k=k) for k in GROUPTYPES})
        self.conf=aux.AttrDict({k: BaseType(k=k) for k in CONFTYPES})

    def ConfPath(self, conftype):
        return Path[conftype]

    def get_dict(self, conftype):
        path=self.ConfPath(conftype)
        return aux.load_dict(path)

    def set_dict(self, conftype, d):
        path = self.ConfPath(conftype)
        aux.save_dict(d, path)

    def resetDict(self,conftype, init=False):
        dd = reg.funcs.stored_confs[conftype]()

        if os.path.isfile(self.ConfPath(conftype)):
            if init:
                return
            else:
                d = self.get_dict(conftype)
        else:
            d = {}

        N0, N1 = len(d), len(dd)

        d.update(dd)

        Ncur = len(d)
        Nnew = Ncur - N0
        Nup = N1 - Nnew
        self.set_dict(conftype, d)
        reg.vprint(f'{conftype}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing', 1)

    def resetConfs(self, conftypes=None, **kwargs):
        if conftypes is None:
            conftypes = CONFTYPES

        for conftype in conftypes:
            self.resetDict(conftype, **kwargs)

    def confIDs(self, conftype):
        d=self.get_dict(conftype)
        return sorted(list(d.keys()))

    @ property
    def RefIDs(self):
        return self.confIDs('Ref')

    @property
    def ModelIDs(self):
        return self.confIDs('Model')

    def get(self, conftype, id):
        d=self.get_dict(conftype)
        if id in d.keys():
            return aux.AttrDict(d[id])
        else:
            reg.vprint(f'{conftype} Configuration {id} does not exist', 1)
            raise ValueError()

    def set(self, conftype, id, conf, mode='overwrite'):
        d=self.get_dict(conftype)
        if id in d.keys() and mode == 'update':
            d[id] = d[id].update_nestdict(conf.flatten())
        else:
            d[id] = aux.AttrDict(conf)
        self.set_dict(conftype, d)
        reg.vprint(f'{conftype} Configuration saved under the id : {id}', 1)

    def delete(self, conftype, id=None):
        if id is not None:
            d=self.get_dict(conftype)
            if id in d.keys():
                d.pop(id, None)
                self.set_dict(conftype, d)
                reg.vprint(f'Deleted {conftype} configuration under the id : {id}', 1)

    def expand(self, conftype, id=None, conf=None):
        if conf is None:
            if id in self.confIDs(conftype):

                conf = self.get(conftype, id)
            else:
                return None
        subks = CONFTYPE_SUBKEYS[conftype]
        if len(subks) > 0:
            for subID, subk in subks.items():
                ids = self.confIDs(subk)
                if subID == 'larva_groups' and subk == 'Model':
                    for k, v in conf['larva_groups'].items():
                        if v.model in ids:
                            v.model = self.get(subk, id=v.model)
                else:
                    if conf[subID] in ids:
                        conf[subID] = self.get(subk, id=conf[subID])

        return conf


    def getExp(self, id, expand=True):
        if expand :
            return self.expand(conftype='Exp', id=id)
        else :
            return self.get(conftype='Exp', id=id)

    def getModel(self, id):
        return self.get(conftype='Model', id=id)

    def getEnv(self, id):
        return self.get(conftype='Env', id=id)

    def setModel(self, id, conf):
        return self.set(conftype='Model', id=id, conf=conf)

    def getGroup(self, id):
        return self.get(conftype='Group', id=id)

    def getRefDir(self, id):
        d = self.get_dict('Ref')
        if id in d.keys():
            return d[id]
        else:
            reg.vprint(f'Reference dataset with ID {id} does not exist. Returning None', 1)
            return None

    def getRef(self, id=None, dir=None):
        if dir is None:
            dir=self.getRefDir(id)
        if dir is not None:
            path = f'{dir}/data/conf.txt'
            if os.path.isfile(path):
                c = aux.load_dict(path)
                if 'id' in c.keys():
                    reg.vprint(f'Loaded existing conf {c.id}', 1)
                    return c
        return None

    def loadRef(self, id, load=False, **kwargs):
        c=self.getRef(id)
        if c is not None:
            d = larvaworld.LarvaDataset(config=c, load_data=False)
            if load:
                d.load(**kwargs)
            reg.vprint(f'Loaded stored reference dataset : {id}', 1)
            return d
        else:
            reg.vprint(f'Failed to load reference dataset : {id}', 1)
            return None


    def setRefID(self, id, dir=None):
        if dir is not None:
            d = self.get_dict('Ref')
            d[id] = dir
            self.set_dict('Ref', d)



    def retrieve_dataset(self,dataset=None, refID=None, dir=None):
        if dataset is None:
            if refID is not None:
                dataset = self.loadRef(refID)
            elif dir is not None:
                dataset = larvaworld.LarvaDataset(dir=f'{reg.DATA_DIR}/{dir}', load_data=False)
            else:
                raise ValueError('Unable to load dataset. Either refID or storage path must be provided. ')
        return dataset

    def imitation_exp(self, refID, model='explorer', **kwargs):
        c = self.getRef(refID)

        kws = {
            'id': 'ImitationGroup',
            'expand': True,
            'sample': refID,
            'model': model,
            'default_color': 'blue',
            'distribution': {'N': c.N},
            'imitation': True,

        }


        exp_conf = reg.get_null('Exp', sim_params=reg.get_null('sim_params', dt=c.dt, duration=c.duration),
                                env_params=c.env_params, larva_groups=reg.full_lg(**kws),experiment='imitation',
                                trials={}, enrichment=reg.par.base_enrich())
        exp_conf.update(**kwargs)
        return exp_conf


stored=StoredConfRegistry()

