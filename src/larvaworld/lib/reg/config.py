import os

import numpy as np
import param

import larvaworld
from larvaworld.lib import reg, aux, util

Path = aux.AttrDict({k : f'{reg.CONF_DIR}/{k}.txt' for k in reg.CONFTYPES})





def build_ConfTypeSubkeys():
    d0 = {k: {} for k in reg.CONFTYPES}
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
    d0 = {k: {} for k in reg.GROUPTYPES}
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


CONFTREE = aux.AttrDict({k : aux.load_dict(Path[k]) for k in reg.CONFTYPES})
#
# def build_conf_tree_expanded():
#     c0 = aux.AttrDict({k : aux.load_dict(Path[k]) for k in reg.CONFTYPES})
#     sk = CONFTYPE_SUBKEYS
#     for confType0 in c0.keys():
#         if confType0 in sk.keys():
#             pairs = sk[confType0]
#             for id, conf in c0[confType0].items():
#                 for subID, confType in pairs.items():
#
#
#                     if subID in conf.keys():
#                         if isinstance(conf[subID], str) and conf[subID] in c0[confType].keys():
#                             conf[subID]=c0[confType][conf[subID]]
#                         elif (subID, confType) == ('larva_groups', 'Model'):
#                             for gID, gConf in conf[subID].items():
#                                 mID=gConf.model
#                                 if mID in c0['Model'].keys():
#                                     gConf.model=c0['Model'][mID]
#                                 else:
#                                     # print(f'{mID} not found')
#                                     pass
#                                     # raise
#     return c0
#
# CONFTREE_EXPANDED = build_conf_tree_expanded()


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
        m = mID if not expand else reg.conf.Model.getID(mID)
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
        self.group=aux.AttrDict({k: BaseType(k=k) for k in reg.GROUPTYPES})
        self.conf=aux.AttrDict({k: BaseType(k=k) for k in reg.CONFTYPES})

    # def ConfPath(self, conftype):
    #     return Path[conftype]
    #
    # def get_dict(self, conftype):
    #     path=self.ConfPath(conftype)
    #     return aux.load_dict(path)
    #
    # def set_dict(self, conftype, d):
    #     path = self.ConfPath(conftype)
    #     aux.save_dict(d, path)

    # def resetDict(self,conftype, init=False):
    #     dd = reg.funcs.stored_confs[conftype]()
    #
    #     if os.path.isfile(self.ConfPath(conftype)):
    #         if init:
    #             return
    #         else:
    #             d = self.get_dict(conftype)
    #     else:
    #         d = {}
    #
    #     N0, N1 = len(d), len(dd)
    #
    #     d.update(dd)
    #
    #     Ncur = len(d)
    #     Nnew = Ncur - N0
    #     Nup = N1 - Nnew
    #     self.set_dict(conftype, d)
    #     reg.vprint(f'{conftype}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing', 1)
    #
    # def resetConfs(self, conftypes=None, **kwargs):
    #     if conftypes is None:
    #         conftypes = reg.CONFTYPES
    #
    #     for conftype in conftypes:
    #         self.resetDict(conftype, **kwargs)

    # def confIDs(self, conftype):
    #     d=self.get_dict(conftype)
    #     return sorted(list(d.keys()))




    # def get(self, conftype, id):
    #     d=self.get_dict(conftype)
    #     if id in d.keys():
    #         return aux.AttrDict(d[id])
    #     else:
    #         reg.vprint(f'{conftype} Configuration {id} does not exist', 1)
    #         raise ValueError()

    # def set(self, conftype, id, conf, mode='overwrite'):
    #     d=self.get_dict(conftype)
    #     if id in d.keys() and mode == 'update':
    #         d[id] = d[id].update_nestdict(conf.flatten())
    #     else:
    #         d[id] = aux.AttrDict(conf)
    #     self.set_dict(conftype, d)
    #     reg.vprint(f'{conftype} Configuration saved under the id : {id}', 1)

    # def delete(self, conftype, id=None):
    #     if id is not None:
    #         d=self.get_dict(conftype)
    #         if id in d.keys():
    #             d.pop(id, None)
    #             self.set_dict(conftype, d)
    #             reg.vprint(f'Deleted {conftype} configuration under the id : {id}', 1)




    # def getRefDir(self, id):
    #     d = self.get_dict('Ref')
    #     if id in d.keys():
    #         return d[id]
    #     else:
    #         reg.vprint(f'Reference dataset with ID {id} does not exist. Returning None', 1)
    #         return None
    #
    # def getRef(self, id=None, dir=None):
    #     if dir is None:
    #         dir=self.getRefDir(id)
    #     if dir is not None:
    #         path = f'{dir}/data/conf.txt'
    #         if os.path.isfile(path):
    #             c = aux.load_dict(path)
    #             if 'id' in c.keys():
    #                 reg.vprint(f'Loaded existing conf {c.id}', 1)
    #                 return c
    #     return None
    #
    # def loadRef(self, id, load=False, **kwargs):
    #     c=self.getRef(id)
    #     if c is not None:
    #         d = larvaworld.LarvaDataset(config=c, load_data=False)
    #         if load:
    #             d.load(**kwargs)
    #         reg.vprint(f'Loaded stored reference dataset : {id}', 1)
    #         return d
    #     else:
    #         reg.vprint(f'Failed to load reference dataset : {id}', 1)
    #         return None







def imitation_exp(refID, model='explorer', **kwargs):
    c = reg.getRef(refID)

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

