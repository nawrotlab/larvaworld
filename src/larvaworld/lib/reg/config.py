import os
import param
from larvaworld.lib import reg, aux, util, decorators


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


CONFTREE = aux.AttrDict({k : aux.load_dict(f'{reg.CONF_DIR}/{k}.txt') for k in reg.CONFTYPES})

def build_conf_tree_expanded():
    c0 = aux.AttrDict({k : aux.load_dict(f'{reg.CONF_DIR}/{k}.txt') for k in reg.CONFTYPES})
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

def update_mdict(mdict, mmdic):
    if mmdic is None or mdict is None:
        return None
    elif not isinstance(mmdic, dict) or not isinstance(mdict, dict):
        return mdict
    else:
        for d, p in mdict.items():
            new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(new_v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        new_v = tuple(new_v)
                p.v = new_v
            else:
                mdict[d] = update_mdict(mdict=p, mmdic=new_v)
        return mdict


def update_existing_mdict(mdict, mmdic):
    if mmdic is None:
        return mdict
    else:
        for d, v in mmdic.items():
            p = mdict[d]
            if isinstance(p, param.Parameterized):
                if type(v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        v = tuple(v)

                p.v = v
            elif isinstance(p, dict) and isinstance(v, dict):
                mdict[d] = update_existing_mdict(mdict=p, mmdic=v)
        return mdict

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


conf =aux.AttrDict({k: BaseType(k=k) for k in reg.CONFTYPES})
group =aux.AttrDict({k: BaseType(k=k) for k in reg.GROUPTYPES})

def loadConf(conftype, id):
    path = reg.Path[conftype]
    d = aux.load_dict(path)
    if id in d.keys():
        return aux.AttrDict(d[id])
    else:
        reg.vprint(f'{conftype} Configuration {id} does not exist',1)
        raise ValueError()

def saveConf(conftype, id, conf, mode='overwrite'):
    path=reg.Path[conftype]
    d = aux.load_dict(path)

    if id in d.keys() and mode == 'update':
        d[id] = d[id].update_nestdict(conf.flatten())
    else:
        d[id] = aux.AttrDict(conf)
    aux.save_dict(d, path)
    reg.vprint(f'{conftype} Configuration saved under the id : {id}', 1)


def deleteConf(conftype, id=None):
    if id is not None:
        path = reg.Path[conftype]
        d = aux.load_dict(path)
        if id in d.keys():
            d.pop(id, None)
            aux.save_dict(d, path)
            reg.vprint(f'Deleted {conftype} configuration under the id : {id}', 1)


def expandConf(conftype, id=None,conf=None):
    if conf is None:
        if id in storedConf(conftype):

            conf = loadConf(conftype, id)
        else :
            return None
    subks=CONFTYPE_SUBKEYS[conftype]
    if len(subks) > 0:
        for subID, subk in subks.items():
            ids=storedConf(subk)
            if subID == 'larva_groups' and subk == 'Model':
                for k, v in conf['larva_groups'].items():
                    if v.model in ids:
                        v.model = loadConf(subk,id=v.model)
            else:
                if conf[subID] in ids:
                    conf[subID] = loadConf(subk,id=conf[subID])

    return conf

def storedConf(conftype):
    path = reg.Path[conftype]
    d = aux.load_dict(path)
    return list(d.keys())

def resetDict(conftype):
    dd = reg.funcs.stored_confs[conftype]()
    path = reg.Path[conftype]
    d = aux.load_dict(path)

    N0, N1 = len(d), len(dd)

    d.update(dd)

    Ncur = len(d)
    Nnew = Ncur - N0
    Nup = N1 - Nnew
    aux.save_dict(d, path)

    reg.vprint(f'{conftype}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing',1)

@decorators.timeit
def resetConfs(conftypes=None):
    if conftypes is None:
        conftypes = reg.CONFTYPES

    for conftype in conftypes:
        resetDict(conftype)

def lgs(mIDs, ids=None, cs=None,**kwargs):

    if ids is None:
        ids = mIDs
    N = len(mIDs)
    if cs is None :
        cs = aux.N_colors(N)
    return aux.AttrDict(aux.merge_dicts([lg(id, c=c, mID=mID, **kwargs) for mID, c, id in zip(mIDs, cs, ids)]))


def lg(id=None, c='black', N=1, mode='uniform', sh='circle', loc=(0.0, 0.0), ors=(0.0, 360.0),
       s=(0.0, 0.0), mID='explorer',age=0.0, epochs={},  o=None,sample = None, expand=False, **kwargs):
    if id is None :
        id=mID
    m=mID if not expand else reg.loadConf(conftype="Model", id=mID)
    if type(s) == float:
        s = (s, s)
    kws = {'kwdic': {
        'distribution': {'N': N, 'scale': s, 'orientation_range': ors, 'loc': loc, 'shape': sh, 'mode': mode},
        'life_history': {'age': age,'epochs': epochs}
    },
           'default_color': c, 'model': m,'sample':sample,  **kwargs}
    if o is not None:
        kws['odor'] = o

    return group.LarvaGroup.entry(id=id, **kws)

def GTRvsS(N=1, age=72.0, q=1.0, h_starved=0.0, sample='None.150controls', substrate_type='standard',pref='',
                navigator=False, expand=False, **kwargs):
    if age==0.0 :
        epochs={}
    else :
        if h_starved==0:
            eps={
                0 : {'start': 0.0, 'stop' : age, 'substate':{'type': substrate_type, 'quality': q}}
            }
        else :
            eps = {
                0: {'start': 0.0, 'stop': age-h_starved, 'substate': {'type': substrate_type, 'quality': q}},
                1: {'start': age-h_starved, 'stop': age, 'substate': {'type': substrate_type, 'quality': 0}},
            }
        epochs={}
        for id,kws in eps.items():
            epochs.update(group.epoch.entry(id=id,**kws))

    kws0 = {
        'kwdic': {
            'distribution': {'N': N, 'scale': (0.005, 0.005)},
            'life_history': {'age': age,
                             'epochs': epochs,
                             },
        'odor':{}
        },
        'sample': sample,
    }

    mcols = ['blue', 'red']
    mID0s = ['rover', 'sitter']
    lgs = {}
    for mID0, mcol in zip(mID0s, mcols):
        id=f'{pref}{mID0.capitalize()}'



        if navigator :
            mID0=f'navigator_{mID0}'
        if expand:
            mID0=loadConf(conftype="Model", id=mID0)



        kws = {
            'default_color': mcol,
            'model': mID0,
            **kws0
        }

        lgs.update(group.LarvaGroup.entry(id, **kws))
    return aux.AttrDict(lgs)

def retrieveRef(id):
    return loadConf('Ref', id)

def loadRef(id, load=False, **kwargs):
    c = loadConf('Ref',id)
    if c is not None:
        from larvaworld.lib.process.dataset import LarvaDataset
        d = LarvaDataset(c.dir, load_data=False)
        if not load:
            reg.vprint(f'Loaded stored reference configuration : {id}')
            return d
        else:
            d.load(**kwargs)
            reg.vprint(f'Loaded stored reference dataset : {id}')
            return d

    else:
        return None

def loadRefD(id, **kwargs):
    return loadRef(id, load=True, **kwargs)


def loadRefDs(ids, **kwargs):
    ds = [loadRefD(id, **kwargs) for id in ids]
    return ds

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

def imitation_exp(sample, model='explorer', idx=0, N=None, duration=None, imitation=True, **kwargs):
    sample_conf = reg.loadConf(id=sample, conftype='Ref')

    base_larva = reg.expandConf(id=model, conftype='Model')
    if imitation:
        exp = 'imitation'
        larva_groups = {
            'ImitationGroup': reg.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                                        imitation=True,
                                        distribution={'N': N})}
    else:
        exp = 'evaluation'
        larva_groups = {
            sample: reg.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                              imitation=False,
                              distribution={'N': N})}
    id = sample_conf.id

    if duration is None:
        duration = sample_conf.duration / 60
    sim_params = reg.get_null('sim_params', timestep=1 / sample_conf['fr'], duration=duration)
    env_params = sample_conf.env_params
    exp_conf = reg.get_null('Exp', sim_params=sim_params, env_params=env_params, larva_groups=larva_groups,
                         trials={}, enrichment=reg.par.base_enrich())
    exp_conf['experiment'] = exp
    exp_conf.update(**kwargs)
    return exp_conf
