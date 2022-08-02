import copy
import json
import pickle
import shutil

import param

import lib.aux.dictsNlists as dNl
from lib.registry import paths


def loadConf(id, conf_type, **kwargs):
    try:
        conf_dict = loadConfDict(conf_type, **kwargs)
        conf = conf_dict[id]
        return dNl.NestDict(conf)
    except:
        raise ValueError(f'{conf_type} Configuration {id} does not exist')


def expandConf(id, conf_type, **kwargs):
    conf = loadConf(id, conf_type, **kwargs)
    try:
        if conf_type == 'Batch':
            conf.exp = expandConf(conf.exp, 'Exp', **kwargs)
        elif conf_type == 'Exp':
            conf.experiment = id
            conf.env_params = expandConf(conf.env_params, 'Env', **kwargs)
            conf.trials = loadConf(conf.trials, 'Trial', **kwargs)
            for k, v in conf.larva_groups.items():
                if type(v.model) == str:
                    v.model = loadConf(v.model, 'Model', **kwargs)
        elif conf_type == 'Ga':
            conf.experiment = id
            conf.env_params = expandConf(conf.env_params, 'Env', **kwargs)
    except:
        pass
    return conf


def loadConfDict(conf_type, use_pickle=False):
    path = paths.path_dict[conf_type]
    if conf_type == 'Ga':
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


def kConfDict(conf_type, **kwargs):
    return list(loadConfDict(conf_type, **kwargs).keys())





def loadRef(id, verbose=1):
    from lib.stor.larva_dataset import LarvaDataset
    d= LarvaDataset(loadConf(id, 'Ref')['dir'], load_data=False)
    if verbose >= 1:
        print(f'Loaded stored reference dataset : {id}')
    return d

def copyConf(id, conf_type):
    return dNl.NestDict(copy.deepcopy(expandConf(id, conf_type)))


def saveConf(conf, conf_type, id=None, mode='overwrite', verbose=1, **kwargs):
    try:
        d = loadConfDict(conf_type, **kwargs)
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
    saveConfDict(d, conf_type, **kwargs)
    if verbose >= 1:
        print(f'{conf_type} Configuration saved under the id : {id}')


def saveConfDict(ConfDict, conf_type, use_pickle=False):
    # from lib.conf.pars.pars import ParDict
    # path = ParDict.path_dict[conf_type]
    path = paths.path_dict[conf_type]
    if conf_type == 'Ga':
        use_pickle = True
    if use_pickle:
        with open(path, 'wb') as fp:
            pickle.dump(ConfDict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "w") as f:
            json.dump(ConfDict, f)


def deleteConf(id, conf_type,**kwargs):
    if conf_type == 'Data':
        DataGroup = loadConf(id, conf_type)
        path = DataGroup['path']
        try:
            shutil.rmtree(path)
        except:
            pass
    d = loadConfDict(conf_type)
    try:
        d.pop(id, None)
        saveConfDict(d, conf_type)
        print(f'Deleted {conf_type} configuration under the id : {id}')
    except:
        pass


def next_idx(exp, type='Exp'):
    # from lib.conf.pars.pars import ParDict
    # F0 = ParDict.path_dict["SimIdx"]
    F0 = paths.path_dict["SimIdx"]
    try:
        with open(F0) as f:
            d = json.load(f)
    except:
        ksExp = kConfDict('Exp')
        ksBatch = kConfDict('Batch')
        ksEssay = kConfDict('Essay')
        ksGA = kConfDict('Ga')
        ksEval = kConfDict('Exp')
        dExp = dict(zip(ksExp, [0] * len(ksExp)))
        dBatch = dict(zip(ksBatch, [0] * len(ksBatch)))
        dEssay = dict(zip(ksEssay, [0] * len(ksEssay)))
        dGA = dict(zip(ksGA, [0] * len(ksGA)))
        dEval = dict(zip(ksEval, [0] * len(ksEval)))
        # batch_idx_dict.update(loadConfDict('Batch'))
        d = {'Exp': dExp,
             'Batch': dBatch,
             'Essay': dEssay,
             'Eval': dEval,
             'Ga': dGA}
    if not type in d.keys():
        d[type] = {}
    if not exp in d[type].keys():
        d[type][exp] = 0
    d[type][exp] += 1
    with open(F0, "w") as fp:
        json.dump(d, fp)
    return d[type][exp]



def modshort(vv):
    mm = vv.brain.modules
    module_dict = {
        'T': 'turner',
        'C': 'crawler',
        'If': 'interference',
        'Im': 'intermitter',
        'O': 'olfactor',
        'To': 'toucher',
        'W': 'windsensor',
        'F': 'feeder',
        'M': 'memory',
    }
    mms = [k for k, v in module_dict.items() if mm[v]]
    pairs = [(['T', 'C', 'If', 'Im'], 'L'), (['L', 'O', 'F'], 'LOF'), (['L', 'O'], 'LO'), (['L', 'F'], 'LF'),
             (['L', 'W'], 'LW'),
             (['LOF', 'M'], 'LOFM'), (['L', 'To', 'M'], 'LToM'), (['L', 'To'], 'LTo'), (['T', 'C', 'If'], 'T:C'), ]
    for (ls, l0) in pairs:
        if all([k in mms for k in ls]):
            for k in ls:
                mms.remove(k)
            mms.append(l0)
    null_Box2D_params = {
        'joint_types': {
            'friction': {'N': 0, 'args': {}},
            'revolute': {'N': 0, 'args': {}},
            'distance': {'N': 0, 'args': {}}
        }
    }

    if vv.Box2D_params != null_Box2D_params:
        mms = ['B'] + mms
    if vv.brain.nengo:
        mms = ['N'] + mms

    if vv.energetics is not None:
        mms.append('E')

    def joinStrings(stringList):
        return ''.join(string for string in stringList)

    fmm = joinStrings(mms)
    return fmm


def imitation_exp(sample, model='explorer', idx=0, N=None, duration=None, imitation=True, **kwargs):
    from lib.registry.pars import preg
    I = preg.init_dict

    sample_conf = preg.loadConf(id=sample, conftype='Ref')

    # env_params = null_dict('env_conf', arena=sample_conf.env_params.arena)
    base_larva = preg.expandConf(id=model, conftype='Model')
    if imitation:
        exp = 'imitation'
        larva_groups = {
            'ImitationGroup': preg.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                                        imitation=True,
                                        distribution={'N': N})}
    else:
        exp = 'evaluation'
        larva_groups = {
            sample: I.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                              imitation=False,
                              distribution={'N': N})}
    id = sample_conf.id

    if duration is None:
        duration = sample_conf.duration / 60
    sim_params = I.get_null('sim_params', timestep=1 / sample_conf['fr'], duration=duration,
                           path=f'single_runs/{exp}', sim_ID=f'{id}_{exp}_{idx}')
    env_params = sample_conf.env_params
    exp_conf = I.get_null('exp_conf', sim_params=sim_params, env_params=env_params, larva_groups=larva_groups,
                         trials={}, enrichment=I.base_enrich())
    exp_conf['experiment'] = exp
    exp_conf.update(**kwargs)
    return exp_conf
