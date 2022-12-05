import os
import time

import numpy as np

from lib.aux import dictsNlists as dNl

VERBOSE = 2


def init_conf():
    from lib.registry.paths import AllConfDict, ExpandedConfDict, build_path_dict, buildSampleDic
    global conf
    conf = AllConfDict()
    global conF
    conF = ExpandedConfDict()

    global Path
    global SampleDic

    Path = build_path_dict()
    SampleDic = buildSampleDic()


def init_data_structure():
    from lib.registry.paths import build_datapath_structure
    global datafunc_dict
    global datapath_dict
    datapath_dict, datafunc_dict = build_datapath_structure()


def reg_dict():

    from lib.registry import conf, distro,models, params,param_definition, parsers,graphs,processing

    d = dNl.NestDict({
        'CT': conf.ConfTypeDict,
        'DD': distro.DistributionRegistry,
        'MD': models.ModelRegistry,
        'PI': param_definition.ParInitDict,
        'DEF': param_definition.ParDefaultDict,
        'GT': conf.GroupTypeDict,
        'GD': graphs.GraphRegistry,
        'parsers': parsers.ParserDict,
        'PF': processing.ProcFuncDict,
        'PD': params.ParamRegistry,
    })
    return d


def init_dicts(ks=None):
    global DicF
    DicF = reg_dict()
    all_ks = list(DicF.keys())

    global Dic
    Dic = dNl.NestDict({kk: None for kk in all_ks})

    load_mode = {}
    # load_mode = {'DEF': {'mode': 'load'}}
    if ks is None:
        ks = all_ks
    for k in ks:

        if k in load_mode.keys():
            kws = load_mode[k]
        else:
            kws = {}
        init_Dic(k, D=Dic, **kws)


def init(ks=None):
    vprint(f'Initializing larvaworld registry', 3)
    init_conf()
    init_data_structure()
    init_dicts(ks=ks)
    vprint(f'Completed larvaworld registry', 3)


def init_Dic(k, D=None, **kws):
    if D is None:
        D = globals()['Dic']
    if D[k] is None or k not in globals():
        D[k] = DicF[k](**kws)
        globals()[k] = D[k]
    return D[k]


def getPar(k=None, p=None, d=None, to_return='d'):
    dd = init_Dic('PD')
    return dd.getPar(k=k, d=d, p=p, to_return=to_return)

def getParDic(k):
    dd = init_Dic('PI')
    return dd.dict[k]


def next_idx(id, conftype='Exp'):
    f = Path.SimIdx
    if not os.path.isfile(f):
        d = dNl.NestDict({k: dNl.NestDict() for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']})
    else:
        d = dNl.load_dict(f, use_pickle=False)

    if not conftype in d.keys():
        d[conftype] = {}
    if not id in d[conftype].keys():
        d[conftype][id] = 0
    d[conftype][id] += 1
    dNl.save_dict(d, f, use_pickle=False)
    return d[conftype][id]


def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)


def datafunc(filepath_key, mode='load'):
    if 'datafunc_dict' not in globals():
        init_data_structure()
    DD = datafunc_dict
    if filepath_key in DD.keys():
        return DD[filepath_key][mode]
    else:
        return None


def datapath(filepath_key, dir=None):
    if 'datapath_dict' not in globals():
        init_data_structure()
    DD = datapath_dict
    if dir is not None and filepath_key in DD.keys():
        return f'{dir}{DD[filepath_key]}'
    else:
        return None


def lgs(**kwargs):
    d = init_Dic('GT')
    return d.dict.LarvaGroup.lgs(**kwargs)

def lg(**kwargs):
    d = init_Dic('GT')
    return d.dict.LarvaGroup.lg_entry(**kwargs)


def get_null(name, **kwargs):
    d = init_Dic('DEF')
    return d.get_null(name=name, **kwargs)


def loadRef(id, load=False, **kwargs):
    c = retrieveRef(id)
    if c is not None:
        from lib.stor.larva_dataset import LarvaDataset
        d = LarvaDataset(c.dir, load_data=False)
        if not load:
            vprint(f'Loaded stored reference configuration : {id}')
            return d
        else:
            d.load(**kwargs)
            vprint(f'Loaded stored reference dataset : {id}')
            return d

    else:
        # self.vprint(f'Ref Configuration {id} does not exist. Returning None')
        return None

def loadRefD(id, **kwargs):
    return loadRef(id, load=True, **kwargs)


def loadRefDs(ids, **kwargs):
    ds = [loadRefD(id, **kwargs) for id in ids]
    return ds



def retrieveRef(id):
    dic = dNl.load_dict(Path.Ref, use_pickle=False)
    if id in dic.keys():
        return dic[id]
    else:
        vprint(f'Ref Configuration {id} does not exist. Returning None', 1)
        return None


def saveRef(id, conf):
    path = Path.Ref
    dic = dNl.load_dict(path, use_pickle=False)
    dic[id] = conf
    dNl.save_dict(dic, path, use_pickle=False)

def deleteRef(id):
    import shutil
    path = Path.Ref
    dic = dNl.load_dict(path, use_pickle=False)
    if id in dic.keys():
        shutil.rmtree(dic[id].dir,ignore_errors=True)
        dic.pop(id,None)
        vprint(f'Deleted Ref Configuration {id}')
        dNl.save_dict(dic, path, use_pickle=False)

def testRef(id):
    from lib.aux.stor_aux import read
    config = retrieveRef(id)
    if config is not None:
        D = config.dir_dict
        dic={}
        for k, d in D.items():
            if d.endswith('.h5') and os.path.exists(d):
                try :
                    t0=time.time()
                    read(d, key=k)
                    dic[k]=np.round(time.time()-t0,2)
                except :
                    dic[k]='FAIL'
        # if k not in D1.keys() :
        print(f'------- Loading times for {id}---------------------')
        print(dic)
        print()


def simRef(id, mID, **kwargs):
    from lib.aux.sample_aux import sim_model
    return sim_model(mID,  refID=id, **kwargs)

def resetConfs(ks=None) :
    Dic.CT.resetConfs(ks=ks)

def loadConf(conftype, id=None):
    return Dic.CT.dict[conftype].loadConf(id=id)

def saveConf(conftype, id, conf):
    return Dic.CT.dict[conftype].saveConf(id=id, conf=conf)

def deleteConf(conftype, id=None):
    return Dic.CT.dict[conftype].deleteConf(id=id)

def expandConf(conftype, id=None):
    return Dic.CT.dict[conftype].expandConf(id=id)

def storedConf(conftype):
    return Dic.CT.dict[conftype].ConfIDs

def get_dist(**kwargs):
    d = init_Dic('DD')
    return d.get_dist(**kwargs)


