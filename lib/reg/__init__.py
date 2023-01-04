import os

VERBOSE =2

def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)
vprint("Initializing larvaworld registry", 2)

from lib.aux import dictsNlists as dNl
from .paths import ROOT_DIR, Path, SampleDic, datapath, datafunc, conftree


from .output import output_dict,set_output, get_reporters
from .units import units
from .facade import funcs
from .parFunc import *
from . import base
from .distro import distro_database,get_dist
from .parDB import par
from .config import conf, group
from .controls import controls
from .models import model
from .parser import parsers
from .graph import graphs
vprint("Registry configured!", 2)


def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)

def get_null(name, **kwargs):
    return par.get_null(name=name, **kwargs)


def lgs(**kwargs):
    # d = init_Dic('GT')
    return group.dict.LarvaGroup.lgs(**kwargs)

def lg(**kwargs):
    # d = init_Dic('GT')
    return group.dict.LarvaGroup.lg_entry(**kwargs)



def loadRef(id, load=False, **kwargs):
    c = retrieveRef(id)
    if c is not None:
        from lib.process.larva_dataset import LarvaDataset
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
    import os
    import time

    import numpy as np
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


# def simRef(id, mID, **kwargs):
#     from lib.aux.sample_aux import sim_model
#     return sim_model(mID,  refID=id, **kwargs)


def loadConf(conftype, id=None):
    return conf.dict[conftype].loadConf(id=id)

def saveConf(conftype, id, conf):
    return conf.dict[conftype].saveConf(id=id, conf=conf)

def deleteConf(conftype, id=None):
    return conf.dict[conftype].deleteConf(id=id)

def expandConf(conftype, id=None):
    return conf.dict[conftype].expandConf(id=id)

def storedConf(conftype):
    return conf.dict[conftype].ConfIDs


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
