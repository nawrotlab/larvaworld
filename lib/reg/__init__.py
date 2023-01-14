VERBOSE =2

def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)
vprint("Initializing larvaworld registry", 2)


vprint("Initializing path registry", 0)
from .paths import *
from .data_structure import datapath, datafunc


vprint("Initializing output registry", 0)
from .output import output_dict,set_output, get_reporters
from .units import units

vprint("Initializing function registry", 0)
from .facade import funcs
from .parFunc import *
from .stored import *
from .distro import distro_database,get_dist

vprint("Initializing parameter registry", 0)
from .parDB import par

vprint("Initializing configuration registry", 0)
from .config import conf0, group, CONFTREE, CONFTREE_EXPANDED, loadConf, saveConf, deleteConf, storedConf, expandConf, lgs, lg
from .controls import controls

vprint("Initializing model registry", 0)
from .models import model
from .parser import parsers

vprint("Initializing graph registry", 0)
from .graph import graphs

vprint("Registry configured!", 2)


def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)

def get_null(name, **kwargs):
    return par.get_null(name=name, **kwargs)






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
    dic = dNl.load_dict(Path.Ref)
    if id in dic.keys():
        return dic[id]
    else:
        vprint(f'Ref Configuration {id} does not exist. Returning None', 1)
        return None


def saveRef(id, conf):
    path = Path.Ref
    dic = dNl.load_dict(path)
    dic[id] = conf
    dNl.save_dict(dic, path)

def deleteRef(id):
    import shutil
    path = Path.Ref
    dic = dNl.load_dict(path)
    if id in dic.keys():
        shutil.rmtree(dic[id].dir,ignore_errors=True)
        dic.pop(id,None)
        vprint(f'Deleted Ref Configuration {id}')
        dNl.save_dict(dic, path)

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




