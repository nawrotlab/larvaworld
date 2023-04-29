import os
from os.path import dirname, abspath

VERBOSE =2
def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)
vprint("Initializing larvaworld registry", 2)


vprint("Initializing path registry", 0)

ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = f'{ROOT_DIR}/data'
SIM_DIR = f'{DATA_DIR}/SimGroup'
BATCH_DIR = f'{SIM_DIR}/batch_runs'
CONF_DIR = f'{ROOT_DIR}/lib/reg/confDicts'

CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']

GROUPTYPES = ['LarvaGroup', 'SourceGroup', 'epoch']

Path = {k : f'{CONF_DIR}/{k}.txt' for k in CONFTYPES}
os.makedirs(CONF_DIR, exist_ok=True)


vprint("Initializing function registry")
from .units import units
from .facade import funcs
from .parFunc import *
from .stored import *
from .distro import distro_database,get_dist

vprint("Initializing parameter registry")
from .parDB import output_keys, par

vprint("Initializing configuration registry")
from .config import conf, group, CONFTREE, CONFTREE_EXPANDED, loadConf, saveConf, deleteConf, storedConf,storedRefs,load_config,Ref_paths, expandConf,resetConfs,GTRvsS, lgs, lg, loadRef, retrieve_dataset, next_idx
from .controls import controls

vprint("Initializing model registry")
from .models import model

vprint("Initializing graph registry")
from .graph import graphs





def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)

def get_null(name, **kwargs):
    return par.get_null(name=name, **kwargs)



resetConfs(init=True)

vprint(f"Registry configured!", 2)