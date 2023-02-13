VERBOSE =2
def vprint(text='', verbose=0):
    if verbose >= VERBOSE:
        print(text)
vprint("Initializing larvaworld registry", 2)


vprint("Initializing path registry", 0)
from os.path import dirname, abspath
ROOT_DIR = dirname(dirname(dirname(abspath(__file__))))
DATA_DIR = f'{ROOT_DIR}/data'
SIM_DIR = f'{DATA_DIR}/SimGroup'
BATCH_DIR = f'{SIM_DIR}/batch_runs'
CONF_DIR = f'{ROOT_DIR}/lib/reg/confDicts'

CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']

GROUPTYPES = ['LarvaGroup', 'SourceGroup', 'epoch']

Path = {k : f'{CONF_DIR}/{k}.txt' for k in CONFTYPES}


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
from .config import conf, group, CONFTREE, CONFTREE_EXPANDED, loadConf, saveConf, deleteConf, storedConf, expandConf,resetConfs,GTRvsS, lgs, lg, retrieveRef, loadRef, loadRefDs, next_idx
from .controls import controls

vprint("Initializing model registry", 0)
from .models import model

vprint("Initializing graph registry", 0)
from .graph import graphs

vprint("Registry configured!", 2)


def getPar(k=None, p=None, d=None, to_return='d'):
    return par.getPar(k=k, d=d, p=p, to_return=to_return)

def get_null(name, **kwargs):
    return par.get_null(name=name, **kwargs)



