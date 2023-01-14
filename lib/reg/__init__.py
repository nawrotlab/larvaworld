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
from .config import conf0, group, CONFTREE, CONFTREE_EXPANDED, loadConf, saveConf, deleteConf, storedConf, expandConf, lgs, lg, loadRef, loadRefDs
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



