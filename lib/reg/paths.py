import os

from lib.aux import dictsNlists as dNl

def get_parent_dir():
    import os
    p=os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    return p

ROOT_DIR=get_parent_dir()

DATA_DIR = f'{ROOT_DIR}/data'
SIM_DIR = f'{DATA_DIR}/SimGroup'
RUN_DIR = f'{SIM_DIR}/single_runs'
BATCH_DIR = f'{SIM_DIR}/batch_runs'
ESSAY_DIR = f'{SIM_DIR}/essays'
DEB_DIR = f'{SIM_DIR}/deb_runs'

CONF_DIR = f'{ROOT_DIR}/lib/reg/confDicts'

GLOSSARY_PATH = f'{CONF_DIR}/glossary.txt'
CONTROLS_PATH = f'{CONF_DIR}/controls.txt'
SimIdx_PATH = f'{CONF_DIR}/SimIdx.txt'


CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']

GROUPTYPES = ['LarvaGroup', 'SourceGroup', 'epoch']

Path = dNl.AttrDict({k : f'{CONF_DIR}/{k}.txt' for k in CONFTYPES})




def next_idx(id, conftype='Exp'):
    f = SimIdx_PATH
    if not os.path.isfile(f):
        d = dNl.AttrDict({k: dNl.AttrDict() for k in ['Exp', 'Batch', 'Essay', 'Eval', 'Ga']})
    else:
        d = dNl.load_dict(f)

    if not conftype in d.keys():
        d[conftype] = {}
    if not id in d[conftype].keys():
        d[conftype][id] = 0
    d[conftype][id] += 1
    dNl.save_dict(d, f)
    return d[conftype][id]







