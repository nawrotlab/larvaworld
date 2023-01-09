import os

from lib.aux import dictsNlists as dNl, naming as nam

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

Path = dNl.NestDict({k : f'{CONF_DIR}/{k}.txt' for k in CONFTYPES})



# def build_conf_tree(ks=CONFTYPES, paths=Path):
#     dd = dNl.NestDict()
#     for k in ks:
#         try:
#             d = dNl.load_dict2(paths[k])
#         except:
#             d= {}
#         dd[k] = d
#     return dd
#
# CONFTREE = build_conf_tree()






#
#
# def buildSampleDic():
#     d =dNl.NestDict(
#         {
#             'length': 'body.initial_length',
#             nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
#             'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
#             nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
#             nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
#             nam.freq('feed'): 'brain.feeder_params.initial_freq',
#             nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler_params.max_scaled_vel',
#             'phi_scaled_velocity_max': 'brain.crawler_params.max_vel_phase',
#             'attenuation': 'brain.interference_params.attenuation',
#             'attenuation_max': 'brain.interference_params.attenuation_max',
#             nam.freq(nam.vel(nam.orient(('front')))): 'brain.turner_params.initial_freq',
#             nam.max('phi_attenuation'): 'brain.interference_params.max_attenuation_phase',
#         }
#     )
#     return dNl.bidict(d)
SAMPLING_PARS = dNl.bidict(
    dNl.NestDict(
        {
            'length': 'body.initial_length',
            nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
            'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
            nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
            nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
            nam.freq('feed'): 'brain.feeder_params.initial_freq',
            nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler_params.max_scaled_vel',
            'phi_scaled_velocity_max': 'brain.crawler_params.max_vel_phase',
            'attenuation': 'brain.interference_params.attenuation',
            'attenuation_max': 'brain.interference_params.attenuation_max',
            nam.freq(nam.vel(nam.orient(('front')))): 'brain.turner_params.initial_freq',
            nam.max('phi_attenuation'): 'brain.interference_params.max_attenuation_phase',
        }
    )
)
# SAMPLING_PARS = buildSampleDic()

def next_idx(id, conftype='Exp'):
    f = SimIdx_PATH
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







