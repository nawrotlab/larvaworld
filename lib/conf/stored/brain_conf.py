'''
Define and store configurations of larva brains and their modules
'''

import copy

import numpy as np
import matplotlib.pyplot as plt

from lib.anal.argparsers import update_exp_conf
from lib.aux.dictsNlists import AttrDict
from lib.conf.base.dtypes import null_dict




from lib.conf.stored.conf import loadConf, kConfDict
from lib.sim.single.single_run import SingleRun

# Crawl-bend interference
CT_exclusive = null_dict('interference', crawler_phi_range=(0.0,2.0), attenuation=0.0)
CT_continuous=null_dict('interference', crawler_phi_range=(0.0,2.0), attenuation=1.0)
CT_constant=null_dict('interference', crawler_phi_range=(0.0,2.0), attenuation=0.5)
CT_phasic = null_dict('interference', crawler_phi_range=(0.5,1.0), attenuation=0.2)

CT_dict={
    'exclusive' : CT_exclusive,
    'continuous' : CT_continuous,
    'constant' : CT_constant,
    'phasic' : CT_phasic,
}

# Turner module
Tcon = null_dict('turner',
                 mode='constant',
                 initial_amp=20.0,
                 amp_range=[-20.0, 20.0],
                 initial_freq=None,
                 freq_range=None,
                 noise=0.0,
                 activation_noise=0.0,
                 base_activation=None,
                 activation_range=None
                 )

T = null_dict('turner')

Tno_noise = null_dict('turner', activation_noise=0.0, noise=0.0)

Tsin = null_dict('turner',
                 mode='sinusoidal',
                 initial_amp=15.0,
                 amp_range=[0.0, 50.0],
                 initial_freq=0.3,
                 freq_range=[0.1, 1.0],
                 noise=0.15,
                 activation_noise=0.5,
                 )
Tsin_no_noise = null_dict('turner',
                          mode='sinusoidal',
                          initial_amp=15.0,
                          amp_range=[0.0, 50.0],
                          initial_freq=0.3,
                          freq_range=[0.1, 1.0],
                          noise=0.0,
                          activation_noise=0.0,
                          )

T_dict = {
    'neural': T,
    'neural*': Tno_noise,
    'sinusoidal': Tsin,
    'sinusoidal*': Tsin_no_noise,
    'constant': Tcon,
}

# Crawler module
C = null_dict('crawler')
C_no_noise = null_dict('crawler', noise=0.0)
Ccon = null_dict('crawler', waveform='constant', initial_amp=0.2)

Ccon_no_noise = null_dict('crawler', waveform='constant', initial_amp=0.2, noise=0.0)

C_dict = {
    'default': C,
    'default*': C_no_noise,
    'constant': Ccon,
    'constant*': Ccon_no_noise,
}

# Intermittency module
Im = null_dict('intermitter')
Im_sampled = null_dict('intermitter')
Im_branch=null_dict('intermitter', mode='branch')
Im_nengo=null_dict('intermitter', mode='nengo')

Im_dict = {
    'default': Im,
    'stochastic': Im,
    'sampled': Im_sampled,
    'branch': Im_branch,
    'nengo': Im_nengo,
}

loco_combs={
    'Levy' : ['constant*', 'constant','exclusive',  'default'],
    'Wystrach_2016' : ['constant*', 'neural*','continuous',  None],
    'Davies_2015' : ['constant*', 'constant','constant',  'default'],
    'Sakagiannis_2022' : ['default*', 'neural*','phasic',  'default']
}

def build_loco(c,t,ct,im, B=None) :
    if B is None :
        B=null_dict('brain')
    else :
        B=copy.deepcopy(B)
    if c is not None :
        B.modules.crawler=True
        B.crawler_params=C_dict[c]
    if t is not None:
        B.modules.turner=True
        B.turner_params=T_dict[t]
    if ct is not None:
        B.modules.interference=True
        B.interference_params=CT_dict[ct]
    if im is not None:
        B.modules.intermitter = True
        B.intermitter_params = Im_dict[im]
    return B

loco_dict={k:build_loco(*v) for k, v in loco_combs.items()}


base_coupling = null_dict('interference', crawler_phi_range=(0.45, 1.0), feeder_phi_range=(0.0, 0.0), attenuation=0.1)

RL_olf_memory = null_dict('memory', Delta=0.1, state_spacePerSide=1, modality='olfaction',
                          gain_space=np.arange(-200.0, 200.0, 50.0).tolist())



def brain(module_shorts, nengo=False, OD=None, **kwargs):
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
    if 'L' in module_shorts:
        module_shorts.remove('L')
        module_shorts += ['T', 'C', 'If', 'Im']
    elif 'LOF' in module_shorts:
        module_shorts.remove('LOF')
        module_shorts += ['T', 'C', 'If', 'Im', 'O', 'F']
    # module_shorts.append('W')
    modules = [module_dict[k] for k in module_shorts]

    modules = null_dict('modules', **{m: True for m in modules})
    d = {'modules': modules}
    for k, v in modules.items():
        p = f'{k}_params'
        if not v:
            d[p] = None
        elif k in list(kwargs.keys()):
            d[p] = kwargs[k]
        elif k == 'interference':
            d[p] = base_coupling
        elif k == 'memory':
            d[p] = RL_olf_memory
        else:
            d[p] = null_dict(k)
        if k == 'olfactor' and d[p] is not None:
            d[p]['odor_dict'] = OD
    d['nengo'] = nengo
    return AttrDict.from_nested_dicts(d)




if __name__ == '__main__':
    # print(kConfDict('Brain'))
    from lib.conf.stored.conf import saveConf, loadRef
    print(loadConf('None.200_controls', 'Ref').bout_distros.stride.keys())

    for k, v in loco_dict.items():
        # print(v.intermitter_params.mode)
        saveConf(v, 'Brain', k)

    # sim_brain(kConfDict('Brain'))