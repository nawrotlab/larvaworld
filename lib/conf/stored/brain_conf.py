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
# CT = null_dict('interference')
# CTtemp={k:None for k,v in CT.items()}
# CT_exclusive = CTtemp.update({'mode':'default', 'attenuation':0.0})
# CT_continuous = CTtemp.update({'mode':'default', 'attenuation':1.0})
# CT_constant = CTtemp.update({'mode':'default', 'attenuation':0.25})
# CT_phasic = CTtemp.update({'mode':'phasic', 'attenuation_min':0.2, 'attenuation_max':0.31, 'max_attenuation_phase':2.4})

CT_exclusive = null_dict('interference', mode='default', attenuation=0.0)
CT_continuous=null_dict('interference', mode='default', attenuation=1.0)
CT_constant=null_dict('interference', mode='default', attenuation=0.25)
CT_phasic = null_dict('interference', mode='phasic', attenuation_min=0.4, attenuation_max=0.2, max_attenuation_phase=2.4)

CT_dict={
    'exclusive' : CT_exclusive,
    'continuous' : CT_continuous,
    'constant' : CT_constant,
    'phasic' : CT_phasic,
}


T = null_dict('turner')
# Ttemp={k:None for k,v in T.items()}

# Tcon = Ttemp.update({'mode':'constant', 'initial_amp':0.964, 'noise':0.1, 'activation_noise':0.1})
# Tcon_no_noise = Ttemp.update({'mode':'constant', 'initial_amp':0.964, 'noise':0.0, 'activation_noise':0.0})
# Turner module
Tcon_no_noise = null_dict('turner',
                 mode='constant',
                 initial_amp=0.964,
                 amp_range=None,
                 initial_freq=None,
                 freq_range=None,
                 noise=0.0,
                 activation_noise=0.0,
                 base_activation=None,
                 activation_range=None
                 )

Tno_noise = null_dict('turner', activation_noise=0.0, noise=0.0)
T_Sak = null_dict('turner', activation_noise=0.0, noise=0.0, base_activation=24)

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
    'Sak': T_Sak,
    'sinusoidal': Tsin,
    'sinusoidal*': Tsin_no_noise,
    # 'constant': Tcon,
    'constant*': Tcon_no_noise,
}

# Crawler module
C = null_dict('crawler')
C_no_noise = null_dict('crawler', noise=0.0)
C_Sak = null_dict('crawler', noise=0.0, initial_freq = 1.37, freq_std=0.0,#freq_std=0.18,
                  step_to_length_mu=0.24, step_to_length_std=0.07, max_vel_phase=3.6)

# Ctemp={k:None for k,v in C.items()}
# Ccon = Ctemp.update({'waveform':'constant', 'initial_amp':0.323, 'noise':0.1})
Ccon = null_dict('crawler', waveform='constant', initial_amp=0.323)

# Ccon_no_noise = Ctemp.update({'waveform':'constant', 'initial_amp':0.323, 'noise':0.0})
Ccon_no_noise = null_dict('crawler', waveform='constant', initial_amp=0.323, noise=0.0)

C_dict = {
    'default': C,
    'default*': C_no_noise,
    'Sak': C_Sak,
    'constant': Ccon,
    'constant*': Ccon_no_noise,
}

# Intermittency module
Im = null_dict('intermitter')
Im_Levy = null_dict('intermitter', run_mode = 'run', stridechain_dist = None,
                    run_dist ={'range': [0.44, 133.0], 'name': 'powerlaw','alpha': 1.53},
                    pause_dist = {'range': [0.125, 16], 'name': 'uniform'})
Im_Davies = null_dict('intermitter', run_mode = 'run', stridechain_dist = None,
                    run_dist ={'range': [0.44, 133.0],'name': 'exponential', 'beta': 0.148},
                    pause_dist = {'range': [0.125, 16],'name': 'exponential','beta': 2.0})
Im_Sak = null_dict('intermitter', run_mode = 'stridechain',run_dist =None,
                   stridechain_dist ={'range': [1.0, 178.0], 'name': 'exponential', 'beta': 0.13784},
                   pause_dist ={'range': [0.125, 16], 'name': 'lognormal','mu': -0.2269, 'sigma': 0.96917}
                   )
Im_sampled = null_dict('intermitter')
Im_branch=null_dict('intermitter', mode='branch')
Im_nengo=null_dict('intermitter', mode='nengo')

Im_dict = {
    'default': Im,
    'Davies': Im_Davies,
    'Levy': Im_Levy,
    'Sak': Im_Sak,
    'stochastic': Im,
    'sampled': Im_sampled,
    'branch': Im_branch,
    'nengo': Im_nengo,
}

loco_combs={
    'Levy' : ['constant*', 'constant*','exclusive',  'Levy'],
    'Wystrach_2016' : ['constant*', 'neural*','continuous',  None],
    'Davies_2015' : ['constant*', 'constant*','constant',  'Davies'],
    'Sakagiannis_2022' : ['Sak', 'Sak','phasic',  'Sak']
}

# def build_loco_offline(c,t,ct,im) :
#     kws={f'{m}_params': mps for mps, m in zip([c,t,ct,im], ['crawler', 'turner', 'interference', 'intermitter'])}
#     if c is not None :
#         B.modules.crawler=True
#         B.crawler_params=C_dict[c]
#     if t is not None:
#         B.modules.turner=True
#         B.turner_params=T_dict[t]
#     if ct is not None:
#         B.modules.interference=True
#         B.interference_params=CT_dict[ct]
#     if im is not None:
#         B.modules.intermitter = True
#         B.intermitter_params = Im_dict[im]
#     L=null_dict('locomotor', **kws)
#     return L

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
    for k,v in B.modules.items() :
        if not v :
            B[f'{k}_params']=None
    # B['bend_correction_coef']=1
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
    from lib.model.modules.locomotor import DefaultLocomotor
    # for k, v in loco_dict.items() :
    # #     print(v)
    # # raise
    # for k, v in loco_dict.items():
    #     L = DefaultLocomotor(dt=0.1, conf=v)
    #     for i in range(5000):
    #         v, fov, feed = L.step(length=0.004)
    # raise


    # print(kConfDict('Brain'))
    from lib.conf.stored.conf import saveConf, loadRef
    # print(loadConf('None.200_controls', 'Ref').bout_distros.stride.keys())

    for k, v in loco_dict.items():
        # print(v.intermitter_params.mode)
        saveConf(v, 'Brain', k)

    # sim_brain(kConfDict('Brain'))