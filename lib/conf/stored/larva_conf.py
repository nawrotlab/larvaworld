'''
The larva model parameters
'''

import numpy as np
from lib.conf.base.dtypes import null_dict

''' Default exploration model'''

Cbas = null_dict('crawler', initial_freq=1.5, step_to_length_mu=0.25, step_to_length_std=0.0)
base_coupling = null_dict('interference', crawler_phi_range=(0.45, 1.0), feeder_phi_range=(0.0, 0.0), attenuation=0.1)

Tsin = null_dict('turner',
                 mode='sinusoidal',
                 initial_amp=15.0,
                 amp_range=[0.0, 50.0],
                 initial_freq=0.3,
                 freq_range=[0.1, 1.0],
                 noise=0.15,
                 activation_noise=0.5,
                 )

Tno_noise = null_dict('turner', activation_noise=0.0, noise=0.0)

Ccon = null_dict('crawler', waveform='constant', initial_amp=0.0012)

RL_olf_memory = null_dict('memory', Delta=0.1, state_spacePerSide=1, mode='olf',
                          gain_space=np.arange(-200.0, 200.0, 50.0).tolist())

RL_touch_memory = null_dict('memory', Delta=0.5, state_spacePerSide=1, mode='touch', train_dur=30, update_dt=0.5,
                            gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=True)

gRL_touch_memory = null_dict('memory', Delta=0.5, state_spacePerSide=1, mode='touch', train_dur=30, update_dt=0.5,
                            gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=False)

OD1 = {'Odor': {'mean': 150.0, 'std': 0.0}}
OD2 = {'CS': {'mean': 150.0, 'std': 0.0}, 'UCS': {'mean': 0.0, 'std': 0.0}}


def Im(EEB):
    if EEB > 0:
        return null_dict('intermitter', feed_bouts=True, EEB=EEB)
    else:
        return null_dict('intermitter', feed_bouts=False, EEB=0.0)


def ImD(pau, str):
    return null_dict('intermitter', pause_dist=pau, stridechain_dist=str)


# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------


def brain(module_shorts, nengo=False, OD=None, **kwargs):
    module_dict = {
        'T': 'turner',
        'C': 'crawler',
        'If': 'interference',
        'Im': 'intermitter',
        'O': 'olfactor',
        'To': 'toucher',
        'F': 'feeder',
        'M': 'memory',
    }
    if 'L' in module_shorts:
        module_shorts.remove('L')
        module_shorts += ['T', 'C', 'If', 'Im']
    elif 'LOF' in module_shorts:
        module_shorts.remove('LOF')
        module_shorts += ['T', 'C', 'If', 'Im', 'O', 'F']
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
    return d


def RvsS_larva(EEB, Nsegs=2, mock=False, hunger_gain=1.0, DEB_dt=1.0, OD=None, **deb_kws):
    if OD is None:
        ms = ['L', 'F']
    else:
        ms = ['LOF']
    b = brain(ms, OD=OD, crawler=Cbas, intermitter=Im(EEB)) if not mock else brain(['Im', 'F'],
                                                                                   intermitter=Im(EEB))
    return null_dict('larva_conf', brain=b, body=null_dict('body', initial_length=0.001, Nsegs=Nsegs),
                     energetics=null_dict('energetics', hunger_as_EEB=True, hunger_gain=hunger_gain, DEB_dt=DEB_dt,
                                          **deb_kws))


def nengo_brain(EEB):
    if EEB > 0:
        f_fr0, f_fr_r = 2.0, (1.0, 3.0)
    else:
        f_fr0, f_fr_r = 0.0, (0.0, 0.0)
    return brain(['L', 'F'],
                 turner=null_dict('turner', initial_freq=0.3, initial_amp=10.0, noise=0.0, freq_range=(0.2, 0.4)),
                 crawler=null_dict('crawler', initial_freq=1.5, initial_amp=0.6, freq_range=(1.2, 1.8),
                                   waveform=None, step_to_length_mu=0.25, step_to_length_std=0.01),
                 feeder=null_dict('feeder', initial_freq=f_fr0, freq_range=f_fr_r),
                 intermitter=Im(EEB),
                 nengo=True
                 )


# -------------------------------------------WHOLE LARVA MODES---------------------------------------------------------


odors3 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base']]
odors5 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
odors2 = [f'{i}_odor' for i in ['Left', 'Right']]

freq_Fed = np.random.normal(1.244, 0.13)
freq_Deprived = np.random.normal(1.4, 0.14)
freq_Starved = np.random.normal(1.35, 0.15)

pause_dist_Fed = {'range': (0.22, 69.0),
                  'name': 'lognormal',
                  'mu': -0.488,
                  'sigma': 0.705}
pause_dist_Deprived = {'range': (0.22, 91.0),
                       'name': 'lognormal',
                       'mu': -0.431,
                       'sigma': 0.79}
pause_dist_Starved = {'range': (0.22, 22.0),
                      'name': 'lognormal',
                      'mu': -0.534,
                      'sigma': 0.733}

stridechain_dist_Fed = {'range': (1, 63),
                        'name': 'lognormal',
                        'mu': 0.987,
                        'sigma': 0.885}
stridechain_dist_Deprived = {'range': (1, 99),
                             'name': 'lognormal',
                             'mu': 1.052,
                             'sigma': 0.978}
stridechain_dist_Starved = {'range': (1, 191),
                            'name': 'lognormal',
                            'mu': 1.227,
                            'sigma': 1.052}

Levy_brain = brain(['L'], turner=Tsin, crawler=Ccon,
                   interference=null_dict('interference', attenuation=0.0),
                   intermitter=ImD({'fit': False, 'range': (0.01, 3.0), 'name': 'uniform'},
                                   {'fit': False, 'range': (1, 120), 'name': 'levy', 'mu': 0, 'sigma': 1})
                   )

brain_3c = brain(['L'],
                 crawler=null_dict('crawler', step_to_length_mu=0.18, step_to_length_std=0.055, initial_freq=1.35,
                                   freq_std=0.14),
                 intermitter=ImD(null_dict('logn_dist', range=(0.22, 56.0), mu=-0.48, sigma=0.74),
                                 null_dict('logn_dist', range=(1, 120), mu=1.1, sigma=0.95)))


def mod(brain, bod={}, energetics=None, phys={}):
    return null_dict('larva_conf', brain=brain,
                     energetics=energetics,
                     body=null_dict('body', **bod),
                     physics=null_dict('physics', **phys),
                     )


larvae = {
    'explorer': mod(brain(['L'])),
    'toucher': mod(brain(['L', 'To'], turner=Tno_noise), bod={'touch_sensors': 0}),
    'RL_toucher_0': mod(brain(['L', 'To', 'M'], turner=Tno_noise, memory=RL_touch_memory), bod={'touch_sensors': 0}),
    'gRL_toucher_0': mod(brain(['L', 'To', 'M'], turner=Tno_noise, memory=gRL_touch_memory), bod={'touch_sensors': 0}),
    'RL_toucher_2': mod(brain(['L', 'To', 'M'], turner=Tno_noise, memory=RL_touch_memory), bod={'touch_sensors': 2}),
    'gRL_toucher_2': mod(brain(['L', 'To', 'M'], turner=Tno_noise, memory=gRL_touch_memory), bod={'touch_sensors': 2}),
    'Levy-walker': mod(Levy_brain),
    'navigator': mod(brain(['L', 'O'], OD=OD1)),
    'navigator_x2': mod(brain(['L', 'O'], OD=OD2)),
    'immobile': mod(brain(['T', 'O'], OD=OD1)),
    'Orco_forager': mod(brain(['L', 'F'], intermitter=Im(0.5))),
    'forager': mod(brain(['LOF'], OD=OD1, intermitter=Im(0.5))),
    'forager_x2': mod(brain(['LOF'], OD=OD2, intermitter=Im(0.5))),
    'RL_navigator': mod(brain(['LOF', 'M'])),
    'RL_forager': mod(brain(['LOF', 'M'], intermitter=Im(0.5))),
    'basic_navigator': mod(brain(['L', 'O'], OD=OD1, turner=Tsin, crawler=Ccon), bod={'Nsegs': 1}),
    'explorer_3con': mod(brain_3c, bod={'initial_length': 3.85 / 1000, 'length_std': 0.35 / 1000}),
    'nengo_feeder': mod(nengo_brain(0.75)),
    'nengo_explorer': mod(nengo_brain(0.0)),
    'imitator': mod(brain(['L']), bod={'initial_length': 0.0045, 'length_std': 0.0001, 'Nsegs': 11},
                    phys={'ang_damping': 1.0, 'body_spring_k': 1.0}),
}
RvsS = {
    'rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover'),
    'navigator_rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover', OD=OD1),
    'mock_rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover', Nsegs=1, mock=True),
    'sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter'),
    'navigator_sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter', OD=OD1),
    'mock_sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter', Nsegs=1, mock=True),
}


def OD(ids: list, means: list, stds=None) -> dict:
    if stds is None:
        stds = np.array([0.0] * len(means))
    odor_dict = {}
    for id, m, s in zip(ids, means, stds):
        odor_dict[id] = {'mean': m,
                         'std': s}
    return odor_dict


gamers = {
    'gamer': mod(brain(['LOF'], OD=OD(odors3, [150.0, 0.0, 0.0]))),
    'gamer-5x': mod(brain(['LOF'], OD=OD(odors5, [150.0, 0.0, 0.0, 0.0, 0.0]))),
    'follower-R': mod(brain(['LOF'], OD=OD(odors2, [150.0, 0.0]))),
    'follower-L': mod(brain(['LOF'], OD=OD(odors2, [0.0, 150.0]))),
}

mod_dict = {
    **larvae,
    **RvsS,
    **gamers,
}

print(mod_dict['rover']['energetics'])