'''
The larva model parameters
'''
import copy

import numpy as np

from lib.aux.dictsNlists import AttrDict
from lib.conf.base.dtypes import null_dict, null_Box2D_params, Box2Djoints

''' Default exploration model'''

Cbas = null_dict('crawler', initial_freq=1.5, step_to_length_mu=0.25, step_to_length_std=0.0)
base_coupling = null_dict('interference', mode='square', crawler_phi_range=(0.45, 1.0), feeder_phi_range=(0.0, 0.0),
                          attenuation=0.1)
phasic_coupling = null_dict('interference', mode='phasic', attenuation_min=0.2, attenuation_max=0.31)

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

Tno_noise = null_dict('turner', activation_noise=0.0, noise=0.0)

Ccon = null_dict('crawler', waveform='constant', initial_amp=0.0012)

Ccon_no_noise = null_dict('crawler', waveform='constant', initial_amp=0.0012, noise=0.0)

RL_olf_memory = null_dict('memory', Delta=0.1, state_spacePerSide=1, modality='olfaction',
                          gain_space=np.arange(-200.0, 200.0, 50.0).tolist())

RL_touch_memory = null_dict('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30, update_dt=0.5,
                            gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=True)

gRL_touch_memory = null_dict('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30, update_dt=0.5,
                             gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=False)

OD1 = {'Odor': {'mean': 150.0, 'std': 0.0}}
OD2 = {'CS': {'mean': 150.0, 'std': 0.0}, 'UCS': {'mean': 0.0, 'std': 0.0}}


def Im(EEB, **kwargs):
    if EEB > 0:
        return null_dict('intermitter', feed_bouts=True, EEB=EEB, **kwargs)
    else:
        return null_dict('intermitter', feed_bouts=False, EEB=0.0, **kwargs)


def ImD(pau, str, run=None, **kwargs):
    return null_dict('intermitter', pause_dist=pau, stridechain_dist=str, run_dist=run, **kwargs)


ImFitted = ImD(
    run_mode='run',
    pau={'fit': False, 'range': (0.125, 15.875), 'name': 'lognormal', 'mu': -0.24223, 'sigma': 0.96498},
    str={'fit': False, 'range': (1, 157), 'name': 'lognormal', 'mu': 1.34411, 'sigma': 1.16138},
    run={'fit': False, 'range': (0.375, 115.9375), 'name': 'powerlaw', 'alpha': 1.48249},
)


# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------


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


def RvsS_larva(EEB=0.5, Nsegs=2, mock=False, hunger_gain=1.0, DEB_dt=10.0, OD=None, gut_kws={}, **deb_kws):
    if OD is None:
        ms = ['L', 'F']
    else:
        ms = ['LOF']
    b = brain(ms, OD=OD, crawler=Cbas, intermitter=Im(EEB)) if not mock else brain(['Im', 'F'],
                                                                                   intermitter=Im(EEB))

    gut = null_dict('gut', **gut_kws)
    deb = null_dict('DEB', hunger_as_EEB=True, hunger_gain=hunger_gain, DEB_dt=DEB_dt, **deb_kws)
    return null_dict('larva_conf', brain=b, body=null_dict('body', initial_length=0.001, Nsegs=Nsegs),
                     energetics={'DEB': deb, 'gut': gut})


def nengo_brain(module_shorts, EEB, OD=None):
    if EEB > 0:
        f_fr0, f_fr_r = 2.0, (1.0, 3.0)
    else:
        f_fr0, f_fr_r = 0.0, (0.0, 0.0)
    return brain(module_shorts,
                 turner=null_dict('turner', initial_freq=0.3, initial_amp=30.0, noise=1.85, activation_noise=0.8,
                                  freq_range=(0.2, 0.4)),
                 crawler=null_dict('crawler', initial_freq=1.5, initial_amp=0.6, freq_range=(1.2, 1.8),
                                   waveform=None, step_to_length_mu=0.25, step_to_length_std=0.01),
                 feeder=null_dict('feeder', initial_freq=f_fr0, freq_range=f_fr_r),
                 # olfactor=olfactor,
                 intermitter=Im(EEB, mode='nengo'),
                 nengo=True,
                 OD=OD
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
                   intermitter=ImD({'fit': False, 'range': (0.01, 3.0), 'name': 'uniform', 'mu': None, 'sigma': None},
                                   {'fit': False, 'range': (1, 120), 'name': 'levy', 'mu': 0, 'sigma': 1})
                   )

brain_3c = brain(['L'],
                 crawler=null_dict('crawler', step_to_length_mu=0.18, step_to_length_std=0.055, initial_freq=1.35,
                                   freq_std=0.14),
                 intermitter=ImD(null_dict('logn_dist', range=(0.22, 56.0), mu=-0.48, sigma=0.74),
                                 null_dict('logn_dist', range=(1, 120), mu=1.1, sigma=0.95)))

brain_phasic = brain(['L'],
                     crawler=null_dict('crawler', step_to_length_mu=0.224, step_to_length_std=0.033, initial_freq=1.418,
                                       max_vel_phase=3.6),
                     turner=Tno_noise,
                     interference=phasic_coupling,
                     intermitter=ImFitted)


def mod(brain=None, bod={}, energetics=None, phys={}, Box2D={}):
    if Box2D == {}:
        Box2D_params = null_Box2D_params
    else:
        Box2D_params = null_dict('Box2D_params', **Box2D)
    return null_dict('larva_conf', brain=brain,
                     energetics=energetics,
                     body=null_dict('body', **bod),
                     physics=null_dict('physics', **phys),
                     Box2D_params=Box2D_params
                     )


def OD(ids: list, means: list, stds=None) -> dict:
    if stds is None:
        stds = np.array([0.0] * len(means))
    odor_dict = {}
    for id, m, s in zip(ids, means, stds):
        odor_dict[id] = {'mean': m,
                         'std': s}
    return odor_dict


def create_mod_dict():
    M0 = mod()

    def add_brain(brain, M0=M0, bod={}, phys={}, Box2D={}):
        M1 = AttrDict.from_nested_dicts(copy.deepcopy(M0))
        M1.brain = brain
        M1.body.update(**bod)
        M1.physics.update(**phys)
        M1.Box2D_params.update(**Box2D)
        return M1

    LOF = brain(['LOF'])
    LOFM = brain(['LOF', 'M'])
    LO = brain(['L', 'O'])
    LO_brute = brain(['L', 'O'], olfactor=null_dict('olfactor', brute_force=True))
    LW = brain(['L', 'W'])
    L = brain(['L'])
    LTo = brain(['L', 'To'], turner=Tno_noise)
    LToM = brain(['L', 'To', 'M'], turner=Tno_noise, memory=RL_touch_memory)
    LToMg = brain(['L', 'To', 'M'], turner=Tno_noise, memory=gRL_touch_memory)
    LTo_brute = brain(['L', 'To'], turner=Tno_noise, toucher=null_dict('toucher', brute_force=True))
    nLO = nengo_brain(['L', 'O'], EEB=0.0)

    def add_OD(OD, B0=LOF):
        B1 = AttrDict.from_nested_dicts(copy.deepcopy(B0))
        B1.olfactor_params.odor_dict = OD
        return B1

    def add_Im(Im, B0=LOFM):
        B1 = AttrDict.from_nested_dicts(copy.deepcopy(B0))
        B1.intermitter_params = Im
        return B1

    explorers = {
        'explorer': add_brain(LW),
        'phasic_explorer': add_brain(brain_phasic),
        'branch_explorer': add_brain(add_Im(Im(0.0, mode='branch'), LW)),
        'nengo_explorer': add_brain(nengo_brain(['L', 'W'], EEB=0.0)),
        'Levy-walker': add_brain(Levy_brain),
        'explorer_3con': add_brain(brain_3c, bod={'initial_length': 3.85 / 1000, 'length_std': 0.35 / 1000}),
        'imitator': add_brain(L, bod={'initial_length': 0.0045, 'length_std': 0.0001, 'Nsegs': 11},
                              phys={'ang_damping': 1.0, 'body_spring_k': 1.0}),

    }

    navigators = {
        'navigator': add_brain(add_OD(OD1, LO)),
        'navigator_brute': add_brain(add_OD(OD1, LO_brute)),
        'navigator_x2': add_brain(add_OD(OD2, LO)),
        'navigator_x2_brute': add_brain(add_OD(OD2, LO_brute)),
        'basic_navigator': add_brain(brain(['L', 'O'], OD=OD1, turner=Tsin, crawler=Ccon), bod={'Nsegs': 1}),
        'continuous_navigator': add_brain(brain(['C', 'T', 'O'], OD=OD1, turner=Tno_noise, crawler=Ccon_no_noise),
                                          bod={'Nsegs': 1}),
        'RL_navigator': add_brain(LOFM),
        'nengo_navigator': add_brain(nLO),
        'nengo_navigator_x2': add_brain(add_OD(OD2, nLO)),
    }

    foragers = {
        'Orco_forager': add_brain(brain(['L', 'F'], intermitter=Im(0.5))),
        'nengo_feeder': add_brain(nengo_brain(['L', 'F'], EEB=0.75)),
        'forager': add_brain(add_Im(Im(0.5), add_OD(OD1))),
        'forager_x2': add_brain(add_Im(Im(0.5), add_OD(OD2))),
        'RL_forager': add_brain(add_Im(Im(0.5), LOFM)),
        'nengo_forager': add_brain(nengo_brain(['LOF'], EEB=0.75, OD=OD1))
    }

    touchers = {
        'toucher': add_brain(LTo, bod={'touch_sensors': []}),
        'toucher_brute': add_brain(LTo_brute, bod={'touch_sensors': []}),
        'RL_toucher_0': add_brain(LToM, bod={'touch_sensors': []}),
        'gRL_toucher_0': add_brain(LToMg, bod={'touch_sensors': []}),
        'RL_toucher_2': add_brain(LToM, bod={'touch_sensors': [0, 2]}),
        'gRL_toucher_2': add_brain(LToMg, bod={'touch_sensors': [0, 2]}),
    }

    other = {
        'immobile': add_brain(brain(['T', 'O'], OD=OD1)),
    }

    RvsS = {
        'rover': RvsS_larva(EEB=0.37, gut_kws={'k_abs': 0.8}),
        'sitter': RvsS_larva(EEB=0.67, gut_kws={'k_abs': 0.4}),
        'old_rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover'),
        'old_sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter'),
        'navigator_rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover', OD=OD1),
        'mock_rover': RvsS_larva(EEB=0.37, absorption=0.5, species='rover', Nsegs=1, mock=True),
        'navigator_sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter', OD=OD1),
        'mock_sitter': RvsS_larva(EEB=0.67, absorption=0.15, species='sitter', Nsegs=1, mock=True),
    }

    gamers = {
        'gamer': add_brain(add_OD(OD(odors3, [150.0, 0.0, 0.0]))),
        'gamer-5x': add_brain(add_OD(OD(odors5, [150.0, 0.0, 0.0, 0.0, 0.0]))),
        'follower-R': add_brain(add_OD(OD(odors2, [150.0, 0.0]))),
        'follower-L': add_brain(add_OD(OD(odors2, [0.0, 150.0]))),
    }
    zebrafish = {
        'zebrafish': add_brain(L,
                               bod={'initial_length': 0.004, 'length_std': 0.0001, 'Nsegs': 2,
                                    'shape': 'zebrafish_larva'},
                               phys={'ang_damping': 1.0, 'body_spring_k': 1.0, 'torque_coef': 0.3},
                               Box2D={
                                   'joint_types': {'revolute': Box2Djoints(N=1, maxMotorTorque=10 ** 5, motorSpeed=1)}})
    }

    grouped_mod_dict = {
        'explorers': explorers,
        'navigators': navigators,
        'foragers': foragers,
        'touchers': touchers,
        'foraging phenotypes': RvsS,
        'games': gamers,
        'zebrafish': zebrafish,
        'other': other,
    }

    return grouped_mod_dict


if __name__ == '__main__':
    M0 = mod()


    def add_brain(brain, M0=M0, bod={}, phys={}, Box2D={}):
        M1 = AttrDict.from_nested_dicts(copy.deepcopy(M0))
        M1.brain = brain
        M1.body.update(**bod)
        M1.physics.update(**phys)
        M1.Box2D_params.update(**Box2D)
        return M1


    zebrafish = {
        'continuous_navigator': add_brain(brain(['C', 'T', 'O'], OD=OD1, turner=Tno_noise, crawler=Ccon_no_noise),
                                          bod={'Nsegs': 1}),
    }
    from lib.conf.stored.conf import saveConf

    for k, v in zebrafish.items():
        saveConf(v, 'Model', k)
