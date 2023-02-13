'''
The larva model parameters
'''
import copy

import numpy as np

from larvaworld.lib import reg, aux

OD1 = {'Odor': {'mean': 150.0, 'std': 0.0}}
OD2 = {'CS': {'mean': 150.0, 'std': 0.0}, 'UCS': {'mean': 0.0, 'std': 0.0}}


def Im(EEB, **kwargs):
    conf = reg.get_null('intermitter', feed_bouts=EEB > 0, EEB=EEB, **kwargs)

    return conf


# -------------------------------------------WHOLE NEURAL MODES---------------------------------------------------------


def brain(ks, nengo=False, OD=None, **kwargs):
    base_coupling = reg.get_null('interference', mode='square', crawler_phi_range=(np.pi / 2, np.pi),
                                  feeder_phi_range=(0.0, 0.0),
                                  attenuation=0.1, attenuation_max=0.6)

    RL_olf_memory = reg.get_null('memory', Delta=0.1, state_spacePerSide=1, modality='olfaction',
                                  gain_space=np.arange(-200.0, 200.0, 50.0).tolist())

    module_dict = {
        'T': 'turner',
        'C': 'crawler',
        'If': 'interference',
        'Im': 'intermitter',
        'O': 'olfactor',
        'To': 'toucher',
        'W': 'windsensor',
        'Th': 'thermosensor',
        'F': 'feeder',
        'M': 'memory',
    }
    if 'L' in ks:
        ks.remove('L')
        ks += ['T', 'C', 'If', 'Im']
    elif 'LOF' in ks:
        ks.remove('LOF')
        ks += ['T', 'C', 'If', 'Im', 'O', 'F']
    modules = [module_dict[k] for k in ks]

    modules = reg.get_null('modules', **{m: True for m in modules})
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
            d[p] = reg.get_null(k)
        if k == 'olfactor' and d[p] is not None:
            d[p]['odor_dict'] = OD
    d['nengo'] = nengo
    return aux.AttrDict(d)


def nengo_brain(module_shorts, EEB, OD=None):
    if EEB > 0:
        f_fr0, f_fr_r = 2.0, (1.0, 3.0)
    else:
        f_fr0, f_fr_r = 0.0, (0.0, 0.0)
    return brain(module_shorts,
                 turner=reg.get_null('turner', initial_freq=0.3, initial_amp=30.0, noise=0.1, activation_noise=0.8,
                                      freq_range=(0.2, 0.4)),
                 crawler=reg.get_null('crawler', initial_freq=1.5, initial_amp=0.6, freq_range=(1.2, 1.8),
                                       mode='realistic', stride_dst_mean=0.25, stride_dst_std=0.01),
                 feeder=reg.get_null('feeder', initial_freq=f_fr0, freq_range=f_fr_r),
                 intermitter=reg.get_null('intermitter', feed_bouts=EEB > 0, EEB=EEB, mode='nengo'),
                 nengo=True,
                 OD=OD
                 )


def build_RvsS(b):
    # RvsS = {}
    # for species, k_abs, EEB in zip(['rover', 'sitter'], [0.8, 0.4], [0.67, 0.37]):
    #     kws0 = {'energetics': {
    #         'DEB': preg.get_null('DEB', hunger_as_EEB=True, hunger_gain=1.0, DEB_dt=10.0, species=species),
    #         'gut': preg.get_null('gut', k_abs=k_abs)},
    #             'body': preg.get_null('body', initial_length=0.001, Nsegs=2)},
    #     brain_kws = {
    #         'intermitter_params': preg.get_null('intermitter', feed_bouts=True, EEB=EEB),
    #         'feeder_params': preg.get_null('feeder'),
    #         'nengo': False
    #     }
    #     mods = ['intermitter', 'feeder']
    #     RvsS[f'mock_{species}']


    def RvsS_larva(species, mock=False, OD=None, l0=0.001):
        if species == 'rover':
            EEB = 0.67
            gut_kws = {'k_abs': 0.8}
        elif species == 'sitter':
            EEB = 0.37
            gut_kws = {'k_abs': 0.4}
        else:
            raise

        mods = ['intermitter', 'feeder']
        kws = {'intermitter_params': reg.get_null('intermitter', feed_bouts=True, EEB=EEB),
               'feeder_params': reg.get_null('feeder'),
               'nengo': False}
        if mock:
            Nsegs = 1
        else:
            Nsegs = 2

            mods2 = ['crawler', 'turner', 'interference']

            if OD is not None:
                mods2 += ['olfactor']

            for mod in mods2:
                key = f'{mod}_params'
                kws[key] = b[key]
                if mod == 'olfactor':
                    kws[key]['odor_dict'] = OD

            mods += mods2

        kws['modules'] = reg.get_null('modules', **{m: True for m in mods})
        bb = reg.get_null('brain', **kws)
        #

        # if not mock :
        #     b = brain(ms, OD=OD, intermitter=Im)
        # else :
        #     b =brain(['Im', 'F'], intermitter=Im)

        gut = reg.get_null('gut', **gut_kws)
        deb = reg.get_null('DEB', hunger_as_EEB=True, hunger_gain=1.0, DEB_dt=10.0, species=species)

        null_Box2D_params = {
            'joint_types': {
                'friction': {'N': 0, 'args': {}},
                'revolute': {'N': 0, 'args': {}},
                'distance': {'N': 0, 'args': {}}
            }
        }

        return reg.get_null('Model', brain=bb, body=reg.get_null('body', initial_length=l0, Nsegs=Nsegs),
                             energetics={'DEB': deb, 'gut': gut}, Box2D_params=null_Box2D_params)

    RvsS = {
        'rover_old': RvsS_larva(species='rover'),
        'sitter_old': RvsS_larva(species='sitter'),
        'navigator_rover': RvsS_larva(species='rover', OD=OD1),
        'mock_rover': RvsS_larva(species='rover', mock=True),
        'navigator_sitter': RvsS_larva(species='sitter', OD=OD1),
        'mock_sitter': RvsS_larva(species='sitter', mock=True),
    }
    return RvsS


def mod(brain=None, bod={}, energetics=None, phys={}, Box2D={}):
    if Box2D == {}:
        null_Box2D_params = {
            'joint_types': {
                'friction': {'N': 0, 'args': {}},
                'revolute': {'N': 0, 'args': {}},
                'distance': {'N': 0, 'args': {}}
            }
        }
        Box2D_params = null_Box2D_params
    else:
        Box2D_params = reg.get_null('Box2D_params', **Box2D)
    return reg.get_null('Model', brain=brain,
                         energetics=energetics,
                         body=reg.get_null('body', **bod),
                         physics=reg.get_null('physics', **phys),
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


def create_mod_dict(b):

    RL_touch_memory = reg.get_null('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30,
                                    update_dt=0.5,
                                    gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=True)

    gRL_touch_memory = reg.get_null('memory', Delta=0.5, state_spacePerSide=1, modality='touch', train_dur=30,
                                     update_dt=0.5,
                                     gain_space=np.round(np.arange(-10, 11, 5), 1).tolist(), state_specific_best=False)

    M0 = mod()

    def add_brain(brain, M0=M0, bod={}, phys={}, Box2D={}):
        M1 = aux.AttrDict(copy.deepcopy(M0))
        M1.brain = brain
        M1.body.update(**bod)
        M1.physics.update(**phys)
        M1.Box2D_params.update(**Box2D)
        return M1

    LOF = brain(['LOF'])
    LOFM = brain(['LOF', 'M'])
    LW = brain(['L', 'W'])
    L = brain(['L'])
    LTo = brain(['L', 'To'], toucher=reg.get_null('toucher', touch_sensors=[]))
    LToM = brain(['L', 'To', 'M'], toucher=reg.get_null('toucher', touch_sensors=[]),
                 memory=RL_touch_memory)
    LToMg = brain(['L', 'To', 'M'], toucher=reg.get_null('toucher', touch_sensors=[]),
                  memory=gRL_touch_memory)
    LTo2M = brain(['L', 'To', 'M'], toucher=reg.get_null('toucher', touch_sensors=[0, 2]),
                  memory=RL_touch_memory)
    LTo2Mg = brain(['L', 'To', 'M'], toucher=reg.get_null('toucher', touch_sensors=[0, 2]),
                   memory=gRL_touch_memory)
    LTo_brute = brain(['L', 'To'], toucher=reg.get_null('toucher', touch_sensors=[], brute_force=True))
    nLO = nengo_brain(['L', 'O'], EEB=0.0)
    LTh = brain(['L', 'Th'])

    def add_OD(OD, B0=LOF):
        B1 = aux.AttrDict(copy.deepcopy(B0))
        B1.olfactor_params.odor_dict = OD
        return B1

    def add_Im(Im, B0=LOFM):
        B1 = aux.AttrDict(copy.deepcopy(B0))
        B1.intermitter_params = Im
        return B1

    explorers = {
        'explorer': add_brain(LW),
        'branch_explorer': add_brain(add_Im(reg.get_null('intermitter', feed_bouts=False, EEB=0, mode='branch'), LW)),
        'nengo_explorer': add_brain(nengo_brain(['L', 'W'], EEB=0.0)),
        'imitator': add_brain(L, bod={'Nsegs': 11}),

    }

    navigators = {
        'RL_navigator': add_brain(LOFM),
        'nengo_navigator': add_brain(nLO),
        'nengo_navigator_x2': add_brain(add_OD(OD2, nLO)),
        'thermo_navigator': add_brain(LTh),
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
        'toucher': add_brain(LTo),
        'toucher_brute': add_brain(LTo_brute),
        'RL_toucher_0': add_brain(LToM),
        'gRL_toucher_0': add_brain(LToMg),
        'RL_toucher_2': add_brain(LTo2M),
        'gRL_toucher_2': add_brain(LTo2Mg),
    }

    other = {
        'immobile': add_brain(brain(['T', 'O'], OD=OD1)),
    }

    odors3 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base']]
    odors5 = [f'{i}_odor' for i in ['Flag', 'Left_base', 'Right_base', 'Left', 'Right']]
    odors2 = [f'{i}_odor' for i in ['Left', 'Right']]

    gamers = {
        'gamer': add_brain(add_OD(OD(odors3, [150.0, 0.0, 0.0]))),
        'gamer-5x': add_brain(add_OD(OD(odors5, [150.0, 0.0, 0.0, 0.0, 0.0]))),
        'follower-R': add_brain(add_OD(OD(odors2, [150.0, 0.0]))),
        'follower-L': add_brain(add_OD(OD(odors2, [0.0, 150.0]))),
    }
    zebrafish = {
        'zebrafish': add_brain(L,
                               bod={'shape': 'zebrafish_larva'},
                               Box2D={
                                   'joint_types': {'revolute': {'N': 1, 'args': {'maxMotorTorque':10 ** 5, 'motorSpeed':1}}}})
    }

    grouped_mod_dict = {
        'explorers': explorers,
        'navigators': navigators,
        'foragers': foragers,
        'touchers': touchers,
        'foraging phenotypes': build_RvsS(b),
        'games': gamers,
        'zebrafish': zebrafish,
        'other': other,
    }

    return grouped_mod_dict

def all_mod_dict():
    dnew = reg.model.autostored_confs
    b = dnew['RE_NEU_SQ_DEF_nav'].brain
    return create_mod_dict(b)

@reg.funcs.stored_conf("Model")
def Model_dict():
    dnew = reg.model.autostored_confs
    b = dnew['RE_NEU_SQ_DEF_nav'].brain
    d = create_mod_dict(b)


    dd = aux.merge_dicts(list(d.values()))

    dnew.update(dd)
    return dnew

@reg.funcs.stored_conf("ModelGroup")
def ModelGroup_dict():
    d = all_mod_dict()
    return aux.AttrDict({k: {'model families': list(v.keys())} for k, v in d.items()})
