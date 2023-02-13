import numpy as np
from larvaworld.lib import reg

def trial_conf(durs=[], qs=[]):
    cumdurs = np.cumsum([0] + durs)
    return {i: reg.get_null('epoch', start=t0, stop=t1, substrate=reg.get_null('substrate', quality=q)) for i, (t0, t1, q) in
            enumerate(zip(cumdurs[:-1], cumdurs[1:], qs))}

@reg.funcs.stored_conf("Trial")
def Trial_dict() :


    d = {
        'default': trial_conf(),
        'odor_preference': trial_conf(
            [5.0] * 8,
            [1.0, 0.0] * 4),
        'odor_preference_short': trial_conf(
            [0.125] * 8,
            [1.0, 0.0] * 4)
    }
    return d


def life_conf(durs=[], qs=[], age=0.0):
    return reg.get_null('Life', epochs=trial_conf(durs, qs), age=age)

@reg.funcs.stored_conf("Life")
def Life_dict() :
    d = {
        'default': life_conf(durs=[0.0], qs=[1.0], age=0.0),
        '72h_q50': life_conf(durs=[72.0], qs=[0.5], age=72.0),
    }
    return d

def body_conf(ps, symmetry='bilateral', **kwargs):
    return reg.get_null('Body', points=ps, symmetry=symmetry, **kwargs)

@reg.funcs.stored_conf("Body")
def Body_dict() :
    d = {
        'drosophila_larva': body_conf([
            (1.0, 0.0),
            (0.9, 0.1),
            (0.05, 0.1),
            (0.0, 0.0),
            (0.05, -0.1),
            (0.9, -0.1)
        ]),
        'zebrafish_larva': body_conf([
            (1.0, 0.0),
            (0.9, 0.25),
            (0.7, 0.25),
            (0.6, 0.005),
            (0.05, 0.005),
            (0.0, 0.0),
            (0.05, -0.005),
            (0.6, -0.005),
            (0.7, -0.25),
            (0.9, -0.25),
        ])
    }
    return d

@reg.funcs.stored_conf("Tree")
def Tree_dict() :
    return {}

@reg.funcs.stored_conf("Source")
def Source_dict() :
    return {}

