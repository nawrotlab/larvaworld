import numpy as np
from lib.conf.base.dtypes import null_dict


def trial_conf(durs=[], qs=[]):
    cumdurs = np.cumsum([0] + durs)
    return {i: null_dict('epoch', start=t0, stop=t1, substrate=null_dict('substrate', quality=q)) for i, (t0, t1, q) in
            enumerate(zip(cumdurs[:-1], cumdurs[1:], qs))}


trial_dict = {
    'default': trial_conf(),
    'odor_preference': trial_conf(
        [5.0] * 8,
        [1.0, 0.0] * 4),
    'odor_preference_short': trial_conf(
        [0.125] * 8,
        [1.0, 0.0] * 4)
}


def life_conf(durs=[], qs=[], age=0.0):
    return null_dict('life_history', epochs=trial_conf(durs, qs), age=age)


life_dict = {
    'default': life_conf(durs=[0.0], qs=[1.0], age=0.0),
    '72h_q50': life_conf(durs=[72.0], qs=[0.5], age=72.0),
}


def body_conf(ps, symmetry='bilateral', **kwargs):
    return null_dict('body_shape', points=ps, symmetry=symmetry, **kwargs)


body_dict = {
    'drosophila_larva': body_conf([(0.9, 0.1), (0.05, 0.1)]),
    'zebrafish_larva': body_conf([(0.9, 0.25), (0.7, 0.25), (0.6, 0.005), (0.05, 0.005)])
}
