import numpy as np

# from lib.registry.dtypes import null_dict
from lib.registry.pars import preg
from lib.registry import reg
null=reg.PI.get_null

def trial_conf(durs=[], qs=[]):
    cumdurs = np.cumsum([0] + durs)
    return {i: null('epoch', start=t0, stop=t1, substrate=null('substrate', quality=q)) for i, (t0, t1, q) in
            enumerate(zip(cumdurs[:-1], cumdurs[1:], qs))}

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
    return null('life_history', epochs=trial_conf(durs, qs), age=age)

def Life_dict() :
    d = {
        'default': life_conf(durs=[0.0], qs=[1.0], age=0.0),
        '72h_q50': life_conf(durs=[72.0], qs=[0.5], age=72.0),
    }
    return d

def body_conf(ps, symmetry='bilateral', **kwargs):
    return null('body_shape', points=ps, symmetry=symmetry, **kwargs)

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

def store_RefPars():
    from lib.aux.dictsNlists import save_dict
    # from lib.conf.pars.pars import ParDict
    import lib.aux.naming as nam
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
        nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler_params.max_scaled_vel',
        'phi_scaled_velocity_max': 'brain.crawler_params.max_vel_phase',
        'attenuation': 'brain.interference_params.attenuation',
        'attenuation_max':  'brain.interference_params.attenuation_max',
        nam.freq(nam.vel(nam.orient(('front')))):  'brain.turner_params.initial_freq',
        nam.max('phi_attenuation'):  'brain.interference_params.max_attenuation_phase',
    }
    save_dict(d, preg.path_dict["ParRef"], use_pickle=False)


if __name__ == '__main__':
    # d=init2par()
    # print(d.keys())

    # store_controls()
    store_RefPars()
