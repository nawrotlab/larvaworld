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


def init_shortcuts():
    draw = {
        'visible trail': 'p',
        '▲ trail duration': '+',
        '▼ trail duration': '-',

        'draw_head': 'h',
        'draw_centroid': 'e',
        'draw_midline': 'm',
        'draw_contour': 'c',
        'draw_sensors': 'j',
    }

    inspect = {
        'focus_mode': 'f',
        'odor gains': 'z',
        'dynamic graph': 'q',
    }

    color = {
        'black_background': 'g',
        'random_colors': 'r',
        'color_behavior': 'b',
    }

    aux = {
        'visible_clock': 't',
        'visible_scale': 'n',
        'visible_state': 's',
        'visible_ids': 'tab',
    }

    screen = {
        'move up': 'UP',
        'move down': 'DOWN',
        'move left': 'LEFT',
        'move right': 'RIGHT',
    }

    sim = {
        'larva_collisions': 'y',
        'pause': 'space',
        'snapshot': 'i',
        'delete item': 'del',

    }

    odorscape = {
        'odor_aura': 'u',
        'windscape': 'w',
        'plot odorscapes': 'o',
        **{f'odorscape {i}': i for i in range(10)},
        # 'move_right': 'RIGHT',
    }

    d = {
        'draw': draw,
        'color': color,
        'aux': aux,
        'screen': screen,
        'simulation': sim,
        'inspect': inspect,
        'landscape': odorscape,
    }

    return d


def init_controls():
    from lib.gui.aux.functions import get_pygame_key
    k = init_shortcuts()
    d = {'keys': {}, 'pygame_keys': {}, 'mouse': {
        'select item': 'left click',
        'add item': 'left click',
        'select item mode': 'right click',
        'inspect item': 'right click',
        'screen zoom in': 'scroll up',
        'screen zoom out': 'scroll down',
    }}
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d['keys'][title] = dic
    d['pygame_keys'] = {k: get_pygame_key(v) for k, v in ds.items()}
    return d


def store_controls():
    d = init_controls()
    from lib.conf.stored.conf import saveConfDict
    saveConfDict(d, 'Settings')


def store_RefPars():
    from lib.aux.dictsNlists import save_dict
    from lib.conf.pars.pars import ParDict
    import lib.aux.naming as nam
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
    }
    save_dict(d, ParDict.path_dict["ParRef"], use_pickle=False)

if __name__ == '__main__':
    # d=init2par()
    # print(d.keys())

    store_controls()
    store_RefPars()