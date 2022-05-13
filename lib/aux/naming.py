import numpy as np


def join(s, p, loc, c='_'):
    if loc == 'suf':
        return f'{p}{c}{s}'
    elif loc == 'pref':
        return f'{s}{c}{p}'

def name(s, ps, loc='suf',c='_') :
    if type(ps) == str:
        if ps == '':
            return s
        else:
            return join(s,ps, loc, c)
    elif type(ps) == list:
        return [join(s,p, loc, c) if p != '' else s for p in ps]

def xy(points, flat=False):
    if type(points) == str:
        if points == '':
            return ['x', 'y']
        else:
            return [f'{points}_x', f'{points}_y']
    elif type(points) == list:
        t = [[f'{p}_x', f'{p}_y'] if p != '' else ['x', 'y'] for p in points]
        return [item for sublist in t for item in sublist] if flat else t


def dst(points):
    s = 'dst'
    if points == '':
        return s
    else:
        return name(s, points, 'suf')


def dst2(points, **kwargs):
    s = 'dst_to'
    return name(s, points, 'pref', **kwargs)


def bearing2(points, **kwargs):
    s = 'bearing_to'
    return name(s, points, 'pref', **kwargs)


def straight_dst(points):
    s = dst('straight')
    return name(s, points, 'suf')


def vel(params):
    s='velocity'
    if params == '':
        return s
    else:
        return name(s, params, 'suf')


def acc(params):
    s = 'acceleration'
    if params == '':
        return s
    else:
        return name(s, params, 'suf')


def lin(params):
    s = 'lin'
    return name(s, params, 'pref')


def scal(params):
    s = 'scaled'
    return name(s, params, 'pref')


def abs(params):
    s = 'abs'
    return name(s, params, 'pref')


def final(params):
    s = 'final'
    return name(s, params, 'pref')


def initial(params):
    s = 'initial'
    return name(s, params, 'pref')


def cum(params):
    s = 'cum'
    return name(s, params, 'pref')


def filt(params):
    s='filt'
    return name(s, params, 'suf')


def min(params):
    s = 'min'
    return name(s, params, 'suf')


def orient(segs):
    s = 'orientation'
    return name(s, segs, 'suf')


def unwrap(segs):
    s = 'unwrapped'
    return name(s, segs, 'suf')


def max(params):
    s = 'max'
    return name(s, params, 'suf')


def freq(params):
    s = 'freq'
    return name(s, params, 'suf')


def start(chunk):
    s = 'start'
    return name(s, chunk, 'suf')


def stop(chunk):
    s = 'stop'
    return name(s, chunk, 'suf')


def dur(chunk):
    s = 'dur'
    return name(s, chunk, 'suf')


def dur_ratio(chunk):
    s = 'dur_ratio'
    return name(s, chunk, 'suf')


def num(chunk):
    s = 'num'
    temp = name(s, chunk, 'pref')
    return name('s', temp, 'suf', c='')


def contact(chunk):
    s = 'contact'
    return name(s, chunk, 'suf')


def id(chunk):
    s = 'id'
    return name(s, chunk, 'suf')


def mean(chunk):
    s = 'mean'
    return name(s, chunk, 'suf')


def std(chunk):
    s = 'std'
    return name(s, chunk, 'suf')


def var(chunk):
    s = 'var'
    return name(s, chunk, 'suf')


def non(chunk):
    s = 'non'
    return name(s, chunk, 'pref')


def length(chunk):
    s = 'length'
    return name(s, chunk, 'suf')


def chain(chunk):
    s = 'chain'
    return name(s, chunk, 'suf', c='')


def overlap_ratio(base_chunk, overlapping_chunk):
    return f'{base_chunk}_{overlapping_chunk}_overlap'


def chunk_track(chunk_name, params):
    s = chunk_name
    return name(s, params, 'pref')


def contour(Nc):
    contour = [f'contour{i}' for i in range(Nc)]
    return contour


def midline(N, type='point'):
    if N >= 2:
        points = ['head'] + [f'{type}{i}' for i in np.arange(2, N, 1)] + ['tail']
    elif N == 1:
        points = ['body']
    else:
        points = []
    return points

def at(p, t) :
    s = f'{p}_at'
    return name(s, t, 'pref')
    # return f'{p}_at_{t}'

