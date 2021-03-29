import numpy as np
from lib.aux import functions as fun


def xy(points, flat=False):
    if type(points) == str:
        return [f'{points}_x', f'{points}_y']
    elif type(points) == list:
        t=[[f'{p}_x', f'{p}_y'] for p in points]
        if flat :
            return fun.flatten_list(t)
        else :
            return t

def dst(points):
    if type(points) == str:
        return f'{points}_dst'
    elif type(points) == list:
        return [f'{point}_dst' for point in points]


def straight_dst(points):
    if type(points) == str:
        return f'{points}_straight_dst'
    elif type(points) == list:
        return [f'{point}_straight_dst' for point in points]


def vel(params):
    if type(params) == str:
        return f'{params}_vel'
    elif type(params) == list:
        return [f'{param}_vel' for param in params]


def acc(params):
    if type(params) == str:
        return f'{params}_acc'
    elif type(params) == list:
        return [f'{param}_acc' for param in params]


def lin(params):
    if type(params) == str:
        return f'lin_{params}'
    elif type(params) == list:
        return [f'lin_{param}' for param in params]


def scal(params):
    if type(params) == str:
        return f'scaled_{params}'
    elif type(params) == list:
        return [f'scaled_{param}' for param in params]


def abs(params):
    if type(params) == str:
        return f'abs_{params}'
    elif type(params) == list:
        return [f'abs_{param}' for param in params]


def final(params):
    if type(params) == str:
        return f'final_{params}'
    elif type(params) == list:
        return [f'final_{param}' for param in params]


def initial(params):
    if type(params) == str:
        return f'initial_{params}'
    elif type(params) == list:
        return [f'initial_{param}' for param in params]


def cum(params):
    if type(params) == str:
        return f'cum_{params}'
    elif type(params) == list:
        return [f'cum_{param}' for param in params]


def filt(params):
    if type(params) == str:
        return f'{params}_filt'
    elif type(params) == list:
        return [f'{param}_filt' for param in params]


def min(params):
    if type(params) == str:
        return f'{params}_min'
    elif type(params) == list:
        return [f'{param}_min' for param in params]


def orient(segs):
    if type(segs) == str:
        return f'{segs}_orientation'
    elif type(segs) == list:
        return [f'{seg}_orientation' for seg in segs]


def unwrap(segs):
    if type(segs) == str:
        return f'{segs}_unwrapped'
    elif type(segs) == list:
        return [f'{seg}_unwrapped' for seg in segs]


def max(params):
    if type(params) == str:
        return f'{params}_max'
    elif type(params) == list:
        return [f'{param}_max' for param in params]


def freq(params):
    if type(params) == str:
        return f'{params}_freq'
    elif type(params) == list:
        return [f'{param}_freq' for param in params]


def start(chunk):
    if type(chunk) == str:
        return f'{chunk}_start'
    elif type(chunk) == list:
        return [f'{c}_start' for c in chunk]


def stop(chunk):
    if type(chunk) == str:
        return f'{chunk}_stop'
    elif type(chunk) == list:
        return [f'{c}_stop' for c in chunk]


def dur(chunk):
    if type(chunk) == str:
        return f'{chunk}_dur'
    elif type(chunk) == list:
        return [f'{c}_dur' for c in chunk]


def dur_ratio(chunk):
    if type(chunk) == str:
        return f'{chunk}_dur_ratio'
    elif type(chunk) == list:
        return [f'{c}_dur_ratio' for c in chunk]


def num(chunk):
    if type(chunk) == str:
        return f'num_{chunk}s'
    elif type(chunk) == list:
        return [f'num_{c}s' for c in chunk]


def contact(chunk):
    if type(chunk) == str:
        return f'{chunk}_contact'
    elif type(chunk) == list:
        return [f'{c}_contact' for c in chunk]


def id(chunk):
    if type(chunk) == str:
        return f'{chunk}_id'
    elif type(chunk) == list:
        return [f'{c}_id' for c in chunk]


def mean(chunk):
    if type(chunk) == str:
        return f'{chunk}_mean'
    elif type(chunk) == list:
        return [f'{c}_mean' for c in chunk]


def std(chunk):
    if type(chunk) == str:
        return f'{chunk}_std'
    elif type(chunk) == list:
        return [f'{c}_std' for c in chunk]


def non(chunk):
    if type(chunk) == str:
        return f'non_{chunk}'
    elif type(chunk) == list:
        return [f'non_{c}' for c in chunk]


def length(chunk):
    if type(chunk) == str:
        return f'{chunk}_length'
    elif type(chunk) == list:
        return [f'{c}_length' for c in chunk]


def chain(chunk):
    if type(chunk) == str:
        return f'{chunk}chain'
    elif type(chunk) == list:
        return [f'{c}chain' for c in chunk]

def overlap_ratio(base_chunk, overlapping_chunk):
    return f'{base_chunk}_{overlapping_chunk}_overlap'

def chunk_track(chunk_name, params):
    if type(params) == str:
        return f'{chunk_name}_{params}'
    elif type(params) == list:
        return [f'{chunk_name}_{param}' for param in params]

def contour(Nc) :
    contour = [f'contour{i}' for i in range(Nc)]
    return contour

def midline(N, type='point') :
    if N >= 2:
        points = ['head'] + [f'{type}{i}' for i in np.arange(2, N, 1)] + ['tail']
    elif N == 1:
        points = ['body']
    else:
        points = []
    return points




