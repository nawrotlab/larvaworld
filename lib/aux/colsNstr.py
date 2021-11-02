import random

import numpy as np
from matplotlib import cm, colors


# function takes in a hex color and outputs it inverted
def invert_color(col, return_self=False):
    if type(col)in [list, tuple] and len(col)==3:
        if not all([0<=i<=1 for i in col]) :
            col=list(np.array(col)/255)
        col = colors.rgb2hex(col)
    elif col[0]!='#' :
        col=colors.cnames[col]
    table = str.maketrans('0123456789abcdef', 'fedcba9876543210')
    col2= '#' + col[1:].lower().translate(table).upper()
    if not return_self :
        return col2
    else :
        return col,col2

def random_colors(n):
    ret = []
    r = int(random.random() * 200)
    g = int(random.random() * 200)
    b = int(random.random() * 200)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append(np.array([r, g, b]))
    return ret


def N_colors(N, as_rgb=False):
    if N == 1:
        cs = ['blue']
    elif N == 2:
        cs = ['red', 'blue']
    elif N == 3:
        cs = ['green', 'blue', 'red']
    elif N == 4:
        cs = ['red', 'blue', 'darkred', 'darkblue']
    elif N == 5:
        cs = ['green', 'red', 'blue', 'darkred', 'darkblue']
    else:
        colormap = cm.get_cmap('brg')
        cs = [colormap(i) for i in np.linspace(0, 1, N)]
    if as_rgb:
        cs = [colorname2tuple(c) for c in cs]
    return cs


def colorname2tuple(name):
    c0 = colors.to_rgb(name)
    c1 = tuple([i * 255 for i in c0])
    return c1



def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever

def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text  # or whatever

import functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def col_range(q, low=(255, 0, 0),high=(255, 255, 255), mul255=False) :
    rr0, gg0, bb0 = q_col1 = np.array(low)/255
    rr1, gg1, bb1 = q_col2 = np.array(high)/255
    qrange = np.array([rr1 - rr0, gg1 - gg0, bb1 - bb0])
    res= q_col1 + np.array([q,q,q]).T * qrange
    res=np.clip(res, a_min=0, a_max=1)
    if mul255 :
        res*=255
    return res


# for q in np.arange(0,1,0.1):
#     print(q, col_range(q, low=(255, 0, 0), high=(0, 128, 0)))

