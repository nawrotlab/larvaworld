import math
import numpy as np
import pandas as pd
import scipy as sp

from larvaworld.lib.aux import nam




def comp_bearing(xs, ys, ors, loc=(0.0, 0.0), in_deg=True):
    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(rads)

def compute_dispersal_solo(xy):
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    N = xy.shape[0]
    valid_idx=np.where(~np.isnan(xy))[0]
    if valid_idx.shape[0]<N*0.2 or valid_idx[0]>N*0.1 or valid_idx[-1]<N*0.9 :
        return np.zeros(N) * np.nan
    else:
        return eudi5x(xy, xy[valid_idx[0]])


def compute_dispersal_multi(xy0, t0,t1,dt):
    s0 = int(t0 / dt)
    s1 = int(t1 / dt)
    xy = xy0.loc[(slice(s0, s1), slice(None)), ['x', 'y']]

    AA = apply_per_level(xy, compute_dispersal_solo)
    Nt = AA.shape[0]
    N = xy0.index.unique('AgentID').values.shape[0]
    Nticks = xy0.index.unique('Step').size


    AA0 = np.zeros([Nticks, N]) * np.nan
    AA0[s0:s0 + Nt, :] = AA

    return AA0.flatten(), Nt

def raw_or_filtered_xy(s, points):
    r = nam.xy(points, flat=True)
    f = nam.filt(r)
    if all(i in s.columns for i in f):
        return f
    elif all(i in s.columns for i in r):
        return r
    else:
        print('No xy coordinates exist. Not computing spatial metrics')
        return


def compute_component_velocity(xy, angles, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x, prepend=[np.nan])
    dy = np.diff(y, prepend=[np.nan])
    d_temp = np.sqrt(dx ** 2 + dy ** 2)
    # This is the angle of the displacement vector relative to x-axis
    rads = np.arctan2(dy, dx)
    # And this is the angle of the displacement vector relative to the front-segment orientation vector
    angles2ref = rads - angles
    angles2ref %= 2 * np.pi
    d = d_temp * np.cos(angles2ref)
    v = d / dt
    if return_dst:
        return v, d
    else:
        return v


def compute_velocity_threshold(v, Nbins=500, max_v=None, kernel_width=0.02):
    import matplotlib.pyplot as plt
    if max_v is None:
        max_v = np.nanmax(v)
    bins = np.linspace(0, max_v, Nbins)
    hist, bin_edges = np.histogram(v, bins=bins, density=True)
    vals = bin_edges[0:-1] + 0.5 * np.diff(bin_edges)
    hist += 1 / len(v)
    hist /= np.sum(hist)
    plt.figure()
    plt.semilogy(vals, hist)
    ker = sp.signal.gaussian(len(vals), kernel_width * Nbins / max_v)
    ker /= np.sum(ker)

    density = np.exp(np.convolve(np.log(hist), ker, 'same'))
    plt.semilogy(vals, density)

    mi, ma = sp.signal.argrelextrema(density, np.less)[0], sp.signal.argrelextrema(density, np.greater)[0]
    try:
        minimum = vals[mi][0]
    except:
        minimum = np.nan
    return minimum





def get_display_dims():
    import pygame
    pygame.init()
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    return int(W * 2 / 3 / 16) * 16, int(H * 2 / 3 / 16) * 16


def get_window_dims(arena_dims):
    X, Y = np.array(arena_dims)
    W0, H0 = get_display_dims()
    R0, R = W0 / H0, X / Y
    if R0 < R:

        return W0, int(W0 / R / 16) * 16
    else:
        return int(H0 * R / 16) * 16, H0


def get_arena_bounds(arena_dims, s=1):
    X, Y = np.array(arena_dims) * s
    return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


def screen2space_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims
    p = (2 * pos[0] / X0 - 1), -(2 * pos[1] / Y0 - 1)
    pp = p[0] * X / 2, p[1] * Y / 2
    return pp


def space2screen_pos(pos, screen_dims, space_dims):
    X, Y = space_dims
    X0, Y0 = screen_dims

    p = pos[0] * 2 / X, pos[1] * 2 / Y
    pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
    return pp


def circle_to_polygon(N, r):
    one_segment = np.pi * 2 / N

    points = [
        (np.sin(one_segment * i) * r,
         np.cos(one_segment * i) * r)
        for i in range(N)]


    return points


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key='end', unit='sec'):
    dfs = []
    for l, d in ddic.items():
        if key == 'end':
            try:
                df = d.endpoint_data
            except:
                df = d.read('end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read('step')
        else :
            raise
        df['DatasetID'] = l
        df['GroupID'] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config['dt'] for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,)) / n, mode='same')



def body_contour(points=[(0.9, 0.1), (0.05, 0.1)], start=(1, 0), stop=(0, 0)):
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy




def apply_per_level(s, func, level='AgentID', **kwargs):
    '''
    Apply a function to each subdataframe of a dataframe after grouping by level

    Args:
        level:
        s: MultiIndex Dataframe with levels : ['Step', 'AgentID']
        func:function to apply on each subdataframe

    Returns:
        A : Array of dimensions [Nticks, Nids]
    '''



    def init_A(Ndims):
        ids = s.index.unique('AgentID').values
        Nids = len(ids)
        N = s.index.unique('Step').size

        if Ndims == 1:
            A = np.zeros([N, Nids]) * np.nan
        elif Ndims == 2:
            A = np.zeros([N, Nids, Ai.shape[1]]) * np.nan
        else:
            raise ValueError('Not implemented')
        return A


    A=None

    for i, (v, ss) in enumerate(s.groupby(level=level)):

        ss = ss.droplevel(level)
        Ai=func(ss, **kwargs)
        if A is None :
            A=init_A(len(Ai.shape))
            # print(i,'ff')
        if level=='AgentID' :
            A[:, i] = Ai
        elif level=='Step' :
            A[i, :] = Ai
    # print(s)
    return A

def unwrap_deg(a) :

    if isinstance(a, pd.Series) :
        a=a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b

def unwrap_rad(a) :

    if isinstance(a, pd.Series) :
        a=a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)])
    return b

def rate(a, dt) :
    if isinstance(a, pd.Series) :
        a=a.values
    v = np.diff(a) / dt
    return np.insert(v, 0, np.nan)

def eudist(xy) :
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    A= np.sqrt(np.nansum(np.diff(xy, axis=0)**2, axis=1))
    A = np.insert(A, 0, 0)
    return A

def eudi5x(a, b):
    return np.sqrt(np.sum((a - np.array(b)) ** 2, axis=1))

def eudiNxN(a,b) :
    b=np.array(b)
    return np.sqrt(np.sum(np.array([a-b[i] for i in range(b.shape[0])]) ** 2, axis=2))

def compute_dst(s,point='') :
    s[nam.dst(point)] = apply_per_level(s[nam.xy(point)], eudist).flatten()



def comp_extrema(a, order=3, threshold=None, return_2D=True) :
    A=a.values
    N=A.shape[0]
    i_min = sp.signal.argrelextrema(A, np.less_equal, order=order)[0]
    i_max = sp.signal.argrelextrema(A, np.greater_equal, order=order)[0]

    i_min_dif = np.diff(i_min, append=order)
    i_max_dif = np.diff(i_max, append=order)
    i_min = i_min[i_min_dif >= order]
    i_max = i_max[i_max_dif >= order]

    if threshold is not None:
        t0 = a.index.min()
        thr_min, thr_max = threshold
        i_min = i_min[a.loc[i_min + t0] < thr_min]
        i_max = i_max[a.loc[i_max + t0] > thr_max]

    if return_2D :
        aa = np.zeros([N, 2]) * np.nan
        aa[i_min, 0] = True
        aa[i_max, 1] = True
    else :
        aa = np.zeros(N) * np.nan
        aa[i_min] = -1
        aa[i_max] = 1
    return aa

