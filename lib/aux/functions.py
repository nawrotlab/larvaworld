import collections
import math
import random
import sys
import time
from collections import deque

import numpy
import pandas as pd
from contextlib import contextmanager
import sys, os
import numpy as np
from fitter import Fitter
from matplotlib import cm, colors
from pypet import ParameterGroup, Parameter
from scipy.signal import butter, sosfiltfilt
import scipy.stats as st
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import scipy as sp
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from shapely.ops import split
from lib.stor.paths import LarvaShape_path


def sigmoid(x):
    return 1 / (1 + math.e ** -x)  # mathematically equivalent, but simpler


def sigmoid_derivative(a):
    return a * (1 - a)


def density_extrema(data, kernel_width=0.02, Nbins=1000):
    min_v, max_v = np.nanmin(data), np.nanmax(data)
    bins = np.linspace(min_v, max_v, Nbins)
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    vals = bin_edges[0:-1] + 0.5 * np.diff(bin_edges)
    hist += 1 / len(data)
    hist /= np.sum(hist)
    # plt.figure()
    # plt.semilogy(vals, hist)

    ker = sp.signal.gaussian(len(vals), kernel_width * Nbins / (max_v - min_v))
    ker /= np.sum(ker)

    density = np.exp(np.convolve(np.log(hist), ker, 'same'))
    # plt.semilogy(vals, density)

    args_min, args_max = sp.signal.argrelextrema(density, np.less)[0], sp.signal.argrelextrema(density, np.greater)[0]
    if len(args_min):
        minima = vals[args_min]
    else:
        minima = []
    if len(args_max):
        maxima = vals[args_max]
    else:
        maxima = []
    return minima, maxima


def _restore_angle(a, d, l, n, num_segments, correction_coef):
    k0 = (l * n / num_segments) / correction_coef
    k1 = (l * (n + 1) / num_segments) / correction_coef
    if d <= k0:
        return a, 0
    elif k0 < d < k1:
        da = 1.0 * a * d / (l / num_segments)
        return a - da, da
    elif k1 <= d:
        return 0, a


def restore_bend(state, d, l, num_segments, correction_coef=1.0):
    nstate = []
    da = 0
    for i, a in enumerate(state):
        na, k = _restore_angle(a + da, d, l, i, num_segments, correction_coef=correction_coef)
        da = k
        nstate.append(na)
    return nstate


def restore_bend_2seg(bend, d, l, correction_coef=1.0):
    k0 = 0.5 * l / correction_coef
    if 0 <= d < k0:
        return bend * (1 - d / k0)
    elif k0 <= d:
        return 0


def circle_to_polygon(sides, radius, rotation=0, translation=None):
    one_segment = np.pi * 2 / sides

    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return np.array(points)


def angle(a, b, c, in_deg=True):
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return np.nan
    if in_deg:
        ang = (math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])) - 180) % 360
        return ang if ang <= 180 else ang - 360
    else:
        ang = (math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])) - np.pi) % (
                2 * np.pi)
        return ang if ang <= np.pi else ang - 2 * np.pi


def angle_to_x_axis(point_1, point_2, in_deg=True):
    # Point 1 is start, point 2 is end of vector
    if np.isnan(point_1).any() or np.isnan(point_2).any():
        return np.nan
    dx, dy = np.array(point_2) - np.array(point_1)
    rads = math.atan2(dy, dx)
    rads %= 2 * np.pi
    if in_deg:
        return math.degrees(rads)
    else:
        return rads


def angle_dif(angle_1, angle_2, in_deg=True):
    dif = angle_1 - angle_2
    if in_deg:
        if dif > 180:
            dif -= 2 * 180
        elif dif < -180:
            dif += 2 * 180
    else:
        if dif > np.pi:
            dif -= np.pi * 2
        elif dif < -np.pi:
            dif += np.pi * 2
    return dif


def angle_sum(angle_1, angle_2, in_deg=True):
    summ = angle_1 + angle_2

    if in_deg:
        summ %= 2 * 180
        if summ > 180:
            summ -= 2 * 180
        elif summ < -180:
            summ += 2 * 180
    else:
        summ %= 2 * np.pi
        if summ > np.pi:
            summ -= np.pi * 2
        elif summ < -np.pi:
            summ += np.pi * 2
    return summ


# ATTENTION : This rotates clockwise
def rotate_around_point(point, radians, origin=[0, 0]):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy


def rotate_around_center(point, radians):
    x, y = point
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = cos_rad * x + sin_rad * y
    qy = -sin_rad * x + cos_rad * y
    return np.array([qx, qy])


def rotate_around_center_multi(points, radians):
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    return np.array([(cos_rad * x + sin_rad * y, -sin_rad * x + cos_rad * y) for x, y in points])


def rotate_multiple_points(points, radians, origin=[0, 0]):
    # points have the form :
    # points=np.array([[1,2],[3,4], [5,6], [7,8]])
    qx, qy = rotate_around_point(points.T, radians, origin=origin)
    return np.vstack((qx, qy)).T


# Eliminate discontinuities from radians time series
def unwrap_deg(ts):
    b = np.copy(ts)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reconstruct_dict(param_group):
    dict = {}
    for p in param_group:
        if type(p) == ParameterGroup:
            d = reconstruct_dict(p)
            dict.update({p.v_name: d})
        elif type(p) == Parameter:
            if p.f_is_empty():
                dict.update({p.v_name: None})
            else:
                dict.update({p.v_name: p.f_get()})

    return dict


def group_list_by_n(l, n):
    if not len(l) % n == 0.0:
        print('List length must be multiple of n')
    else:
        k = int(len(l) / n)
        nl = []
    for i in range(k):
        nl.append(l[i * n:(i + 1) * n])
    return nl


def weib(x, n, a):
    return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)


def parse_array_at_nans(array):
    array = np.insert(array, 0, np.nan)
    array = np.insert(array, -1, np.nan)
    dif = np.diff(np.isnan(array).astype(int))
    de = np.where(dif == 1)[0]
    ds = np.where(dif == -1)[0]
    # c = array
    # c_filt = np.full_like(c, np.nan)
    # d = np.where(np.isnan(c))[0]
    # ds = [i + 1 for i in d if i + 1 not in d and not i + 1 == len(c)]
    # de = [i for i in d if i - 1 not in d and not i - 1 == -1]
    # if not np.isnan(array[0]):
    #     ds = np.insert(ds, 0, 0)
    # if not np.isnan(array[-1]):
    #     de = np.append([de, len(array)])
    return ds, de


# def apply_filter_to_array_with_nans(array, critical_freq, fr, N=1):
#     sos = butter(N=N, Wn=critical_freq, btype='lowpass', analog=False, fs=fr, output='sos')
#     # print(type(array))
#     # The array chunks must be longer than padlen=6
#     padlen = 6
#     return apply_sos_filter_to_array_with_nans(array=array, sos=sos, padlen=padlen)


def apply_sos_filter_to_array_with_nans(array, sos, padlen=6):
    try:
        array_filt = np.full_like(array, np.nan)
        ds, de = parse_array_at_nans(array)
        for s, e in zip(ds, de):
            k = array[s:e]
            if len(k) > padlen:
                k_filt = sosfiltfilt(sos, k, padlen=padlen)
                array_filt[s:e] = k_filt
        return array_filt
    except:
        array_filt = sosfiltfilt(sos, array, padlen=padlen)
        # raise ValueError
        return array_filt


def apply_filter_to_array_with_nans_multidim(array, freq, fr, N=1):
    sos = butter(N=N, Wn=freq, btype='lowpass', analog=False, fs=fr, output='sos')
    # The array chunks must be longer than padlen=6
    padlen = 6
    # 2-dimensional array must have each timeseries in different column
    if array.ndim == 1:
        return apply_sos_filter_to_array_with_nans(array=array, sos=sos, padlen=padlen)
    elif array.ndim == 2:
        return np.array([apply_sos_filter_to_array_with_nans(array=array[:, i], sos=sos, padlen=padlen) for i in
                         range(array.shape[1])]).T
    elif array.ndim == 3:
        return np.transpose([apply_filter_to_array_with_nans_multidim(array[:, :, i], freq, fr, N=1) for i in
                             range(array.shape[2])], (1, 2, 0))
    else:
        raise ValueError('Method implement for up to 3-dimensional array')


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    return a_set & b_set


# def rotate_point(origin, point, radians):
#     """
#     Rotate a point counterclockwise by a given angle_to_x_axis around a given origin.
#     CAUTION : points as [x,y]
#     The angle_to_x_axis should be given in radians.
#     """
#     # ox, oy = origin
#     # px, py = point
#     ox = origin[0]
#     oy = origin[1]
#     px = point[0]
#     py = point[1]
#
#     qx = ox + math.cos(radians) * (px - ox) - math.sin(radians) * (py - oy)
#     qy = oy + math.sin(radians) * (px - ox) + math.cos(radians) * (py - oy)
#     return [qx, qy]
#     # return qx, qy
# k=rotate_around_point(point=[-1,0], radians=-np.pi/2, origin=[10, 10])
# print(k)

# def center_of_mass(X):
#     # calculate center of mass of a closed polygon
#     x = X[:, 0]
#     y = X[:, 1]
#     g = (x[:-1] * y[1:] - x[1:] * y[:-1])
#     A = 0.5 * g.sum()
#     cx = ((x[:-1] + x[1:]) * g).sum()
#     cy = ((y[:-1] + y[1:]) * g).sum()
#     return 1. / (6 * A) * np.array([cx, cy])
def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def compute_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def inside_polygon(points,tank_polygon):
    return all([tank_polygon.contains(Point(x, y)) for x, y in points])
    # # // p is your point, p.x is the x coord, p.y is the y coord
    # Xmin, Xmax, Ymin, Ymax = space_edges_for_screen
    # # x, y = point
    #
    # if all([(x < Xmin or x > Xmax or y < Ymin or y > Ymax) for x, y in points]):
    #     # // Definitely not within the polygon!
    #     return False
    # else:
    #     # point = Point(x, y)
    #     # print(polygon.contains(point))
    #     return all([Polygon(vertices).contains(Point(x, y)) for x, y in points])


def border_collision(line, border_lines):
    return any([line.crosses(l) for l in border_lines])

def larva_collision(line, larva_bodies):
    return any([line.intersects(p) for p in larva_bodies])


def timeit(exec_s, globals={}, N=10000):
    t = time.time()
    for i in range(N):
        exec(exec_s, globals)
    return time.time() - t


def time2algorithms(alg1, alg2, globals={}, args={}, N=10000):
    '''
    example use :
    time2algorithms(alg1="rotate_point(**args)",
                    alg2="rotate_around_point(**args)",
                    globals=locals(),
                    args={  'origin': (-3.4, -5.3),
                            'radians': np.pi / 14,
                            'point': (-6.5, 4.3)},
                    N=100000)
    '''

    t = time.time()
    for i in range(N):
        exec(alg1, globals.update(args))
    r1 = time.time() - t
    t = time.time()
    for i in range(N):
        exec(alg2, globals.update(args))
    r2 = time.time() - t
    print(r1)
    print(r2)


def fit_geom_distribution(data):
    data = pd.Series(data)
    """Model data by finding best fit distribution to data"""
    x = np.arange(np.min(data), np.max(data) + 1, 1).astype(int)
    y = np.zeros(len(x)).astype(int)

    counts = data.value_counts()
    for i, k in enumerate(x):
        if k in counts.index.values.astype(int):
            y[i] = int(counts.loc[k])
    y = y / len(data)
    # print(y)

    mean = data.mean()
    p = 1 / mean

    # Calculate fitted PDF and error with fit in distribution
    pdf = st.geom.pmf(x, p=p)
    sse = np.sum(np.power(y - pdf, 2.0)) / len(y)
    print(f'geom distribution fitted with SSE :{sse}')
    return p, sse


def fit_powerlaw_distribution(data):
    f = Fitter(data)
    f.distributions = ['powerlaw']
    f.fit()
    k = f.get_best()
    alpha = list(k.values())[0][0]
    return alpha


def erase_overlap(data):
    # Data are of the form [[start1,stop1], [start2,stop2], ..., [startN,stopN]]
    # Data is numpy array
    u0, c0 = np.unique(data[:, 0], return_counts=True)
    dup0 = u0[c0 > 1]
    u1, c1 = np.unique(data[:, 1], return_counts=True)
    dup0 = u1[c1 > 1]

    pass


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def downsample_2d_array(a, step):
    return a[::step, ::step]


def compute_velocity(xy, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    v = d / dt
    if return_dst:
        return v, d
    else:
        return v


def compute_component_velocity(xy, angles, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    d_temp = np.sqrt(dx ** 2 + dy ** 2)
    # This is the angle of the displacement vector relative to x-axis
    rads = np.arctan2(dy, dx)
    # And this is the angle of the displacement vector relative to the front-segment orientation vector
    angles2ref = rads - angles[:-1]
    angles2ref %= 2 * np.pi
    d = d_temp * np.cos(angles2ref)
    v = d / dt
    if return_dst:
        return v, d
    else:
        return v


def compute_velocity_threshold(v, Nbins=500, max_v=None, kernel_width=0.02):
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


def match_larva_ids(s, dl=None, max_t=5 * 60, max_s=20, pars=None, e=None, min_Nids=1, max_Niters=1000):
    t_r = np.linspace(0, max_t, max_Niters)
    s_r = np.linspace(0, max_s, max_Niters)
    if dl is None:
        ls = None
    else:
        ls = e['length']
    if pars is None:
        pars = s.columns.values.tolist()
    ss = s.dropna().reset_index(level='Step', drop=False)

    ids, mins, maxs, first_xy, last_xy = get_extrema(ss, pars)
    for i in range(max_Niters):
        ds, dt = s_r[i], t_r[i]
        print(i, len(ids), ds, dt)
        # nexts_t = get_temporal_nexts0(ids, mins, maxs, dt)
        nexts_sp = get_spatial_nexts0(ids, ds, first_xy, last_xy)
        # N_t = np.sum([len(next) for next in nexts_t])
        N_s = np.sum([len(next) for next in nexts_sp])
        # if N_t > 0:
        if N_s > 0:
            # nexts = get_spatial_nexts1(ids, nexts_t, ds, first_xy, last_xy, dl, ls)
            nexts = get_temporal_nexts1(ids, nexts_sp, mins, maxs, dt)
            # N_s = np.sum([len(next) for next in nexts])
            N_t = np.sum([len(next) for next in nexts])
            # if N_s > 0:
            if N_t > 0:
                # print(N_t)
                taken = []
                pairs = dict()
                for id, next in zip(ids, nexts):
                    next = [idx for idx in next if idx not in taken]
                    if dl is not None:
                        next = [idx for idx in next if np.abs(ls[id] - ls[idx]) < dl]
                    if len(next) == 0:
                        continue
                    elif len(next) == 1:
                        best_next = next[0]
                        taken.append(best_next)
                    elif len(next) > 1:
                        errors = [np.sum(np.abs(last_xy[id] - first_xy[idx])) for idx in next]
                        indmin = np.argmin(errors)
                        best_next = next[indmin]
                    pairs[best_next] = id
                while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
                    for id1, id2 in pairs.items():
                        if id2 in list(pairs.keys()):
                            pairs.update({id1: pairs[id2]})
                ss.rename(index=pairs, inplace=True)
                if dl is not None:
                    for id1, id2 in pairs.items():
                        v = ss['spinelength'].loc[id2].values

                        ls[id2] = np.nanmean(v)
                        ls.drop([id1], inplace=True)
                # print(pairs)
                ids, mins, maxs, first_xy, last_xy = update_extrema(pairs, ids, mins, maxs, first_xy, last_xy)
                if len(ids) <= min_Nids:
                    break
    # inds, dt, ds = 0, 0, 0
    # while len(ids) > min_Nids and (dt<max_t or ds<max_s) :
    #     inds+=1
    #     print(inds, len(ids), dt, ds)
    #     # Compute extrema
    #     # mins = ss['Step'].groupby('AgentID').min()
    #     # maxs = ss['Step'].groupby('AgentID').max()
    #
    #     # first_xy, last_xy= {},{}
    #     # for id in ids :
    #     #     first_xy[id] = ss[pars].xs(id).dropna().values[0,:]
    #     #     last_xy[id] = ss[pars].xs(id).dropna().values[-1,:]
    #
    #     # pairs_found=False
    #     for i in range(max_Niters):
    #     # for s_i in range(max_Niters):
    #         ds, dt =s_r[i], t_r[i]
    #         nexts_sp=get_spatial_nexts(ids, ds,first_xy, last_xy, dl0, ls)
    #         N_s=np.sum([len(next) for next in nexts_sp])
    #         # if N_s==0 :
    #         #     ddst += dst
    #         if N_s > 0 :
    #             nexts = get_temporal_nexts(ids, nexts_sp, mins, maxs, dt)
    #             N_t = np.sum([len(next) for next in nexts])
    #             if N_t > 0:
    #         # else :
    #         #     for t_i in range(max_Niters):
    #         #         dt=t_r[t_i]
    #         #         nexts = get_temporal_nexts(ids, nexts_sp, mins, maxs, dt)
    #         #         N_t = np.sum([len(next) for next in nexts])
    #                 # if N_t == 0:
    #                 #     ddur += dur
    #                 # if N_t > 0 :
    #                 # else :
    #                 # pairs_found = True
    #                 taken = []
    #                 pairs = dict()
    #                 for id, next in zip(ids, nexts):
    #                     next = [idx for idx in next if idx not in taken]
    #                     if len(next) == 0:
    #                         continue
    #                     elif len(next) == 1:
    #                         best_next = next[0]
    #                         taken.append(best_next)
    #                     elif len(next) > 1:
    #                         errors = []
    #                         for idx in next:
    #                             if dl0 is None:
    #                                 error = np.sum(np.abs(last_xy[id] - first_xy[idx]))
    #                             else:
    #                                 error = np.abs(ls[id] - ls[idx])
    #                             errors.append(error)
    #                         indmin = np.argmin(errors)
    #                         best_next = next[indmin]
    #                     pairs[best_next] = id
    #                 while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
    #                     for id1, id2 in pairs.items():
    #                         if id2 in list(pairs.keys()):
    #                             pairs.update({id1: pairs[id2]})
    #
    #                 ss.rename(index=pairs, inplace=True)
    #                 ids = ss.index.unique().tolist()
    #                 print(pairs)
    #                 break
    #             if pairs_found:
    #                 break
    # else :
    #     ddst += dst

    # break
    # break

    # break
    # break
    # sss= ss.reset_index(drop=False).set_index(keys=['Step', 'AgentID'], drop=True)
    # print(any(sss.index.duplicated()))
    # print(sss[sss.index.duplicated()].index)
    # print(sss.loc[(1324, 'Larva_10092'), 'head_x'])
    # print(sss.loc[(1324, 'Larva_10110'), 'head_x'])

    # print(pairs)
    # print(nexts)
    # print(best_nexts)
    # break
    print('Finalizing dataset')
    ss.reset_index(drop=False, inplace=True)
    ss.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
    # ss.sort_index(level=['Step', 'AgentID'], inplace=True)
    return ss


def get_spatial_nexts0(ids, ddst, first_xy, last_xy):
    nexts = [[idx for idx in ids if (idx != id and all(np.abs(last_xy[id] - first_xy[idx]) < ddst))] for id in ids]
    return nexts


def get_spatial_nexts1(ids, nexts, ddst, first_xy, last_xy):
    nexts = [[idx for idx in next if (all(np.abs(last_xy[id] - first_xy[idx]) < ddst))] for id, next in zip(ids, nexts)]
    return nexts


def get_temporal_nexts1(ids, nexts, mins, maxs, ddur):
    nexts = [[idx for idx in next if 0 < mins[idx] - maxs[id] < ddur] for id, next in zip(ids, nexts)]
    return nexts


def get_temporal_nexts0(ids, mins, maxs, ddur):
    nexts = [[idx for idx in ids if (idx != id and 0 < mins[idx] - maxs[id] < ddur)] for id in ids]
    return nexts


def get_extrema(ss, pars):
    ids = ss.index.unique().tolist()
    mins = ss['Step'].groupby('AgentID').min()
    maxs = ss['Step'].groupby('AgentID').max()
    first_xy, last_xy = {}, {}
    for id in ids:
        first_xy[id] = ss[pars].xs(id).dropna().values[0, :]
        last_xy[id] = ss[pars].xs(id).dropna().values[-1, :]
    return ids, mins, maxs, first_xy, last_xy


def update_extrema(pairs, ids, mins, maxs, first_xy, last_xy):
    for id2 in pairs.values():
        n_ids = [id for id in pairs.keys() if pairs[id] == id2] + [id2]
        n_mins = [mins[id] for id in n_ids]
        min_id = n_ids[np.argmin(n_mins)]
        n_maxs = [maxs[id] for id in n_ids]
        max_id = n_ids[np.argmax(n_maxs)]
        mins[id2], first_xy[id2] = mins[min_id], first_xy[min_id]
        maxs[id2], last_xy[id2] = maxs[max_id], last_xy[max_id]
    for id1 in pairs.keys():
        del mins[id1]
        del maxs[id1]
        del first_xy[id1]
        del last_xy[id1]
        ids.remove(id1)
    return ids, mins, maxs, first_xy, last_xy


def positions_in_circle(r, N):
    angles = np.linspace(0, np.pi * 2, N + 1)[:-1]
    # print(angles)
    p = np.array([(np.cos(a) * r, np.sin(a) * r) for a in angles])
    return p


def generate_orientations(num_identical, circle_parsing, iterations):
    orientations = sum(([[i] * num_identical for i in np.arange(circle_parsing) * 2 * np.pi / circle_parsing]),
                       []) * iterations
    return orientations


def generate_positions_on_xaxis(num_identical, num_starting_points, step, starting_x):
    positions = sum(
        ([(np.round(x, 2), 0.0)] * num_identical for x in np.arange(num_starting_points) * step + starting_x), [])
    return positions


def depth(d):
    queue = deque([(id(d), d, 1)])
    memo = set()
    while queue:
        id_, o, level = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            queue += ((id(v), v, level + 1) for v in o.values())
    return level


def print_dict(d):
    l = depth(d)
    for k, v in d.items():
        if isinstance(v, dict):
            print('----' * l, k, '----' * l)
            print_dict(v)
        else:
            print(k, ':', v)
    print()


def dict_to_file(dictionary, filename):
    orig_stdout = sys.stdout
    f = open(filename, 'w')
    sys.stdout = f
    print_dict(dictionary)
    sys.stdout = orig_stdout
    f.close()
    # sys.stdout = open(filename, 'w')
    # sys.stdout = stdout
    # with open(filename, 'w') as sys.stdout: print_dict(dictionary)


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

def round2significant(a, significant_digits) :
    return round(a, significant_digits - int(math.floor(math.log10(abs(a)))) - 1)

# Create a bilaterally symmetrical 2D contour with the long axis along x axis

# Arguments :
#   points : the points above x axis through which the contour passes
#   start : the front end of the body
#   stop : the rear end of the body
def body(points, start=[1, 0], stop=[0, 0]):
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy


# Segment body in N segments of given ratios via vertical lines
def segment_body(N, xy0, seg_ratio=None, centered=True):
    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = [1 / N] * N

    # Create a polygon from the given body contour
    p = Polygon(xy0)
    # Get maximum x value of contour
    y0 = np.max(p.exterior.coords.xy[1])

    # Segment body via vertical lines
    ps = [p]
    for i, (r, cum_r) in enumerate(zip(seg_ratio, np.cumsum(seg_ratio))):
        l = LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
        new_ps = []
        for p in ps:
            new_p = [new_p for new_p in split(p, l)]
            new_ps += new_p
        ps = new_ps

    # Sort segments so that front segments come first
    ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

    # Transform to 2D array of coords
    ps = [p.exterior.coords.xy for p in ps]
    ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

    # Center segments around 0,0
    if centered:
        for i, (r, cum_r) in enumerate(zip(seg_ratio, np.cumsum(seg_ratio))):
            ps[i] -= [(1 - cum_r) + r / 2, 0]

    # Put front point at the start of segment vertices. Drop duplicate rows
    for i in range(len(ps)):
        if i == 0:
            ind = np.argmax(ps[i][:, 0])
            ps[i] = np.flip(np.roll(ps[i], -ind - 1, axis=0), axis=0)
        else:
            ps[i] = np.flip(np.roll(ps[i], 1, axis=0), axis=0)
        _, idx = np.unique(ps[i], axis=0, return_index=True)
        ps[i] = ps[i][np.sort(idx)]
    # ps[0]=np.vstack([ps[0], np.array(ps[0][0,:])])
    return ps


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def unique_list(l):
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]


def agents_spatial_query(pos, radius, agent_list):
    if len(agent_list) == 0:
        return []
    agent_positions = np.array([agent.get_position() for agent in agent_list])
    agent_radii = np.array([agent.get_radius() for agent in agent_list])
    dsts = np.linalg.norm(agent_positions - pos, axis=1) - agent_radii
    inds = np.where(dsts <= radius)[0]
    return [agent_list[i] for i in inds]

def agent_dict2list(dic) :
    l=[]
    for id, pars in dic.items() :
        pars['unique_id']=id
        l.append(pars)
    return l

def agent_list2dict(l) :
    d={}
    for a in l :
        id=a['unique_id']
        a.pop('unique_id')
        d[id]=a
    return d

def compute_dst(point1, point2) :
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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
    if as_rgb :
        cs=[colors.to_rgb(c) for c in cs]
        cs=[tuple([i*255 for i in c]) for c in cs]
    return cs

def LvsRtoggle(side) :
    if side=='Left' :
        return 'Right'
    elif side=='Right':
        return 'Left'
    else :
        raise ValueError(f'Argument {side} is neither Left nor Right')

