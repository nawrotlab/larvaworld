import math
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
from scipy.spatial import ConvexHull

from lib.aux import naming as nam


def single_parametric_interpolate(obj_x_loc, obj_y_loc, numPts=50):
    n = len(obj_x_loc)
    vi = [[obj_x_loc[(i + 1) % n] - obj_x_loc[i],
           obj_y_loc[(i + 1) % n] - obj_y_loc[i]] for i in range(n)]
    si = [np.linalg.norm(v) for v in vi]
    di = np.linspace(0, sum(si), numPts, endpoint=False)
    new_points = []
    for d in di:
        for i, s in enumerate(si):
            if d > s:
                d -= s
            else:
                break
        l = d / s
        new_points.append((obj_x_loc[i] + l * vi[i][0],
                           obj_y_loc[i] + l * vi[i][1]))
    return new_points


def xy_along_circle(N, loc, radius):
    angles = np.linspace(0, np.pi * 2, N + 1)[:-1]
    p = [(loc[0] + np.cos(a) * radius[0], loc[1] + np.sin(a) * radius[1]) for a in angles]
    return p


def xy_along_rect(N, loc, scale):
    x0, y0 = -scale
    x1, y1 = scale
    rext_x = [loc[0] + x for x in [x0, x1, x1, x0]]
    rext_y = [loc[1] + y for y in [y0, y0, y1, y1]]
    p = single_parametric_interpolate(rext_x, rext_y, numPts=N)
    return p


def xy_uniform_circle(N, loc, scale):
    angles = np.random.uniform(0, 2 * np.pi, N).tolist()
    xs = np.random.uniform(0, scale[0] ** 2, N) ** 0.5 * np.cos(angles)
    ys = np.random.uniform(0, scale[1] ** 2, N) ** 0.5 * np.sin(angles)
    p = [(loc[0] + x, loc[1] + y) for a, x, y in zip(angles, xs, ys)]
    return p

def xy_grid(grid_dims, area, loc=(0.0, 0.0)) :
    x0, y0 = loc
    X,Y=area
    Nx, Ny=grid_dims
    dx,dy=X/Nx, Y/Ny
    grid = np.meshgrid(np.linspace(x0-X/2+dx/2,x0+X/2+dx/2, Nx), np.linspace(y0-Y/2+dy/2,y0+Y/2+dy/2, Ny))
    cartprod = np.stack(grid, axis=-1).reshape(-1, 2)

    # Convert to list of tuples
    return list(map(tuple, cartprod))


def generate_xy_distro(mode, shape, N, loc=(0.0, 0.0), scale=(0.0, 0.0), area=None):
    loc, scale = np.array(loc), np.array(scale)
    if mode == 'uniform':
        if shape in ['circle', 'oval']:
            return xy_uniform_circle(N=N, loc=loc, scale=scale)
        elif shape == 'rect':
            return list(map(tuple, np.random.uniform(low=-scale, high=scale, size=(N, 2)) + loc))
    elif mode == 'normal':
        return np.random.normal(loc=loc, scale=scale / 2, size=(N, 2)).tolist()
    elif mode == 'periphery':
        if shape in ['circle', 'oval']:
            return xy_along_circle(N, loc=loc, radius=scale)
        elif shape == 'rect':
            return xy_along_rect(N, loc=loc, scale=scale)
    elif mode == 'grid':
        if type(N) == tuple:
            grid_dims = N
        else:
            Nx = int(np.sqrt(N))
            Ny = int(N / Nx)
            if Nx * Ny != N:
                raise
            grid_dims=(Nx,Ny)
        if area is None :
            area=scale
        return xy_grid(grid_dims, loc=loc, area=area)
    else:
        raise ValueError(f'XY distribution {mode} not implemented.')

def generate_xyNor_distro(d):
    N = d.N
    a1, a2 = np.deg2rad(d.orientation_range)
    ors = np.random.uniform(low=a1, high=a2, size=N).tolist()
    ps = generate_xy_distro(N=N, **{k: d[k] for k in ['mode', 'shape', 'loc', 'scale']})
    return ps, ors



def eudis5(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def eudi5x(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def xy_projection(point, angle: float, distance: float):
    return [point[0] + math.cos(angle) * distance,
        point[1] + math.sin(angle) * distance]


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

def comp_dst(s,c,point):
    xy_params = raw_or_filtered_xy(s, point)
    D = np.zeros([c.Nticks, c.N])
    for i, id in enumerate(c.agent_ids):
        xy=s[xy_params].xs(id, level='AgentID').values
        D[1:, i] = np.sqrt(np.diff(xy[:, 0]) ** 2 + np.diff(xy[:, 1]) ** 2)
    s[nam.dst(point)] = D.flatten()


def convex_hull(xs=None, ys=None, N=None, interp_nans=True):
    Nrows, Ncols = xs.shape
    xs = [xs[i][~np.isnan(xs[i])] for i in range(Nrows)]
    ys = [ys[i][~np.isnan(ys[i])] for i in range(Nrows)]
    ps = [np.vstack((xs[i], ys[i])).T for i in range(Nrows)]
    xxs = np.zeros((Nrows, N))
    xxs[:] = np.nan
    yys = np.zeros((Nrows, N))
    yys[:] = np.nan

    for i, p in enumerate(ps):
        if len(p) > 0:
            try:
                b = p[ConvexHull(p).vertices]
                s = np.min([b.shape[0], N])
                xxs[i, :s] = b[:s, 0]
                yys[i, :s] = b[:s, 1]
                if interp_nans:
                    xxs[i] = interpolate_nans(xxs[i])
                    yys[i] = interpolate_nans(yys[i])
            except:
                pass
    return xxs, yys


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


def compute_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def comp_bearing(xs, ys, ors, loc=(0.0, 0.0), in_deg=True):
    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(rads)

def dsp_single(xy0, s0, s1, dt):
    ids = xy0.index.unique('AgentID').values
    Nids=len(ids)
    t0 = int(s0 / dt)
    t1 = int(s1 / dt)
    Nt = t1 - t0
    xy = xy0.loc[(slice(t0, t1), slice(None)), ['x', 'y']]
    AA = np.zeros([Nt, Nids]) * np.nan
    fails=0
    for i, id in (enumerate(ids)):
        xy_i = xy.xs(id, level='AgentID')
        try:
            AA[:, i] = eudi5x(xy_i.values[1:], xy_i.dropna().values[0])
        except:
            fails+=1
            pass
    print(f'In {fails} out of {Nids} tracks failed to set origin point')

    trange = np.arange(t0, t1, 1)
    dsp_ar = np.zeros([Nt, 3]) * np.nan
    dsp_ar[:, 0] = np.nanquantile(AA, q=0.5, axis=1)
    dsp_ar[:, 1] = np.nanquantile(AA, q=0.75, axis=1)
    dsp_ar[:, 2] = np.nanquantile(AA, q=0.25, axis=1)
    df = pd.DataFrame(dsp_ar, index=trange, columns=['median', 'upper', 'lower'])
    return AA,df


def compute_velocity(xy, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    v = d / dt
    v = np.insert(v, 0, np.nan)
    d = np.insert(d, 0, np.nan)
    if return_dst:
        return v, d
    else:
        return v


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

    mi, ma = argrelextrema(density, np.less)[0], argrelextrema(density, np.greater)[0]
    try:
        minimum = vals[mi][0]
    except:
        minimum = np.nan
    return minimum


def comp_rate(s,c, p, pv=None):
    if pv is None :
        pv = nam.vel(p)
    V = np.zeros([c.Nticks, c.N]) * np.nan
    for i, id in enumerate(c.agent_ids):
        V[1:, i] = np.diff(s[p].xs(id, level='AgentID').values) / c.dt

    s[pv] = V.flatten()
