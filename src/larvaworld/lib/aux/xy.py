import math
import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt


from larvaworld.lib.aux import naming as nam


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




def eudist(xy) :
    A= np.sqrt(np.nansum(np.diff(xy, axis=0)**2, axis=1))
    A = np.insert(A, 0, 0)
    return A

def eudi5x(a, b):
    return np.sqrt(np.sum((a - np.array(b)) ** 2, axis=1))


def xy_projection(point, angle: float, distance: float):
    return [point[0] + math.cos(angle) * distance,
        point[1] + math.sin(angle) * distance]





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
                b = p[sp.spatial.ConvexHull(p).vertices]
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
        xy_valid =xy.dropna().values
        xy = xy.values
    else :
        xy_valid=xy[~np.isnan(xy)]
    if xy_valid.shape[0]>1 :
        return eudi5x(xy, xy_valid[0])
    else:
        return np.zeros(xy.shape[0]) * np.nan

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


def circle_to_polygon(sides, radius, rotation=0, translation=None):
    one_segment = np.pi * 2 / sides

    points = [
        (np.sin(one_segment * i + rotation) * radius,
         np.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return np.array(points)


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
                df = d.read(key='end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read(key='step')
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


def mdict2df(mdict, columns=['symbol', 'value', 'description']):
    data = []
    for k, p in mdict.items():
        entry = [getattr(p, col) for col in columns]
        data.append(entry)
    df = pd.DataFrame(data, columns=columns)
    df.set_index(columns[0], inplace=True)
    return df


def body_contour(points=[(0.9, 0.1), (0.05, 0.1)], start=(1, 0), stop=(0, 0)):
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy

from shapely.geometry import Point, Polygon

def body_contour2(points=[(0.9, 0.1), (0.05, 0.1)], start=(1, 0), stop=(0, 0)):
    ps1 = [Point(p) for p in points]
    ps2 = [Point(p.x,-p.y) for p in ps1]
    ps2.reverse()
    ps=[Point(start)] + ps1 + [Point(stop)] + ps2
    pol=Polygon([[p.x,p.y] for p in ps])
    return pol


def apply_per_level(s, func, level='AgentID', **kwargs):
    '''
    Apply a function to each subdataframe of a dataframe after grouping by level

    Args:
        s: MultiIndex Dataframe with levels : ['Step', 'AgentID']
        func:function to apply on each subdataframe

    Returns:
        A : Array of dimensions [Nticks, Nids]
    '''

    ids = s.index.unique('AgentID').values
    N = s.index.unique('Step').size
    A = np.zeros([N, len(ids)]) * np.nan

    for i, (v, ss) in enumerate(s.groupby(level=level)):
        ss = ss.droplevel(level)
        if level=='AgentID' :
            A[:, i] = func(ss, **kwargs)
        elif level=='Step' :
            A[i, :] = func(ss, **kwargs)
    return A


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


def compute_dst(s,point='') :
    s[nam.dst(point)] = apply_per_level(s[nam.xy(point)], eudist).flatten()

def epochs_by_thr(a, thr, lower=True, min_size=1) :
    if lower:
        b=np.where(a<= thr,1,0)
    else:
        b=np.where(a>= thr,1,0)
    c=np.where(b[:-1] != b[1:])[0]
    if b[0]==1 and c[0]!=0:
        c=np.insert(c,0,0)
    if c.shape[0]%2==1 :
        c=np.append(c,a.shape[0])
    epochs = c.reshape(-1,2)
    return epochs[(np.diff(epochs)>=min_size).T[0]]

