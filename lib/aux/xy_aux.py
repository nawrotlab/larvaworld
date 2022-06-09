import math

import numpy as np

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
    # print(r, N, loc)
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


def generate_xy_distro(mode, shape, N, loc=(0.0, 0.0), scale=(0.0, 0.0)):
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
            return xy_along_rect(N, loc, scale)
    else:
        raise ValueError(f'XY distribution {mode} not implemented.')


def eudis5(v1, v2):
    dist = [(a - b) ** 2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def eudi5x(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def xy_projection(point, angle: float, distance: float):
    return [point[0] + math.cos(angle) * distance,
        point[1] + math.sin(angle) * distance]


def comp_rate(s,c, p, pv=None):
    if pv is None :
        pv = nam.vel(p)
    V = np.zeros([c.Nticks, c.N]) * np.nan
    # print(V.shape)
    # print(V.flatten().shape)
    # print(s[p].values.shape)
    for i, id in enumerate(c.agent_ids):
        # print(np.diff(s[p].xs(id, level='AgentID').values).shape)
        V[1:, i] = np.diff(s[p].xs(id, level='AgentID').values) / c.dt

    s[pv] = V.flatten()





def raw_or_filtered_xy(s, points):
    r = nam.xy(points, flat=True)
    f = nam.filt(r)
    if all(i in s.columns for i in f):
        # print('Using filtered xy coordinates')
        return f
    elif all(i in s.columns for i in r):
        # print('Using raw xy coordinates')
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
