import numpy as np


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
    ors = (np.random.uniform(low=a1, high=a2, size=N)%(2*np.pi)).tolist()
    ps = generate_xy_distro(N=N, mode=d.mode,shape=d.shape, loc=d.loc, scale=d.scale)
    # ps = generate_xy_distro(N=N, **{k: d[k] for k in ['mode', 'shape', 'loc', 'scale']})
    return ps, ors





