import numpy as np
import pandas as pd

from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux





@reg.funcs.preproc("filter_f")
def filter(s, c, filter_f=2, inplace=True, recompute=False, **kwargs):
    if filter_f in ['', None, np.nan]:
        return
    if 'filtered_at' in c and not recompute:
        print(
            f'Dataset already filtered at {c["filtered_at"]}. If you want to apply additional filter set recompute to True')
        return
    c['filtered_at'] = filter_f

    points = nam.midline(c.Npoints, type='point') + ['centroid', '']
    pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    data = np.dstack(list(s[pars].groupby('AgentID').apply(pd.DataFrame.to_numpy)))
    f_array = aux.apply_filter_to_array_with_nans_multidim(data, freq=filter_f, fr=1 / c.dt)
    fpars = nam.filt(pars) if not inplace else pars
    for j, p in enumerate(fpars):
        s[p] = f_array[:, j, :].flatten()
    print(f'All spatial parameters filtered at {filter_f} Hz')

@reg.funcs.preproc("interpolate_nans")
def interpolate_nan_values(s, c, pars=None, **kwargs):
    if pars is None:
        points = nam.midline(c.Npoints, type='point') + ['centroid', ''] + nam.contour(
            c.Ncontour)  # changed from N and Nc to N[0] and Nc[0] as comma above was turning them into tuples, which the naming function does not accept.
        pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    for p in pars:
        for id in s.index.unique('AgentID').values:
            s.loc[(slice(None), id), p] = aux.interpolate_nans(s[p].xs(id, level='AgentID', drop_level=True).values)
    print('All parameters interpolated')

@reg.funcs.preproc("rescale_by")
def rescale(s, e, c, recompute=False, rescale_by=1.0, **kwargs):
    # print(rescale_by)
    # raise
    if rescale_by in ['', None, np.nan]:
        return
    if 'rescaled_by' in c and not recompute:
        print(
            f'Dataset already rescaled by {c["rescaled_by"]}. If you want to rescale again set recompute to True')
        return
    c['rescaled_by'] = rescale_by
    points = nam.midline(c.Npoints, type='point') + ['centroid', '']
    contour_pars = nam.xy(nam.contour(c.Ncontour), flat=True)
    pars = nam.xy(points, flat=True) + nam.dst(points) + nam.vel(points) + nam.acc(points) + [
        'spinelength'] + contour_pars
    lin_pars = [p for p in pars if p in s.columns]
    for p in lin_pars:
        s[p] = s[p].apply(lambda x: x * rescale_by)
    if 'length' in e.columns:
        e['length'] = e['length'].apply(lambda x: x * rescale_by)
    print(f'Dataset rescaled by {rescale_by}.')

@reg.funcs.preproc("drop_collisions")
def exclude_rows(s, e, c, flag='collision_flag',  accepted=[0], rejected=None, **kwargs):
    if accepted is not None:
        s.loc[s[flag] != accepted[0]] = np.nan
    if rejected is not None:
        s.loc[s[flag] == rejected[0]] = np.nan

    for id in s.index.unique('AgentID').values:
        e.loc[id, 'cum_dur'] = len(s.xs(id, level='AgentID', drop_level=True).dropna()) * c.dt

    print(f'Rows excluded according to {flag}.')


@reg.funcs.proc("traj_colors")
def generate_traj_colors(s, sp_vel=None, ang_vel=None, **kwargs):
    N = len(s.index.unique('Step'))
    if sp_vel is None:
        sp_vel = 'scaled_velocity'
    if ang_vel is None:
        ang_vel = 'front_orientation_velocity'
    pars = [sp_vel, ang_vel]
    edge_colors = [[(255, 0, 0), (0, 255, 0)], [(255, 0, 0), (0, 255, 0)]]
    labels = ['lin_color', 'ang_color']
    lims = [0.8, 300]
    for p, c, l, lim in zip(pars, edge_colors, labels, lims):
        if p in s.columns:
            (r1, b1, g1), (r2, b2, g2) = c
            r, b, g = r2 - r1, b2 - b1, g2 - g1
            temp = np.clip(s[p].abs().values / lim, a_min=0, a_max=1)
            s[l] = [(r1 + r * t, b1 + b * t, g1 + g * t) for t in temp]
        else:
            s[l] = [(np.nan, np.nan, np.nan)] * N
    # return s

@reg.funcs.proc("PI")
def comp_dataPI(s,e,c, **kwargs):
    # from lib.process.angular import angular_processing
    from larvaworld.lib.process.spatial import comp_PI, comp_PI2
    if 'x' in e.keys():
        px = 'x'
        xs = e[px].values
    elif nam.final('x') in e.keys():
        px = nam.final('x')
        xs = e[px].values
    elif 'x' in s.keys():
        px = 'x'
        xs = s[px].dropna().groupby('AgentID').last().values
    elif 'centroid_x' in s.keys():
        px = 'centroid_x'
        xs = s[px].dropna().groupby('AgentID').last().values
    else:
        raise ValueError('No x coordinate found')
    PI, N, N_l, N_r = comp_PI(xs=xs, arena_xdim=c.env_params.arena.dims[0], return_num=True,
                              return_all=True)
    c.PI = {'PI': PI, 'N': N, 'N_l': N_l, 'N_r': N_r}
    try:
        c.PI2 = comp_PI2(xys=s[nam.xy('')], arena_xdim=c.env_params.arena.dims[0])
    except:
        pass


