import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


from lib.registry.base import BaseConfDict
from lib.registry.pars import preg

from lib.aux import dictsNlists as dNl, naming as nam, sim_aux, xy_aux, stdout


def comp_extrema(s, dt, parameters, interval_in_sec, threshold_in_std=None, abs_threshold=None):
    if abs_threshold is None:
        abs_threshold = [+np.inf, -np.inf]
    order = np.round(interval_in_sec / dt).astype(int)
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Npars = len(parameters)
    Nticks = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()

    min_array = np.ones([Nticks, Npars, Nids]) * np.nan
    max_array = np.ones([Nticks, Npars, Nids]) * np.nan
    for i, p in enumerate(parameters):
        p_min, p_max = nam.min(p), nam.max(p)
        s[p_min] = np.nan
        s[p_max] = np.nan
        d = s[p]
        std = d.std()
        mu = d.mean()
        if threshold_in_std is not None:
            thr_min = mu - threshold_in_std * std
            thr_max = mu + threshold_in_std * std
        else:
            thr_min, thr_max = abs_threshold
        for j, id in enumerate(ids):
            df = d.xs(id, level='AgentID', drop_level=True)
            i_min = argrelextrema(df.values, np.less_equal, order=order)[0]
            i_max = argrelextrema(df.values, np.greater_equal, order=order)[0]

            i_min_dif = np.diff(i_min, append=order)
            i_max_dif = np.diff(i_max, append=order)
            i_min = i_min[i_min_dif >= order]
            i_max = i_max[i_max_dif >= order]

            i_min = i_min[df.loc[i_min + t0] < thr_min]
            i_max = i_max[df.loc[i_max + t0] > thr_max]

            min_array[i_min, i, j] = True
            max_array[i_max, i, j] = True

        s[p_min] = min_array[:, i, :].flatten()
        s[p_max] = max_array[:, i, :].flatten()


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
    f_array = sim_aux.apply_filter_to_array_with_nans_multidim(data, freq=filter_f, fr=1 / c.dt)
    fpars = nam.filt(pars) if not inplace else pars
    for j, p in enumerate(fpars):
        s[p] = f_array[:, j, :].flatten()
    print(f'All spatial parameters filtered at {filter_f} Hz')


def interpolate_nan_values(s, c, pars=None, **kwargs):
    if pars is None:
        points = nam.midline(c.Npoints, type='point') + ['centroid', ''] + nam.contour(
            c.Ncontour)  # changed from N and Nc to N[0] and Nc[0] as comma above was turning them into tuples, which the naming function does not accept.
        pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    for p in pars:
        for id in s.index.unique('AgentID').values:
            s.loc[(slice(None), id), p] = xy_aux.interpolate_nans(s[p].xs(id, level='AgentID', drop_level=True).values)
    print('All parameters interpolated')


def rescale(s, e, c, recompute=False, rescale_by=1.0, **kwargs):
    # if Npoints is None:
    #     Npoints = c['Npoints']
    # if Ncontour is None:
    #     Ncontour = c['Ncontour']
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


def exclude_rows(s, e, c, flag='collision_flag',  accepted=[0], rejected=None, **kwargs):
    if accepted is not None:
        s.loc[s[flag] != accepted[0]] = np.nan
    if rejected is not None:
        s.loc[s[flag] == rejected[0]] = np.nan

    for id in s.index.unique('AgentID').values:
        e.loc[id, preg.getPar('cum_t')] = len(s.xs(id, level='AgentID', drop_level=True).dropna()) * c.dt

    print(f'Rows excluded according to {flag}.')



def generate_traj_colors(s, sp_vel=None, ang_vel=None, **kwargs):
    N = len(s.index.unique('Step'))
    if sp_vel is None:
        sp_vel = preg.getPar('sv')
    if ang_vel is None:
        ang_vel = preg.getPar('fov')
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


def comp_dataPI(s,e,c):
    # from lib.process.angular import angular_processing
    from lib.process.spatial import comp_PI, comp_PI2
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
    PI, N, N_l, N_r = comp_PI(xs=xs, arena_xdim=c.env_params.arena.arena_dims[0], return_num=True,
                              return_all=True)
    c.PI = {'PI': PI, 'N': N, 'N_l': N_l, 'N_r': N_r}
    try:
        c.PI2 = comp_PI2(xys=s[nam.xy('')], arena_xdim=c.env_params.arena.arena_dims[0])
    except:
        pass

def processing_funcs():
    from lib.process.angular import angular_processing
    from lib.process.spatial import spatial_processing, comp_source_metrics, comp_dispersion, comp_straightness_index, comp_wind
    func_dict = dNl.NestDict({
        'angular': angular_processing,
        'spatial': spatial_processing,
        'source': comp_source_metrics,
        'wind': comp_wind,
        'dispersion': comp_dispersion,
        'tortuosity': comp_straightness_index,
        'PI': comp_dataPI,
        'traj_colors': generate_traj_colors,

    })
    return func_dict

def preproccesing_funcs():
    from lib.process.spatial import align_trajectories
    func_dict = dNl.NestDict({
        'rescale_by': rescale,
        'drop_collisions': exclude_rows,
        'interpolate_nans': interpolate_nan_values,
        'filter_f': filter,
        'transposition': align_trajectories,
        # 'tortuosity': align_trajectories,

    })
    return func_dict

def annotation_funcs():
    from lib.process.spatial import align_trajectories
    from lib.anal.fitting import fit_epochs, get_bout_distros
    from lib.process.aux import comp_chunk_dicts

    klist=[
    ['chunk_dicts', comp_chunk_dicts, ['s', 'e', 'c']],
    ['grouped_epochs', dNl.group_epoch_dicts, ['chunk_dicts']],
    ['fitted_epochs', fit_epochs, ['grouped_epochs']],
    ['bout_distros', get_bout_distros, ['fitted_epochs']]
        ]

    func_dict = dNl.NestDict({k[0] : {'func' : k[1], 'required_ks' :k[2]} for k in klist})



    return func_dict


#
# class ProcFuncDict:
#     def __init__(self, load=False):
#         self.dict_path = preg.paths['ProcFuncDict']
#         if not load:
#             self.dict = proc_func_dict()
#             self.predict = preproc_func_dict()
#             # dNl.save_dict(self.dict, self.dict_path)
#         else:
#             self.dict = dNl.load_dict(self.dict_path)

# procfunc_dict=ProcFuncDict()

class ProcFuncDict(BaseConfDict):

    def build(self):
        d=dNl.NestDict({'preproc' : preproccesing_funcs(), 'proc' : processing_funcs(),
                        'annotation' : annotation_funcs()})
        return d

