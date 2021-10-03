import numpy as np
from typing import List, Tuple, Union
import pandas as pd
from siunits import BaseUnit, Composite, DerivedUnit

import lib.aux.dictsNlists
from lib.aux import colsNstr as fun, naming as nam
from lib.aux.collecting import output_keys
from lib.gui.aux.functions import get_pygame_key
from lib.stor import paths

def maxNdigits(array, Min=None) :
    N= len(max(array.astype(str), key=len))
    if Min is not None :
        N=max([N,Min])
    return N

def base_dtype(t):
    if t in [float, Tuple[float], List[float], List[Tuple[float]]]:
        base_t = float
    elif t in [int, Tuple[int], List[int], List[Tuple[int]]]:
        base_t = int
    else:
        base_t = t
    return base_t


def par(name, t=float, v=None, vs=None, min=None, max=None, dv=None, aux_values=None,
        return_dtype=True, Ndigits=None):
    cur_dtype = base_dtype(t)
    if vs is None:
        if any([arg is not None for arg in [min, max, dv]]):
            if cur_dtype in [float, int]:
                if min is None:
                    min = 0
                if max is None:
                    max = 1
                if dv is None:
                    if cur_dtype == int:
                        dv = 1
                    elif cur_dtype == float:
                        dv = 0.1

                array = np.arange(min, max + dv, dv)
                if cur_dtype == float:
                    Ndecimals = len(str(format(dv, 'f')).split('.')[1])
                    array = np.round(array, Ndecimals)
                vs = array.astype(cur_dtype)
                Ndigits = maxNdigits(vs, 4)
                vs = vs.tolist()

    if aux_values is not None and vs is not None:
        vs += aux_values
    d = {'initial_value': v, 'values': vs}
    if return_dtype:
        d['dtype'] = t
    d['Ndigits'] = Ndigits
    # print({name: d})
    return {name: d}


def par_dict(name, d0=None, **kwargs):
    if d0 is None:
        d0 = init_pars()[name]
    # print(d0)
    d = {}
    for n, v in d0.items():
        try:
            entry = par(n, **v, **kwargs)
        except:
            # print(n, d0[n])
            entry = {n: {'dtype': dict, 'content': par_dict(n, d0=d0[n])}}
        # print(entry)
        d.update(entry)
    # return {name : d}
    return d


def par_dict_from_df(name, df):
    df = df.where(pd.notnull(df), None)
    d = {}
    for n in df.index:
        # print(df.loc[n])
        entry = par(n, **df.loc[n])
        d.update(entry)
    return {name: d}


def pars_to_df(d):
    df = pd.DataFrame.from_dict(d, orient='index',
                                columns=['dtype', 'initial_value', 'value_list', 'min', 'max', 'interval'])
    df.index.name = 'name'
    df = df.where(pd.notnull(df), None)


def init_vis_dtypes2():
    d = {}
    d['render'] = {
        'mode': {'t': str, 'v': 'video', 'vs': [None, 'video', 'image']},
        'image_mode': {'t': str, 'vs': [None, 'final', 'snapshots', 'overlap']},
        'video_speed': {'t': int, 'v': 60, 'min': 1, 'max': 100},
        'media_name': {'t': str},
        'show_display': {'t': bool, 'v': True},
    }

    d['draw'] = {
        'draw_head': {'t': bool, 'v': False},
        'draw_centroid': {'t': bool, 'v': False},
        'draw_midline': {'t': bool, 'v': True},
        'draw_contour': {'t': bool, 'v': True},
        'draw_sensors': {'t': bool, 'v': False},
        'trails': {'t': bool, 'v': False},
        'trajectory_dt': {'max': 100.0},
    }

    d['color'] = {
        'black_background': {'t': bool, 'v': False},
        'random_colors': {'t': bool, 'v': False},
        'color_behavior': {'t': bool, 'v': False},
    }

    d['aux'] = {
        'visible_clock': {'t': bool, 'v': True},
        'visible_scale': {'t': bool, 'v': True},
        'visible_state': {'t': bool, 'v': False},
        'visible_ids': {'t': bool, 'v': False},
    }
    d['visualization'] = {
        'render': d['render'],
        'draw': d['draw'],
        'color': d['color'],
        'aux': d['aux'],
    }
    # d['visualization'] = {'dtype': dict, 'content': {
    #     'render': {'dtype': dict, 'content':vis_render_dtypes},
    #     'draw': {'dtype': dict, 'content':vis_draw_dtypes},
    #     'color': {'dtype': dict, 'content':vis_color_dtypes},
    #     'aux': {'dtype': dict, 'content':vis_aux_dtypes},
    # }}
    return d


proc_type_keys = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI']
bout_keys = ['stride', 'pause', 'turn']

# Compound densities (g/cm**3)
substrate_dict = {
    'agar': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 16 / 1000,
        'cornmeal': 0
    },
    'standard': {
        'glucose': 100 / 1000,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 50 / 1000,
        'agar': 16 / 1000,
        'cornmeal': 0
        # 'KPO4': 0.1/1000,
        # 'Na_K_tartrate': 8/1000,
        # 'NaCl': 0.5/1000,
        # 'MgCl2': 0.5/1000,
        # 'Fe2(SO4)3': 0.5/1000,
    },
    'cornmeal': {
        'glucose': 517 / 17000,
        'dextrose': 1033 / 17000,
        'saccharose': 0,
        'yeast': 0,
        'agar': 93 / 17000,
        'cornmeal': 1716 / 17000
    },
    'PED_tracker': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 2 / 200,
        'yeast': 3 * 0.05 * 0.125 / 0.1,
        'agar': 500 * 2 / 200,
        'cornmeal': 0
    },
    #     [1] M. E. Wosniack, N. Hu, J. Gjorgjieva, and J. Berni, “Adaptation of Drosophila larva foraging in response to changes in food distribution,” bioRxiv, p. 2021.06.21.449222, 2021.
    'cornmeal2': {
        'glucose': 0,
        'dextrose': 450 / 6400,
        'saccharose': 0,
        'yeast': 90 / 6400,
        'agar': 42 / 6400,
        'cornmeal': 420 / 6400
    },
    'sucrose': {
        'glucose': 3.42 / 200,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 0.8 / 200,
        'cornmeal': 0
    }
    # 'apple_juice': {
    #         'glucose': 0.342/200,
    #         'dextrose': 0,
    #         'saccharose': 0,
    #         'yeast': 0,
    #         'agar': 0.8 / 200,
    #         'cornmeal': 0,
    #         'apple_juice': 1.05*5/200,
    #     },

}
null_bout_dist = {
    'fit': True,
    'range': None,
    'name': None,
    'mu': None,
    'sigma': None,
    # 'c': None
}


def init_pars():
    from lib.conf.conf import loadConfDict

    bout_dist_dtypes = {
        'fit': {'t': bool, 'v': True},
        'range': {'t': Tuple[float], 'max': 100.0},
        'name': {'t': str,
                 'vs': ['powerlaw', 'exponential', 'lognormal', 'lognormal-powerlaw', 'levy', 'normal', 'uniform']},
        'mu': {'max': 10.0},
        'sigma': {'max': 10.0},
    }

    d = {
        'xy': {'t': Tuple[float], 'v': (0.0, 0.0), 'min': -1.0, 'max': 1.0},
        'odor': {
            'odor_id': {'t': str},
            'odor_intensity': {'max': 10.0},
            'odor_spread': {'max': 10.0}
        },
        'food': {
            'radius': {'v': 0.003, 'max': 0.1, 'dv': 0.001},
            'amount': {'v': 0.0, 'max': 1.0},
            'quality': {'v': 1.0, 'max': 1.0},
            'can_be_carried': {'t': bool, 'v': False},
            'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
        },
        'food_grid':
            {'unique_id': {'t': str, 'v': 'Food_grid', },
             'grid_dims': {'t': Tuple[int], 'v': (20, 20), 'min': 10, 'max': 200},
             'initial_value': {'v': 0.001, 'max': 0.1, 'dv': 0.001},
             'distribution': {'t': str, 'v': 'uniform', 'vs': ['uniform']},
             'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
             },
        'life':
            {
                'epochs': {'t': List[Tuple[float]], 'max': 250.0, 'dv': 0.1},
                'epoch_qs': {'t': List[float], 'max': 1.0},
                'hours_as_larva': {'v': 0.0, 'max': 250.0, 'dv': 1.0},
                'substrate_quality': {'v': 1.0, 'max': 1.0, 'dv': 0.05},
                'substrate_type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
            },
        'odorscape': {'odorscape': {'t': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion']},
                      'grid_dims': {'t': Tuple[int], 'min': 10, 'max': 100},
                      'evap_const': {'max': 1.0},
                      'gaussian_sigma': {'t': Tuple[float], 'max': 1.0}
                      },
        'odor_gains': {
            'unique_id': {'t': str},
            'mean': {'max': 1000.0, 'dv': 10.0},
            'std': {'max': 10.0, 'dv': 1.0}
        },
        'optimization': {
            'fit_par': {'t': str},
            'minimize': {'t': bool, 'v': True},
            'threshold': {'v': 0.001, 'max': 0.01, 'dv': 0.0001},
            'max_Nsims': {'t': int, 'v': 7, 'max': 100},
            'Nbest': {'t': int, 'v': 3, 'max': 10},
            'operations': {
                'mean': {'t': bool, 'v': True},
                'std': {'t': bool, 'v': False},
                'abs': {'t': bool, 'v': False}
            },
        },
        'batch_methods': {
            'run': {'t': str, 'v': 'default', 'vs': ['null', 'default', 'deb', 'odor_preference', 'exp_fit']},
            'post': {'t': str, 'v': 'default', 'vs': ['null', 'default']},
            'final': {'t': str, 'v': 'null', 'vs': ['null', 'scatterplots', 'deb', 'odor_preference']}
        },
        'space_search_par': {
            'range': {'t': Tuple[float], 'max': 100.0, 'min': -100.0, 'dv': 1.0},
            'Ngrid': {'t': int, 'max': 100},
            'values': {'t': List[float], 'min': -100.0, 'max': 100.0}
        },
        'space_search': {'pars': {'t': List[str]},
                         'ranges': {'t': List[Tuple[float]], 'max': 100.0, 'min': -100.0, 'dv': 1.0},
                         'Ngrid': {'t': int, 'max': 100}},
        # 'body': {'initial_length': {'v': 'sample', 'max': 0.01, 'dv': 0.0001, 'aux_values': ['sample']},
        'body': {'initial_length': {'v': 'sample', 'max': 0.01, 'dv': 0.0001, 'aux_values': ['sample']},
                 'length_std': {'v': 0.0, 'max': 0.001, 'dv': 0.0001, 'aux_values': ['sample']},
                 'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12},
                 'seg_ratio': {'max': 1.0},  # [5 / 11, 6 / 11]
                 'touch_sensors': {'t': bool, 'v': False},
                 },
        'arena': {'arena_dims': {'t': Tuple[float], 'v': (0.1, 0.1), 'max': 1.0, 'dv': 0.01},
                  'arena_shape': {'t': str, 'v': 'circular', 'vs': ['circular', 'rectangular']}
                  },
        'physics': {
            'torque_coef': {'v': 0.41, 'max': 5.0, 'dv': 0.01},
            'ang_damping': {'v': 2.5, 'max': 10.0},
            'body_spring_k': {'v': 0.02, 'max': 1.0, 'dv': 0.01},
            'bend_correction_coef': {'v': 1.4, 'max': 10.0},
        },
        'energetics': {'f_decay': {'v': 0.1, 'max': 1.0, 'dv': 0.1},
                       'absorption': {'max': 1.0},
                       'hunger_as_EEB': {'t': bool, 'v': True},
                       'hunger_gain': {'v': 0.0, 'max': 1.0},
                       'deb_on': {'t': bool, 'v': True},
                       'assimilation_mode': {'t': str, 'v': 'gut', 'vs': ['sim', 'gut', 'deb']},
                       'DEB_dt': {'max': 1.0}},
        'crawler': {'waveform': {'t': str, 'v': 'realistic', 'vs': ['realistic', 'square', 'gaussian', 'constant']},
                    'freq_range': {'t': Tuple[float], 'v': (0.5, 2.5), 'max': 2.0},
                    'initial_freq': {'v': 'sample', 'max': 10.0, 'aux_values': ['sample']},  # From D1 fit
                    'freq_std': {'v': 0.0, 'max': 1.0},  # From D1 fit
                    'step_to_length_mu': {'v': 'sample', 'max': 1.0, 'dv': 0.01, 'aux_values': ['sample']},
                    # From D1 fit
                    'step_to_length_std': {'v': 'sample', 'max': 1.0, 'aux_values': ['sample']},  # From D1 fit
                    'initial_amp': {'max': 2.0},
                    'noise': {'v': 0.1, 'max': 1.0, 'dv': 0.01},
                    'max_vel_phase': {'v': 1.0, 'max': 2.0}
                    },
        'turner': {'mode': {'t': str, 'v': 'neural', 'vs': ['', 'neural', 'sinusoidal']},
                   'base_activation': {'v': 20.0, 'max': 100.0, 'dv': 1.0},
                   'activation_range': {'t': Tuple[float],'v': (10.0,40.0), 'max': 100.0, 'dv': 1.0},
                   'noise': {'v': 0.15, 'max': 10.0},
                   'activation_noise': {'v': 0.5, 'max': 10.0},
                   'initial_amp': {'max': 20.0},
                   'amp_range': {'t': Tuple[float], 'max': 20.0},
                   'initial_freq': {'max': 2.0},
                   'freq_range': {'t': Tuple[float], 'max': 2.0},
                   },
        'interference': {
            'crawler_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2.0},  # np.pi * 0.55,  # 0.9, #,
            'feeder_phi_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 2.0},
            'attenuation': {'v': 1.0, 'max': 1.0}
        },
        'intermitter': {
            'pause_dist': bout_dist_dtypes,
            # 'pause_dist': dict,
            'stridechain_dist': bout_dist_dtypes,
            # 'stridechain_dist': dict,
            'crawl_bouts': {'t': bool, 'v': True},
            'feed_bouts': {'t': bool, 'v': False},
            'crawl_freq': {'v': 1.43, 'max': 2.0, 'dv': 0.01},
            'feed_freq': {'v': 2.0, 'max': 4.0, 'dv': 0.01},
            'feeder_reoccurence_rate': {'max': 1.0},
            'EEB_decay': {'v': 1.0, 'max': 2.0},
            'EEB': {'v': 0.0, 'max': 1.0}},
        'olfactor': {
            'perception': {'t': str, 'v': 'log', 'vs': ['log', 'linear']},
            'olfactor_noise': {'v': 0.0, 'max': 1.0},
            'decay_coef': {'v': 0.0, 'max': 2.0}},
        'feeder': {'freq_range': {'t': Tuple[float], 'v': (1.0, 3.0), 'max': 4.0},
                   'initial_freq': {'v': 2.0, 'max': 4.0},
                   'feed_radius': {'v': 0.1, 'max': 10.0},
                   'V_bite': {'v': 0.0002, 'max': 0.001, 'dv': 0.0001}},
        'memory': {'DeltadCon': {'v': 2.0, 'max': 10.0},
                   'state_spacePerOdorSide': {'t': int, 'v': 0, 'max': 20},
                   'gain_space': {'t': List[float], 'v': [-300.0, -50.0, 50.0, 300.0], 'min': 1000.0, 'max': 1000.0,
                                  'dv': 1.0},
                   'decay_coef_space': {'t': List[float], 'max': 10.0},
                   'update_dt': {'v': 1.0, 'max': 10.0, 'dv': 1.0},
                   'alpha': {'v': 0.05, 'max': 1.0, 'dv': 0.01},
                   'gamma': {'v': 0.6, 'max': 1.0},
                   'epsilon': {'v': 0.3, 'max': 1.0},
                   'train_dur': {'v': 20.0, 'max': 100.0},
                   },
        'modules': {'turner': {'t': bool, 'v': False},
                    'crawler': {'t': bool, 'v': False},
                    'interference': {'t': bool, 'v': False},
                    'intermitter': {'t': bool, 'v': False},
                    'olfactor': {'t': bool, 'v': False},
                    'feeder': {'t': bool, 'v': False},
                    'memory': {'t': bool, 'v': False}},

        'sim_params': {
            'sim_ID': {'t': str},
            'path': {'t': str},
            'duration': {'v': 1.0, 'max': 100.0},
            'timestep': {'v': 0.1, 'max': 0.4, 'dv': 0.05},
            'Box2D': {'t': bool, 'v': False},
            'sample': {'t': str, 'v': 'reference', 'vs': list(loadConfDict('Ref').keys())}
        },
        'essay_params': {
            'essay_ID': {'t': str},
            'path': {'t': str},
            'N': {'t': int, 'min': 1, 'max': 10}
        },
        'logn_dist': {
            'range': {'t': Tuple[float], 'v': (0.0, 2.0), 'max': 10.0, 'dv': 1.0},
            'name': {'t': str, 'v': 'lognormal', 'vs': ['lognormal']},
            'mu': {'v': 1.0, 'max': 10.0},
            'sigma': {'v': 0.0, 'max': 10.0},
            'fit' : {'t' : bool, 'v' : False}
        },
        'par': {
            'p': {'t': str},
            'u': {'t': Union[BaseUnit, Composite, DerivedUnit]},
            'k': {'t': str},
            's': {'t': str},
            'o': {'t': type},
            'lim': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 4.0},
            'd': {'t': str},
            'l': {'t': str},
            'exists': {'t': bool, 'v': False},
            'func': {'t': str},
            'const': {'t': str},
            'operator': {'t': str, 'vs': [None, 'diff', 'cum', 'max', 'min', 'mean', 'std', 'final']},
            # 'diff': bool,
            # 'cum': bool,
            'k0': {'t': str},
            'k_num': {'t': str},
            'k_den': {'t': str},
            'dst2source': {'t': Tuple[float], 'min': -100.0, 'max': 100.0},
            'or2source': {'t': Tuple[float], 'min': -180.0, 'max': 180.0},
            'dispersion': {'t': bool, 'v': False},
            'wrap_mode': {'t': str, 'vs': [None, 'zero', 'positive']}
        },

        'build_conf': {
            'min_duration_in_sec': {'v': 170.0, 'max': 3600.0, 'dv': 1.0},
            'min_end_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0},
            'start_time_in_sec': {'v': 0.0, 'max': 3600.0, 'dv': 1.0},
            'max_Nagents': {'t': int, 'v': 500, 'max': 500},
            'save_mode': {'t': str, 'v': 'semifull', 'vs': ['minimal', 'semifull', 'full', 'points']},
        },
        'substrate': {k: {'v': v, 'max': 1000.0, 'dv': 5.0} for k, v in substrate_dict['standard'].items()},
        'output': {n: {'t': bool, 'v': False} for n in output_keys}
    }

    d['brain'] = {
        'modules' : d['modules'],
        **{f'{m}_params': d[m] for m in d['modules'].keys()},
        'nengo' : {'t': bool, 'v': False}
    }

    d['larva_conf'] ={
        'brain' : d['brain'],
        'body' : d['body'],
        'energetics' : d['energetics'],
        'physics' : d['physics'],
    }

    d['preprocessing'] = {
        'rescale_by': {'max': 10.0},
        'drop_collisions': {'t': bool, 'v': False},
        'interpolate_nans': {'t': bool, 'v': False},
        'filter_f': {'max': 10.0},
        'transposition': {'t': str, 'vs': ['', 'origin', 'arena', 'center']}
    }
    d['processing'] = {
        'types': {t: {'t': bool, 'v': False} for t in proc_type_keys},
        'dsp_starts': {'t': List[float], 'max': 200.0, 'dv': 1.0},
        'dsp_stops': {'t': List[float], 'max': 200.0, 'dv': 1.0},
        'tor_durs': {'t': List[int], 'max': 100, 'dv': 1}}
    d['annotation'] = {'bouts': {b: {'t': bool, 'v': False} for b in bout_keys},
                       'track_point': {'t': str},
                       'track_pars': {'t': List[str]},
                       'chunk_pars': {'t': List[str]},
                       'vel_par': {'t': str},
                       'ang_vel_par': {'t': str},
                       'bend_vel_par': {'t': str},
                       'min_ang': {'v': 0.0, 'max': 180.0, 'dv': 1.0},
                       'min_ang_vel': {'v': 0.0, 'max': 1000.0, 'dv': 1.0},
                       'non_chunks': {'t': bool, 'v': False}}
    d['enrich_aux'] = {'recompute': {'t': bool, 'v': False},
                       'mode': {'t': str, 'v': 'minimal', 'vs': ['minimal', 'full']},
                       'source': {'t': Tuple[float], 'min': -10.0, 'max': 10.0}
                       }
    d['to_drop'] = {'groups': {n: {'t': bool, 'v': False} for n in
                               ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn',
                                'turn','unused']}}
    d['enrichment'] = {k: d[k] for k in
                       ['preprocessing', 'processing', 'annotation', 'enrich_aux', 'to_drop']}

    d['food_params'] = {'source_groups': {'t': dict, 'v': {}},
                        'food_grid': {'t': dict},
                        'source_units': {'t': dict, 'v': {}}
                        }

    d['env_conf'] = {'arena': d['arena'],
                     'border_list': {'t': dict, 'v': {}},
                     'food_params': d['food_params'],
                     'larva_groups': {'t': dict, 'v': {}},
                     'odorscape': d['odorscape']}

    d['exp_conf'] = {'env_params': {'t': str, 'vs': list(loadConfDict('Env').keys())},
                     'sim_params': d['sim_params'],
                     'life_params': {'t': str, 'v': 'default', 'vs': list(loadConfDict('Life').keys())},
                     'collections': {'t': List[str], 'v': ['pose']},
                     'enrichment': d['enrichment'],
                     'experiment': {'t': str, 'vs': list(loadConfDict('Exp').keys())},
                     }
    d['batch_setup'] = {
        'batch_id': {'t': str},
        'save_hdf5': {'t': bool, 'v': False},
    }
    d['batch_conf'] = {'exp': {'t': str},
                       'space_search': d['space_search'],
                       'batch_methods': d['batch_methods'],
                       'optimization': d['optimization'],
                       'exp_kws': {'t': dict, 'v': {'enrichment': d['enrichment']}},
                       'post_kws': {'t': dict, 'v': {}},
                       'proc_kws': {'t': dict, 'v': {}},
                       'save_hdf5': {'t': bool, 'v': False}
                       }

    d['tracker'] = {
        'resolution': {'fr': {'v': 10.0, 'max': 100.0},
                       'Npoints': {'t': int, 'v': 1, 'max': 20},
                       'Ncontour': {'t': int, 'v': 0, 'max': 100}
                       },
        'filesystem': {
            'read_sequence': {'t': List[str]},
            'read_metadata': {'t': bool, 'v': False},
            'detect': {
                'folder': {'pref': {'t': str}, 'suf': {'t': str}},
                'file': {'pref': {'t': str}, 'suf': {'t': str}, 'sep': {'t': str}}
            }
        },
        'arena': d['arena']
    }
    d['parameterization'] = {'bend': {'t': str, 'v': 'from_angles', 'vs': ['from_angles', 'from_vectors']},
                             'front_vector': {'t': Tuple[int], 'v': (1, 2), 'min': -12, 'max': 12},
                             'rear_vector': {'t': Tuple[int], 'v': (-2, -1), 'min': -12, 'max': 12},
                             'front_body_ratio': {'v': 0.5, 'max': 1.0},
                             'point_idx': {'t': int, 'v': 1, 'min': -1, 'max': 12},
                             'use_component_vel': {'t': bool, 'v': False},
                             'scaled_vel_threshold': {'v': 0.2, 'max': 1.0}}

    d['spatial_distro'] = {
        'mode': {'t': str, 'v': 'normal', 'vs': ['normal', 'periphery', 'uniform']},
        'shape': {'t': str, 'v': 'circle', 'vs': ['circle', 'rect', 'oval']},
        'N': {'t': int, 'v': 10, 'max': 1000},
        'loc': d['xy'],
        'scale': d['xy'],
    }

    d['larva_distro'] = {
        **d['spatial_distro'],
        'orientation_range': {'t': Tuple[float], 'v': (0.0, 360.0), 'min': 0.0, 'max': 360.0, 'dv': 1.0}
    }

    d['larva_model'] = {'t': str, 'v': 'explorer', 'vs': list(loadConfDict('Model').keys())}

    d['Larva_DISTRO'] = {
        'model': d['larva_model'],
        **d['larva_distro'],
    }

    d['LarvaGroup'] = {
        'model': d['larva_model'],
        'life': d['life'],
        'sample': {'t': str, 'v': 'controls.exploration'},
        'default_color': {'t': str, 'v': 'black'},
        'imitation': {'t': bool, 'v': False},
        'distribution': d['larva_distro'],
        'odor': d['odor']
    }

    d['agent'] = {
        # 'unique_id': {'t': str},
        'group': {'t': str, 'v': ''},

    }

    d['source'] = {
        **d['agent'],
        'default_color': {'t': str, 'v': 'green'},
        'pos': d['xy'],
        **d['food'],
        'odor': d['odor']
    }

    d['SourceGroup'] = {
        'distribution': d['spatial_distro'],
        'default_color': {'t': str, 'v': 'green'},
        **d['food'],
        'odor': d['odor']
    }

    d['border_list'] = {
        'default_color': {'t': str, 'v': 'green'},
        'points': {'t': List[Tuple[float]], 'min': -1.0, 'max': 1.0},
    }
    d['Source_DISTRO'] = d['spatial_distro']

    d.update(init_vis_dtypes2())

    # d['visualization'] = init_vis_dtypes2()
    d['replay'] = {
        # 'arena_pars': d['arena'],
        'env_params': {'t': str, 'vs': list(loadConfDict('Env').keys()), 'aux_values': ['']},
        'track_point': {'t': int, 'v': -1, 'min': -1, 'max': 12},
        'dynamic_color': {'t': str, 'vs': [None, 'lin_color', 'ang_color']},
        'agent_ids': {'t': List[str]},
        'time_range': {'t': Tuple[float], 'max': 1000.0, 'dv': 1.0},
        'transposition': {'t': str, 'vs': [None, 'origin', 'arena', 'center']},
        'fix_point': {'t': int, 'min': 1, 'max': 12},
        'secondary_fix_point': {'t': int, 'vs': [-1, 1]},
        'use_background': {'t': bool, 'v': False},
        'draw_Nsegs': {'t': int, 'min': 1, 'max': 12}
    }






    return d


def null_dict(n, key='initial_value', **kwargs):
    def v0(d):
        null = {}
        for k, v in d.items():
            if key in v:
                null[k] = v[key]
            else:
                null[k] = v0(v['content'])
        return null

    dic = par_dict(n)
    dic2 = v0(dic)
    dic2.update(kwargs)
    return dic2

def get_dict(n, **kwargs) :
    dic = null_dict(n)
    if n in ['visualization', 'enrichment']:
        for k, v in dic.items():
            if k in list(kwargs.keys()):
                dic[k] = kwargs[k]
            elif type(v) == dict:
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic[k][k0] = kwargs[k0]
    return dic


def enrichment_dict(source=None, types=[], bouts=[]):
    aux = null_dict('enrich_aux', source=source)
    proc = null_dict('processing', types={n: True if n in types else False for n in proc_type_keys})
    annot = null_dict('annotation', bouts={n: True if n in bouts else False for n in bout_keys})
    dic = null_dict('enrichment', processing=proc, annotation=annot, enrich_aux=aux)
    return dic


def init_shortcuts():
    draw = {
        'visible trail': 'p',
        '▲ trail duration': '+',
        '▼ trail duration': '-',

        'draw_head': 'h',
        'draw_centroid': 'e',
        'draw_midline': 'm',
        'draw_contour': 'c',
        'draw_sensors': 'j',
    }

    inspect = {
        'focus_mode': 'f',
        'odor gains': 'w',
        'dynamic graph': 'q',
    }

    color = {
        'black_background': 'g',
        'random_colors': 'r',
        'color_behavior': 'b',
    }

    aux = {
        'visible_clock': 't',
        'visible_scale': 'n',
        'visible_state': 's',
        'visible_ids': 'tab',
    }

    screen = {
        'move up': 'UP',
        'move down': 'DOWN',
        'move left': 'LEFT',
        'move right': 'RIGHT',
    }

    sim = {
        'larva_collisions': 'y',
        'pause': 'space',
        'snapshot': 'i',
        'delete item': 'del',

    }

    odorscape = {

        'plot odorscapes': 'o',
        **{f'odorscape {i}': i for i in range(10)},
        # 'move_right': 'RIGHT',
    }

    d = {
        'draw': draw,
        'color': color,
        'aux': aux,
        'screen': screen,
        'simulation': sim,
        'inspect': inspect,
        'odorscape': odorscape,
    }

    return d


def init_controls():
    k = init_shortcuts()
    d = {'keys': {}, 'pygame_keys': {}, 'mouse': {
        'select item': 'left click',
        'add item': 'left click',
        'select item mode': 'right click',
        'inspect item': 'right click',
        'screen zoom in': 'scroll up',
        'screen zoom out': 'scroll down',
    }}
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d['keys'][title] = dic
    d['pygame_keys'] = {k: get_pygame_key(v) for k, v in ds.items()}
    return d


def store_controls():
    d = init_controls()
    from lib.conf.conf import saveConfDict
    saveConfDict(d, 'Settings')


def store_RefPars():
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_mu',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
        # **{p: p for p in ['initial_x', 'initial_y', 'initial_front_orientation']}
    }
    lib.aux.dictsNlists.save_dict(d, paths.RefParsFile, use_pickle=False)


if __name__ == '__main__':
    store_controls()
    store_RefPars()
    # print(null_dict('enrichment'))

