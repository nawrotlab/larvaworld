import numpy as np
from typing import List, Tuple, Union
import pandas as pd
from siunits import BaseUnit, Composite, DerivedUnit

from lib.aux import functions as fun, naming as nam
from lib.aux.collecting import output_keys
from lib.aux.functions import get_pygame_key
from lib.stor import paths


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
                Ndigits = fun.maxNdigits(vs, 4)
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
        'odor': {
            'odor_id': {'t': str},
            'odor_intensity': {'max': 10.0},
            'odor_spread': {'max': 10.0}
        },
        'food': {
            'radius': {'v': 0.003, 'max': 1.0, 'dv': 0.01},
            'amount': {'v': 0.0, 'max': 10.0},
            'quality': {'v': 1.0, 'max': 1.0},
            'can_be_carried': {'t': bool, 'v': False},
            'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
        },
        'food_grid':
            {'unique_id': {'t': str, 'v': 'Food_grid', },
             'grid_dims': {'t': Tuple[int], 'v': (20, 20), 'min': 10, 'max': 200},
             'initial_value': {'v': 0.001, 'max': 1.0, 'dv': 0.001},
             'distribution': {'t': str, 'v': 'uniform', 'vs': ['uniform']},
             'type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
             },
        'life':
            {
                'epochs': {'t': List[Tuple[float]], 'max': 250.0, 'dv': 0.1},
                'epoch_qs': {'t': List[float], 'max': 1.0},
                'hours_as_larva': {'v': 0.0, 'max': 250.0, 'dv': 0.1},
                'substrate_quality': {'v': 1.0, 'max': 1.0, 'dv': 0.01},
                'substrate_type': {'t': str, 'v': 'standard', 'vs': list(substrate_dict.keys())}
            },
        'odorscape': {'odorscape': {'t': str, 'v': 'Gaussian', 'vs': ['Gaussian', 'Diffusion']},
                      'grid_dims': {'t': Tuple[int], 'min': 10, 'max': 100},
                      'evap_const': {'max': 1.0},
                      'gaussian_sigma': {'t': Tuple[float], 'max': 10.0}
                      },
        'odor_gains': {
            'unique_id': {'t': str},
            'mean': {'max': 1000.0},
            'std': {'max': 10.0}
        },
        'optimization': {
            'fit_par': {'t': str},
            'minimize': {'t': bool, 'v': True},
            'threshold': {'v': 0.001, 'max': 1.0, 'dv': 0.00001},
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
        'space_search': {'pars': {'t': List[str]},
                         'ranges': {'t': List[Tuple[float]], 'max': 100.0, 'min': -100.0, 'dv': 1.0},
                         'Ngrid': {'t': int, 'max': 100}},
        'body': {'initial_length': {'v': 0.0045, 'max': 0.01, 'dv': 0.0001, 'aux_values': ['sample']},
                 'length_std': {'v': 0.0001, 'max': 0.001, 'dv': 0.00001, 'aux_values': ['sample']},
                 'Nsegs': {'t': int, 'v': 2, 'min': 1, 'max': 12},
                 'seg_ratio': {'max': 1.0},  # [5 / 11, 6 / 11]
                 'touch_sensors': {'t': bool, 'v': False},
                 },
        'arena': {'arena_dims': {'t': Tuple[float], 'v': (0.1, 0.1), 'max': 10.0, 'dv': 0.01},
                  'arena_shape': {'t': str, 'v': 'circular', 'vs': ['circular', 'rectangular']}
                  },
        'physics': {
            'torque_coef': {'v': 0.41, 'max': 5.0, 'dv': 0.01},
            'ang_damping': {'v': 2.5, 'max': 10.0},
            'body_spring_k': {'v': 0.02, 'max': 1.0, 'dv': 0.01},
            'bend_correction_coef': {'v': 1.4, 'max': 10.0},
        },
        'energetics': {'f_decay': {'v': 0.1, 'max': 1.0, 'dv': 0.01},
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
                   'base_activation': {'max': 100.0, 'dv': 1.0},
                   'activation_range': {'t': Tuple[float], 'max': 100.0, 'dv': 1.0},
                   'noise': {'v': 0.0, 'max': 10.0},
                   'activation_noise': {'v': 0.0, 'max': 10.0},
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
        'feeder': {'freq_range': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': 4.0},
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
            'sigma': {'v': 0.0, 'max': 10.0}
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
        'preprocessing': {
            'rescale_by': {'max': 10.0},
            'drop_collisions': {'t': bool, 'v': False},
            'interpolate_nans': {'t': bool, 'v': False},
            'filter_f': {'max': 10.0},
            'transposition': {'t': str, 'vs': ['', 'origin', 'arena', 'center']},
        },
        'processing': {
            'types': {t: {'t': bool, 'v': False} for t in
                      ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI']},
            'dsp_starts': {'t': List[float], 'max': 200.0, 'dv': 1.0},
            'dsp_stops': {'t': List[float], 'max': 200.0, 'dv': 1.0},
            'tor_durs': {'t': List[int], 'max': 100, 'dv': 1}},
        'annotation': {'bouts': {t: {'t': bool, 'v': False} for t in ['stride', 'pause', 'turn']},
                       'track_point': {'t': str},
                       'track_pars': {'t': List[str]},
                       'chunk_pars': {'t': List[str]},
                       'vel_par': {'t': str},
                       'ang_vel_par': {'t': str},
                       'bend_vel_par': {'t': str},
                       'min_ang': {'v': 0.0, 'max': 180.0, 'dv': 1.0},
                       'min_ang_vel': {'v': 0.0,'max': 1000.0, 'dv': 1.0},
                       'non_chunks': {'t': bool, 'v': False}},
        'enrich_aux': {'recompute': {'t': bool, 'v': False},
                       'mode': {'t': str, 'v': 'minimal', 'vs': ['minimal', 'full']},
                       'source': {'t': Tuple[float], 'min': -10.0, 'max': 10.0},
                       },
        'to_drop': {'groups': {n: {'t': bool, 'v': False} for n in
                               ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn',
                                'turn',
                                'unused']}},
        'build_conf': {
            'min_duration_in_sec': {'v' : 170.0, 'max': 3600.0, 'dv': 0.1},
            'min_end_time_in_sec': {'v' : 0.0, 'max': 3600.0, 'dv': 0.1},
            'start_time_in_sec': {'v' : 0.0, 'max': 3600.0, 'dv': 0.1},
            'max_Nagents': {'t': int, 'v': 12, 'max': 1000},
            'save_mode': {'t': str, 'v': 'semifull', 'vs': ['minimal', 'semifull', 'full', 'points']},
        },
        'substrate': {k: {'v': v, 'max': 1000.0} for k, v in substrate_dict['standard'].items()},
        'output': {n: {'t': bool, 'v': False} for n in output_keys}
    }

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

    d.update(init_vis_dtypes2())

    # d['visualization'] = init_vis_dtypes2()
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
                     'enrichment': d['enrichment']
                     }


    d['batch_setup'] = {
        'batch_id': {'t': str},
        # 'batch_path': {'t': str},
        'save_hdf5': {'t': bool, 'v': False},
    }
    d['batch_conf'] = {'exp': {'t': str},
                       'space_search': d['space_search'],
                       'batch_methods': d['batch_methods'],
                       'optimization': d['optimization'],
                       'exp_kws': {'t': dict, 'v': {'enrichment' : d['enrichment']}},
                       'post_kws': {'t': dict, 'v': {}},
                       'proc_kws': {'t': dict, 'v': {}},
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
        'loc': {'t': Tuple[float], 'v': (0.0, 0.0), 'min': -100.0, 'max': 100.0},
        'scale': {'t': Tuple[float], 'v': (0.0, 0.0), 'min': -100.0, 'max': 100.0},
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
        'sample': {'t': any, 'v' : paths.RefConf},
        'default_color': {'t': str, 'v' : 'black'},
        'imitation': {'t': bool, 'v': False},
        # 'placement' : {'t': str, 'v': 'distribution', 'vs': ['distribution', 'imitation']},
        'distribution': d['larva_distro'],
        # **d['odor']
        'odor' : d['odor']
    }

    d['agent'] = {
            # 'unique_id': {'t': str},
            'group': {'t': str, 'v' : ''},

        }

    d['source'] = {
        **d['agent'],
        'default_color': {'t': str, 'v': 'green'},
        'pos': {'t': Tuple[float], 'v': (0.0, 0.0), 'max': -100.0, 'max': 100.0, 'dv': 0.01},
        **d['food'],
        'odor': d['odor']
    }

    d['SourceGroup']={
        'distribution': d['spatial_distro'],
        'default_color': {'t': str, 'v': 'green'},
        **d['food'],
        'odor': d['odor']
    }

    d['border_list'] = {
        'default_color': {'t': str, 'v': 'green'},
        'points' : {'t': List[Tuple[float]], 'min' : -1.0, 'max' :1.0},
    }
    d['Source_DISTRO'] = {**d['spatial_distro']}

    return d

def null_dict(n, key='initial_value', **kwargs) :
    def v0(d):
        null = {}
        for k,v in d.items():
            if key in v:
                null[k] = v[key]
            else:
                null[k] = v0(v['content'])
        return null
    dic = par_dict(n)
    dic2=v0(dic)
    dic2.update(kwargs)
    return dic2


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


def processing_types(types=[], dtypes=False):
    l = ['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI']
    if dtypes:
        return {t: bool for t in l}
    else:
        d = {}
        for t in l:
            d[t] = True if t in types else False
        return d


def annotation_bouts(types=[], dtypes=False):
    l = ['stride', 'pause', 'turn']
    if dtypes:
        return {t: bool for t in l}
    else:
        d = {}
        for t in l:
            d[t] = True if t in types else False
        return d


def init_dicts():
    d = {
        'odor':
            {'odor_id': None,
             'odor_intensity': 0.0,
             'odor_spread': None
             },
        'food':
            {
                'radius': 0.003,
                'amount': 0.0,
                'quality': 1.0,
                # 'shape_vertices': None,
                'can_be_carried': False,
                'type': 'standard',
            },
        'food_grid':
            {
                'unique_id': 'Food_grid',
                'grid_dims': (20, 20),
                'initial_value': 10 ** -3,
                'distribution': 'uniform',
                'type': 'standard',
            },
        'arena':
            {
                'arena_dims': (0.1, 0.1),
                # 'arena_xdim': 0.1,
                # 'arena_ydim': 0.1,
                'arena_shape': 'circular'
            },
        'life':
            {
                'epochs': None,
                'epoch_qs': None,
                'hours_as_larva': 0.0,
                'substrate_quality': 1.0,
                'substrate_type': 'standard',

            },
        'odorscape': {'odorscape': 'Gaussian',
                      'grid_dims': None,
                      'evap_const': None,
                      'gaussian_sigma': None,
                      },
        'odor_gains': {
            'unique_id': None,
            'mean': None,
            'std': None
        },

        'optimization': {
            'fit_par': None,
            'operations': {
                'mean': True,
                'std': False,
                'abs': False,
            },
            'minimize': True,
            'threshold': 0.001,
            'max_Nsims': 40,
            'Nbest': 4
        },
        'batch_methods': {
            'run': 'default',
            'post': 'default',
            'final': 'null',
        },
        'space_search': {'pars': None,
                         'ranges': None,
                         'Ngrid': None},

        'body': {'initial_length': 0.0045,
                 'length_std': 0.0001,
                 'Nsegs': 2,
                 'seg_ratio': None,
                 'touch_sensors': False,
                 },
        'physics': {
            'torque_coef': 0.41,
            'ang_damping': 2.5,
            'body_spring_k': 0.02,
            'bend_correction_coef': 1.4,
        },
        'energetics': {'f_decay': 0.1,  # 0.1,  # 0.3
                       'absorption': None,
                       'hunger_as_EEB': True,
                       'hunger_gain': 0.0,
                       'deb_on': True,
                       'assimilation_mode': 'gut',
                       'DEB_dt': None
                       # 'DEB_dt' : 60.0
                       },

        'crawler': {'waveform': 'realistic',
                    'freq_range': [0.5, 2.5],
                    'initial_freq': 'sample',  # From D1 fit
                    'freq_std': 0.0,
                    'step_to_length_mu': 'sample',  # From D1 fit
                    'step_to_length_std': 'sample',  # From D1 fit
                    'initial_amp': None,
                    'noise': 0.01,
                    'max_vel_phase': 1.0
                    },
        'turner': {'mode': None,
                   'base_activation': None,
                   'activation_range': None,
                   'noise': 0.0,
                   'activation_noise': 0.0,
                   'initial_amp': None,
                   'amp_range': None,
                   'initial_freq': None,
                   'freq_range': None,
                   },
        'interference': {
            'crawler_phi_range': [0.0, 0.0],  # np.pi * 0.55,  # 0.9, #,
            'feeder_phi_range': [0.0, 0.0],
            'attenuation': 1.0
        },
        'intermitter': {
            'pause_dist': null_bout_dist,
            'stridechain_dist': null_bout_dist,
            'crawl_bouts': True,
            'feed_bouts': False,
            'crawl_freq': 10 / 7,
            'feed_freq': 2.0,
            'feeder_reoccurence_rate': None,
            'EEB_decay': 1.0,
            'EEB': 0.0},
        'olfactor': {
            'perception': 'log',
            'olfactor_noise': 0.0,
            'decay_coef': 0.5},
        'feeder': {'freq_range': [1.0, 3.0],
                   'initial_freq': 2.0,
                   'feed_radius': 0.1,
                   'V_bite': 0.0002},
        'memory': {'DeltadCon': 0.1,
                   'state_spacePerOdorSide': 0,
                   'gain_space': [-300.0, -50.0, 50.0, 300.0],
                   'decay_coef_space': None,
                   'update_dt': 1,
                   'alpha': 0.05,
                   'gamma': 0.6,
                   'epsilon': 0.3,
                   'train_dur': 20.0,
                   },
        'modules': {'turner': False,
                    'crawler': False,
                    'interference': False,
                    'intermitter': False,
                    'olfactor': False,
                    'feeder': False,
                    'memory': False},
        'sim_params': {
            'sim_ID': None,
            'path': 'single_runs',
            'duration': 1.0,
            'timestep': 0.1,
            'Box2D': False,
            'sample': 'reference'
        },
        'essay_params': {
            'essay_ID': None,
            'path': 'essays',
            'N': None,
            # 'duration': 1.0,
            # 'timestep': 0.1,
            # 'Box2D': False,
            # 'sample': 'reference'
        },
        'logn_dist': {
            'range': (0.0, 2.0),
            'name': 'lognormal',
            'mu': 1.0,
            'sigma': 0.0
        },
        # 'levy_dist': {
        #     'range': (0.0, 2.0),
        #     'name': 'levy',
        #     'mu': 1.0,
        #     'sigma': 0.0
        # },
        'par': {
            'p': None,
            'u': None,
            'k': None,
            's': None,
            'o': None,
            'lim': None,
            'd': None,
            'l': None,
            'exists': True,
            'func': None,
            'const': None,
            'operator': None,
            # 'diff': False,
            # 'cum': False,
            'k0': None,
            'k_num': None,
            'k_den': None,
            'dst2source': None,
            'or2source': None,
            'dispersion': False,
            'wrap_mode': None
        },
        'preprocessing': {
            'rescale_by': None,
            'drop_collisions': False,
            'interpolate_nans': False,
            'filter_f': None,
            'transposition': None,
        },
        'processing': {
            'types': processing_types(['angular', 'spatial', 'dispersion', 'tortuosity']),
            'dsp_starts': None, 'dsp_stops': None,
            'tor_durs': None},
        'annotation': {'bouts': annotation_bouts(types=['stride', 'pause', 'turn']), 'track_point': None,
                       'track_pars': None, 'chunk_pars': None,
                       'vel_par': None, 'ang_vel_par': None, 'bend_vel_par': None, 'min_ang': 0.0, 'min_ang_vel': 0.0,
                       'non_chunks': False},
        'enrich_aux': {'recompute': False,
                       'mode': 'minimal',
                       'source': None,
                       },
        'to_drop': {'groups': {n: False for n in
                               ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn',
                                'turn',
                                'unused']}},
        'build_conf': {
            'min_duration_in_sec': 10.0,
            'min_end_time_in_sec': 0.0,
            'start_time_in_sec': 0.0,
            'max_Nagents': 1000,
            # 'save_mode': 'minimal'
            'save_mode': 'semifull'
        },
        'substrate': substrate_dict['standard']

    }
    d['replay'] = {
        # 'arena_pars': d['arena'],
        'env_params': None,
        'track_point': -1,
        'dynamic_color': None,
        'agent_ids': None,
        'time_range': None,
        'transposition': None,
        'fix_point': None,
        'secondary_fix_point': None,
        'use_background': False,
        'draw_Nsegs': None
    }
    d['visualization'] = init_null_vis()
    d['enrichment'] = {k: d[k] for k in
                       ['preprocessing', 'processing', 'annotation', 'enrich_aux', 'to_drop']}

    d['exp_conf'] = {'env_params': None,
                     'sim_params': d['sim_params'],
                     'life_params': 'default',
                     'collections': ['pose'],
                     'enrichment': d['enrichment']
                     }

    d['batch_conf'] = {'exp': None,
                       'space_search': d['space_search'],
                       'batch_methods': d['batch_methods'],
                       'optimization': None,
                       'exp_kws': {'save_data_flag': False},
                       'post_kws': {},
                       }

    d['food_params'] = {'source_groups': {},
                        'food_grid': None,
                        'source_units': {}}

    d['tracker'] = {
        'resolution': {'fr': 10.0,
                       'Npoints': 1,
                       'Ncontour': 0
                       },
        'filesystem': {
            'read_sequence': None,
            'read_metadata': False,
            'detect': {
                'folder': {'pref': None, 'suf': None},
                'file': {'pref': None, 'suf': None, 'sep': None}
            }
        },
        'arena': d['arena']
    }
    d['parameterization'] = {'bend': 'from_angles',
                             'front_vector': (1, 2),
                             'rear_vector': (-2, -1),
                             'front_body_ratio': 0.5,
                             'point_idx': 1,
                             'use_component_vel': False,
                             'scaled_vel_threshold': 0.2}

    return d


typing_to_str_dict = {
    'Tuple[float, float]': Tuple[float, float],
    'Tuple[int, int]': Tuple[int, int],
    'List[float]': List[float],
    'List[int]': List[int],
    'List[str]': List[str],
    'List[tuple]': List[tuple],
    'Union[Tuple[float, float], Tuple[int, int]]': Union[Tuple[float, float], Tuple[int, int]],
    'Union[BaseUnit, Composite, DerivedUnit]': Union[BaseUnit, Composite, DerivedUnit],
}


def init_agent_dtypes(class_name, dtypes_dict):
    dtypes = {
        'unique_id': str,
        'default_color': str,
        # 'group': str,
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dtypes = {**dtypes, 'group': str, **dtypes_dict['odor']}
    elif class_name in ['Source', 'Food']:
        dtypes = {**dtypes, 'group': str, **dtypes_dict['odor'], **dtypes_dict['food'], 'can_be_carried': bool,
                  'pos': Tuple[float, float]}
    elif class_name in ['Border']:
        dtypes = {**dtypes,
                  'width': fun.value_list(end=0.1, steps=1000, decimals=4),
                  'points': List[tuple]}
    return dtypes


def init_dtypes():
    from lib.conf.conf import loadConfDict

    bout_dist_dtypes = {
        'fit': bool,
        'range': (0.0, 100.0),
        # 'name': str,
        'name': ['powerlaw', 'exponential', 'lognormal', 'lognormal-powerlaw', 'levy', 'normal', 'uniform'],
        'mu': float,
        'sigma': float,
        # 'c': float
    }

    d = {
        'odor':
            {'odor_id': str,
             'odor_intensity': fun.value_list(end=1000.0, steps=10000, decimals=2),
             'odor_spread': fun.value_list(end=10.0, steps=100000, decimals=5)
             },
        'food':
            {
                'radius': fun.value_list(end=10.0, steps=10000, decimals=4),
                'amount': fun.value_list(end=100.0, steps=1000, decimals=2),
                'quality': fun.value_list(),
                # 'shape_vertices': List[tuple],
                'can_be_carried': bool,
                'type': list(substrate_dict.keys())
            },
        'food_grid':
            {
                'unique_id': str,
                'grid_dims': (10, 1000),
                'initial_value': fun.value_list(start=0.0, end=1.0, steps=10000, decimals=4),
                'distribution': ['uniform'],
                'type': list(substrate_dict.keys())
            },
        'arena': {'arena_dims': (0.0, 10.0), 'arena_shape': ['circular', 'rectangular']},
        'life':
            {
                'epochs': {'type': List[tuple], 'values': fun.value_list()},
                'epoch_qs': {'type': list, 'values': fun.value_list()},
                'hours_as_larva': {'type': float, 'values': fun.value_list(end=250, steps=25100, decimals=2)},
                'substrate_quality': {'type': float, 'values': fun.value_list()},
                'substrate_type': list(substrate_dict.keys()),
            },
        'odorscape': {'odorscape': ['Gaussian', 'Diffusion'],
                      'grid_dims': (10, 1000),
                      'evap_const': fun.value_list(),
                      'gaussian_sigma': (0.0, 10.0),
                      },
        'odor_gains': {
            'unique_id': str,
            'mean': float,
            'std': float
        },

        'optimization': {
            'fit_par': str,
            'operations': {
                'mean': bool,
                'std': bool,
                'abs': bool,
            },
            'minimize': bool,
            'threshold': fun.value_list(0.000001, 1.0, steps=1000000, decimals=6),
            'max_Nsims': fun.value_list(2, 1002, steps=1000, integer=True),
            'Nbest': fun.value_list(2, 42, steps=40, integer=True)
        },
        'batch_methods': {
            'run': ['null', 'default', 'deb', 'odor_preference'],
            'post': ['null', 'default'],
            'final': ['null', 'scatterplots', 'deb', 'odor_preference'],
        },
        'space_search': {'pars': str,
                         'ranges': Union[Tuple[float, float], Tuple[int, int]],
                         'Ngrid': int},

        'visualization': init_vis_dtypes(),
        'body': {'initial_length': fun.value_list(0.0, 0.01, steps=100, decimals=4),
                 'length_std': fun.value_list(0.0, 0.01, steps=100, decimals=4),
                 'Nsegs': fun.value_list(1, 12, steps=12, integer=True),
                 'seg_ratio': List[float],  # [5 / 11, 6 / 11]
                 'touch_sensors': bool,
                 },
        'physics': {
            'torque_coef': fun.value_list(),
            'ang_damping': fun.value_list(),
            'body_spring_k': fun.value_list(),
            'bend_correction_coef': fun.value_list(),
        },
        'energetics': {'f_decay': fun.value_list(),
                       'absorption': fun.value_list(),
                       'hunger_as_EEB': bool,
                       'hunger_gain': fun.value_list(),
                       'deb_on': bool,
                       'assimilation_mode': ['sim', 'gut', 'deb'],
                       'DEB_dt': fun.value_list()},
        'crawler': {'waveform': ['realistic', 'square', 'gaussian', 'constant'],
                    'freq_range': (0.0, 2.0),
                    'initial_freq': fun.value_list(end=2.0, steps=200),  # From D1 fit
                    'freq_std': fun.value_list(end=2.0, steps=200),  # From D1 fit
                    'step_to_length_mu': fun.value_list(),  # From D1 fit
                    'step_to_length_std': fun.value_list(),  # From D1 fit
                    'initial_amp': fun.value_list(end=2.0, steps=200),
                    'noise': fun.value_list(),
                    'max_vel_phase': fun.value_list(end=2.0, steps=200)
                    },
        'turner': {'mode': ['', 'neural', 'sinusoidal'],
                   'base_activation': fun.value_list(end=100.0, steps=1000, decimals=1),
                   'activation_range': (0.0, 100.0),
                   'noise': fun.value_list(),
                   'activation_noise': fun.value_list(),
                   'initial_amp': fun.value_list(end=2.0, steps=200),
                   'amp_range': (0.0, 2.0),
                   'initial_freq': fun.value_list(end=2.0, steps=200),
                   'freq_range': (0.0, 2.0),
                   },
        'interference': {
            'crawler_phi_range': (0.0, 2.0),  # np.pi * 0.55,  # 0.9, #,
            'feeder_phi_range': (0.0, 2.0),
            'attenuation': fun.value_list()
        },
        'intermitter': {
            'pause_dist': bout_dist_dtypes,
            # 'pause_dist': dict,
            'stridechain_dist': bout_dist_dtypes,
            # 'stridechain_dist': dict,
            'crawl_bouts': bool,
            'feed_bouts': bool,
            'crawl_freq': fun.value_list(end=2.0, steps=200),
            'feed_freq': fun.value_list(end=4.0, steps=400),
            'feeder_reoccurence_rate': fun.value_list(),
            'EEB_decay': fun.value_list(end=2.0, steps=200),
            'EEB': fun.value_list()},
        'olfactor': {
            'perception': ['log', 'linear'],
            'olfactor_noise': fun.value_list(),
            'decay_coef': fun.value_list(end=2.0, steps=200)},
        'feeder': {'freq_range': (0.0, 4.0),
                   'initial_freq': fun.value_list(end=4.0, steps=400),
                   'feed_radius': fun.value_list(start=0.01, end=1.0, steps=1000, decimals=2),
                   'V_bite': fun.value_list(start=0.0001, end=0.01, steps=1000, decimals=4)},
        'memory': {'DeltadCon': float,
                   'state_spacePerOdorSide': int,
                   'gain_space': List[float],
                   'decay_coef_space': List[float],
                   'update_dt': float,
                   'alpha': float,
                   'gamma': float,
                   'epsilon': float,
                   'train_dur': float,
                   },
        'modules': {'turner': bool,
                    'crawler': bool,
                    'interference': bool,
                    'intermitter': bool,
                    'olfactor': bool,
                    'feeder': bool,
                    'memory': bool},

        'sim_params': {
            'sim_ID': str,
            'path': str,
            'duration': {'type': float, 'values': fun.value_list(end=2000.0, steps=20001, decimals=1)},
            'timestep': {'type': float, 'values': fun.value_list()},
            'Box2D': bool,
            'sample': list(loadConfDict('Ref').keys())
        },
        'essay_params': {
            'essay_ID': str,
            'path': str,
            'N': fun.value_list(1, 100, steps=100, integer=True)
        },
        'logn_dist': {
            'range': Tuple[float, float],
            'name': 'lognormal',
            'mu': float,
            'sigma': float
        },
        'par': {
            'p': str,
            'u': Union[BaseUnit, Composite, DerivedUnit],
            'k': str,
            's': str,
            'o': type,
            'lim': Tuple[float, float],
            'd': str,
            'l': str,
            'exists': bool,
            'func': any,
            'const': any,
            'operator': [None, 'diff', 'cum', 'max', 'min', 'mean', 'std', 'final'],
            # 'diff': bool,
            # 'cum': bool,
            'k0': str,
            'k_num': str,
            'k_den': str,
            'dst2source': Tuple[float, float],
            'or2source': Tuple[float, float],
            'dispersion': bool,
            'wrap_mode': [None, 'zero', 'positive']
        },
        'preprocessing': {
            'rescale_by': fun.value_list(end=100.0, steps=100000, decimals=3),
            'drop_collisions': bool,
            'interpolate_nans': bool,
            'filter_f': fun.value_list(end=10.0, steps=10000, decimals=3),
            'transposition': ['', 'origin', 'arena', 'center'],
        },
        'processing': {
            'types': processing_types(),
            'dsp_starts': {'type': list, 'values': fun.value_list(start=0, end=180, steps=181, integer=True)},
            'dsp_stops': {'type': list, 'values': fun.value_list(start=0, end=180, steps=181, integer=True)},
            'tor_durs': {'type': list, 'values': fun.value_list(start=0, end=180, steps=181, integer=True)}},
        'annotation': {'bouts': annotation_bouts(dtypes=True),
                       'track_point': str,
                       'track_pars': List[str], 'chunk_pars': List[str],
                       'vel_par': str, 'ang_vel_par': str, 'bend_vel_par': str,
                       'min_ang': fun.value_list(end=180.0, steps=1900, decimals=1),
                       'min_ang_vel': fun.value_list(end=1000.0, steps=10001, decimals=1),
                       'non_chunks': bool},
        'enrich_aux': {'recompute': bool,
                       'mode': ['minimal', 'full'],
                       'source': (-100.0, 100.0),
                       },
        'to_drop': {'groups': {n: bool for n in
                               ['midline', 'contour', 'stride', 'non_stride', 'stridechain', 'pause', 'Lturn', 'Rturn',
                                'turn',
                                'unused']}},
        'build_conf': {
            'min_duration_in_sec': fun.value_list(start=0.0, end=3600.0, steps=36000, decimals=1),
            'min_end_time_in_sec': fun.value_list(start=0.0, end=3600.0, steps=36000, decimals=1),
            'start_time_in_sec': fun.value_list(start=0.0, end=3600.0, steps=36000, decimals=1),
            'max_Nagents': fun.value_list(start=1, end=1000, steps=1000, integer=True),
            'save_mode': ['minimal', 'semifull', 'full', 'points'],
        },
        'substrate': {k: float for k in substrate_dict['standard'].keys()}

    }
    d['replay'] = {
        # 'arena_pars': d['arena'],
        'env_params': [''] + list(loadConfDict('Env').keys()),
        'track_point': {'type': int, 'values': np.arange(1, 13).tolist()},
        'dynamic_color': [None, 'lin_color', 'ang_color'],
        'agent_ids': list,
        'time_range': (0.0, 3600.0),
        'transposition': [None, 'origin', 'arena', 'center'],
        'fix_point': {'type': int, 'values': np.arange(1, 13).tolist()},
        'secondary_fix_point': ['', 1, -1],
        'use_background': bool,
        'draw_Nsegs': {'type': int, 'values': np.arange(1, 13).tolist()},
    }
    d['enrichment'] = {k: d[k] for k in
                       ['preprocessing', 'processing', 'annotation', 'enrich_aux', 'to_drop']}
    d['exp_conf'] = {'env_params': str,
                     'sim_params': dict,
                     'life_params': str,
                     'collections': List[str],
                     'enrichment': dict,
                     }
    d['batch_conf'] = {'exp': str,
                       'space_search': dict,
                       'batch_methods': dict,
                       'optimization': dict,
                       'exp_kws': dict,
                       'post_kws': dict,
                       }
    d['food_params'] = {'source_groups': dict,
                        'food_grid': dict,
                        'source_units': dict}
    d['tracker'] = {
        'resolution': {'fr': fun.value_list(0.0, 20.0, steps=2100, decimals=2),
                       'Npoints': fun.value_list(1, 15, steps=15, integer=True),
                       'Ncontour': fun.value_list(0, 30, steps=31, integer=True),
                       },
        'filesystem': {
            'read_sequence': List[str],
            'read_metadata': bool,
            'detect': {
                'folder': {'pref': str, 'suf': str},
                'file': {'pref': str, 'suf': str, 'sep': str}
            }
        },
        'arena': d['arena']
    }
    d['parameterization'] = {'bend': ['from_angles', 'from_vectors'],
                             'front_vector': {'type': tuple,
                                              'values': fun.value_list(start=-12, end=12, steps=25, integer=True)},
                             # 'front_vector': (-12, 12),
                             'rear_vector': {'type': tuple,
                                             'values': fun.value_list(start=-12, end=12, steps=25, integer=True)},
                             # 'rear_vector': (-12, 12),
                             'front_body_ratio': fun.value_list(),
                             'point_idx': fun.value_list(-1, 12, steps=14, integer=True),
                             'use_component_vel': bool,
                             'scaled_vel_threshold': fun.value_list()}

    for c in ['Larva', 'LarvaSim', 'LarvaReplay', 'Border', 'Source', 'Food']:
        d[c] = init_agent_dtypes(c, d)

    return d


def store_dtypes():
    d1 = init_dicts()
    d2 = init_dtypes()

    d22 = fun.replace_in_dict(d2, replace_d=typing_to_str_dict, inverse=True)

    d = {'null_dicts': d1, 'dtypes': d22}
    fun.save_dict(d, paths.Dtypes_path, use_pickle=True)


def load_dtypes():
    d = fun.load_dict(paths.Dtypes_path, use_pickle=True)
    d1 = d['null_dicts']
    d2 = d['dtypes']
    d22 = fun.replace_in_dict(d2, replace_d=typing_to_str_dict, inverse=False)
    return d1, d22


def init_vis_dtypes():
    vis_render_dtypes = {
        'mode': [None, 'video', 'image'],
        'image_mode': [None, 'final', 'snapshots', 'overlap'],
        'video_speed': fun.value_list(1, 60, steps=60, integer=True),
        'media_name': str,
        'show_display': bool,
    }

    vis_draw_dtypes = {
        'draw_head': bool,
        'draw_centroid': bool,
        'draw_midline': bool,
        'draw_contour': bool,
        'draw_sensors': bool,
        'trails': bool,
        'trajectory_dt': fun.value_list(0.0, 100.0, steps=1000, decimals=1),
    }

    vis_color_dtypes = {
        'black_background': bool,
        'random_colors': bool,
        'color_behavior': bool,
    }

    vis_aux_dtypes = {
        'visible_clock': bool,
        'visible_scale': bool,
        'visible_state': bool,
        'visible_ids': bool,
    }

    vis_dtypes = {
        'render': vis_render_dtypes,
        'draw': vis_draw_dtypes,
        'color': vis_color_dtypes,
        'aux': vis_aux_dtypes,
    }
    return vis_dtypes


def init_null_vis():
    null_vis_render = {
        'mode': None,
        'image_mode': None,
        'video_speed': 1,
        'media_name': None,
        'show_display': True,
    }

    null_vis_draw = {
        'draw_head': False,
        'draw_centroid': False,
        'draw_midline': True,
        'draw_contour': True,
        'draw_sensors': True,
        'trails': False,
        'trajectory_dt': 0.0,
    }

    null_vis_color = {
        'black_background': False,
        'random_colors': False,
        'color_behavior': False,
    }

    null_vis_aux = {
        'visible_clock': True,
        'visible_scale': True,
        'visible_state': True,
        'visible_ids': False,
    }

    null_vis = {
        'render': null_vis_render,
        'draw': null_vis_draw,
        'color': null_vis_color,
        'aux': null_vis_aux,
    }
    return null_vis


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

def store_RefPars() :
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_mu',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
        # **{p: p for p in ['initial_x', 'initial_y', 'initial_front_orientation']}
    }
    fun.save_dict(d, paths.RefParsFile, use_pickle=False)


if __name__ == '__main__':
    store_dtypes()
    store_controls()
    store_RefPars()
    # print(null_dict('enrichment'))


