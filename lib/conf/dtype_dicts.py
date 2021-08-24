import copy
from typing import List, Tuple, Union
import numpy as np
from siunits import BaseUnit, Composite, DerivedUnit

import lib.aux.functions as fun

# from lib.conf.par import runtime_pars

vis_render_dtypes = {
    'mode': [None, 'video', 'image'],
    'image_mode': [None, 'final', 'snapshots', 'overlap'],
    'video_speed': fun.value_list(1,60, steps=60, integer=True),
    'media_name': str,
    'show_display': bool,
}

vis_draw_dtypes = {
    'draw_head': bool,
    'draw_centroid': bool,
    'draw_midline': bool,
    'draw_contour': bool,
    'trajectories': bool,
    'trajectory_dt': fun.value_list(0.0,100.0, steps=1000, decimals=1),
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
    'trajectories': False,
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

shortcut_vis_draw = {
    # 'trajectory_dt' : ['MINUS', 'PLUS'],
    'visible trail': 'p',
    '▲ trail duration': '+',
    '▼ trail duration': '-',

    'draw_head': 'h',
    'draw_centroid': 'e',
    'draw_midline': 'm',
    'draw_contour': 'c'
}

shortcut_inspect = {
    'focus_mode': 'f',
    'odor gains': 'w',
    'dynamic graph': 'q',
}

shortcut_vis_color = {
    'black_background': 'g',
    'random_colors': 'r',
    'color_behavior': 'b',
}

shortcut_vis_aux = {
    'visible_clock': 't',
    'visible_scale': 'n',
    'visible_state': 'sigma',
    'visible_ids': 'tab',
}

shortcut_moving = {
    'move up': 'UP',
    'move down': 'DOWN',
    'move left': 'LEFT',
    'move right': 'RIGHT',
}

shortcut_sim = {
    'larva_collisions': 'y',
    'pause': 'space',
    'snapshot': 'i',
    'delete item': 'del',

}

shortcut_odorscape = {

    'plot odorscapes': 'o',
    **{f'odorscape {i}': i for i in range(10)},
    # 'move_right': 'RIGHT',
}

# null_shortcut = {
#    # 'shortcut_vis_render': shortcut_vis_render,
#     'shortcut_vis_draw': shortcut_vis_draw,
#     'shortcut_vis_color': shortcut_vis_color,
#     'shortcut_vis_aux': shortcut_vis_aux,
#     'shortcut_moving': shortcut_moving,
# }

default_shortcuts = {
    **shortcut_vis_draw,
    **shortcut_vis_color,
    **shortcut_vis_aux,
    **shortcut_moving,
    **shortcut_sim,
    **shortcut_inspect,
    **shortcut_odorscape,
}

default_shortcuts = {
    'draw': shortcut_vis_draw,
    'color': shortcut_vis_color,
    'aux': shortcut_vis_aux,
    'screen': shortcut_moving,
    'simulation': shortcut_sim,
    'inspect': shortcut_inspect,
    'odorscape': shortcut_odorscape,
}

mouse_controls = {
    'select item': 'left click',
    'add item': 'left click',
    'select item mode': 'right click',
    'inspect item': 'right click',
    'screen zoom in': 'scroll up',
    'screen zoom out': 'scroll down',
}


def get_pygame_key(key):
    pygame_keys = {
        'BackSpace': 'BACKSPACE',
        'tab': 'TAB',
        'del': 'DELETE',
        'clear': 'CLEAR',
        'Return': 'RETURN',
        'Escape': 'ESCAPE',
        'space': 'SPACE',
        'exclam': 'EXCLAIM',
        'quotedbl': 'QUOTEDBL',
        '+': 'PLUS',
        'comma': 'COMMA',
        '-': 'MINUS',
        'period': 'PERIOD',
        'slash': 'SLASH',
        'numbersign': 'HASH',
        'Down:': 'DOWN',
        'Up:': 'UP',
        'Right:': 'RIGHT',
        'Left:': 'LEFT',
        'dollar': 'DOLLAR',
        'ampersand': 'AMPERSAND',
        'parenleft': 'LEFTPAREN',
        'parenright': 'RIGHTPAREN',
        'asterisk': 'ASTERISK',
    }
    return f'K_{pygame_keys[key]}' if key in list(pygame_keys.keys()) else f'K_{key}'


# default_shortcuts_pygame = {k: get_pygame_key(v) for k, v in default_shortcuts.items()}


def get_dict(name, class_name=None, basic=True, as_entry=False, **kwargs):
    # print(name)
    if name in list(all_null_dicts.keys()):
        d = all_null_dicts[name]
    elif name == 'distro':
        d = null_distro(class_name=class_name, basic=basic)
    elif name == 'agent':
        d = null_agent(class_name=class_name)
    dic = copy.deepcopy(d)
    if name in ['visualization', 'enrichment']:
        for k, v in dic.items():
            if k in list(kwargs.keys()):
                dic[k] = kwargs[k]
            elif type(v) == dict:
                for k0, v0 in v.items():
                    if k0 in list(kwargs.keys()):
                        dic[k][k0] = kwargs[k0]
    else:
        dic.update(kwargs)
    if as_entry:
        if name == 'distro' and 'group' in list(dic.keys()):
            id = dic['group']
        elif 'unique_id' in list(dic.keys()):
            id = dic['unique_id']
            dic.pop('unique_id')
        dic = {id: dic}
    return dic


def get_distro(class_name, **kwargs):
    distro = null_distro(class_name)
    distro.update(**kwargs)
    return distro


def null_distro(class_name, basic=True):
    distro = {
        'mode': None,
        'shape': None,
        'N': 0,
        'loc': (0.0, 0.0),
        'scale': (0.0, 0.0),
    }
    if class_name == 'Larva':
        distro = {**distro, 'orientation_range': (0.0, 360.0), 'model': None}
    if not basic:
        distro = {**distro, **get_dict('agent', class_name=class_name)}
        for p in ['unique_id', 'pos']:
            try:
                distro.pop(p)
            except:
                pass
    return distro


# Compound densities (g/cm**3)
substrate_dict = {
'agar': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 16 / 1000,
        'cornmeal': 0,
    },
    'standard': {
        'glucose': 100 / 1000,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 50 / 1000,
        'agar': 16 / 1000,
        'cornmeal': 0,
    },
    'cornmeal': {
        'glucose': 517 / 17000,
        'dextrose': 1033 / 17000,
        'saccharose': 0,
        'yeast': 0,
        'agar': 93 / 17000,
        'cornmeal': 1716 / 17000,
    },
    'PED_tracker': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 2 / 200,
        'yeast': 3 * 0.05 * 0.125 / 0.1,
        'agar': 500 * 2 / 200,
        'cornmeal': 0,
    }
}

null_bout_dist = {
    'fit': True,
    'range': None,
    'name': None,
    'mu': None,
    'sigma': None,
    # 'c': None
}

all_null_dicts = {
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
            'shape_vertices': None,
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
            'arena_xdim': 0.1,
            'arena_ydim': 0.1,
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
    'odor_gain': {
        'unique_id': None,
        'mean': None,
        'std': None
    },
    'replay':
        {
            'arena_pars': {
                'arena_xdim': 0.1,
                'arena_ydim': 0.1,
                'arena_shape': 'circular'
            },
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
    'visualization': null_vis,
    'body': {'initial_length': 0.0045,
             'length_std': 0.0001,
             'Nsegs': 2,
             'seg_ratio': None
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
                'crawler_noise': 0.01,
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
        # 'pause_dist': 'fit',
        'pause_dist': null_bout_dist,
                    # 'stridechain_dist': 'fit',
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
    'feeder': {'feeder_freq_range': [1.0, 3.0],
               'feeder_initial_freq': 2.0,
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
               'train_dur': 20,
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
        'filter_f': None
    },
    'processing': {'types': None,
                   'dsp_starts': None, 'dsp_stops': None,
                   'tor_durs': None},
    'annotation': {'bouts': None, 'track_point': None,
                   'track_pars': None, 'chunk_pars': None,
                   'vel_par': None, 'ang_vel_par': None, 'bend_vel_par': None, 'min_ang': 0.0,
                   'non_chunks': False},
    'enrich_aux': {'recompute': False,
                   'mode': 'minimal',
                   'source': None,
                   },
    'substrate': substrate_dict['standard']

}
all_null_dicts['enrichment'] = {k: all_null_dicts[k] for k in
                                ['preprocessing', 'processing', 'annotation', 'enrich_aux']}

all_null_dicts['exp_conf'] = {'env_params': None,
                              'sim_params': get_dict('sim_params'),
                              'life_params': 'default',
                              # 'life_params': get_dict('life'),
                              'collections': ['pose'],
                              'enrichment': get_dict('enrichment')
                              }

all_null_dicts['batch_conf'] = {'exp': None,
                                'space_search': get_dict('space_search'),
                                'batch_methods': get_dict('batch_methods'),
                                'optimization': None,
                                'exp_kws': {'save_data_flag': False},
                                'post_kws': {},
                                }

all_null_dicts['food_params'] = {'source_groups': {},
                                 'food_grid': None,
                                 'source_units': {}}


def base_enrich(**kwargs):
    d = {'types': ['angular', 'spatial', 'source', 'dispersion', 'tortuosity'],
         'dsp_starts': [0, 20], 'dsp_stops': [40, 80], 'tor_durs': [2, 5, 10, 20],
         'min_ang': 5.0, 'bouts': ['stride', 'pause', 'turn']
         }

    d.update(**kwargs)
    return get_dict('enrichment', **d)


def get_dict_dtypes(name, **kwargs):
    from lib.conf import par_conf
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

    all_dtypes = {
        'odor':
            {'odor_id': str,
             'odor_intensity': fun.value_list(end=1000.0, steps=10000, decimals=2),
             'odor_spread': fun.value_list(end=100.0, steps=1000, decimals=2)
             },
        'food':
            {
                'radius': fun.value_list(end=10.0, steps=100, decimals=2),
                'amount': fun.value_list(end=100.0, steps=1000, decimals=2),
                'quality': fun.value_list(),
                'shape_vertices': List[Tuple[float, float]],
                'can_be_carried': bool,
                'type': list(substrate_dict.keys())
            },
        'food_grid':
            {
                'unique_id': str,
                'grid_dims': (10,1000),
                'initial_value': fun.value_list(start=0.0, end=1.0, steps=10000, decimals=4),
                'distribution': ['uniform'],
                'type': list(substrate_dict.keys())
            },
        'arena':
            {
                'arena_xdim': fun.value_list(end=10.0, steps=1000, decimals=3),
                'arena_ydim': fun.value_list(end=10.0, steps=1000, decimals=3),
                'arena_shape': ['circular', 'rectangular']
            },
        'life':
            {
                'epochs': List[Tuple[float, float]],
                'epoch_qs': List[float],
                'hours_as_larva': fun.value_list(end=250, steps=250, integer=True),
                'substrate_quality': fun.value_list(),
                'substrate_type': list(substrate_dict.keys()),
            },
        'odorscape': {'odorscape': ['Gaussian', 'Diffusion'],
                      'grid_dims': (10, 1000),
                      'evap_const': fun.value_list(),
                      'gaussian_sigma': (0.0, 10.0),
                      },
        'odor_gain': {
            'unique_id': str,
            'mean': float,
            'std': float
        },
        'replay': {
            'arena_pars': {
                'arena_xdim': float,
                'arena_ydim': float,
                'arena_shape': ['circular', 'rectangular']
            },
            'env_params': [''] + list(loadConfDict('Env').keys()),
            'track_point': fun.value_list(1,12, steps=12, integer=True),
            'dynamic_color': [None, 'lin_color', 'ang_color'],
            'agent_ids': list,
            'time_range': (0.0,3600.0),
            'transposition': [None, 'origin', 'arena', 'center'],
            'fix_point': fun.value_list(1,12, steps=12, integer=True),
            'secondary_fix_point': ['', 1, -1],
            'use_background': bool,
            'draw_Nsegs': fun.value_list(1,12, steps=12, integer=True),
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
            'max_Nsims': fun.value_list(2,1002, steps=1000, integer=True),
            'Nbest': fun.value_list(2,42, steps=40, integer=True)
        },
        'batch_methods': {
            'run': ['null', 'default', 'deb', 'odor_preference'],
            'post': ['null', 'default'],
            'final': ['null', 'scatterplots', 'deb', 'odor_preference'],
        },
        'space_search': {'pars': str,
                         'ranges': Tuple[float, float],
                         'Ngrid': int},

        'visualization': vis_dtypes,
        'body': {'initial_length': fun.value_list(0.0,0.01, steps=100, decimals=4),
                 'length_std': fun.value_list(0.0,0.01, steps=100, decimals=4),
                 'Nsegs':  fun.value_list(1,12, steps=12, integer=True),
                 'seg_ratio': List[float]  # [5 / 11, 6 / 11]
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
                    'freq_range': (0.0,2.0),
                    'initial_freq': fun.value_list(end=2.0, steps=200),  # From D1 fit
                    'freq_std': fun.value_list(end=2.0, steps=200),  # From D1 fit
                    'step_to_length_mu': fun.value_list(),  # From D1 fit
                    'step_to_length_std': fun.value_list(),  # From D1 fit
                    'initial_amp': fun.value_list(end=2.0, steps=200),
                    'crawler_noise': fun.value_list(),
                    'max_vel_phase': fun.value_list(end=2.0, steps=200)
                    },
        'turner': {'mode': ['', 'neural', 'sinusoidal'],
                   'base_activation': fun.value_list(end=100.0, steps=1000, decimals=1),
                   'activation_range': (0.0,100.0),
                   'noise': fun.value_list(),
                   'activation_noise': fun.value_list(),
                   'initial_amp': fun.value_list(end=2.0, steps=200),
                   'amp_range': (0.0,2.0),
                   'initial_freq': fun.value_list(end=2.0, steps=200),
                   'freq_range': (0.0,2.0),
                   },
        'interference': {
            'crawler_phi_range': (0.0, 2.0),  # np.pi * 0.55,  # 0.9, #,
            # 'crawler_phi_range': Tuple[float, float],  # np.pi * 0.55,  # 0.9, #,
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
        'feeder': {'feeder_freq_range':(0.0, 4.0),
                   'feeder_initial_freq': fun.value_list(end=4.0, steps=400),
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
            'duration': np.round(np.arange(0.0, 2000.1, 0.1), 1).tolist(),
            'timestep': np.round(np.arange(0.01, 1.01, 0.01), 2).tolist(),
            'Box2D': bool,
            'sample': list(loadConfDict('Ref').keys())
        },
        'essay_params': {
            'essay_ID': str,
            'path': str,
            'N': fun.value_list(1,100, steps=100, integer=True),
            # 'duration': np.round(np.arange(0.0, 2000.1, 0.1), 1).tolist(),
            # 'timestep': np.round(np.arange(0.01, 1.01, 0.01), 2).tolist(),
            # 'Box2D': bool,
            # 'sample': list(loadConfDict('Ref').keys())
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
            'rescale_by': float,
            'drop_collisions': bool,
            'interpolate_nans': bool,
            'filter_f': float
        },
        'processing': {'types': ['angular', 'spatial', 'source', 'dispersion', 'tortuosity'],
                       'dsp_starts': List[float], 'dsp_stops': List[float],
                       'tor_durs': List[float]},
        'annotation': {'bouts': ['stride', 'pause', 'turn'], 'track_point': str,
                       'track_pars': List[str], 'chunk_pars': List[str],
                       'vel_par': str, 'ang_vel_par': str, 'bend_vel_par': str, 'min_ang': float,
                       'non_chunks': bool},
        'enrich_aux': {'recompute': bool,
                       'mode': ['minimal', 'full'],
                       'source': Tuple[float, float],
                       },
        'substrate': {k: float for k in substrate_dict['standard'].keys()}

    }
    all_dtypes['enrichment'] = {k: all_dtypes[k] for k in ['preprocessing', 'processing', 'annotation', 'enrich_aux']}
    all_dtypes['exp_conf'] = {'env_params': str,
                              'sim_params': dict,
                              'life_params': str,
                              'collections': List[str],
                              'enrichment': dict,
                              }
    all_dtypes['batch_conf'] = {'exp': str,
                                'space_search': dict,
                                'batch_methods': dict,
                                'optimization': dict,
                                'exp_kws': dict,
                                'post_kws': dict,
                                }
    all_dtypes['food_params'] = {'source_groups': dict,
                                 'food_grid': dict,
                                 'source_units': dict}
    if name in list(all_dtypes.keys()):
        return all_dtypes[name]
    elif name == 'distro':
        return get_distro_dtypes(**kwargs)
    elif name == 'agent':
        return get_agent_dtypes(**kwargs)


module_keys = list(get_dict('modules').keys())


def get_agent_dtypes(class_name):
    dtypes = {
        'unique_id': str,
        'default_color': str,
        # 'group': str,
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dtypes = {**dtypes,'group': str, **get_dict_dtypes('odor')}
    elif class_name in ['Source', 'Food']:
        dtypes = {**dtypes,'group': str, **get_dict_dtypes('odor'), **get_dict_dtypes('food'), 'can_be_carried': bool,
                  'pos': Tuple[float, float]}
    elif class_name in ['Border']:
        dtypes = {**dtypes, 'width': float, 'points': List[Tuple[float, float]]}
    return dtypes


def null_agent(class_name):
    dic = {
        'unique_id': None,
        # 'default_color': 'black',
        'group': '',
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dic = {**dic, **get_dict('odor'), 'default_color': 'black'}
    elif class_name in ['Source', 'Food']:
        dic = {**dic, **get_dict('odor'), **get_dict('food'), 'default_color': 'green', 'pos': (0.0, 0.0)}
    elif class_name in ['Border']:
        dic = {**dic, 'width': 0.001, 'points': None, 'default_color': 'grey'}
    return dic


def get_distro_dtypes(class_name, basic=True):
    from lib.conf.conf import loadConfDict
    dtypes = {
        'mode': ['normal', 'periphery', 'uniform'],
        'shape': ['circle', 'rect', 'oval'],
        'N': int,
        'loc': Tuple[float, float],
        'scale': Tuple[float, float],
    }
    if class_name == 'Larva':
        dtypes = {**dtypes, 'orientation_range': Tuple[float, float], 'model': list(loadConfDict('Model').keys())}
    if not basic:
        dtypes = {**dtypes, **get_dict_dtypes('agent', class_name=class_name)}
        for p in ['unique_id', 'pos']:
            try:
                dtypes.pop(p)
            except:
                pass
    return dtypes


def sim_dict(sim_ID=None, duration=3, dt=0.1, path=None, Box2D=False, exp_type=None):
    from lib.conf.conf import next_idx
    if exp_type is not None:
        if sim_ID is None:
            sim_ID = f'{exp_type}_{next_idx(exp_type)}'
        if path is None:
            path = f'single_runs/{exp_type}'
    return {
        'sim_ID': sim_ID,
        'duration': duration,
        'timestep': dt,
        'path': path,
        'Box2D': Box2D,
    }


def brain_dict(modules, nengo=False, odor_dict=None, **kwargs):
    modules = get_dict('modules', **{m: True for m in modules})
    d = {'modules': modules}
    for k, v in modules.items():
        p = f'{k}_params'
        if not v:
            d[p] = None
        elif k in list(kwargs.keys()):
            d[p] = kwargs[k]
        else:
            d[p] = get_dict(k)
        if k == 'olfactor' and d[p] is not None:
            d[p]['odor_dict'] = odor_dict
    d['nengo'] = nengo
    return d


def larva_dict(brain, **kwargs):
    d = {'brain': brain}
    for k in ['energetics', 'physics', 'body', 'odor']:
        if k in list(kwargs.keys()):
            d[k] = kwargs[k]
        elif k == 'energetics':
            d[k] = None
        else:
            d[k] = get_dict(k)
    return d


def new_odor_dict(ids: list, means: list, stds=None) -> dict:
    if stds is None:
        stds = np.array([0.0] * len(means))
    odor_dict = {}
    for id, m, s in zip(ids, means, stds):
        odor_dict[id] = {'mean': m,
                         'std': s}
    return odor_dict
