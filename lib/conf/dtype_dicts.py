import copy
from typing import List, Tuple
import numpy as np

vis_render_dtypes = {
    'mode': [None, 'video', 'image'],
    'image_mode': [None, 'final', 'snapshots', 'overlap'],
    'video_speed': int,
    'media_name': str,
    'show_display': bool,
}

vis_draw_dtypes = {
    'draw_head': bool,
    'draw_centroid': bool,
    'draw_midline': bool,
    'draw_contour': bool,
    'trajectories': bool,
    'trajectory_dt': float,
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


def get_dict(name, class_name=None, basic=True, as_entry=False, **kwargs):
    if name in list(all_null_dicts.keys()):
        d = all_null_dicts[name]
    elif name == 'distro':
        d = null_distro(class_name=class_name, basic=basic)
    elif name == 'agent':
        d = null_agent(class_name=class_name)
    dic = copy.deepcopy(d)
    if name == 'visualization':
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
        },
    'food_grid':
        {
            'unique_id': 'Food_grid',
            'grid_dims': (50, 50),
            'initial_value': 10 ** -3,
            'distribution': 'uniform',
        },
    'arena':
        {
            'arena_xdim': 0.1,
            'arena_ydim': 0.1,
            'arena_shape': 'circular'
        },
    'life':
        {
            'starvation_hours': None,
            'hours_as_larva': 0.0,
            'deb_base_f': 1.0

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
             'length_std': 0.0,
             'Nsegs': 2,
             'seg_ratio': None
             },
    'physics': {
        'torque_coef': 0.41,
        'ang_damping': 2.5,
        'body_spring_k': 0.02,
        'bend_correction_coef': 1.4,
    },
    'energetics': {'f_decay_coef': 0.1,  # 0.1,  # 0.3
                   'absorption_c': 0.5,
                   'hunger_affects_balance': True,
                   'hunger_sensitivity': 10.0,
                   'deb_on': True},
    'crawler': {'waveform': 'realistic',
                'freq_range': [0.5, 2.5],
                'initial_freq': 'sample',  # From D1 fit
                'step_to_length_mu': 'sample',  # From D1 fit
                'step_to_length_std': 'sample',  # From D1 fit
                'initial_amp': None,
                'crawler_noise': 0.0,
                'max_vel_phase': 1
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
        'attenuation_ratio': 1.0
    },
    'intermitter': {'pause_dist': 'fit',
                    'stridechain_dist': 'fit',
                    'intermittent_crawler': False,
                    'intermittent_feeder': False,
                    'EEB_decay_coef': 1.0,
                    'EEB': 1.0},
    'olfactor': {
        'perception': 'log',
        'olfactor_noise': 0.0,
        'decay_coef': 0.5},
    'feeder': {'feeder_freq_range': [1.0, 3.0],
               'feeder_initial_freq': 2.0,
               'feed_radius': 0.1,
               'max_feed_amount_ratio': 0.00001},
    'memory': {'DeltadCon': 0.1,
               'state_spacePerOdorSide': 0,
               'gain_space': [-300.0, -50.0, 50.0, 300.0],
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
                'memory': False}
}


def get_dict_dtypes(name, **kwargs):
    from lib.conf import par_conf
    from lib.conf.conf import loadConfDict
    all_dtypes = {
        'odor':
            {'odor_id': str,
             'odor_intensity': float,
             'odor_spread': float
             },
        'food':
            {
                'radius': float,
                'amount': float,
                'quality': float,
                'shape_vertices': List[Tuple[float, float]],
                'can_be_carried': bool,
            },
        'food_grid':
            {
                'unique_id': str,
                'grid_dims': Tuple[int, int],
                'initial_value': float,
                'distribution': ['uniform'],
            },
        'arena':
            {
                'arena_xdim': float,
                'arena_ydim': float,
                'arena_shape': ['circular', 'rectangular']
            },
        'life':
            {
                'starvation_hours': List[Tuple[float, float]],
                'hours_as_larva': float,
                'deb_base_f': float

            },
        'odorscape': {'odorscape': ['Gaussian', 'Diffusion'],
                      'grid_dims': tuple,
                      'evap_const': float,
                      'gaussian_sigma': Tuple[float, float],
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
            'track_point': int,
            'dynamic_color': [None, 'lin_color', 'ang_color'],
            'agent_ids': list,
            'time_range': Tuple[float, float],
            'transposition': [None, 'origin', 'arena', 'center'],
            'fix_point': int,
            'secondary_fix_point': ['', 1, -1],
            'use_background': bool,
            'draw_Nsegs': int,
        },
        'optimization': {
            'fit_par': par_conf.get_runtime_pars(),
            'operations': {
                'mean': bool,
                'std': bool,
                'abs': bool,
            },
            'minimize': bool,
            'threshold': float,
            'max_Nsims': int,
            'Nbest': int
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
        'body': {'initial_length': float,
                 'length_std': float,
                 'Nsegs': int,
                 'seg_ratio': List[float]  # [5 / 11, 6 / 11]
                 },
        'physics': {
            'torque_coef': float,
            'ang_damping': float,
            'body_spring_k': float,
            'bend_correction_coef': float,
        },
        'energetics': {'f_decay_coef': float,
                       'absorption_c': float,
                       'hunger_affects_balance': bool,
                       'hunger_sensitivity': float,
                       'deb_on': bool},
        'crawler': {'waveform': ['realistic', 'square', 'gaussian', 'constant'],
                    'freq_range': Tuple[float, float],
                    'initial_freq': float,  # From D1 fit
                    'step_to_length_mu': float,  # From D1 fit
                    'step_to_length_std': float,  # From D1 fit
                    'initial_amp': float,
                    'crawler_noise': float,
                    'max_vel_phase': float
                    },
        'turner': {'mode': ['', 'neural', 'sinusoidal'],
                   'base_activation': float,
                   'activation_range': Tuple[float, float],
                   'noise': float,
                   'activation_noise': float,
                   'initial_amp': float,
                   'amp_range': Tuple[float, float],
                   'initial_freq': float,
                   'freq_range': Tuple[float, float],
                   },
        'interference': {
            'crawler_phi_range': Tuple[float, float],  # np.pi * 0.55,  # 0.9, #,
            'feeder_phi_range': Tuple[float, float],
            'attenuation_ratio': float
        },
        'intermitter': {'pause_dist': dict,
                        'stridechain_dist': dict,
                        'intermittent_crawler': bool,
                        'intermittent_feeder': bool,
                        'EEB_decay_coef': float,
                        'EEB': float},
        'olfactor': {
            'perception': ['log', 'linear'],
            'olfactor_noise': float,
            'decay_coef': float},
        'feeder': {'feeder_freq_range': Tuple[float, float],
                   'feeder_initial_freq': float,
                   'feed_radius': float,
                   'max_feed_amount_ratio': float},
        'memory': {'DeltadCon': float,
                   'state_spacePerOdorSide': int,
                   'gain_space': List[float],
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
                    'memory': bool}

    }
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
        'group': str,
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dtypes = {**dtypes, **get_dict_dtypes('odor')}
    elif class_name in ['Source', 'Food']:
        dtypes = {**dtypes, **get_dict_dtypes('odor'), **get_dict_dtypes('food'), 'can_be_carried': bool,
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


def sim_dict(sim_id=None, sim_dur=3, dt=0.1, path=None, Box2D=False, exp_type=None):
    from lib.conf.conf import next_idx
    if exp_type is not None:
        if sim_id is None:
            sim_id = f'{exp_type}_{next_idx(exp_type)}'
        if path is None:
            path = f'single_runs/{exp_type}'
    return {
        'sim_id': sim_id,
        'sim_dur': sim_dur,
        'dt': dt,
        'path': path,
        'Box2D': Box2D
    }
