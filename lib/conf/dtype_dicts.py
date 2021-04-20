from typing import List, Tuple
import numpy as np

from lib.conf import par_conf
from lib.conf.conf import loadConfDict, next_idx

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


def get_dict(name, class_name=None, **kwargs):
    if name in list(all_null_dicts.keys()):
        dic = all_null_dicts[name]
    elif name == 'distro':
        dic = null_distro(class_name=class_name)
    elif name == 'agent':
        dic = null_agent(class_name=class_name)
    dic.update(**kwargs)
    return dic


def get_distro(class_name, **kwargs):
    distro = null_distro(class_name)
    distro.update(**kwargs)
    return distro


def null_distro(class_name):
    distro = {
        'mode': None,
        'shape': None,
        'N': 0,
        'loc': (0.0, 0.0),
        'scale': (0.0, 0.0),
    }
    if class_name == 'Larva':
        distro = {**distro, 'orientation_range': (0.0, 360.0), 'model': None}
    return distro


all_null_dicts = {
    'odor':
        {'odor_id': None,
         'odor_intensity': 0.0,
         'odor_spread': None
         },
    'food':
        {
            'radius': 0.001,
            'amount': 0.0,
            'quality': 1.0,
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
    'visualization': null_vis

}


def get_dict_dtypes(name, **kwargs):
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

        'visualization': vis_dtypes

    }
    if name in list(all_dtypes.keys()):
        return all_dtypes[name]
    elif name == 'distro':
        return get_distro_dtypes(**kwargs)
    elif name == 'agent':
        return get_agent_dtypes(**kwargs)


def get_agent_dtypes(class_name):
    dtypes = {
        'unique_id': str,
        'default_color': str,
        'group': str,
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dtypes = {**dtypes, **get_dict_dtypes('odor')}
    elif class_name in ['Source', 'Food']:
        dtypes = {**dtypes, **get_dict_dtypes('odor'), **get_dict_dtypes('food')}
    elif class_name in ['Border']:
        dtypes = {**dtypes, 'width': float, 'points': List[Tuple[float, float]]}
    return dtypes


def null_agent(class_name):
    dic = {
        'unique_id': None,
        'default_color': 'black',
        'group': '',
    }
    if class_name in ['Larva', 'LarvaSim', 'LarvaReplay']:
        dic = {**dic, **get_dict('odor')}
    elif class_name in ['Source', 'Food']:
        dic = {**dic, **get_dict('odor'), **get_dict('food')}
    elif class_name in ['Border']:
        dic = {**dic, 'width': 0.001, 'points': None}
    return dic


def get_distro_dtypes(class_name, basic=True):
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
    return dtypes

def sim_dict(sim_id=None, sim_dur=3, dt=0.1, path=None, Box2D=False, exp_type=None):
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
