from typing import List, Tuple
import numpy as np

from lib.conf import par_conf
from lib.conf.conf import loadConfDict, next_idx

odor_dtypes = {'odor_id': str,
               'odor_intensity': float,
               'odor_spread': float}

odor_null_distro = {'odor_id': None,
                    'odor_intensity': 0.0,
                    'odor_spread': None}

food_dtypes = {
    'radius': float,
    'amount': float,
    'quality': float,
    # **odor_dtypes
}
food_null_distro = {
    'radius': 0.001,
    'amount': 0.0,
    'quality': 1.0,
    # **odor_dtypes
}

basic_dtypes = {
    'unique_id': str,
    'default_color': str,
    'group': str,
    'pos': Tuple[float, float]
}

source_dtypes = {
    **basic_dtypes,
    **food_dtypes,
    **odor_dtypes
}

odor_gain_pars = {
    'unique_id': str,
    'mean': float,
    'std': float
}

larva_dtypes = {
    **basic_dtypes,
    **odor_dtypes
}
border_dtypes = {
    'unique_id': str,
    'default_color': str,
    'width': float,
    'points': List[Tuple[float, float]]
}

agent_dtypes = {'Food': source_dtypes,
                'LarvaSim': larva_dtypes,
                'LarvaReplay': larva_dtypes,
                'Border': border_dtypes,
                }

arena_dtypes = {'arena_xdim': float,
                'arena_ydim': float,
                'arena_shape': ['circular', 'rectangular']}

odorscape_dtypes = {'odorscape': ['Gaussian', 'Diffusion'],
                    'grid_dims': tuple,
                    'evap_const': float,
                    'gaussian_sigma': Tuple[float, float],
                    }

operation_dtypes = {
    'mean': bool,
    'std': bool,
    'abs': bool,
}
optimization_dtypes = {
    'fit_par': par_conf.get_runtime_pars(),
    'operations': operation_dtypes,
    'minimize': bool,
    'threshold': float,
    'max_Nsims': int,
    'Nbest': int
}

batch_method_dtypes = {
    'run': ['null', 'default', 'deb', 'odor_preference'],
    'post': ['null', 'default'],
    'final': ['null', 'scatterplots', 'deb', 'odor_preference'],
}

space_search_dtypes = {'pars': str,
                       'ranges': Tuple[float, float],
                       'Ngrid': int}

# method_pars_dict = {
#
# }
life_dtypes = {
    'starvation_hours': List[Tuple[float, float]],
    'hours_as_larva': float,
    'deb_base_f': float

}

vis_render_dtypes = {
    'mode': ['', 'video', 'image'],
    'image_mode': ['final', 'snapshots', 'overlap'],
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

# vis_dtypes = {
#     **vis_render_dtypes,
#     **vis_draw_dtypes,
#     **vis_color_dtypes,
#     **vis_aux_dtypes,
# }

vis_dtypes = {
    'render': vis_render_dtypes,
    'draw': vis_draw_dtypes,
    'color': vis_color_dtypes,
    'aux': vis_aux_dtypes,
}


def get_vis_kwargs_dict(mode='video', image_mode='final', video_speed=1, show_display=True, media_name=None,
                        draw_head=False, draw_centroid=False, draw_midline=True, draw_contour=True,
                        trajectories=False, trajectory_dt=0.0,
                        black_background=False, random_colors=False, color_behavior=False,
                        visible_clock=True, visible_state=True, visible_scale=True, visible_ids=False,
                        ):
    dic = {
        'render': {'mode': mode,
                   'image_mode': image_mode,
                   'video_speed': video_speed,
                   'media_name': media_name,
                   'show_display': show_display},
        'draw': {'draw_head': draw_head,
                 'draw_centroid': draw_centroid,
                 'draw_midline': draw_midline,
                 'draw_contour': draw_contour,
                 'trajectories': trajectories,
                 'trajectory_dt': trajectory_dt},
        'color': {'black_background': black_background,
                  'random_colors': random_colors,
                  'color_behavior': color_behavior},
        'aux': {'visible_clock': visible_clock,
                'visible_scale': visible_scale,
                'visible_state': visible_state,
                'visible_ids': visible_ids}
    }

    return dic


replay_dtypes = {
    'arena_pars': arena_dtypes,
    'env_params': [''] + list(loadConfDict('Env').keys()),
    'track_point': int,
    # 'spinepoints': bool,
    # 'centroid': bool,
    # 'contours': bool,
    'dynamic_color': ['', 'lin_color', 'ang_color'],
    'agent_ids': list,
    'time_range': Tuple[float, float],
    'transposition': ['', 'origin', 'arena', 'center'],
    'fix_point': int,
    'secondary_fix_point': ['', 1, -1],
    'use_background': bool,
    'draw_Nsegs': int,
}


def get_replay_kwargs_dict(arena_pars=None,
                           env_params=None,
                           track_point=None,
                           # spinepoints=True, centroid=True, contours=True,
                           dynamic_color=None,
                           agent_ids=None,
                           time_range=None,
                           transposition=None, fix_point=None, secondary_fix_point=None, use_background=False,
                           draw_Nsegs=None):
    dic = {
        'arena_pars': arena_pars,
        'env_params': env_params,
        'track_point': track_point,
        # 'spinepoints': spinepoints,
        # 'centroid': centroid,
        # 'contours': contours,
        'dynamic_color': dynamic_color,
        'agent_ids': agent_ids,
        'time_range': time_range,
        'transposition': transposition,
        'fix_point': fix_point,
        'secondary_fix_point': secondary_fix_point,
        'use_background': use_background,
        'draw_Nsegs': draw_Nsegs,
        # 'centroid': centroid,

    }
    return dic



def basic_null_distro(class_name):
    distro = {
        'mode': None,
        'shape': None,
        'N': 0,
        'loc': (0.0, 0.0),
        'scale': 0.0,
    }
    if class_name == 'Larva':
        distro = {**distro, 'orientation_range': (0.0, 360.0)}
    return distro


def basic_distro_types(class_name):
    dtypes = {
        'mode': [
            'normal',
            'periphery',
            'uniform',
            # 'uniform_circ',

        ],
        'shape' : ['circle', 'rect'],
        'N': int,
        'loc': Tuple[float, float],
        'scale': Tuple[float, float],
    }
    if class_name == 'Larva':
        dtypes = {**dtypes, 'orientation_range': Tuple[float, float]}
    return dtypes


def distro_dtypes(class_name, basic=False):
    basic_dtypes = basic_distro_types(class_name)
    if basic:
        return basic_dtypes

    if class_name == 'Food':
        return {
            'group': str,
            'default_color': str,
            **basic_dtypes,
            **food_dtypes,
            **odor_dtypes
        }
    elif class_name == 'Larva':
        from lib.conf.conf import loadConfDict
        return {
            'group': str,
            'default_color': str,
            'model': list(loadConfDict('Model').keys()),
            **basic_dtypes,
            **odor_dtypes
        }


def null_distro(class_name, basic=False):
    basic_distro = basic_null_distro(class_name)
    if basic:
        return basic_distro

    if class_name == 'Food':
        return {
            'group': '',
            'default_color': 'green',
            **basic_distro,
            **food_null_distro,
            **odor_null_distro
        }
    elif class_name == 'Larva':
        from lib.conf.conf import loadConfDict
        return {
            'group': '',
            'default_color': 'black',
            'model': None,
            **basic_distro,
            **odor_null_distro
        }


#
#
# names = [
#             'trajectory_dt',
#             'trajectories',
#             'focus_mode',
#             'draw_centroid',
#             'draw_head',
#             'draw_midline',
#             'draw_contour',
#             'visible_clock',
#             'visible_ids',
#             'visible_state',
#             'color_behavior',
#             'random_colors',
#             'black_background',
#             'larva_collisions',
#             'zoom',
#             'snapshot #',
#             'odorscape #'
#         ]
#
#
# input_default={}
# for name in names :
#     input_default[name] = True
#
# print(input_default)
#
# keyboard_shortcut = {}
# for name in names :
#     keyboard_shortcut[name] = 'e'
# print(keyboard_shortcut)

def life_dict(f=1, age=0, starvation=None):
    return {'deb_base_f': f,
            'hours_as_larva': age,
            'starvation_hours': starvation
            }


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
