from typing import List, Tuple

from lib.conf import par_conf
from lib.conf.conf import loadConfDict

odor_pars = {'odor_id': str,
             'odor_intensity': float,
             'odor_spread': float}

base_food_pars = {
    'radius': float,
    'amount': float,
    'quality': float,
    **odor_pars
}

food_pars = {
    'unique_id': str,
    # 'default_color': str,
    'pos': Tuple[float, float],
    **base_food_pars
}

odor_gain_pars = {
    'unique_id': str,
    'mean': float,
    'std': float
}

larva_pars = {
    'unique_id': str,
    'group': str,
    **odor_pars
}
border_pars = {
    'unique_id': str,
    'width': float,
    'points': List[Tuple[float, float]]
}

agent_pars = {'Food': food_pars,
              'LarvaSim': larva_pars,
              'LarvaReplay': larva_pars,
              'Border': border_pars,
              }

arena_pars_dict = {'arena_xdim': float,
                   'arena_ydim': float,
                   'arena_shape': ['circular', 'rectangular']}

odorscape_pars_dict = {'odorscape': ['Gaussian', 'Diffusion'],
                       'grid_dims': Tuple[float, float],
                       'evap_const': float,
                       'gaussian_sigma': Tuple[float, float],
                       }

opt_pars_dict = {
    'fit_par': par_conf.get_runtime_pars(),
    # 'fit_par': str,
    'minimize': bool,
    'threshold': float,
    'max_Nsims': int,
    'Nbest': int
}

batch_methods_dict = {
    'run': ['null', 'default', 'deb', 'odor_preference'],
    'post': ['null', 'default'],
    'final': ['null', 'scatterplots', 'deb', 'odor_preference'],
}

# space_pars_dict = {'pars': List[str],
#                    'ranges': List[Tuple[float, float]],
#                    'Ngrid': List[int]}

space_pars_dict = {'pars': str,
                   'ranges': Tuple[float, float],
                   'Ngrid': int}

# method_pars_dict = {
#
# }
life_pars_dict = {
    'starvation_hours': List[Tuple[float, float]],
    'hours_as_larva': float,
    'deb_base_f': float

}

vis_render_dict = {
    'mode': ['', 'video', 'image'],
    'image_mode': ['final', 'snapshots', 'overlap'],
    'video_speed': int,
    'media_name': str,
    'show_display': bool,
}

vis_draw_dict = {
    'draw_head': bool,
    'draw_centroid': bool,
    'draw_midline': bool,
    'draw_contour': bool,
    'trajectories': bool,
    'trajectory_dt': float,
}

vis_color_dict = {
    'black_background': bool,
    'random_colors': bool,
    'color_behavior': bool,
}

vis_aux_dict = {
    'visible_clock': bool,
    'visible_scale': bool,
    'visible_state': bool,
    'visible_ids': bool,
}

vis_pars_dict0 = {
    **vis_render_dict,
    **vis_draw_dict,
    **vis_color_dict,
    **vis_aux_dict,
}

vis_pars_dict = {
    'render': vis_render_dict,
    'draw': vis_draw_dict,
    'color': vis_color_dict,
    'aux': vis_aux_dict,
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


replay_pars_dict = {
    'arena_pars': arena_pars_dict,
    'env_params': ['']+list(loadConfDict('Env').keys()),
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
                           transposition=None, fix_point=None, secondary_fix_point=None,use_background=False,
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


def distro_pars(class_name):
    larva_distros = [
        'normal',
        'defined',
        'identical',
        'uniform',
        'uniform_circ',
        'spiral',
        'facing_right'
    ]

    food_distros = [
        'normal',
        # 'defined',
        'uniform'
    ]

    agent_distros = {
        'Larva': larva_distros,
        'Food': food_distros,
    }

    common_distro_pars = {
        'group': str,
        'default_color': str,
        'mode': agent_distros[class_name],
        'N': int,
        'loc': Tuple[float, float],
        'scale': float,
    }
    if class_name == 'Food':
        return {
            **common_distro_pars,
            **base_food_pars,
            # 'pars': base_food_pars,
        }
    elif class_name == 'Larva':
        from lib.conf.conf import loadConfDict
        return {
            'model': list(loadConfDict('Model').keys()),
            **common_distro_pars,
            'orientation': float

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
