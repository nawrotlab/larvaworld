from typing import List, Tuple

from lib.conf import par_conf

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
    'run' : ['null', 'default', 'deb', 'odor_preference'],
    'post' : ['null', 'default'],
    'final' : ['null', 'scatterplots', 'deb', 'odor_preference'],
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
