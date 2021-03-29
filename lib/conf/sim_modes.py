from typing import List, Tuple

full_traj = {'trajectories': True,
             'trail_decay_in_sec': 0}

trail_traj = {'trajectories': True,
              'trail_decay_in_sec': 10}

no_traj = {'trajectories': False,
           'trail_decay_in_sec': None}

draw_default = {'color_behavior': False,
                'draw_head': False,
                'draw_contour': False,
                'random_larva_colors': False}

draw_behavior = {'color_behavior': True,
                 'draw_head': False,
                 'draw_contour': False,
                 'random_larva_colors': False}

draw_colors = {'color_behavior': False,
               'draw_head': False,
               'draw_contour': False,
               'random_larva_colors': True}

draw_on_black = {'color_behavior': False,
                 'draw_head': False,
                 'draw_contour': False,
                 'black_background': True,
                 'random_larva_colors': False}

default_sim = {
    # 'dt': 1 / 6,
    'dt': 1 / 10,
    # 'dt': 1 / 16,
    # 'dt': 1 / 200,
    'sim_time_in_min': 3.0,
    'Box2D' : False,
}


food_pars = {
    'unique_id': str,
    'pos': Tuple[float, float],
    'radius': float,
    'amount': float,
    'quality': float,
    'odor_id': str,
    'odor_intensity': float,
    'odor_spread': float
}
larva_pars = {
    'unique_id': str,
}
border_pars = {
    'unique_id': str,
    'width': float,
    'points': List[Tuple[float, float]]
}

agent_pars = {'Food' : food_pars,
              'LarvaSim' : larva_pars,
              'LarvaReplay' : larva_pars,
              'Border' : border_pars,
              }