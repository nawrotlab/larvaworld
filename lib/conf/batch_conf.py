def optimization(fit_par, minimize=True, threshold=0.0001, max_Nsims=10, Nbest=4, operations={'mean' : True, 'std': False, 'abs' : False}):
    return {
        'fit_par': fit_par,
        'operations' : operations,
        'minimize': minimize,
        'threshold': threshold,
        'max_Nsims': max_Nsims,
        'Nbest': Nbest
    }


def batch_methods(run='default', post='default', final='null'):
    return {'run': run,
            'post': post,
            'final': final}


odor_pref_batch = {
    'exp': 'odor_preference',
    'space_search': {
        'pars': ['CS.mean', 'UCS.mean'],
        'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
        'Ngrid': [3, 3]
    },
    'methods': batch_methods(run='odor_preference', post='null', final='odor_preference'),
    'optimization': None,
    'run_kwargs' : {'save_data_flag': False}
}

chemorbit_batch = {
    'exp': 'chemotaxis_local',
    'space_search': {
        'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3]
    },
    'methods': batch_methods(),
    'optimization': optimization('scaled_dispersion'),
    'run_kwargs' : {'save_data_flag': False}
}

food_grid_batch = {
    'exp': 'food_grid',
    'space_search': {
        'pars': ['EEB', 'EEB_decay_coef'],
        'ranges': [(0.0, 1.0), (0.1, 2.0)],
        'Ngrid': [6, 6]
    },
    'methods': batch_methods(),
    'optimization': optimization('amount_eaten'),
    'run_kwargs' : {'save_data_flag': False}
}

chemotax_batch = {
    'exp': 'chemotaxis_approach',
    'space_search': {
        'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3]
    },
    'methods': batch_methods(),
    'optimization': optimization('final_scaled_dst_to_chemotax_odor'),
    'run_kwargs' : {'save_data_flag': False}
}

uniform_food_batch = {
    'exp': 'uniform_food',
    'space_search': {
        'pars': ['EEB', 'feeder_interference_free_window'],
        'ranges': [(0.0, 1.0), (0.0, 1.0)],
        'Ngrid': [3, 3]
    },
    'methods': batch_methods(),
    'optimization': optimization('amount_eaten'),
    'run_kwargs' : {'save_data_flag': False}
}

rovers_sitters_batch = {
    'exp': 'rovers_sitters',
    'space_search': {
        'pars': ['deb_base_f', 'hours_as_larva'],
        'ranges': [(0.5, 0.8), (0, 100)],
        'Ngrid': [2, 2]
    },
    'methods': batch_methods(run='deb', post='null', final='deb'),
    'optimization': None,
    'run_kwargs' : {'save_data_flag': True}
}


growth_batch = {
    'exp': 'growth',
    'space_search': {
        'pars': ['EEB', 'hunger_sensitivity'],
        'ranges': [(0.5, 0.8), (0.0, 0.0)],
        'Ngrid': [8, 1]
    },
    'methods': batch_methods(),
    'optimization': optimization('deb_f_deviation', max_Nsims=20, operations={'mean':True, 'abs':True}),
    'run_kwargs' : {'save_data_flag': False}
}
