import sys
import numpy as np

from lib.sim.batch_lib import *

# sys.path.insert(0, '../../..')

odor_pref_batch = {
    'pars': ['CS.mean', 'UCS.mean'],
    'ranges': np.array([
        [-100.0, 100.0],
        [-100.0, 100.0]
    ]),
    'process_method': PI_computation,
    'post_process_method': null_post_processing,
    'final_process_method': heat_map_generation,
    'space_method': grid_search_dict,
    # 'space_method': generate_gain_space,
    'batch_config': None,
    'post_kwargs': {},
    'run_kwargs': {}
}

chemorbit_batch = {
    'pars': ['base_activation',
             'activation_noise'],
    'ranges': np.array([[10.0, 30.0],
                        [0.0, 1.0]]),
    'process_method': default_processing,
    'post_process_method': post_processing,
    'final_process_method': null_final_processing,
    'space_method': grid_search_dict,
    'batch_config': {'fit_par': 'final_scaled_dst_to_center',
                     'minimize': True,
                     'threshold': 0.1},
    'post_kwargs': {},
    'run_kwargs': {}
}

chemotax_batch = {
    'pars': ['base_activation',
             'activation_noise'],
    'ranges': np.array([[10.0, 30.0],
                        [0.0, 1.0]]),
    'process_method': default_processing,
    'post_process_method': post_processing,
    'final_process_method': null_final_processing,
    'space_method': grid_search_dict,
    'batch_config': {'fit_par': 'final_scaled_dst_to_chemotax_odor',
                     'minimize': True,
                     'threshold': 0.1},
    'post_kwargs': {},
    'run_kwargs': {}
}

feed_scatter_batch = {
    'pars': ['explore2exploit_bias',
             'feeder_interference_free_window'],
    'ranges': np.array([[0.0, 1.0],
                        [0.0, 1.0]]),
    'process_method': default_processing,
    'post_process_method': post_processing,
    'final_process_method': null_final_processing,
    'space_method': grid_search_dict,
    'batch_config': {'fit_par': 'amount_eaten',
                     'minimize': False,
                     'threshold': 10.0},
    'post_kwargs': {},
    'run_kwargs': {}
}

feed_grid_batch = {
    'pars': ['explore2exploit_bias',
             'feeder_reoccurence_rate_on_success'],
    'ranges': np.array([[0.2, 0.8],
                        [0.05, 0.95]]),
    'process_method': null_processing,
    'post_process_method': null_post_processing,
    'final_process_method': end_scatter_generation,
    'space_method': grid_search_dict,
    'batch_config': {
        'end_parshorts_1': ['f_am', 'cum_sd'],
        'end_parshorts_2': ['fee_N', 'str_N'],
        'end_parshorts_3': ['fee_tr', 'str_tr']
    },
    'post_kwargs': {},
    'run_kwargs': {'save_data_flag': True}

}

growth_batch = {
    'pars': [
        'EEB',
        'f_decay_coef',
        # 'f_increment'
    ],
    'ranges': np.array([
        [0.3, 0.8],
        [0.1, 0.6],
        # [0.8, 1.2]
    ]),
    'process_method': deb_processing,
    'post_process_method': null_post_processing,
    'final_process_method': deb_analysis,
    'space_method': grid_search_dict,
    'batch_config': None,
    'post_kwargs': {},
    'run_kwargs': {'save_data_flag': True}
}

growth_2x_batch = {
    'pars': [
        'EEB',
        'hunger_sensitivity',
        # 'f_increment'
    ],
    'ranges': np.array([
        [0.2, 0.99],
        [0.0, 0.0],
        # [0.8, 1.2]
    ]),
    'process_method': deb_processing,
    'post_process_method': post_processing,
    'final_process_method': null_final_processing,
    'space_method': grid_search_dict,
    'batch_config': {'fit_par': 'deb_f_mean_deviation',
                     'minimize': True,
                     'threshold': 0.00001},
    'post_kwargs': {},
    'run_kwargs': {'save_data_flag': True}
}

# Just change the 'larva_pars' to growing_sitter, in growth_2x in exp_modes
# python batch_run.py growth_2x -N 6 -t 5 -id_b test333 -Ngrd 8 1 -Nmax 20 -Nbst 4



batch_types = {
    'odor_pref': odor_pref_batch,
    'chemorbit': chemorbit_batch,
    'chemotax': chemotax_batch,
    'feed_scatter': feed_scatter_batch,
    'feed_grid': feed_grid_batch,
    'growth': growth_batch,
    'growth_2x': growth_2x_batch,
}
