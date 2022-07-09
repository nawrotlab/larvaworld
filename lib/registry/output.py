
output_dict = {
    'olfactor': {
        'step': ['c_odor1', 'dc_odor1', 'c_odor2', 'dc_odor2', 'A_olf', 'A_T', 'I_T', 'A_C'],
        'endpoint': []},

    'thermo': {
        'step': ['temp_W', 'dtemp_W', 'temp_C', 'dtemp_C', 'A_therm'],
        'endpoint': []},

    'toucher': {
        'step': ['A_touch', 'A_tur', 'Act_tur', 'cum_f_det', 'on_food_tr', 'on_food'],
        'endpoint': ['on_food_tr']},

    'wind': {
        'step': ['A_wind'],
        'endpoint': []},

    'feeder': {
        'step': ['l', 'm', 'f_am', 'sf_am', 'EEB'],
        'endpoint': ['l', 'm', 'f_am', 'sf_am', 'on_food_tr']
    },

    'gut': {'step': ['sf_am_Vg', 'sf_am_V',  'f_am_V', 'sf_am_A', 'sf_am_M', 'sf_abs_M', 'f_abs_M', 'sf_faeces_M', 'f_faeces_M',
                     'f_am'],
            'endpoint': ['sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M', 'sf_abs_M', 'f_abs_M', 'sf_faeces_M',
                         'f_faeces_M', 'f_am']},
    'pose': {'step': ['x', 'y', 'b', 'fo', 'ro'],
             'endpoint': ['l', 'cum_t', 'x']},
    'memory': {'step': [],
               'endpoint': [],
               'tables': {'best_gains': ['unique_id', 'first_odor_best_gain', 'second_odor_best_gain', 'cum_reward',
                                         'best_olfactor_decay']}},
    'midline': None,
    'contour': None,
    # 'source_vincinity': {'step': [], 'endpoint': ['d_cent_fin']},
    # 'source_approach': {'step': [], 'endpoint': ['d_chem_fin']},
}

output_keys = list(output_dict.keys())