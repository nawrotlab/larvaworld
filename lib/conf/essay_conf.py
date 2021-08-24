from lib.conf.larva_conf import *
from lib.conf.env_conf import *
from lib.conf import dtype_dicts as dtypes
from lib.stor import paths

rover_sitter_essay = {'experiments':{
    'pathlength': {
        'exp_types': ['rovers_sitters_on_standard', 'rovers_sitters_on_agar'],
        'durations': [20, 20]
    },
    'intake': {
        'exp_types': ['rovers_sitters_on_standard'] * 3,
        'durations': [10, 15, 20]
    },
    'starvation': {
        'exp_types': [
            'rovers_sitters_on_standard',
            'rovers_sitters_on_standard_1h_prestarved',
            'rovers_sitters_on_standard_2h_prestarved',
            'rovers_sitters_on_standard_3h_prestarved',
            'rovers_sitters_on_standard_4h_prestarved',
        ],
        'durations': [5] * 5
    },
    'quality': {
        'exp_types': [
            'rovers_sitters_on_standard',
            'rovers_sitters_on_standard_q75',
            'rovers_sitters_on_standard_q50',
            'rovers_sitters_on_standard_q25',
            'rovers_sitters_on_standard_q15',
        ],
        'durations': [5] * 5
    },
    'refeeding': {
        'exp_types': [
            'rovers_sitters_on_standard_3h_prestarved'
        ],
        'durations': [120]
    }
},
'exp_fig_folder' : paths.RoverSitterFigFolder}

essay_dict = {'roversVSsitters': rover_sitter_essay}
