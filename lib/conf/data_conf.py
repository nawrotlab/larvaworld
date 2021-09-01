import numpy as np
from lib.aux import naming as nam
import lib.conf.env_conf as env

PaisiosParConf = {'bend': 'from_vectors',
                  'front_vector': (2,4),
                  # 'front_vector_start': 2,
                  # 'front_vector_stop': 4,
                  'rear_vector_start': (7,11),
                  # 'rear_vector_start': 7,
                  # 'rear_vector_stop': 11,
                  'front_body_ratio': 0.5,
                  'point_idx': 6,
                  'use_component_vel': False,
                  'scaled_vel_threshold': 0.2}

SinglepointParConf = {'bend': None,
                      'front_vector': (None,None),
                      # 'front_vector_start': None,
                      # 'front_vector_stop': None,
                      'rear_vector': (None,None),
                      # 'rear_vector_start': None,
                      # 'rear_vector_stop': None,
                      'front_body_ratio': None,
                      'point_idx': 0,
                      'use_component_vel': False,
                      'scaled_vel_threshold': None}

SchleyerParConf = {'bend': 'from_angles',
                   'front_vector': (1,2),
                   # 'front_vector_start': 1,
                   # 'front_vector_stop': 2,
                   'rear_vector': (7,11),
                   # 'rear_vector_start': 7,
                   # 'rear_vector_stop': 11,
                   'front_body_ratio': 0.5,
                   'point_idx': np.nan,
                   'use_component_vel': False,
                   'scaled_vel_threshold': 0.2}

JovanicParConf = {'bend': 'from_angles',
                  'front_vector': (2,3),
                  # 'front_vector_start': 2,
                  # 'front_vector_stop': 3,
                  'rear_vector': (7,8),
                  # 'rear_vector_start': 7,
                  # 'rear_vector_stop': 8,
                  'front_body_ratio': 0.5,
                  'point_idx': 8,
                  'use_component_vel': False,
                  'scaled_vel_threshold': 0.2}

SimParConf = {'bend': 'from_angles',
              'front_vector': (1,2),
              # 'front_vector_start': 1,
              # 'front_vector_stop': 2,
              'rear_vector': (-2,-1),
              # 'rear_vector_start': -2,
              # 'rear_vector_stop': -1,
              'point_idx': np.nan,
              'use_component_vel': False,
              'scaled_vel_threshold': 0.2}

SimDataConf = {'fr': 16.0,
               'Npoints': 3,
               'Ncontour': 0
               }

SimEnrichConf = {
    'preprocessing': {
        'rescale_by': None,
        'drop_collisions': False,
        'interpolate_nans': False,
        'filter_f': None
    },
    # 'drop_contour': False,
    # 'drop_unused_pars': True,
    'processing': ['angular', 'spatial'],
    'to_drop': ['unused'],
    'dispersion_starts': [0],
    'bouts': ['turn', 'stride', 'pause'],
    'mode': 'minimal'}

SimConf = {'id': 'SimConf',
           'data': SimDataConf,
           'par': 'SimParConf',
           'build': None,
           'enrich': SimEnrichConf}

SchleyerDataConf = {'fr': 16.0,
                    'Npoints': 12,
                    'Ncontour': 22
                    }

Schleyer_raw_cols = ['Step'] + \
                    nam.xy(nam.midline(SchleyerDataConf['Npoints'])[::-1], flat=True) + \
                    nam.xy(nam.contour(SchleyerDataConf['Ncontour']), flat=True) + \
                    nam.xy('centroid') + \
                    ['blob_orientation', 'area', 'grey_value', 'raw_spinelength', 'width', 'perimeter',
                     'collision_flag']

Sims_raw_cols = ['Step'] + nam.xy('centroid')

SchleyerEnrichConf = {'preprocessing': {
        'rescale_by': 1.0,
        'drop_collisions': True,
        'interpolate_nans': False,
        'filter_f': 2.0
    },
    'processing': {'types': {'angular':True, 'spatial':True, 'source':False, 'dispersion':True, 'tortuosity':True},
                   'dsp_starts': [0, 20], 'dsp_stops': [40, 80, 120],
                   'tor_durs': [2, 5, 10, 20]},
    'annotation': {'bouts': {'stride':True, 'pause':True, 'turn':True}, 'track_point': None,
                   'track_pars': None, 'chunk_pars': None,
                   'vel_par': None, 'ang_vel_par': None, 'bend_vel_par': None, 'min_ang': 5.0,
                   'non_chunks': False},
    'enrich_aux': {'recompute': False,
                   'mode': 'minimal',
                   'source': None,
                   },
# 'to_drop': [],
}

SchleyerConf = {'id': 'SchleyerConf',
                'data': SchleyerDataConf,
                'par': 'SchleyerParConf',
                'build': {'read_sequence': Schleyer_raw_cols,
                          'read_metadata': True},
                'enrich': SchleyerEnrichConf}

SchleyerGroup = {
    'id': 'SchleyerGroup',
    'conf': 'SchleyerConf',
    'path': 'SchleyerGroup',
    'arena_pars': env.dish(0.15),
    'subgroups': ['no_odor', 'Ntrials', 'odor_conc', 'FRU_conc', 'new-reward-punishment'],
    'detect': {
        'folder' : {'pref' : 'box', 'suf' : None},
        'file' : {'pref' : None, 'suf' : None, 'sep' : None}
    }
}

JovanicDataConf = {'fr': 11.27,
                   'Npoints': 11,
                   'Ncontour': 0}


JovanicEnrichConf = {
'preprocessing': {
        'rescale_by': 1.0,
        'drop_collisions': False,
        'interpolate_nans': False,
        'filter_f': 2.0
    },
    'processing': {'types': {'angular':True, 'spatial':True, 'source':False, 'dispersion':True, 'tortuosity':True},
                   'dsp_starts': [0, 20], 'dsp_stops': [40, 80, 120],
                   'tor_durs': [2, 5, 10, 20]},
    'annotation': {'bouts': {'stride':True, 'pause':True, 'turn':True}, 'track_point': None,
                   'track_pars': None, 'chunk_pars': None,
                   'vel_par': None, 'ang_vel_par': None, 'bend_vel_par': None, 'min_ang': 5.0,
                   'non_chunks': False},
    'enrich_aux': {'recompute': False,
                   'mode': 'minimal',
                   'source': None,
                   },
# 'to_drop': [],
}


JovanicConf = {'id': 'JovanicConf',
               'data': JovanicDataConf,
               'par': 'JovanicParConf',
               'build': {'read_sequence': None,
                         'read_metadata': False},
               'enrich': JovanicEnrichConf}

JovanicGroup = {
    'id': 'JovanicGroup',
    'conf': 'JovanicConf',
    'path': 'JovanicGroup',
    'genotypes': ['AttP2@UAS_TNT', 'AttP240@UAS_TNT'],
    'subgroups': ['AttP2@UAS_TNT', 'AttP240@UAS_TNT', 'FoodPatches'],
    'conditions': ['Fed', 'ProteinDeprived', 'Starved'],
    'arena_pars': env.arena(0.193, 0.193),
'detect': {
        'folder' : {'pref' : None, 'suf' : None},
        'file' : {'pref' : None, 'suf' : 'larvaid.txt', 'sep' : '_'}
    }
}

TestGroup = {
    'id': 'TestGroup',
    'conf': 'SchleyerConf',
    'path': 'TestGroup',
    'subgroups': [],
    'arena_pars': env.dish(0.15)
}

SimGroup = {
    'id': 'SimGroup',
    'conf': 'SimConf',
    'path': 'SimGroup',
    'subgroups': ['single_runs', 'batch_runs'],
    'arena_pars': None
}
