from lib.aux import naming as nam
from lib.conf.base.dtypes import null_dict, base_enrich, enr_dict, arena, metric_def


import_par_confs = {
    'SchleyerParConf': metric_def(ang= {'rv': (7, 11)}),
    'JovanicParConf': metric_def(ang= {'fv': (2, 3), 'rv': (7, 8)}),
    'PaisiosParConf': metric_def(ang= {'b' : 'from_vectors','fv': (2, 4), 'rv': (7, 11)}, sp={'point_idx' : 6}),
    'SinglepointParConf': metric_def(ang= {'b' :None}, sp={'point_idx' : 0}),
    'SimParConf': metric_def(),
}

importformats = [
    {
        'id': 'Schleyer lab',
        'path': 'SchleyerGroup',
        'tracker': {
            'resolution': {'fr': 16.0,
                           'Npoints': 12,
                           'Ncontour': 22
                           },

            'filesystem': {
                'read_sequence': ['Step'] + nam.xy(nam.midline(12)[::-1], flat=True) + nam.xy(nam.contour(22),
                                                                                              flat=True) + nam.xy(
                    'centroid') + \
                                 ['blob_orientation', 'area', 'grey_value', 'raw_spinelength', 'width', 'perimeter',
                                  'collision_flag'],
                'read_metadata': True,
                # 'detect': {
                'folder': {'pref': 'box', 'suf': None},
                'file': {'pref': None, 'suf': None, 'sep': None}
                # }
            },
            'arena': arena(0.15)
        },
        # 'parameterization': parconf(rear_vector=(7, 11)),
        'enrichment': base_enrich(pre_kws={'filter_f': 2.0, 'drop_collisions': True, 'rescale_by': 0.001},
                                  def_kws={'ang': {'rv': (7, 11)}}),

    },
    {
        'id': 'Jovanic lab',
        'path': 'JovanicGroup',
        'tracker': {
            'resolution': {'fr': 11.27,
                           'Npoints': 11,
                           'Ncontour': 30},

            'filesystem': {
                'read_sequence': None,
                'read_metadata': False,
                'folder': {'pref': None, 'suf': None},
                'file': {'pref': None, 'suf': 'larvaid.txt', 'sep': '_'}
            },
            'arena': arena(0.193, 0.193)
        },
        # 'parameterization': parconf(front_vector=(2, 3), rear_vector=(7, 8)),
        'enrichment': base_enrich(pre_kws={'filter_f': 2.0, 'rescale_by': 0.001, 'transposition': 'arena'},
                                  def_kws={'ang': {'fv': (2, 3), 'rv': (7, 8)}}),
    },
    {
        'id': 'Berni lab',
        'path': 'BerniGroup',
        'tracker': {
            'resolution': {'fr': 2,
                           'Npoints': 1,
                           'Ncontour': 0},

            'filesystem': {
                'read_sequence': ['Date', 'x', 'y'],
                'read_metadata': False,
                # 'detect': {
                'folder': {'pref': None, 'suf': None},
                'file': {'pref': None, 'suf': None, 'sep': '_-_'}
                # }

            },
            'arena': arena(0.24, 0.24)
        },
        # 'parameterization': parconf(bend=None, point_idx=0),
        'enrichment': enr_dict(pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
                               def_kws={'ang': {'b': None}, 'sp': {'point_idx': 0}}),
    },

        {
        'id': 'Arguello lab',
        'path': 'ArguelloGroup',
        'tracker': {
            'resolution': {'fr': 10,
                           'Npoints': 5,
                           'Ncontour': 0},

            'filesystem': {
                'read_sequence': ['Date', 'head_x', 'head_y', 'spinepoint_1_x', 'spinepoint_1_y', 'spinepoint_2_x','spinepoint_2_y', 'spinepoint_3_x', 'spinepoint_3_y', 'tail_x', 'tail_y', 'centroid_x', 'centroid_y'],
                'read_metadata': False,
                # 'detect': {
                'folder': {'pref': None, 'suf': None},
                'file': {'pref': None, 'suf': None, 'sep': '_-_'}
                # }

            },
            'arena': arena(0.17, 0.17)
        },
        # 'parameterization': parconf(bend=None, point_idx=0),
        'enrichment': enr_dict(pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
                               def_kws={'ang': {'b': None}, 'sp': {'point_idx': 0}}),
        }
]
