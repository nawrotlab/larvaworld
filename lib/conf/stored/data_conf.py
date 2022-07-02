from lib.aux import naming as nam
from lib.registry.pars import preg
from lib.aux import dictsNlists as dNl

def metric_def(ang={}, sp={}, **kwargs):
    def ang_def(b='from_angles', fv=(1, 2), rv=(-2, -1), **kwargs):
        return preg.get_null('ang_definition', bend=b, front_vector=fv, rear_vector=rv, **kwargs)

    # def metric_def(ang={}, sp={}, dsp={}, tor={}, str={}, pau={}, tur={}) :
    return preg.get_null('metric_definition',
                         angular=ang_def(**ang),
                         spatial=preg.get_null('spatial_definition', **sp),
                         **kwargs
                         )


import_par_confs = {
    'SchleyerParConf': metric_def(ang={'b': 'from_vectors', 'fv': (2, 6), 'rv': (7, 11)}, sp={'point_idx': 9}),
    'JovanicParConf': metric_def(ang={'b': 'from_vectors', 'fv': (2, 6), 'rv': (6, 10)}, sp={'point_idx': 8}),
    'PaisiosParConf': metric_def(ang={'b': 'from_vectors', 'fv': (2, 4), 'rv': (7, 11)}, sp={'point_idx': 6}),
    'SinglepointParConf': metric_def(ang={'b': None}, sp={'point_idx': 0}),
    'SimParConf': metric_def(),
}

# importformats2 = [
#     {
#         'id': 'Schleyer lab',
#         'path': 'SchleyerGroup',
#         'tracker': {
#             'resolution': {'fr': 16.0,
#                            'Npoints': 12,
#                            'Ncontour': 22
#                            },
#
#             'filesystem': {
#                 'read_sequence': ['Step'] + nam.xy(nam.midline(12)[::-1], flat=True) + nam.xy(nam.contour(22),
#                                                                                               flat=True) + nam.xy(
#                     'centroid') + \
#                                  ['blob_orientation', 'area', 'grey_value', 'raw_spinelength', 'width', 'perimeter',
#                                   'collision_flag'],
#                 'read_metadata': True,
#                 # 'detect': {
#                 'folder': {'pref': 'box', 'suf': None},
#                 'file': {'pref': None, 'suf': None, 'sep': None}
#                 # }
#             },
#             'arena': preg.arena(0.15)
#         },
#         # 'parameterization': parconf(rear_vector=(7, 11)),
#         'enrichment': preg.base_enrich(pre_kws={'filter_f': 2.0, 'drop_collisions': True, 'rescale_by': 0.001},
#                                        metric_definition=import_par_confs['SchleyerParConf']),
#
#     },
#     {
#         'id': 'Jovanic lab',
#         'path': 'JovanicGroup',
#         'tracker': {
#             'resolution': {'fr': 11.27,
#                            'Npoints': 11,
#                            'Ncontour': 0},
#
#             'filesystem': {
#                 'read_sequence': None,
#                 'read_metadata': False,
#                 'folder': {'pref': None, 'suf': None},
#                 'file': {'pref': None, 'suf': 'larvaid.txt', 'sep': '_'}
#             },
#             'arena': preg.arena(0.193, 0.193)
#         },
#         # 'parameterization': parconf(front_vector=(2, 3), rear_vector=(7, 8)),
#         'enrichment': preg.base_enrich(pre_kws={'filter_f': 2.0, 'rescale_by': 0.001, 'transposition': 'arena'},
#                                        metric_definition=import_par_confs['JovanicParConf']),
#     },
#     {
#         'id': 'Berni lab',
#         'path': 'BerniGroup',
#         'tracker': {
#             'resolution': {'fr': 2,
#                            'Npoints': 1,
#                            'Ncontour': 0},
#
#             'filesystem': {
#                 'read_sequence': ['Date', 'x', 'y'],
#                 'read_metadata': False,
#                 # 'detect': {
#                 'folder': {'pref': None, 'suf': None},
#                 'file': {'pref': None, 'suf': None, 'sep': '_-_'}
#                 # }
#
#             },
#             'arena': preg.arena(0.24, 0.24)
#         },
#         # 'parameterization': parconf(bend=None, point_idx=0),
#         'enrichment': preg.enr_dict(pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
#                                     metric_definition=import_par_confs['SinglepointParConf']),
#     },
#
#     {
#         'id': 'Arguello lab',
#         'path': 'ArguelloGroup',
#         'tracker': {
#             'resolution': {'fr': 10,
#                            'Npoints': 5,
#                            'Ncontour': 0},
#
#             'filesystem': {
#                 'read_sequence': ['Date', 'head_x', 'head_y', 'spinepoint_1_x', 'spinepoint_1_y', 'spinepoint_2_x',
#                                   'spinepoint_2_y', 'spinepoint_2_x', 'spinepoint_2_y', 'spinepoint_3_x',
#                                   'spinepoint_3_y', 'tail_x', 'tail_y', 'centroid_x', 'centroid_y'],
#                 'read_metadata': False,
#                 # 'detect': {
#                 'folder': {'pref': None, 'suf': None},
#                 'file': {'pref': None, 'suf': None, 'sep': '_-_'}
#                 # }
#
#             },
#             'arena': preg.arena(0.17, 0.17)
#         },
#         # 'parameterization': parconf(bend=None, point_idx=0),
#         'enrichment': preg.enr_dict(pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
#                                     metric_definition=import_par_confs['SinglepointParConf']),
#     }
# ]

Scl_kws = {
    'resolution.fr': 16.0,
    'resolution.Npoints': 12,
    'resolution.Ncontour': 22,
    'filesystem.read_sequence': ['Step'] + nam.xy(nam.midline(12)[::-1], flat=True) + nam.xy(nam.contour(22),
                                                                                             flat=True) + nam.xy(
        'centroid') + \
                                ['blob_orientation', 'area', 'grey_value', 'raw_spinelength', 'width', 'perimeter',
                                 'collision_flag'],
    'filesystem.read_metadata': True,
    'filesystem.folder.pref': 'box',
    'arena.arena_shape': 'circular',
    'arena.arena_dims': (0.15, 0.15),

}
Jov_kws = {
    'resolution.fr': 11.27,
    'resolution.Npoints': 11,
    'resolution.Ncontour': 0,
    'filesystem.file.suf': 'larvaid.txt',
    'filesystem.file.sep': '_',
    'arena.arena_shape': 'rectangular',
    'arena.arena_dims': (0.193, 0.193),

}
Ber_kws = {
    'resolution.fr': 2.0,
    'resolution.Npoints': 1,
    'resolution.Ncontour': 0,
    'filesystem.read_sequence': ['Date', 'x', 'y'],
    'filesystem.file.sep': '_-_',
    'arena.arena_shape': 'rectangular',
    'arena.arena_dims': (0.24, 0.24),

}
Arg_kws = {
    'resolution.fr': 10.0,
    'resolution.Npoints': 5,
    'resolution.Ncontour': 0,
    'filesystem.read_sequence': ['Date', 'head_x', 'head_y', 'spinepoint_1_x', 'spinepoint_1_y', 'spinepoint_2_x',
                                 'spinepoint_2_y', 'spinepoint_2_x', 'spinepoint_2_y', 'spinepoint_3_x',
                                 'spinepoint_3_y', 'tail_x', 'tail_y', 'centroid_x', 'centroid_y'],
    # 'filesystem.file.suf': 'larvaid.txt',
    'filesystem.file.sep': '_-_',
    'arena.arena_shape': 'rectangular',
    'arena.arena_dims': (0.17, 0.17),

}


T0 = preg.get_null('tracker')
Jov_T = dNl.update_nestdict(T0, Jov_kws)
Scl_T = dNl.update_nestdict(T0, Scl_kws)
Ber_T = dNl.update_nestdict(T0, Ber_kws)
Arg_T = dNl.update_nestdict(T0, Arg_kws)

importformats = {
    'Schleyer lab': {
        # 'id': 'Schleyer lab',
        'path': 'SchleyerGroup',
        'tracker': Scl_T,
        # 'parameterization': parconf(rear_vector=(7, 11)),
        'enrichment': preg.base_enrich(pre_kws={'filter_f': 2.0, 'drop_collisions': True, 'rescale_by': 0.001},
                                       metric_definition=import_par_confs['SchleyerParConf']),

    },
    'Jovanic lab': {
        # 'id': 'Jovanic lab',
        'path': 'JovanicGroup',
        'tracker': Jov_T,
        # 'parameterization': parconf(front_vector=(2, 3), rear_vector=(7, 8)),
        'enrichment': preg.base_enrich(
            pre_kws={'filter_f': 2.0, 'rescale_by': 0.001, 'transposition': 'arena'},
            metric_definition=import_par_confs['JovanicParConf']),
    },
    'Berni lab': {
        # 'id': 'Berni lab',
        'path': 'BerniGroup',
        'tracker': Ber_T,
        # 'parameterization': parconf(bend=None, point_idx=0),
        'enrichment': preg.enr_dict(
            pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
            metric_definition=import_par_confs['SinglepointParConf']),
    },

    'Arguello lab': {
        'id': 'Arguello lab',
        'path': 'ArguelloGroup',
        'tracker': Arg_T,
        # 'parameterization': parconf(bend=None, point_idx=0),
        'enrichment': preg.enr_dict(
            pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
            metric_definition=import_par_confs['SinglepointParConf']),
    }
}

if __name__ == '__main__':
    T0 = preg.get_null('metric_definition')
    print(T0)
    pass
