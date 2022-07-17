from lib.registry.pars import preg
from lib.aux import dictsNlists as dNl, colsNstr as cNs, naming as nam




def Tracker_dict():
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

    T0 = preg.init_dict.get_null('tracker')

    d = {
        'Schleyer': dNl.update_nestdict(T0, Scl_kws),
        'Jovanic': dNl.update_nestdict(T0, Jov_kws),
        'Berni': dNl.update_nestdict(T0, Ber_kws),
        'Arguello': dNl.update_nestdict(T0, Arg_kws)}
    return dNl.NestDict(d)


# tracker_formats = build_tracker_formats()


def Ref_dict():
    from lib.stor.larva_dataset import LarvaDataset
    DATA = preg.path_dict["DATA"]
    dds = [
        [f'{DATA}/JovanicGroup/processed/3_conditions/AttP{g}@UAS_TNT/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = dNl.flatten_list(dds)
    dds.append(f'{DATA}/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/exploration')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/200_controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/10_controls')
    entries = {}
    for dr in dds:
        try:
            d = LarvaDataset(dr, load_data=False)
            entry = d.save_config(add_reference=True, return_entry=True)
            entries.update(entry)
        except:
            pass
    return dNl.NestDict(entries)


def Group_dict():
    import_par_confs = {
        'SchleyerParConf': preg.init_dict.metric_def(ang={'b': 'from_vectors', 'fv': (2, 6), 'rv': (7, 11)},
                                                     sp={'point_idx': 9}),
        'JovanicParConf': preg.init_dict.metric_def(ang={'b': 'from_vectors', 'fv': (2, 6), 'rv': (6, 10)},
                                                    sp={'point_idx': 8}),
        'PaisiosParConf': preg.init_dict.metric_def(ang={'b': 'from_vectors', 'fv': (2, 4), 'rv': (7, 11)},
                                                    sp={'point_idx': 6}),
        'SinglepointParConf': preg.init_dict.metric_def(ang={'b': None}, sp={'point_idx': 0}),
        'SimParConf': preg.init_dict.metric_def(),
    }

    tracker_formats = Tracker_dict()

    d = {
        'Schleyer lab': {
            # 'id': 'Schleyer lab',
            'path': 'SchleyerGroup',
            'tracker': tracker_formats['Schleyer'],
            # 'parameterization': parconf(rear_vector=(7, 11)),
            'enrichment': preg.base_enrich(pre_kws={'filter_f': 2.0, 'drop_collisions': True, 'rescale_by': 0.001},
                                           metric_definition=import_par_confs['SchleyerParConf']),

        },
        'Jovanic lab': {
            # 'id': 'Jovanic lab',
            'path': 'JovanicGroup',
            'tracker': tracker_formats['Jovanic'],
            # 'parameterization': parconf(front_vector=(2, 3), rear_vector=(7, 8)),
            'enrichment': preg.base_enrich(
                pre_kws={'filter_f': 2.0, 'rescale_by': 0.001, 'transposition': 'arena'},
                metric_definition=import_par_confs['JovanicParConf']),
        },
        'Berni lab': {
            # 'id': 'Berni lab',
            'path': 'BerniGroup',
            'tracker': tracker_formats['Berni'],
            # 'parameterization': parconf(bend=None, point_idx=0),
            'enrichment': preg.enr_dict(
                pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
                metric_definition=import_par_confs['SinglepointParConf']),
        },

        'Arguello lab': {
            'id': 'Arguello lab',
            'path': 'ArguelloGroup',
            'tracker': tracker_formats['Arguello'],
            # 'parameterization': parconf(bend=None, point_idx=0),
            'enrichment': preg.enr_dict(
                pre_kws={'filter_f': 0.1, 'rescale_by': 0.001, 'transposition': 'arena'},
                metric_definition=import_par_confs['SinglepointParConf']),
        }
    }
    return d


if __name__ == '__main__':
    # T0 = preg.get_null('metric_definition')
    # print(T0)
    pass
