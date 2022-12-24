from lib.aux import dictsNlists as dNl, colsNstr as cNs, naming as nam
from lib import reg


@reg.funcs.stored_conf("Tracker")
def Tracker_dict():
    dkws={'Schleyer': {
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

    },
        'Jovanic': {
        'resolution.fr': 11.27,
        'resolution.Npoints': 11,
        'resolution.Ncontour': 0,
        'filesystem.file.suf': 'larvaid.txt',
        'filesystem.file.sep': '_',
        'arena.arena_shape': 'rectangular',
        'arena.arena_dims': (0.193, 0.193),

    },
        'Berni': {
        'resolution.fr': 2.0,
        'resolution.Npoints': 1,
        'resolution.Ncontour': 0,
        'filesystem.read_sequence': ['Date', 'x', 'y'],
        'filesystem.file.sep': '_-_',
        'arena.arena_shape': 'rectangular',
        'arena.arena_dims': (0.24, 0.24),

    },
        'Arguello': {
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

    }}

    return dNl.NestDict({k:reg.get_null('tracker', kws=kws) for k,kws in dkws.items()})


@reg.funcs.stored_conf("Ref")
def Ref_dict(DATA=None):
    from lib.stor.larva_dataset import LarvaDataset
    from lib.stor.config import update_config
    if DATA is None :
        DATA = reg.Path["DATA"]
    dds = [
        [f'{DATA}/JovanicGroup/processed/AttP{g}/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = dNl.flatten_list(dds)
    # dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/200controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/exploration/dish')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/150controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/40controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/naive_locomotion/20controls')
    entries = {}
    for dr in dds:
        try:
            d = LarvaDataset(dr, load_data=False)
            d.load(step=False)
            refID=d.retrieveRefID()
            conf=update_config(d, d.config)
            entries[refID]=conf
            # print(f'Added reference dataset under ID {refID}')
        except:
            # print(f'Failed to retrieve reference dataset from path {dr}')
            pass
    return dNl.NestDict(entries)




@reg.funcs.stored_conf("Group")
def Group_dict():
    def Enr_dict():
        kws = {'metric_definition': [
            'angular.bend',
            'angular.front_vector',
            'angular.rear_vector',
            'spatial.point_idx'], 'preprocessing': ['filter_f', 'rescale_by', 'drop_collisions', 'transposition']}
        dkws = {
            'Schleyer': [['from_vectors', (2, 6), (7, 11), 9], [2.0, 0.001, True, None]],
            'Jovanic': [['from_vectors', (2, 6), (6, 10), 8], [2.0, 0.001, False, 'arena']],
            # 'Paisios': ['from_vectors',(2, 4), (7, 11),6],
            'Arguello': [['from_vectors', (1, 3), (3, 5), -1], [0.1, None, False, 'arena']],
            # 'Singlepoint': [None,None, None,0],
            'Berni': [[None, None, None, 0], [0.1, None, False, 'arena']],
            # 'Sim': [],
        }
        kw_list = dNl.flatten_list([[f'{k0}.{k}' for k in ks] for i, (k0, ks) in enumerate(kws.items())])
        enr = {}
        for i, (k, vs) in enumerate(dkws.items()):
            v_list = dNl.flatten_list(vs)
            dF = dict(zip(kw_list, v_list))
            enr[k] = reg.get_null('enrichment', kws=dF)
        return dNl.NestDict(enr)


    tracker_dic = Tracker_dict()
    enr_dic = Enr_dict()
    d = dNl.NestDict({f'{k} lab': {'path': f'{reg.Path["DATA"]}/{k}Group',
                             'tracker': tr,
                             'enrichment': enr_dic[k]} for k, tr in tracker_dic.items()})

    return d


# if __name__ == '__main__':
    # I = preg.init_dict
