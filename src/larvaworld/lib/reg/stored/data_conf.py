import json
import os

from larvaworld.lib.aux import naming as nam
from larvaworld.lib import reg, aux

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
        'arena.shape': 'circular',
        'arena.dims': (0.15, 0.15),

    },
        'Jovanic': {
        'resolution.fr': 11.27,
        'resolution.Npoints': 11,
        'resolution.Ncontour': 0,
        'filesystem.file.suf': 'larvaid.txt',
        'filesystem.file.sep': '_',
        'arena.shape': 'rectangular',
        'arena.dims': (0.193, 0.193),

    },
        'Berni': {
        'resolution.fr': 2.0,
        'resolution.Npoints': 1,
        'resolution.Ncontour': 0,
        'filesystem.read_sequence': ['Date', 'x', 'y'],
        'filesystem.file.sep': '_-_',
        'arena.shape': 'rectangular',
        'arena.dims': (0.24, 0.24),

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
        'arena.shape': 'rectangular',
        'arena.dims': (0.17, 0.17),

    }}

    return aux.AttrDict({k:reg.par.null('Tracker', **kws) for k,kws in dkws.items()})


@reg.funcs.stored_conf("Ref")
def Ref_dict(DATA=None):

    if DATA is None :
        DATA = reg.DATA_DIR
    dds = [
        [f'{DATA}/JovanicGroup/processed/AttP{g}/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = aux.flatten_list(dds)
    dds.append(f'{DATA}/SchleyerGroup/processed/exploration/dish')
    dds.append(f'{DATA}/SchleyerGroup/processed/exploration/40controls')
    dds.append(f'{DATA}/SchleyerGroup/processed/no_odor/150controls')
    entries = {}
    for dr in dds:
        f = reg.datapath('conf', dr)
        if os.path.isfile(f):
            try:
                with open(f) as tfp:
                    c = json.load(tfp)
                c = aux.AttrDict(c)
                entries[c.refID] = c
                # print(f'Added reference dataset under ID {c.refID}')
        # try:
        #     d = LarvaDataset(dr, load_data=False)
        #     d.load(step=False)
        #     refID=d.retrieveRefID()
        #     conf=update_config(d, d.config)
        #     entries[refID]=conf
            #
            except:
            # print(f'Failed to retrieve reference dataset from path {dr}')
                pass
    return aux.AttrDict(entries)




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
        kw_list = aux.flatten_list([[f'{k0}.{k}' for k in ks] for i, (k0, ks) in enumerate(kws.items())])
        enr = {}
        for i, (k, vs) in enumerate(dkws.items()):
            v_list = aux.flatten_list(vs)
            dF = dict(zip(kw_list, v_list))
            enr[k] = reg.par.null('enrichment', **dF)
        return aux.AttrDict(enr)


    tracker_dic = Tracker_dict()
    enr_dic = Enr_dict()
    d = aux.AttrDict({f'{k} lab': {'path': f'{reg.DATA_DIR}/{k}Group',
                             'Tracker': tr,
                             'enrichment': enr_dic[k]} for k, tr in tracker_dic.items()})

    return d


# if __name__ == '__main__':
    # I = preg.init_dict
