import json
import os

from larvaworld.lib.aux import nam
from larvaworld.lib import reg, aux
from larvaworld.lib.param import Resolution, Filesystem, TrackerOps, PreprocessConf


@reg.funcs.stored_conf("LabFormat")
def LabFormat_dict():
    d={
        'Schleyer': {
            'tracker' : TrackerOps(XY_unit='mm', fr=16.0,Npoints=12,Ncontour=22,
                                       front_vector=(2, 6), rear_vector=(7, 11), point_idx=9),
    'filesystem' : Filesystem(**{
        'read_sequence': ['Step'] + nam.xy(nam.midline(12)[::-1], flat=True) + nam.xy(nam.contour(22),
                                                                                      flat=True) + nam.xy(
            'centroid') + \
                         ['blob_orientation', 'area', 'grey_value', 'raw_spinelength', 'width', 'perimeter',
                          'collision_flag'],
        'read_metadata': True,
        'folder_pref': 'box'}),
            'env_params' : reg.gen.Env(
        arena=reg.gen.Arena(dims=(0.15, 0.15), geometry='circular')),
    'preprocess' : PreprocessConf(filter_f=2.0, rescale_by=0.001, drop_collisions=True)
    },


        'Jovanic': {
            'tracker':TrackerOps(XY_unit ='mm',
        fr= 1/0.07,constant_framerate=False,
        Npoints= 11,
        Ncontour= 0, front_vector=(2, 6), rear_vector=(6,10),point_idx= 9),
            'filesystem':Filesystem(**{
        'file_suf': 'larvaid.txt',
        'file_sep': '_'}),
            'env_params':reg.gen.Env(arena=reg.gen.Arena(dims=(0.193, 0.193),geometry='rectangular')),
                                   'preprocess':PreprocessConf(filter_f=2.0,rescale_by=0.001,transposition='arena')
            },
        'Berni': {
            'tracker':TrackerOps(fr= 2.0,
        Npoints=1,front_vector=(1,1), rear_vector=(1,1),point_idx= 1),
            'filesystem':Filesystem(**{
        'read_sequence': ['Date', 'x', 'y'],
        'file_sep': '_-_'}),
            'env_params':reg.gen.Env(arena=reg.gen.Arena(dims=(0.24, 0.24),geometry='rectangular')),
                                'preprocess':PreprocessConf(filter_f=0.1,transposition='arena')
            },
        'Arguello': {'tracker':TrackerOps(fr=10.0,
        Npoints= 5,front_vector=(1,3), rear_vector=(3,5),point_idx= -1),
                     'filesystem':Filesystem(**{
        'read_sequence': ['Date', 'head_x', 'head_y', 'spinepoint_1_x', 'spinepoint_1_y', 'spinepoint_2_x',
                                     'spinepoint_2_y', 'spinepoint_2_x', 'spinepoint_2_y', 'spinepoint_3_x',
                                     'spinepoint_3_y', 'tail_x', 'tail_y', 'centroid_x', 'centroid_y'],
        # 'filesystem.file.suf': 'larvaid.txt',
        'file_sep': '_-_'}),
                     'env_params':reg.gen.Env(arena=reg.gen.Arena(dims=(0.17, 0.17),geometry='rectangular')),
                                    'preprocess':PreprocessConf(filter_f=0.1,transposition='arena')
                     }

}

    return aux.AttrDict({k:reg.gen.LabFormat(labID=k,**kws).nestedConf for k,kws in d.items()})
    # return aux.AttrDict({k:kws.nestedConf for k,kws in dkws.items()})


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
        f = f'{dr}/data/conf.txt'
        # f = reg.datapath('conf', dr)
        if os.path.isfile(f):
            try:
                with open(f) as tfp:
                    c = json.load(tfp)
                c = aux.AttrDict(c)
                entries[c.refID] = c.dir
            except:
                pass
    return aux.AttrDict(entries)





