import os


def get_parent_dir():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, '../..')
    return p


DataFolder = f'{get_parent_dir()}/data'
ConfFolder = f'{DataFolder}/configurations'
SimFolder = f'{DataFolder}/SimGroup'
SingleRunFolder = f'{SimFolder}/single_runs'
BatchRunFolder = f'{SimFolder}/batch_runs'

DebFolder = f'{SimFolder}/deb_runs'
Deb_path = f'{get_parent_dir()}/lib/sim/deb_drosophila.csv'
LarvaShape_path = f'{get_parent_dir()}/lib/conf/larva_shape.csv'

DataGroups_path = f'{ConfFolder}/DataGroups.txt'
DataConfs_path = f'{ConfFolder}/DataConfs.txt'
ParConfs_path = f'{ConfFolder}/ParConfs.txt'
SimIdx_path = f'{ConfFolder}/SimIdx.txt'
EnvConfs_path = f'{ConfFolder}/EnvConfs.txt'
ExpConfs_path = f'{ConfFolder}/ExpConfs.txt'
ModelConfs_path = f'{ConfFolder}/ModelConfs.txt'
ParDb_path = f'{ConfFolder}/ParDatabase.csv'

RefFolder = f'{DataFolder}/reference'
Ref_path = f'{RefFolder}/data/reference.csv'
Ref_fits = f'{RefFolder}/data/bout_fits.csv'
