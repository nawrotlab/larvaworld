import os


def get_parent_dir():
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, '../..')
    return p


DataFolder = f'{get_parent_dir()}/data'

# ConfFolder = f'{DataFolder}/configurations'
SimFolder = f'{DataFolder}/SimGroup'
SingleRunFolder = f'{SimFolder}/single_runs'
BatchRunFolder = f'{SimFolder}/batch_runs'

DebFolder = f'{SimFolder}/deb_runs'
Deb_path = f'{get_parent_dir()}/lib/sim/deb_drosophila.csv'


RefFolder = f'{DataFolder}/reference'
Ref_path = f'{RefFolder}/data/reference.csv'
Ref_fits = f'{RefFolder}/data/bout_fits.csv'


ConfFolder = f'{get_parent_dir()}/lib/conf/stored_confs'
SimIdx_path = f'{ConfFolder}/SimIdx.txt'
ParDb_path = f'{ConfFolder}/ParDatabase.csv'
ParShelve_path = f'{ConfFolder}/ParShelve'
LarvaShape_path = f'{ConfFolder}/larva_shape.csv'
conf_paths = {
    'Data': f'{ConfFolder}/DataConfs.txt',
    'Group': f'{ConfFolder}/DataGroups.txt',
    'Env': f'{ConfFolder}/EnvConfs.txt',
    'Par': f'{ConfFolder}/ParConfs.txt',
    'Exp': f'{ConfFolder}/ExpConfs.txt',
    'Model': f'{ConfFolder}/ModelConfs.txt',
    'Batch': f'{ConfFolder}/BatchConfs.txt',
}
