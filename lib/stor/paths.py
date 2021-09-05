import os


def get_parent_dir():
    p=os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    # p = os.path.join(p, '../..')
    return p


DataFolder = f'{get_parent_dir()}/data'

# ConfFolder = f'{DataFolder}/configurations'
SimFolder = f'{DataFolder}/SimGroup'
SingleRunFolder = f'{SimFolder}/single_runs'
BatchRunFolder = f'{SimFolder}/batch_runs'

DebFolder = f'{SimFolder}/deb_runs'
Deb_paths={n : f'{get_parent_dir()}/lib/model/DEB/models/deb_{n}.csv' for n in ['rover', 'sitter', 'default']}


RefFolder = f'{DataFolder}/SampleGroup'


ConfFolder = f'{get_parent_dir()}/lib/conf/stored_confs'
SimIdx_path = f'{ConfFolder}/SimIdx.txt'
ParDb_path = f'{ConfFolder}/ParDatabase.csv'
ParShelve_path = f'{ConfFolder}/ParShelve'
ParDict_path = f'{ConfFolder}/ParDict.csv'
ParDf_path= f'{ConfFolder}/ParDf.csv'
ParPdf_path= f'{ConfFolder}/ParPdf.pdf'
UnitDict_path = f'{ConfFolder}/UnitDict.csv'
LarvaShape_path = f'{ConfFolder}/larva_shape.csv'
conf_paths = {
    'Data': f'{ConfFolder}/DataConfs.txt',
    'Group': f'{ConfFolder}/DataGroups.txt',
    'Env': f'{ConfFolder}/EnvConfs.txt',
    'Par': f'{ConfFolder}/ParConfs.txt',
    'Exp': f'{ConfFolder}/ExpConfs.txt',
    'Essay': f'{ConfFolder}/EssayConfs.txt',
    'Model': f'{ConfFolder}/ModelConfs.txt',
    'Batch': f'{ConfFolder}/BatchConfs.txt',
    'Settings': f'{ConfFolder}/SetConfs.txt',
    'Ref': f"{ConfFolder}/ReferenceDatasets.txt",
    'Life': f"{ConfFolder}/LifeConfs.txt",
}

Dtypes_path=f'{ConfFolder}/dtypes.txt'
NullDicts_path=f'{ConfFolder}/null_dicts.txt'
# Controls_path=f'{ConfFolder}/controls.txt'

MediaFolder = f'{get_parent_dir()}/lib/media'
VideoSlideFolder = f'{MediaFolder}/video_slides'
IntroSlideFolder = f'{MediaFolder}/intro_slides'
TutorialSlideFolder = f'{MediaFolder}/tutorial_slides'

ModelFigFolder = f'{MediaFolder}/model_figures'
ExpFigFolder = f'{MediaFolder}/exp_figures'
RoverSitterFigFolder = f'{ExpFigFolder}/roversVSsitters'
OdorPrefFigFolder = f'{ExpFigFolder}/odor_preference'

new_format = False
# new_format = True