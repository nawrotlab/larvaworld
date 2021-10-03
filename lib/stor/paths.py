import os

from lib.aux.dictsNlists import load_dict

def get_parent_dir():
    p=os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    # p = os.path.join(p, '../..')
    return p

F0=get_parent_dir()


DataFolder = f'{F0}/data'
RunFolder = f'{F0}/run'
GuiFolder = f'{F0}/lib/gui'
GuiTest = f'{GuiFolder}/gui_speed_test.csv'

SimFolder = f'{DataFolder}/SimGroup'
SingleRunFolder = f'{SimFolder}/single_runs'
BatchRunFolder = f'{SimFolder}/batch_runs'
EssayFolder = f'{SimFolder}/essays'
ExecConfFile = f'{RunFolder}/exec_conf.txt'
ExecFile = f'{RunFolder}/exec_run.py'

DebFolder = f'{SimFolder}/deb_runs'
Deb_paths={n : f'{F0}/lib/model/DEB/models/deb_{n}.csv' for n in ['rover', 'sitter', 'default']}


RefFolder = f'{DataFolder}/SampleGroup'

ConfFolder = f'{F0}/lib/conf/stored_confs'
RefParsFile =f'{ConfFolder}/RefPars.txt'


SimIdx_path = f'{ConfFolder}/SimIdx.txt'
ParDb_path = f'{ConfFolder}/ParDatabase.csv'
ParShelve_path = f'{ConfFolder}/ParShelve'
ParDict_path = f'{ConfFolder}/ParDict.csv'
ParDf_path= f'{ConfFolder}/ParDf.csv'
ParPdf_path= f'{ConfFolder}/ParPdf.pdf'
UnitDict_path = f'{ConfFolder}/UnitDict.csv'
conf_paths = {
    'Data': f'{ConfFolder}/DataConfs.txt',
    'Group': f'{ConfFolder}/DataGroups.txt',
    'Env': f'{ConfFolder}/EnvConfs.txt',
    'Par': f'{ConfFolder}/ParConfs.txt',
    'Exp': f'{ConfFolder}/ExpConfs.txt',
    'ExpGroup': f'{ConfFolder}/ExpGroupConfs.txt',
    'Essay': f'{ConfFolder}/EssayConfs.txt',
    'Model': f'{ConfFolder}/ModelConfs.txt',
    'Batch': f'{ConfFolder}/BatchConfs.txt',
    'Settings': f'{ConfFolder}/SetConfs.txt',
    'Ref': f"{ConfFolder}/ReferenceDatasets.txt",
    'Life': f"{ConfFolder}/LifeConfs.txt",
}

# Dtypes_path=f'{ConfFolder}/DataTypes.txt'

MediaFolder = f'{F0}/lib/media'
VideoSlideFolder = f'{MediaFolder}/video_slides'
IntroSlideFolder = f'{MediaFolder}/intro_slides'
TutorialSlideFolder = f'{MediaFolder}/tutorial_slides'

ModelFigFolder = f'{MediaFolder}/model_figures'
ExpFigFolder = f'{MediaFolder}/exp_figures'
RoverSitterFigFolder = f'{ExpFigFolder}/roversVSsitters'
OdorPrefFigFolder = f'{ExpFigFolder}/odor_preference'

new_format = False
# new_format = True

