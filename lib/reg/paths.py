

CONFTYPES = ['Ref', 'Model', 'ModelGroup', 'Env', 'Exp', 'ExpGroup', 'Essay', 'Batch', 'Ga', 'Tracker',
                          'Group', 'Trial', 'Life', 'Body', 'Tree', 'Source']


def get_parent_dir():
    import os
    p=os.path.abspath(__file__)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    p = os.path.dirname(p)
    return p

ROOT_DIR=get_parent_dir()

# raise
from lib.aux import dictsNlists as dNl, naming as nam
def build_path_dict(F0=ROOT_DIR) :

    # Fixed behavior
    RF = f'{F0}/lib/sim/exec'
    GF = f'{F0}/lib/gui'
    CF = f'{F0}/lib/conf/conf_dicts'
    MF = f'{F0}/lib/media'

    exp_paths = {
        'RvsS': f'{MF}/exp_figures/roversVSsitters',
        'odor_pref': f'{MF}/exp_figures/odor_preference',
    }

    media_paths = {
        'videos': f'{MF}/video_slides',
        'intro': f'{MF}/intro_slides',
        'tutorials': f'{MF}/tutorial_slides',
        'model': f'{MF}/model_figures',
        'model_tables': f'{MF}/model_tables',
        'model_summaries': f'{MF}/model_summaries',
        'exp_figs': f'{MF}/exp_figures',
        'icons': f'{MF}/icons',
        'ga_scene': f'{MF}/ga_scenes',
    }

    par_paths = {
        'ParDb': f'{CF}/ParDatabase.csv',
        'ParShelve': f'{CF}/ParShelve',
        'ParDict': f'{CF}/ParDict.csv',
        'ParamComputeFunctionRegistry': f'{CF}/ParamComputeFunctionRegistry.csv',
        'ProcFuncDict': f'{CF}/ProcFuncDict.csv',
        'ParInitDict': f'{CF}/ParInitDict.csv',
        'ParDefaultDict': f'{CF}/ParDefaultDict.csv',
        'ParserDict': f'{CF}/ParserDict.csv',
        'BaseConfDict': f'{CF}/BaseConfDict.csv',
        'DistDict': f'{CF}/DistDict.csv',
        'LarvaConfDict': f'{CF}/LarvaConfDict.csv',
        'ConfTypeDict': f'{CF}/ConfTypeDict.csv',
        'ParDf': f'{CF}/ParDf.csv',
        'ParPdf': f'{CF}/ParPdf.pdf',
        'Unit': f'{CF}/UnitDict.csv',
        'ParRef': f'{CF}/RefPars.txt',
        'ParGlossary': f'{CF}/ParGlossary.csv',
        'ParGlossaryTxT': f'{CF}/ParGlossaryTxT.txt',
        'Par': f'{CF}/ParConfs.txt',
        'controls': f'{CF}/controls.txt',
        'SimIdx': f'{CF}/SimIdx.txt',
    }

    conf_paths={k : f'{F0}/lib/conf/confDicts/{k}.txt' for k in CONFTYPES}
    # conf_paths = {
    #     'Group': f'{CF}/DataGroups.txt',
    #     'Tracker': f'{CF}/TrackerFormats.txt',
    #     'Env': f'{CF}/EnvConfs.txt',
    #
    #     'Exp': f'{CF}/ExpConfs.txt',
    #     'Ga': f'{CF}/GaConfs.txt',
    #     'ExpGroup': f'{CF}/ExpGroupConfs.txt',
    #     'Essay': f'{CF}/EssayConfs.txt',
    #     'Source': f'{CF}/SourceConfs.txt',
    #     'Model': f'{CF}/ModelConfs.txt',
    #     'ModelGroup': f'{CF}/ModelGroupConfs.txt',
    #     'Batch': f'{CF}/BatchConfs.txt',
    #     # 'Settings': f'{CF}/SetConfs.txt',
    #     'controls': f'{CF}/controls.txt',
    #     'Ref': f"{CF}/ReferenceDatasets.txt",
    #     'Trial': f"{CF}/TrialConfs.txt",
    #     'Life': f"{CF}/LifeConfs.txt",
    #     'Body': f"{CF}/BodyConfs.txt",
    #     'Brain': f"{CF}/BrainConfs.txt",
    #     'SimIdx': f'{CF}/SimIdx.txt',
    #     'Tree': f'{CF}/ParTree.txt',
    # }



    data_paths = {
        'DEB': f'{F0}/data/SimGroup/deb_runs',
        'DEB_MODS': {n: f'{F0}/lib/model/DEB/models/deb_{n}.csv' for n in ['rover', 'sitter', 'default']},
        'REF': f'{F0}/data/SampleGroup',
        'EXEC': f'{RF}/exec_run.py',
        'EXECONF': f'{RF}/exec_conf.txt',
        'BATCH': f'{F0}/data/SimGroup/batch_runs',
        'ESSAY': f'{F0}/data/SimGroup/essays',
        'RUN': f'{F0}/data/SimGroup/single_runs',
        'SIM': f'{F0}/data/SimGroup',

        'DATA': f'{F0}/data',
        'GUITEST': f'{GF}/gui_speed_test.csv',
    }



    dic = {**par_paths, **conf_paths, **exp_paths, **media_paths, **data_paths}
    dic['parent']=F0
    return dNl.NestDict(dic)

Path = build_path_dict()

def buildSampleDic():
    d =dNl.NestDict(
        {
            'length': 'body.initial_length',
            nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
            'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
            nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_mean',
            nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.stride_dst_std',
            nam.freq('feed'): 'brain.feeder_params.initial_freq',
            nam.max(nam.chunk_track('stride', nam.scal(nam.vel('')))): 'brain.crawler_params.max_scaled_vel',
            'phi_scaled_velocity_max': 'brain.crawler_params.max_vel_phase',
            'attenuation': 'brain.interference_params.attenuation',
            'attenuation_max': 'brain.interference_params.attenuation_max',
            nam.freq(nam.vel(nam.orient(('front')))): 'brain.turner_params.initial_freq',
            nam.max('phi_attenuation'): 'brain.interference_params.max_attenuation_phase',
        }
    )
    return dNl.bidict(d)
    # save_dict(d, preg.path_dict["ParRef"], use_pickle=False)
SampleDic = buildSampleDic()




def build_datapath_structure():
    kd = dNl.NestDict()
    kd.solo_dicts = ['bouts', 'foraging', 'deb', 'nengo']

    kd.folders = {
        'parent': ['data', 'plots', 'visuals', 'aux', 'model'],
        'data': ['individuals'],
        'individuals': kd.solo_dicts,
        'plots': ['model_tables', 'model_summaries'],
        'model': ['GAoptimization', 'evaluation'],

    }

    h5base = ['end', 'step']
    kd.h5step = ['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']
    h5aux = ['derived', 'traj', 'aux', 'vel_definition', 'tables', 'food', 'distro']

    kd.h5 = h5base + kd.h5step + h5aux

    confs = ['conf', 'sim_conf', 'log']
    dics1 = ['chunk_dicts', 'grouped_epochs', 'pooled_epochs', 'cycle_curves', 'dsp', 'fit']
    dics2 = ['ExpFitter']

    kd.dic = dics1 + dics2 + confs

    datapath_dict = build_datapath_dict(kd)
    datafunc_dict = build_datafunc_dict(kd)
    return datapath_dict, datafunc_dict


def build_datapath_dict(kd):
    d = dNl.NestDict()
    d.parent = ''
    for k0, ks in kd.folders.items():
        for k in ks:
            d[k] = f'{d[k0]}/{k}'

    for k in kd.h5:
        d[k] = f'{d.data}/{k}.h5'
    for k in kd.dic:
        d[k] = f'{d.data}/{k}.txt'
    return d


def build_datafunc_dict(kd):
    from lib.aux.stor_aux import read, storeH5

    func_dic0 = {'h5':
                     {'load': read, 'save': storeH5},
                 'dic': {'load': dNl.load_dict, 'save': dNl.save_dict},

                 'solo_dicts': {'load': dNl.loadSoloDics, 'save': dNl.storeSoloDics}
                 }
    dic = {}
    for k, funcs in func_dic0.items():
        ddic = {kk: funcs for kk in kd[k]}
        dic.update(ddic)

    return dNl.NestDict(dic)


datapath_dict, datafunc_dict = build_datapath_structure()


def datapath(filepath_key, dir=None):
    DD = datapath_dict
    if dir is not None and filepath_key in DD.keys():
        return f'{dir}{DD[filepath_key]}'
    else:
        return None


def datafunc(filepath_key, mode='load'):
    DD = datafunc_dict
    if filepath_key in DD.keys():
        return DD[filepath_key][mode]
    else:
        return None


def AllConfDict(F0=ROOT_DIR):
    dd = dNl.NestDict()


    for k in CONFTYPES:
        path=f'{F0}/lib/conf/confDicts/{k}.txt'
        try:
            d = dNl.load_dict(path, use_pickle=False)
        except:
            try:
                d = dNl.load_dict(path, use_pickle=True)
            except:
                # print(f'{k} configuration directory not found')
                d= {}
        dd[k] = d
    return dd

conftree = AllConfDict()


def ConfSubkeyDict():
    d1 = dNl.NestDict({
        'Batch': {'exp': 'Exp'},
        'Ga': {'env_params': 'Env'},
        'Exp': {'env_params': 'Env',
                'trials': 'Trial',
                'larva_groups': 'Model',
                }
    })
    return d1

def ExpandedConfDict():
    c0=AllConfDict()
    sk=ConfSubkeyDict()
    for confType0 in c0.keys():
        if confType0 in sk.keys():
            pairs = sk[confType0]
            for id, conf in c0[confType0].items():
                for subID, confType in pairs.items():


                    if subID in conf.keys():
                        if isinstance(conf[subID], str) and conf[subID] in c0[confType].keys():
                            conf[subID]=c0[confType][conf[subID]]
                        elif (subID, confType) == ('larva_groups', 'Model'):
                            for gID, gConf in conf[subID].items():
                                mID=gConf.model
                                if mID in c0['Model'].keys():
                                    gConf.model=c0['Model'][mID]
                                else:
                                    # print(f'{mID} not found')
                                    pass
                                    # raise
    return c0

