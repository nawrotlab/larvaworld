import copy
import json
import shutil

from lib.conf.init_dtypes import enrichment_dict
from lib.stor import paths



def loadConf(id, conf_type):
    try:
        conf_dict = loadConfDict(conf_type)
        conf = conf_dict[id]
        return conf
    except:
        raise ValueError(f'{conf_type} Configuration {id} does not exist')

def expandConf(id, conf_type):
    conf = loadConf(id, conf_type)
    # print(conf.keys(), id)
    try:
        if conf_type=='Batch' :
            conf['exp'] = expandConf(conf['exp'], 'Exp')
        elif conf_type=='Exp' :
            conf['env_params']=expandConf(conf['env_params'], 'Env')
            conf['life_params'] = loadConf(conf['life_params'], 'Life')
        elif conf_type=='Env' :
            for k, v in conf['larva_groups'].items():
                if type(v['model']) == str:
                    v['model'] = loadConf(v['model'], 'Model')
    except :
        pass
    return conf


def loadConfDict(conf_type):
    # from lib.stor.paths import conf_paths
    try :
        with open(paths.path(conf_type)) as tfp:
            Conf_dict = json.load(tfp)
        return Conf_dict
    except :
        return {}


def saveConf(conf, conf_type, id=None, mode='overwrite'):
    try:
        conf_dict = loadConfDict(conf_type)
    except:
        conf_dict = {}
    if id is None:
        id = conf['id']

    if id in list(conf_dict.keys()):
        for k, v in conf.items():
            if type(k) == dict and k in list(conf_dict[id].keys()) and mode == 'update':
                conf_dict[id][k].update(conf[k])
            else:
                conf_dict[id][k] = v
    else:
        conf_dict[id] = conf
    saveConfDict(conf_dict, conf_type)
    print(f'{conf_type} Configuration saved under the id : {id}')


def saveConfDict(ConfDict, conf_type):
    with open(paths.path(conf_type), "w") as fp:
        json.dump(ConfDict, fp)


def deleteConf(id, conf_type):
    if conf_type == 'Data':
        DataGroup = loadConf(id, conf_type)
        path = DataGroup['path']
        try:
            shutil.rmtree(path)
        except:
            pass
    conf_dict = loadConfDict(conf_type)
    try:
        conf_dict.pop(id, None)
        saveConfDict(conf_dict, conf_type)
        print(f'Deleted {conf_type} configuration under the id : {id}')
    except:
        pass





def next_idx(exp, type='single'):
    f0=paths.path('SimIdx')
    try:
        with open(f0) as tfp:
            idx_dict = json.load(tfp)
    except:
        exp_names = list(loadConfDict('Exp').keys())
        batch_names = list(loadConfDict('Batch').keys())
        essay_names = list(loadConfDict('Essay').keys())
        exp_idx_dict = dict(zip(exp_names, [0] * len(exp_names)))
        batch_idx_dict = dict(zip(batch_names, [0] * len(batch_names)))
        essay_idx_dict = dict(zip(essay_names, [0] * len(essay_names)))
        # batch_idx_dict.update(loadConfDict('Batch'))
        idx_dict = {'single': exp_idx_dict,
                    'batch': batch_idx_dict,
                    'essay' : essay_idx_dict}
    if not exp in idx_dict[type].keys():
        idx_dict[type][exp] = 0
    idx_dict[type][exp] += 1
    with open(f0, "w") as fp:
        json.dump(idx_dict, fp)
    return idx_dict[type][exp]


def store_reference_data_confs() :
    from lib.stor.larva_dataset import LarvaDataset
    from lib.aux.dictsNlists import flatten_list

    DATA=paths.path('DATA')

    dds = [
        [f'{DATA}/JovanicGroup/processed/3_conditions/AttP{g}@UAS_TNT/{c}' for g
         in ['2', '240']] for c in ['Fed', 'Deprived', 'Starved']]
    dds = flatten_list(dds)
    dds.append(f'{DATA}/SchleyerGroup/processed/FRUvsQUI/Naive->PUR/EM/exploration')
    for dr in dds:
        d = LarvaDataset(dr, load_data=False)
        # # c = d.config
        # del d.config['agent_ids']
        # d.config['bout_distros']['stride']=d.config['bout_distros']['stride']['best']
        # d.config['bout_distros']['pause']=d.config['bout_distros']['pause']['best']
        d.save_config(add_reference=True)

def store_confs(keys=None) :
    if keys is None :
        keys=['Ref','Data', 'Model', 'Env', 'Exp']

    if 'Ref' in keys :
        store_reference_data_confs()
    if 'Data' in keys :
        import lib.conf.data_conf as dat
        dat_list = [
            dat.SchleyerConf,
            dat.JovanicConf,
            dat.SimConf,
        ]
        for d in dat_list:
            saveConf(d, 'Data')

        par_conf_dict = {
            'SchleyerParConf': dat.SchleyerParConf,
            'JovanicParConf': dat.JovanicParConf,
            'PaisiosParConf': dat.PaisiosParConf,
            'SinglepointParConf': dat.SinglepointParConf,
            'SimParConf': dat.SimParConf,
        }
        for k, v in par_conf_dict.items():
            saveConf(v, 'Par', k)
        group_list = [
            dat.SchleyerFormat,
            dat.JovanicFormat,
            dat.BerniFormat,
        ]
        for g in group_list:
            saveConf(g, 'Group')
    if 'Model' in keys:
        from lib.conf.larva_conf import mod_dict
        for k, v in mod_dict.items():
            saveConf(v, 'Model', k)
    if 'Env' in keys :
        from lib.conf.env_conf import env_dict
        for k, v in env_dict.items():
            saveConf(v, 'Env', k)
    if 'Exp' in keys :
        import lib.conf.exp_conf as exp
        import lib.conf.essay_conf as essay
        import lib.conf.batch_conf as bat
        from lib.aux.dictsNlists import merge_dicts

        d = exp.grouped_exp_dict
        exp_dict = merge_dicts(list(d.values()))
        exp_group_dict = {k: {'simulations': list(v.keys())} for k, v in d.items()}
        for k, v in exp_dict.items():
            saveConf(v, 'Exp', k)
        for k, v in exp_group_dict.items():
            saveConf(v, 'ExpGroup', k)

        for k, v in essay.essay_dict.items():
            saveConf(v, 'Essay', k)

        for k, v in bat.batch_dict.items():
            saveConf(v, 'Batch', k)






























# if __name__ == '__main__':
#     init_confs()

def imitation_exp(config, model='explorer', idx=0, **kwargs):
    from lib.conf.init_dtypes import null_dict
    if type(config)==str :
        config=loadConf(config, 'Ref')
    # f = ExpFitter(config)

    id = config['id']
    base_larva = expandConf(model, 'Model')

    sim_params = {
        'timestep': 1/config['fr'],
        'duration': config['duration'] / 60,
        'path': 'single_runs/imitation',
        'sim_ID': f'{id}_imitation_{idx}',
        # 'sample': id,
        'Box2D': False
    }
    env_params =null_dict('env_conf', arena=config['env_params']['arena'], larva_groups={'ImitationGroup': null_dict('LarvaGroup', sample= config, model= base_larva, default_color = 'blue', imitation=True, distribution=None)})

    exp_conf=null_dict('exp_conf', sim_params=sim_params, env_params=env_params, life_params=null_dict('life'),
                       enrichment=enrichment_dict(types=['angular', 'spatial','dispersion', 'tortuosity'],
                                                                    bouts=['stride', 'pause', 'turn']))
    # print(config)
    # exp_conf = expandConf(exp, 'Exp')
    # exp_conf['env_params']['larva_groups'] = {'ImitationGroup': null_dict('LarvaGroup', sample= config, model= base_larva, default_color = 'blue', imitation=True, distribution=None)}
    # exp_conf['env_params']['arena'] = config['env_params']['arena']
    # exp_conf['sim_params'] = sim_params
    exp_conf['experiment'] = 'imitation'
    exp_conf.update(**kwargs)
    # print(exp_conf.keys())
    return exp_conf


def get_exp_conf(exp_type, sim_params, life_params=None, N=None, larva_model=None):
    conf = copy.deepcopy(expandConf(exp_type, 'Exp'))
    # print(conf['sample'])
    for k in list(conf['env_params']['larva_groups'].keys()):
        if N is not None:
            conf['env_params']['larva_groups'][k]['N'] = N
        if larva_model is not None:
            conf['env_params']['larva_groups'][k]['model'] = loadConf(larva_model, 'Model')
    if life_params is not None:
        conf['life_params'] = life_params

    if sim_params['sim_ID'] is None:
        idx = next_idx(exp_type)
        sim_params['sim_ID'] = f'{exp_type}_{idx}'
    if sim_params['path'] is None:
        sim_params['path'] = f'single_runs/{exp_type}'
    if sim_params['duration'] is None:
        sim_params['duration'] = conf['sim_params']['duration']
    conf['sim_params'] = sim_params
    conf['experiment'] = exp_type

    return conf