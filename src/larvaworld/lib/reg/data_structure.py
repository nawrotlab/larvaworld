from larvaworld.lib import aux

def build_datapath_structure():
    kd = aux.AttrDict()
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
    d = aux.AttrDict()
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

    func_dic0 = {'h5':
                     {'load': aux.read, 'save': aux.storeH5},
                 'dic': {'load': aux.load_dict, 'save': aux.save_dict},

                 'solo_dicts': {'load': aux.loadSoloDics, 'save': aux.storeSoloDics}
                 }
    dic = {}
    for k, funcs in func_dic0.items():
        ddic = {kk: funcs for kk in kd[k]}
        dic.update(ddic)

    return aux.AttrDict(dic)


DATAPATH_DIC, DATAFUNC_DIC = build_datapath_structure()


def datapath(filepath_key, dir=None):
    if dir is not None and filepath_key in DATAPATH_DIC.keys():
        return f'{dir}{DATAPATH_DIC[filepath_key]}'
    else:
        return None


def datafunc(filepath_key, mode='load'):
    if filepath_key in DATAFUNC_DIC.keys():
        return DATAFUNC_DIC[filepath_key][mode]
    else:
        return None