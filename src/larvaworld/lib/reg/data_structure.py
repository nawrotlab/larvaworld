from larvaworld.lib import aux


def build_datapath_structure():
    # kd = aux.AttrDict()

    folders = {
        'parent': ['data', 'plots', 'visuals', 'aux', 'model'],
        'data': ['individuals'],
        'individuals': ['bouts', 'foraging', 'deb', 'nengo'],
        'plots': ['model_tables', 'model_summaries'],
        'model': ['GAoptimization', 'evaluation'],

    }

    h5base = ['end', 'step']
    h5step = ['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']
    h5aux = ['derived', 'traj', 'aux', 'vel_definition', 'tables', 'food', 'distro']

    # kd.h5 = h5base + kd.h5step + h5aux

    confs = ['conf', 'sim_conf', 'log']
    dics1 = ['chunk_dicts', 'grouped_epochs', 'pooled_epochs', 'cycle_curves', 'dsp', 'fit']
    dics2 = ['ExpFitter']

    # kd.dic = dics1 + dics2 + confs

    d = aux.AttrDict()
    d.parent = ''
    for k0, ks in folders.items():
        for k in ks:
            d[k] = f'{d[k0]}/{k}'

    for k in h5base + h5step + h5aux:
        d[k] = f'{d.data}/{k}.h5'
    for k in dics1 + dics2 + confs:
        d[k] = f'{d.data}/{k}.txt'
    return d


DATAPATH_DIC = build_datapath_structure()


def datapath(filepath_key, dir=None):
    if dir is not None and filepath_key in DATAPATH_DIC.keys():
        return f'{dir}{DATAPATH_DIC[filepath_key]}'
    else:
        return None


