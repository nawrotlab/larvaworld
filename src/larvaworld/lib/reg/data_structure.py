from larvaworld.lib import aux


def build_datapath_structure():
    folders = {
        'parent': ['data', 'plots', 'visuals', 'model'],
        'data': ['individuals'],
        'individuals': ['bouts','foraging', 'deb', 'nengo'],
        'plots': ['model_tables', 'model_summaries'],
        'model': ['GAoptimization', 'evaluation'],

    }


    d = aux.AttrDict()
    d.parent = ''
    for k0, ks in folders.items():
        for k in ks:
            d[k] = f'{d[k0]}/{k}'

    for k in ['end', 'step', 'vel_definition', 'distro']:
        d[k] = f'{d.data}/{k}.h5'
    for k in ['conf', 'chunk_dicts', 'grouped_epochs', 'pooled_epochs', 'cycle_curves', 'dsp', 'fit']:
        d[k] = f'{d.data}/{k}.txt'
    return d


DATAPATH_DIC = build_datapath_structure()


def datapath(filepath_key, dir=None):
    if dir is not None and filepath_key in DATAPATH_DIC.keys():
        return f'{dir}{DATAPATH_DIC[filepath_key]}'
    else:
        return None


