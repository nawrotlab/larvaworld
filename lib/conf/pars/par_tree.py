import numpy as np
import pandas as pd

from lib.aux import dictsNlists as dNl


def tree_dict(d, parent_key='', sep='.'):
    cols = ['parent', 'key', 'text', 'values']
    entries = []
    keys = []

    def add(item):
        entry = dict(zip(cols, item))
        if not entry['key'] in keys:
            entries.append(entry)
            keys.append(entry['key'])

    add(['', parent_key, parent_key, [' ']])

    def tree_dict0(d, parent_key='', sep='.'):
        import collections

        for k, v in d.items():
            new_key = parent_key + sep + k
            if isinstance(v, collections.MutableMapping):
                add([parent_key, new_key, k, [' ']])
                if len(v) > 0:
                    tree_dict0(v, new_key, sep=sep)
            else:
                add([parent_key, new_key, k, [v]])
        return entries

    return tree_dict0(d, parent_key, sep)


def pars_to_tree(name):
    from lib.conf.pars.pars import ParDict
    from lib.conf.base.dtypes import par, par_dict
    from lib.aux.par_aux import dtype_name

    invalid = []
    valid = []

    def add_entry(k4, v4, parent):
        key = f'{parent}.{k4}'
        if 'content' in v4.keys():
            dd = v4['content']
            if key not in valid:
                data.append([parent, key, k4, None, dict, None, k4])
                valid.append(key)
            for k1, v1 in dd.items():
                add_entry(k1, v1, key)
        else:
            entry = [parent, key, k4] + [v4[c] for c in columns[3:]]
            data.append(entry)
            valid.append(key)

    def add_multientry0(d, k0, name):
        key = f'{name}.{k0}'
        if key not in valid:
            data.append([name, key, k0, None, dict, None, k0])
            valid.append(key)
        for k1, v1 in d.items():
            add_entry(k1, v1, key)

    data = []
    columns = ['parent', 'key', 'text', 'initial_value', 'dtype', 'tooltip', 'disp']
    columns2 = ['parent', 'key', 'text', 'default_value', 'dtype', 'description', 'name']
    P = ParDict.init_dict[name]
    data.append(['root', name, name, None, dict, None, name])
    valid.append(name)
    for k0, v0 in P.items():
        if v0 is None :
            continue

        try:
            d = par(k0, **v0)
            add_entry(k0, d[k0], name)
        except:
            # d0 = P.get(k0, None)
            # if d0 is None:
            #     # from lib.conf.base.init_pars import InitDict
            #     d0 = ParDict.init_dict[k0]
            d = par_dict(d0=v0)
            add_multientry0(d, k0, name)
    ddf = pd.DataFrame(data, columns=columns2)
    if 'dtype' in columns2:
        ddf['dtype'] = [dtype_name(v) for v in ddf['dtype']]
    ddf = ddf.fillna(value=' ')
    ddf = ddf.replace({}, ' ')
    return ddf



def multiconf_to_tree(ids, conftype):
    from lib.conf.stored.conf import expandConf
    dfs = []
    for i, id in enumerate(ids):
        conf = expandConf(id, conftype)
        entries = tree_dict(d=conf, parent_key=id)
        df = pd.DataFrame.from_records(entries, index=['parent', 'key', 'text'])
        dfs.append(df)
    ind0 = []
    for df in dfs:
        for ind in df.index.values:
            if ind not in ind0:
                ind0.append(ind)
    vs = np.zeros([len(ind0), len(ids)]) * np.nan
    df0 = pd.DataFrame(vs, index=ind0, columns=ids)
    for id, df in zip(ids, dfs):
        for key in df.index:
            df0[id].loc[key] = df['values'].loc[key][0]
    df0.reset_index(inplace=True)
    df0['values'] = [df0[id] for id in ids]
    df0.drop(ids, axis=1)
    return df0.to_dict(orient='records')
    # return comp_entries


if __name__ == '__main__':

    name = 'crawler'
    # mIDs = ['PHIonNEU', 'SQonNEU', 'PHIonSIN', 'SQonSIN']
    # for mID in mIDs:
    #     m=loadConf(mID, 'Model')
    #     print(m)
    ddf=pars_to_tree(name)
    print(ddf.values)