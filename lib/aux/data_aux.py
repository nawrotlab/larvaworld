import numpy as np
import pandas as pd
import param

from lib.aux.par_aux import sub


def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key='end', unit='sec'):
    dfs = []
    for l, d in ddic.items():
        if key == 'end':
            try:
                df = d.endpoint_data
            except:
                df = d.read(key='end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read(key='step')
        df['DatasetID'] = l
        df['GroupID'] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config['dt'] for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    # ret = np.cumsum(a, dtype=float)
    # ret[n:] = ret[n:] - ret[:-n]
    return np.convolve(a, np.ones((n,)) / n, mode='same')
    # return ret[n - 1:] / n


def arrange_index_labels(index):
    from lib.aux import dictsNlists as dNl
    Nks = index.value_counts(sort=False)

    def merge(k, Nk):
        Nk1 = int((Nk - 1) / 2)
        Nk2 = Nk - 1 - Nk1
        return [''] * Nk1 + [k.upper()] + [''] * Nk2

    new = dNl.flatten_list([merge(k, Nk) for i, (k, Nk) in enumerate(Nks.items())])
    return new


def mdict2df(mdict, columns=['symbol', 'value', 'description']):
    data = []
    for k, p in mdict.items():
        entry = [getattr(p, col) for col in columns]
        data.append(entry)
    df = pd.DataFrame(data, columns=columns)
    df.set_index(columns[0], inplace=True)
    return df


def init2mdict(d0):
    from lib.aux import dictsNlists as dNl
    from lib.registry.par_dict import preparePar
    from lib.registry.par import v_descriptor
    # d = {}

    def check(D0):
        D = {}
        for kk, vv in D0.items():
            if not isinstance(vv, dict):
                pass
            elif 'dtype' in vv.keys() and vv['dtype'] == dict:
                mdict = check(vv)
                vv0 = {kkk: vvv for kkk, vvv in vv.items() if kkk not in mdict.keys()}
                if 'v0' not in vv0.keys():
                    vv0['v0'] = gConf(mdict)
                prepar = preparePar(p=kk, mdict=mdict, **vv0)
                p = v_descriptor(**prepar)
                D[kk] = p

            elif any([a in vv.keys() for a in ['symbol', 'h', 'label', 'disp', 'k']]):
                prepar = preparePar(p=kk, **vv)
                p = v_descriptor(**prepar)
                D[kk] = p

            else:

                # label = f'{kk} conf'
                # if 'v0' in vv.keys() :
                #     v0=vv['v0']
                # else :
                #     v0=None
                # if 'k' in vv.keys() :
                #     k=vv['k']
                # else :
                #     k=kk
                # vparfunc = vdicpar(mdict, h=f'The {kk} conf', lab=label, v0=v0)
                # kws = {
                #     'name': kk,
                #     'p': kk,
                #     'd': kk,
                #     'k': k,
                #     'disp': label,
                #     'sym': sub(k, 'conf'),
                #     'codename': kk,
                #     'dtype': dict,
                #     # 'func': func,
                #     # 'u': ureg.dimensionless,
                #     # 'u_name': None,
                #     # 'required_ks': [],
                #     'vparfunc': vparfunc,
                #     # 'dv': None,
                #     'v0': v0,
                #
                # }
                # p = v_descriptor(**kws)
                D[kk] = check(vv)
        return D

    d = check(d0)
    return dNl.NestDict(d)


def gConf(mdict, **kwargs):
    if mdict is None:
        return None


    elif isinstance(mdict, param.Parameterized):
        return mdict.v
    elif isinstance(mdict, dict):
        from lib.aux import dictsNlists as dNl
        conf = dNl.NestDict()
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                conf[d] = p.v
            else:
                conf[d] = gConf(mdict=p)
            conf = dNl.update_existingdict(conf, kwargs)
        # conf.update(kwargs)
        return conf
    else:
        return mdict


def update_mdict(mdict, mmdic):
    if mmdic is None or mdict is None:
        return None
    elif not isinstance(mmdic, dict) or not isinstance(mdict, dict):
        return mdict
    else:
        for d, p in mdict.items():
            new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(new_v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        new_v = tuple(new_v)
                p.v = new_v
            else:
                mdict[d] = update_mdict(mdict=p, mmdic=new_v)
        return mdict


def update_existing_mdict(mdict, mmdic):
    if mmdic is None:
        return mdict
    else:
        for d, v in mmdic.items():
            p = mdict[d]

            # new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        v = tuple(v)

                p.v = v
            elif isinstance(p, dict) and isinstance(v, dict):
                mdict[d] = update_existing_mdict(mdict=p, mmdic=v)
        return mdict


def get_ks(d0, k0=None, ks=[]):
    for k, p in d0.items():
        if k0 is not None:
            k = f'{k0}.{k}'
        if isinstance(p, param.Parameterized):

            ks.append(k)
        else:
            ks = get_ks(p, k0=k, ks=ks)
    return ks


