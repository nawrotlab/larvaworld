import numpy as np

from lib.aux.dictsNlists import common_member


def match_larva_ids(s, e, pars=None, wl=100, wt=1, ws=0.5, max_error=600, Nidx=20, **kwargs):
    pairs= {}

    def eval(t0, xy0, l0, t1, xy1, l1):
        tt = t1 - t0
        if tt <= 0:
            return max_error * 2
        ll = np.abs(l1 - l0)
        dd = np.sqrt(np.sum((xy1 - xy0) ** 2))
        return wt * tt + wl * ll + ws * dd

    ls = e['length']
    if pars is None:
        pars = s.columns.values.tolist()
    s.reset_index(level='Step', drop=False, inplace=True)
    s['Step'] = s['Step'].values.astype(int)
    ids, mins, maxs, first_xy, last_xy, durs = get_extrema(s, pars)
    while Nidx <= len(ids):
        cur_er, id0, id1 = max_error, None, None
        t0s = maxs.nsmallest(Nidx)
        t1s=mins.loc[mins>t0s.min()].nsmallest(Nidx)
        if len(t1s)>0 :
            for i in range(Nidx):
                cur_id0, t0 = t0s.index[i], t0s.values[i]
                xy0, l0 = last_xy[cur_id0], ls[cur_id0]
                ee = [eval(t0, xy0, l0, mins[id], first_xy[id], ls[id]) for id in t1s.index]
                temp_err=np.min(ee)
                if temp_err < cur_er:
                    cur_er, id0, id1 = temp_err, cur_id0, t1s.index[np.argmin(ee)]
        if id0 is not None:
            pairs[id0]=id1
            ls[id1] = (ls[id0]*durs[id0]+ls[id1]*durs[id1])/(durs[id0]+durs[id1])
            durs[id1]+=durs[id0]
            del durs[id0]
            ls.drop([id0], inplace=True)
            ids, mins, maxs, first_xy, last_xy = update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy)
            print(len(ids), int(cur_er))
        else :
            Nidx += 1
    print('Finalizing dataset')
    while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
        for id0,id1 in pairs.items() :
            if id1 in pairs.keys() :
                pairs[id0]=pairs[id1]
                break
    s.rename(index=pairs, inplace=True)
    s.reset_index(drop=False, inplace=True)
    s.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
    return s


def get_extrema(ss, pars):
    ids = ss.index.unique().tolist()

    mins = ss['Step'].groupby('AgentID').min()
    maxs = ss['Step'].groupby('AgentID').max()
    durs = ss['Step'].groupby('AgentID').count()
    first_xy, last_xy = {}, {}
    for id in ids:
        first_xy[id] = ss[pars].xs(id).dropna().values[0, :]
        last_xy[id] = ss[pars].xs(id).dropna().values[-1, :]
    return ids, mins, maxs, first_xy, last_xy, durs


def update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy):
    mins[id1], first_xy[id1] = mins[id0], first_xy[id0]
    del mins[id0]
    del maxs[id0]
    del first_xy[id0]
    del last_xy[id0]
    ids.remove(id0)
    return ids, mins, maxs, first_xy, last_xy