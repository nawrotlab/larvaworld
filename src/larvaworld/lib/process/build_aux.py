import numpy as np
import pandas as pd

from larvaworld.lib import aux


def df_from_csvs(pref,Npoints=11,Ncontour=0, max_Nagents=None, time_slice=None, min_duration_in_sec=0.0):
    t='t'
    aID='AgentID'

    kws = {'header': None, 'sep': '\t'}
    par_list = [pd.read_csv(f'{pref}_{suf}.txt', **kws) for suf in ['larvaid', 't', 'x_spine', 'y_spine']]

    columns = [aID, t] + aux.nam.xy(aux.nam.midline(Npoints, type='point'), xsNys=True, flat=True)
    try:
        states = pd.read_csv(f'{pref}_state.txt', **kws)
        par_list.append(states)
        columns.append('state')
    except:
        pass

    if Ncontour > 0:
        try:
            xcs = pd.read_csv(f'{pref}_x_contour.txt', **kws)
            ycs = pd.read_csv(f'{pref}_y_contour.txt', **kws)
            xcs, ycs = aux.convex_hull(xs=xcs.values, ys=ycs.values, N=Ncontour)
            xcs = pd.DataFrame(xcs, index=None)
            ycs = pd.DataFrame(ycs, index=None)
            par_list += [xcs, ycs]
            columns+=aux.nam.xy(aux.nam.contour(Ncontour), xsNys=True, flat=True)
        except:
            pass

    df = pd.concat(par_list, axis=1, sort=False)
    df.columns=columns
    df.set_index(keys=[aID], inplace=True, drop=True)

    if time_slice is not None:
        tmin, tmax = time_slice
        df = df[df[t] < tmax]
        df = df[df[t] >= tmin]
    df = df.loc[df[t].groupby(aID).last() - df[t].groupby(aID).first() > min_duration_in_sec]
    if max_Nagents is not None:
        df = df.loc[df['head_x'].dropna().groupby(aID).count().nlargest(max_Nagents).index]
    df.sort_index(inplace=True)
    return df

def match_larva_ids(s, e, pars, wl=100, wt=0.1, ws=0.5, max_error=600, Nidx=20, verbose=1, **kwargs):
    t = 't'
    aID = 'AgentID'
    # s.reset_index(level=t, drop=False, inplace=True)
    s[t] = s[t].values.astype(float)

    pairs = {}

    def common_member(a, b):
        a_set = set(a)
        b_set = set(b)
        return a_set & b_set

    def eval(t0, xy0, l0, t1, xy1, l1):
        tt = t1 - t0
        if tt <= 0:
            return max_error * 2
        ll = np.abs(l1 - l0)
        dd = np.sqrt(np.sum((xy1 - xy0) ** 2))
        # print(tt,ll,dd)
        return wt * tt + wl * ll + ws * dd

    def get_extrema(ss, pars):
        ids = ss.index.unique().tolist()

        mins = ss[t].groupby(aID).min()
        maxs = ss[t].groupby(aID).max()
        durs = ss[t].groupby(aID).count()
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

    ls = e['length']

    ids, mins, maxs, first_xy, last_xy, durs = get_extrema(s, pars)
    Nids0 = len(ids)
    while Nidx <= len(ids):
        cur_er, id0, id1 = max_error, None, None
        t0s = maxs.nsmallest(Nidx)
        t1s = mins.loc[mins > t0s.min()].nsmallest(Nidx)
        if len(t1s) > 0:
            for i in range(Nidx):
                cur_id0, t0 = t0s.index[i], t0s.values[i]
                xy0, l0 = last_xy[cur_id0], ls[cur_id0]
                ee = [eval(t0, xy0, l0, mins[id], first_xy[id], ls[id]) for id in t1s.index]
                temp_err = np.min(ee)
                if temp_err < cur_er:
                    cur_er, id0, id1 = temp_err, cur_id0, t1s.index[np.argmin(ee)]
        if id0 is not None:
            pairs[id0] = id1
            ls[id1] = (ls[id0] * durs[id0] + ls[id1] * durs[id1]) / (durs[id0] + durs[id1])
            durs[id1] += durs[id0]
            del durs[id0]
            ls.drop([id0], inplace=True)
            ids, mins, maxs, first_xy, last_xy = update_extrema(id0, id1, ids, mins, maxs, first_xy, last_xy)
            if verbose >= 2:
                print(len(ids), int(cur_er))
        else:
            Nidx += 1
    Nids1 = len(ids)
    if verbose >= 2:
        print('Finalizing dataset')
    while len(common_member(list(pairs.keys()), list(pairs.values()))) > 0:
        for id0, id1 in pairs.items():
            if id1 in pairs.keys():
                pairs[id0] = pairs[id1]
                break
    s.rename(index=pairs, inplace=True)
    # s.reset_index(drop=False, inplace=True)
    # s.set_index(keys=[t, aID], inplace=True, drop=True)
    if verbose >= 1:
        print(f'**--- Track IDs reduced from {Nids0} to {Nids1} by the matchIDs algorithm -----')
    return s
