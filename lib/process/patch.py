import pandas as pd
import numpy as np


import lib.aux.naming as nam
from lib.process.store import store_aux_dataset

def comp_chunk_bearing(s, c, chunk, **kwargs):
    from lib.aux.xy_aux import comp_bearing

    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    ho = nam.unwrap(nam.orient('front'))
    ho0s = s[nam.at(ho, c0)].dropna().values
    ho1s = s[nam.at(ho, c1)].dropna().values
    for n, pos in c.sources.items():
        b = nam.bearing2(n)
        b0_par = nam.at(b, c0)
        b1_par = nam.at(b, c1)
        db_par = nam.chunk_track(chunk, b)
        b0 = comp_bearing(s[nam.at('x', c0)].dropna().values, s[nam.at('y', c0)].dropna().values, ho0s, loc=pos)
        b1 = comp_bearing(s[nam.at('x', c1)].dropna().values, s[nam.at('y', c1)].dropna().values, ho1s, loc=pos)
        s[b0_par] = np.nan
        s.loc[s[c0] == True, b0_par] = b0
        s[b1_par] = np.nan
        s.loc[s[c1] == True, b1_par] = b1
        s[db_par] = np.nan
        s.loc[s[c1] == True, db_par] = np.abs(b0) - np.abs(b1)
        store_aux_dataset(s, pars=[b0_par, b1_par, db_par], type='distro', file=c.aux_dir)
        print(f'Bearing to source {n} during {chunk} computed')


def comp_patch_metrics(s, e, **kwargs):
    # v=nam.vel('')
    # v_mu=nam.mean(v)
    cum_t = nam.cum('dur')
    on = 'on_food'
    off = 'off_food'
    on_tr = nam.dur_ratio(on)
    s_on=s[s[on] == True]
    s_off=s[s[on] == False]
    # e[f'{v_mu}_{on}'] = s_on[v].dropna().groupby('AgentID').mean()
    # e[f'{v_mu}_{off}'] = s_off[v].dropna().groupby('AgentID').mean()
    for c in ['run', 'Lturn','turn', 'pause']:
        dur = nam.dur(c)
        cdur = nam.cum(dur)
        N = nam.num(c)

        e[f'{N}_{on}'] = s_on[dur].groupby('AgentID').count()
        e[f'{N}_{off}'] = s_off[dur].groupby('AgentID').count()

        e[f'{cdur}_{on}'] = s_on[dur].groupby('AgentID').sum()
        e[f'{cdur}_{off}'] = s_off[dur].groupby('AgentID').sum()

        e[f'{nam.dur_ratio(c)}_{on}'] = e[f'{cdur}_{on}'] / e[cum_t] / e[on_tr]
        e[f'{nam.dur_ratio(c)}_{off}'] = e[f'{cdur}_{off}'] / e[cum_t] / (1 - e[on_tr])
        e[f'{nam.mean(N)}_{on}'] = 60*e[f'{N}_{on}'] / e[cum_t] / e[on_tr]
        e[f'{nam.mean(N)}_{off}'] = 60*e[f'{N}_{off}'] / e[cum_t] / (1 - e[on_tr])

    dst = nam.dst('')
    cdst = nam.cum(dst)
    v_mu = nam.mean(nam.vel(''))
    e[f'{cdst}_{on}'] = s[s[on] == True][dst].dropna().groupby('AgentID').sum()
    e[f'{cdst}_{off}'] = s[s[on] == False][dst].dropna().groupby('AgentID').sum()

    e[f'{v_mu}_{on}'] = e[f'{cdst}_{on}'] / e[cum_t] / e[on_tr]
    e[f'{v_mu}_{off}'] = e[f'{cdst}_{off}'] / e[cum_t] / (1 - e[on_tr])
    e['handedness_score'] = e[nam.num('Lturn')] / e[nam.num('turn')]
    e[f'handedness_score_{on}'] = e[f"{nam.num('Lturn')}_{on}"] / e[f"{nam.num('turn')}_{on}"]
    e[f'handedness_score_{off}'] = e[f"{nam.num('Lturn')}_{off}"] / e[f"{nam.num('turn')}_{off}"]


def comp_patch(s,e,c):
    on = 'on_food'
    if on in s.columns and nam.dur_ratio(on) in e.columns:
        comp_patch_metrics(s, e)
    for b in ['stride', 'pause', 'turn']:
        try:
            comp_chunk_bearing(s, c, chunk=b)
            if b == 'turn':
                comp_chunk_bearing(s,  c, chunk='Lturn')
                comp_chunk_bearing(s, c, chunk='Rturn')
        except:
            pass
