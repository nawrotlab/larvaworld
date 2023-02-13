import pandas as pd
import numpy as np

from larvaworld.lib import reg, aux

def comp_chunk_bearing(s, c, chunk, **kwargs):

    c0 = aux.nam.start(chunk)
    c1 = aux.nam.stop(chunk)
    ho = aux.nam.unwrap(aux.nam.orient('front'))
    ho0s = s[aux.nam.at(ho, c0)].dropna().values
    ho1s = s[aux.nam.at(ho, c1)].dropna().values
    for n, pos in c.sources.items():
        b = aux.nam.bearing2(n)
        b0_par = aux.nam.at(b, c0)
        b1_par = aux.nam.at(b, c1)
        db_par = aux.nam.chunk_track(chunk, b)
        b0 = aux.comp_bearing(s[aux.nam.at('x', c0)].dropna().values, s[aux.nam.at('y', c0)].dropna().values, ho0s, loc=pos)
        b1 = aux.comp_bearing(s[aux.nam.at('x', c1)].dropna().values, s[aux.nam.at('y', c1)].dropna().values, ho1s, loc=pos)
        s[b0_par] = np.nan
        s.loc[s[c0] == True, b0_par] = b0
        s[b1_par] = np.nan
        s.loc[s[c1] == True, b1_par] = b1
        s[db_par] = np.nan
        s.loc[s[c1] == True, db_par] = np.abs(b0) - np.abs(b1)
        aux.store_distros(s, pars=[b0_par, b1_par, db_par], parent_dir=c.dir)
        print(f'Bearing to source {n} during {chunk} computed')


def comp_patch_metrics(s, e, **kwargs):

    cum_t = aux.nam.cum('dur')
    on = 'on_food'
    off = 'off_food'
    on_tr = aux.nam.dur_ratio(on)
    on_cumt = aux.nam.cum(aux.nam.dur(on))
    off_cumt = aux.nam.cum(aux.nam.dur(off))
    s_on = s[s[on] == True]
    s_off = s[s[on] == False]

    e[on_cumt] = e[cum_t] * e[on_tr]
    e[off_cumt] = e[cum_t] * (1 - e[on_tr])

    for c in ['Lturn', 'turn', 'pause']:
        dur = aux.nam.dur(c)
        cdur = aux.nam.cum(dur)
        cdur_on = f'{cdur}_{on}'
        cdur_off = f'{cdur}_{off}'
        N = aux.nam.num(c)

        e[f'{N}_{on}'] = s_on[dur].groupby('AgentID').count()
        e[f'{N}_{off}'] = s_off[dur].groupby('AgentID').count()

        e[cdur_on] = s_on[dur].groupby('AgentID').sum()
        e[cdur_off] = s_off[dur].groupby('AgentID').sum()

        e[f'{aux.nam.dur_ratio(c)}_{on}'] = e[cdur_on] / e[on_cumt]
        e[f'{aux.nam.dur_ratio(c)}_{off}'] = e[cdur_off] / e[off_cumt]
        e[f'{aux.nam.mean(N)}_{on}'] = e[f'{N}_{on}'] / e[on_cumt]
        e[f'{aux.nam.mean(N)}_{off}'] = e[f'{N}_{off}'] / e[off_cumt]

    dst = aux.nam.dst('')
    cdst = aux.nam.cum(dst)
    cdst_on = f'{cdst}_{on}'
    cdst_off = f'{cdst}_{off}'
    v_mu = aux.nam.mean(aux.nam.vel(''))
    e[cdst_on] = s_on[dst].dropna().groupby('AgentID').sum()
    e[cdst_off] = s_off[dst].dropna().groupby('AgentID').sum()

    e[f'{v_mu}_{on}'] = e[cdst_on] / e[on_cumt]
    e[f'{v_mu}_{off}'] = e[cdst_off] / e[off_cumt]
    # e['handedness_score'] = e[aux.nam.num('Lturn')] / e[aux.nam.num('turn')]
    e[f'handedness_score_{on}'] = e[f"{aux.nam.num('Lturn')}_{on}"] / e[f"{aux.nam.num('turn')}_{on}"]
    e[f'handedness_score_{off}'] = e[f"{aux.nam.num('Lturn')}_{off}"] / e[f"{aux.nam.num('turn')}_{off}"]

@reg.funcs.annotation("source_attraction")
def comp_bearing_to_source(s, e, c, **kwargs):
    for b in ['stride', 'pause', 'turn']:
        try:
            comp_chunk_bearing(s, c, chunk=b)
            if b == 'turn':
                comp_chunk_bearing(s, c, chunk='Lturn')
                comp_chunk_bearing(s, c, chunk='Rturn')
        except:
            pass

@reg.funcs.annotation("patch_residency")
def comp_time_on_patch(s, e, c, **kwargs):
    on = 'on_food'
    if on in s.columns and aux.nam.dur_ratio(on) in e.columns:
        comp_patch_metrics(s, e)
