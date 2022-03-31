import copy
import os

from scipy.stats import ks_2samp
from typing import Union

import numpy as np
from matplotlib import cm
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt

import lib.aux.dictsNlists as dNl
import lib.aux.naming as nam
from lib.anal.fitting import std_norm, minmax
from lib.anal.plot_aux import plot_single_bout, dataset_legend
from lib.aux.colsNstr import N_colors
from lib.aux.combining import render_mpl_table

from lib.conf.base.par import getPar, ParDict
from lib.conf.stored.conf import loadRef
from lib.process.aux import annotation, fit_bouts
from lib.process.spatial import scale_to_length, comp_straightness_index, comp_dispersion
from lib.process.store import get_dsp

dic = ParDict(mode='load').dict
dst, v, sv, acc, sa, fou, rou, fo, ro, b,fov, rov, bv,foa, roa, ba, x, y, l,dsp, dsp_0_40,dsp_0_40_mu,dsp_0_40_max,sdsp, sdsp_0_40,sdsp_0_40_mu,sdsp_0_40_max,str_fov_mu,run_fov_mu,pau_fov_mu, str_fov_std,pau_fov_std,sstr_d_mu, sstr_d_std,str_d_mu, str_d_std, str_sv_mu, pau_sv_mu,str_v_mu,run_v_mu,run_sv_mu, pau_v_mu, str_tr,run_tr, pau_tr,Ltur_tr,Rtur_tr, Ltur_fou,Rtur_fou, run_t_min, cum_t, run_t, run_dst, pau_t= [dic[k]['d'] for k in ['d','v', 'sv','a','sa','fou', 'rou', 'fo', 'ro', 'b', 'fov', 'rov', 'bv', 'foa', 'roa', 'ba', 'x', 'y', 'l',"dsp", "dsp_0_40","dsp_0_40_mu","dsp_0_40_max","sdsp", "sdsp_0_40","sdsp_0_40_mu","sdsp_0_40_max",'str_fov_mu','run_fov_mu','pau_fov_mu', 'str_fov_std','pau_fov_std', 'sstr_d_mu', 'sstr_d_std','str_d_mu', 'str_d_std', 'str_sv_mu', 'pau_sv_mu','str_v_mu', 'run_v_mu','run_sv_mu','pau_v_mu', 'str_tr','run_tr','pau_tr','Ltur_tr','Rtur_tr', 'Ltur_fou','Rtur_fou', 'run_t_min', 'cum_t',  'run_t', 'run_d', 'pau_t']]
l, v_mu,cum_d, sv_mu, fov_mu, b_mu = [dic[k]['d'] for k in ['l', 'v_mu', "cum_d", "sv_mu",'fov_mu','b_mu']]
tors=tor2,tor2_mu,tor2_std,tor5,tor5_mu,tor5_std,tor10,tor10_mu,tor10_std,tor20,tor20_mu,tor20_std= [dic[k]['d'] for k in ["tor2","tor2_mu","tor2_std","tor5","tor5_mu","tor5_std","tor10","tor10_mu","tor10_std","tor20","tor20_mu","tor20_std"]]
fsv, ffov =  [dic[k]['d'] for k in ['fsv','ffov']]

att='attenuation'
att_max, att_min, phi_att_max, phi_sv_max = nam.max(att), nam.min(att), nam.max(f'phi_{att}'), nam.max(f'phi_{sv}')

def adapt_conf(conf0, ee, c):
    conf = copy.deepcopy(conf0)
    dic={
        run_sv_mu : ee[str_sv_mu] if str_sv_mu in ee.index else ee[run_v_mu] / ee[l],
        run_v_mu : ee[run_v_mu],
        run_t_min : ee[run_t_min],
        'ang_vel_headcast' : np.deg2rad(ee[pau_fov_mu]),
        'theta_min_headcast' :  np.deg2rad(ee['headcast_q25_amp']),
        'theta_max_headcast' : np.deg2rad(ee['headcast_q75_amp']),
        'theta_max_weathervane' : np.deg2rad(ee['weathervane_q75_amp']),
        'ang_vel_weathervane' : np.deg2rad(ee[run_fov_mu]),
        'turner_input_constant' : ee['turner_input_constant'],
        'run_dist' : c.bout_distros.run_dur,
        'pause_dist' : c.bout_distros.pause_dur,
        'stridechain_dist' : c.bout_distros.run_count,
        'initial_freq' : ee[fsv],
        'step_mu' : ee[sstr_d_mu],
        'step_std' : ee[sstr_d_std],
        'attenuation_min' : ee[att_min],
        'attenuation_max' : ee[att_max],
        'max_vel_phase' : ee[phi_sv_max],
    }
    for k, v in dic.items() :
        if k in conf.keys():
            conf[k]=v

    return conf


def sim_locomotor(L, N, dt, df_cols, e_id):
    aL = np.ones([N, len(df_cols)]) * np.nan
    aL[0, :] = 0
    xy0s = e_id[nam.initial([fo, x, y])]
    if not None in xy0s:
        aL[0, 3:6] = xy0s
    cur_fo = np.deg2rad(aL[0, 3])
    cur_x = aL[0, 4]
    cur_y = aL[0, 5]
    for i in range(N - 1):
        lin, ang, feed = L.step(length=e_id[l])
        cur_d = lin * dt
        cur_fo += ang * dt
        cur_x += np.cos(cur_fo) * cur_d
        cur_y += np.sin(cur_fo) * cur_d
        aL[i + 1, :7] = [lin, np.rad2deg(ang), cur_d, np.rad2deg(cur_fo), cur_x, cur_y, np.rad2deg(L.bend)]

    return aL


def sim_dataset(d,s, e, c, ids, loco_id, loco_func, loco_conf, adapted):

    N = c.Nticks
    strides_enabled = True if 'initial_freq' in loco_conf.keys() else False
    vel_thr = c.vel_thr if c.Npoints == 1 else None

    # Define columns for the simulated dataset
    df_cols, = getPar(['v', 'fov', 'd', 'fo', 'x', 'y', 'b'], to_return=['d'])
    Nids = len(ids)
    Ncols = len(df_cols)

    l = "length"
    if l not in e.columns:
        e[l] = np.ones(Nids) * 0.004
    # empty=np.ones([c.Nticks*len(ids), len(df_cols)])*np.nan
    cc = dNl.AttrDict.from_nested_dicts({
        'id': loco_id,
        'dt': c.dt,
        'Nticks': N,
        'Npoints': c.Npoints,
        'point': '',

    })

    e_ps = [l, nam.initial(x), nam.initial(y), nam.initial(fo)]
    ee = pd.DataFrame(e[e_ps].loc[ids], index=ids, columns=e_ps)
    ee[cum_t] = N * c.dt

    aaL = np.zeros([N, Nids, Ncols]) * np.nan

    for jj, id in enumerate(ids):
        ee_id = e.loc[id]

        # Build locomotor instance of given model configuration adapted to the specific experimental larva
        if adapted:
            loco_conf = adapt_conf(loco_conf, ee_id, c)
        L = loco_func(dt=c.dt, **loco_conf)
        aaL[:, jj, :] = sim_locomotor(L, N, c.dt, df_cols, ee_id)

    df_index = pd.MultiIndex.from_product([np.arange(N), ids], names=['Step', 'AgentID'])
    ss = pd.DataFrame(aaL.reshape([N * Nids, Ncols]), index=df_index, columns=df_cols)
    ss[bv] = ss[b].groupby('AgentID').diff() / c.dt
    ss[ba] = ss[bv].groupby('AgentID').diff() / c.dt
    ss[foa] = ss[fov].groupby('AgentID').diff() / c.dt
    ss[acc] = ss[v].groupby('AgentID').diff() / c.dt

    ee[v_mu] = ss[v].dropna().groupby('AgentID').mean()
    ee[cum_d] = ss[dst].dropna().groupby('AgentID').sum()

    scale_to_length(ss, ee, keys=['d', 'v', 'a', 'v_mu'])

    comp_dispersion(ss, ee, cc.dt, cc.point, dsp_starts=[0], dsp_stops=[40])
    comp_straightness_index(ss, c.dt, e=ee, tor_durs=[5, 20])
    aux_dic = annotation(ss, ee, cc, strides_enabled=strides_enabled, vel_thr=vel_thr, save_to=d.dir_dict.chunk_dicts)
    bout_dic = fit_bouts(aux_dic, loco_id, cc, save_to=d.dir_dict['group_bout_dicts'])

    return ss, ee, cc, bout_dic


def eval_dataset(s, ss, e, ee, end_ps, distro_ps):
    # print(e.index, ee.index)
    eval_func = np.median

    ids = e.index.values
    Eend, Edistro = {}, {}
    Eend_pool, Edistro_pool = {}, {}

    # Averages
    ps1, ps1l = getPar(end_ps, to_return=['d', 'l'])
    for p, pl in zip(ps1, ps1l):
        Eend[pl] = None
        Eend_pool[pl] = None
        if p in e.columns and p in ee.columns:
            # error_end[pl]=eval_func(np.abs(e[p] - ee[p]))
            Eend[pl] = ((e[p] - ee[p]) ** 2).mean() ** .5
            Eend_pool[pl] = ks_2samp(e[p].values, ee[p].values)[0]

    # Distributions
    N = 20
    ps2, ps2l = getPar(distro_ps, to_return=['d', 'l'])
    for p, pl in zip(ps2, ps2l):
        Edistro[pl] = None
        Edistro_pool[pl] = None
        if p in s.columns and p in ss.columns:
            pps = []
            for id in ids:
                sp, ssp = s[p].xs(id, level="AgentID").dropna().values, ss[p].xs(id, level="AgentID").dropna().values
                if sp.shape[0] > N and ssp.shape[0] > N:
                    pps.append(ks_2samp(sp, ssp)[0])
                # else :
                #     print(p, id)
            Edistro[pl] = eval_func(pps)
            spp, sspp = s[p].dropna().values, ss[p].dropna().values
            if spp.shape[0] > N and sspp.shape[0] > N:
                Edistro_pool[pl] = ks_2samp(spp, sspp)[0]


    return Eend, Eend_pool, Edistro, Edistro_pool

def run_locomotor_evaluation(d, locomotor_models, Nids=None,end_ps=None, distro_ps=None, save_to=None,
                             stridechain_duration=False) :
    if save_to is not None :
        os.makedirs(save_to, exist_ok=True)

    # Select the most complete experimental larvae

    s, e, c = d.step_data, d.endpoint_data, d.config
    if Nids is None:
        ids=c.agent_ids
        Nids=len(ids)
    else :
        ids = e.nlargest(Nids, 'cum_dur').index.values
    # Individual-specific model fitting
    s_exp = s.loc[(slice(None), ids), slice(None)]
    e_exp = e.loc[ids]

    # A dictionary to keep all datasets
    loco_dict = {'experiment': {'step': s_exp, 'end': e_exp}}
    bout_dict = {'experiment': d.load_group_bout_dict()}
    Eend, Edistro = {}, {}
    Eend_pool, Edistro_pool = {}, {}

    # Evaluation metrics
    if end_ps is None:
        end_ps = ['fsv', 'ffov', 'run_tr', 'pau_tr', 'v_mu', 'run_v_mu', 'pau_v_mu', 'run_a_mu', 'pau_a_mu', 'cum_d',
              'dsp_0_40_mu', 'dsp_0_40_max', 'tor5_mu', 'tor5_std', 'tor20_mu', 'tor20_std',
              'run_fov_mu', 'run_fov_std', 'pau_fov_mu', 'pau_fov_std', 'run_foa_mu', 'pau_foa_mu']
    if distro_ps is None:
        distro_ps = ['v', 'a', 'b', 'bv', 'ba', 'fov', 'foa', 'tor5', 'tor20', 'run_d', 'run_t', 'pau_t', 'tur_t',
                 'tur_fou', 'tur_fov_max']

    for ii, (loco_id, (func, conf, adapted)) in enumerate(locomotor_models.items()):
        print(f'Evaluating model {loco_id} on {Nids} larvae from dataset {d.id}')
        ss, ee, cc, bouts = sim_dataset(d, s_exp, e_exp, c, ids, loco_id, func, conf, adapted)

        Eend[loco_id], Eend_pool[loco_id], Edistro[loco_id], Edistro_pool[
            loco_id] = eval_dataset(s_exp, ss, e_exp, ee, end_ps, distro_ps)

        loco_dict[loco_id] = {'step': ss, 'end': ee}

        bout_dict[loco_id] = bouts

    Eend = pd.DataFrame.from_records(Eend).T
    Edistro = pd.DataFrame.from_records(Edistro).T
    Eend_pool = pd.DataFrame.from_records(Eend_pool).T
    Edistro_pool = pd.DataFrame.from_records(Edistro_pool).T
    error_dict={
        'individual endpoint':Eend, 'pooled endpoint':Eend_pool, 'individual distribution':Edistro, 'pooled distribution':Edistro_pool
    }

    error_tables(error_dict, save_to=save_to)
    error_barplots(error_dict, normalization='raw', save_to=save_to)
    error_barplots(error_dict, normalization='minmax', save_to=save_to)
    plot_trajectories(loco_dict, save_to=save_to)
    plot_bouts(bout_dict, save_to=save_to, stridechain_duration=stridechain_duration)
    plot_comparative_dispersion(loco_dict, c=c,save_to=save_to)
    return error_dict, loco_dict, bout_dict

def error_tables(error_dict, save_to=None) :
    dic={}
    for k, v in error_dict.items() :
        ax, fig=render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min', title=k)
        dic[k]=fig
        if save_to is not None :
            plt.savefig(f'{save_to}/error_{k}.pdf', dpi=300)
    return dic
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')

def error_barplots(error_dict, normalization='minmax', save_to=None) :
    # Plot the errors
    df1, df2, df3, df4 = error_dict['individual endpoint'], error_dict['pooled endpoint'], error_dict['individual distribution'], error_dict['pooled distribution']
    if normalization=='minmax' :
        df1, df2, df3, df4 = minmax(df1), minmax(df2), minmax(df3), minmax(df4)
    elif normalization=='std' :
        df1, df2, df3, df4 = std_norm(df1), std_norm(df2), std_norm(df3), std_norm(df4)
    dfs = dict(zip([f'{normalization} solo', f'{normalization} pooled', 'KS solo', 'KS pooled'], [df1, df2, df3, df4]))

    fig, axs = plt.subplots(4, 1, figsize=(20, 15), sharex=True)
    axs = axs.ravel()
    for ii, (lab, df) in enumerate(dfs.items()):
        df.plot(kind='bar', ax=axs[ii], ylabel=lab, rot=0, legend=False)
        if ii in [0, 2]:
            axs[ii].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.subplots_adjust(hspace=0.05, top=0.99, bottom=0.15)
    if save_to is not None:
        fig.savefig(f'{save_to}/error_barplots_{normalization}.pdf', dpi=300)
    ddf = {}
    for ii, (lab, df) in enumerate(dfs.items()):
        ddf[lab] = df.mean(axis=1)
    df0 = pd.DataFrame.from_dict(ddf)

    axx, figg =render_mpl_table(df0, highlighted_cells='col_min')
    if save_to is not None:
        figg.savefig(f'{save_to}/mean_error_table.pdf', dpi=300)
    return fig

def plot_trajectories(loco_dict, save_to=None, show=False) :
    # Plot the trajectories
    fig, axs = plt.subplots(len(loco_dict) // 2 + 1, 2, figsize=(20, 35), sharex=True, sharey=True)
    axs = axs.ravel()
    for ii, (lab, sim) in enumerate(loco_dict.items()):
        for id in sim['step'].index.unique('AgentID').values:
            xy = sim['step'][['x', 'y']].xs(id, level="AgentID").values
            axs[ii].plot(xy[:, 0], xy[:, 1])
        axs[ii].set_title(lab)
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_trajectories.pdf', dpi=300)
    if show :
        plt.show()
    return fig

def plot_bouts(bout_dict, save_to=None, show=False, plot_fits='best', stridechain_duration=False, legend_outside=False,
               axs=None, fig=None) :
    bout_dict= dNl.AttrDict.from_nested_dicts(bout_dict)
    cols = N_colors(len(bout_dict))
    if axs is None and fig is None :
        fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for j, (k, v) in enumerate(bout_dict.items()):
        kws = {
            'marker': 'o' if k == 'experiment' else '.',
            'plot_fits': plot_fits,
            'label': k,
            'color': cols[j],
            'legend_outside': legend_outside,
            'axs': axs,
            'x0': None

        }
        if v.pause_dur is not None:
            bout = 'pauses'
            fit_dic = v.pause_dur
            print(k, bout, fit_dic.best.pause_dur.best)
            plot_single_bout(fit_dic=fit_dic,discr=False, bout=bout, i=1, **kws)
        if stridechain_duration and v.run_dur is not None:
            bout = 'runs'
            fit_dic = v.run_dur
            print(k, bout, fit_dic.best.run_dur.best)
            plot_single_bout(fit_dic=fit_dic,discr=False, bout=bout, i=0, **kws)
        elif not stridechain_duration and v.run_count is not None:
            bout = 'stridechains'
            fit_dic = v.run_count
            print(k, bout, fit_dic.best.run_count.best)
            plot_single_bout(fit_dic=fit_dic,discr=True, bout=bout, i=0, **kws)
        print()
    axs[1].yaxis.set_visible(False)
    if len(bout_dict.keys())>1 :
        dataset_legend(bout_dict.keys(), cols, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
    if save_to is not None:
        # fig.savefig(f'{save_to}/comparative_bouts.eps', dpi=300)
        fig.savefig(f'{save_to}/comparative_bouts.pdf', dpi=300)
        # fig.savefig(f'{save_to}/comparative_bouts.png', dpi=300)
    if show :
        plt.show()
    # return fig, axs

def plot_comparative_dispersion(loco_dict, c,**kwargs) :
    from lib.anal.plot_aux import BasePlot, plot_mean_and_range
    from lib.aux.colsNstr import random_colors

    P = BasePlot(name='comparative_dispersal', **kwargs)
    P.build(figsize=(30, 10))
    r0, r1 = (0, 40)
    p = f'dispersion_{r0}_{r1}'
    t0, t1 = int(r0 * c.fr), int(r1 * c.fr)
    x = np.linspace(r0, r1, t1 - t0)
    # cols=N_colors(len(loco_dict))
    cols = ['black', 'lightgreen', 'green', 'red', 'darkred', 'orange', 'lightblue', 'blue', 'darkblue', 'magenta',
            'cyan', 'orange', 'purple'][:len(loco_dict)]
    lws = [8] + [3] * (len(loco_dict) - 1)
    # cols=random_colors(len(loco_dict))
    for lw, col, (lab, dic) in zip(lws, cols, loco_dict.items()):
        ddsp = get_dsp(dic['step'], p)
        mean = ddsp['median'].values[t0:t1]
        lb = ddsp['upper'].values[t0:t1]
        ub = ddsp['lower'].values[t0:t1]
        P.axs[0].fill_between(x, ub, lb, color=col, alpha=.2)
        P.axs[0].plot(x, mean, col, label=lab, linewidth=lw, alpha=1.0)
    P.conf_ax(xlab='time, $sec$', ylab=r'dispersion $(mm)$', xlim=[x[0], x[-1]], ylim=[0, None], xMaxN=4, yMaxN=4,
              leg_loc='upper left')
    P.adjust((0.2 / 1, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


if __name__ == '__main__':
    refID = 'None.100controls'
    # refID='None.Sims2019_controls'
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config
    trange = np.arange(0, c.Nticks * c.dt, c.dt)



    physics0_args = {
        'torque_coef': 1,
        'ang_damp_coef': 1,
        'body_spring_k': 1,
    }
    physics1_args = {
        'torque_coef': 0.4,
        'ang_damp_coef': 2.5,
        'body_spring_k': 0.25,
    }
    physics2_args = {
        'torque_coef': 1.77,
        'ang_damp_coef': 5.0,
        'body_spring_k': 0.39,
    }

    lat_osc_args = {
        'w_ee': 3.0,
        'w_ce': 0.1,
        'w_ec': 4.0,
        'w_cc': 4.0,
        'm': 100.0,
        'n': 2.0,
    }

    Lev = {
        run_v_mu: e[run_v_mu].mean(),
        'ang_vel_headcast': np.deg2rad(e[pau_fov_mu].mean()),  # 60,
        'run_dist': {'range': [1, 172.125],
                     'name': 'powerlaw',
                     'alpha': 1.46},
        'pause_dist': {'range': [0.4, 2.0],
                       'name': 'uniform'},
    }



    Wys = {
        run_v_mu: 0.001,  # in m/s
        'turner_input_constant': 19.0,
        "bend_correction_coef": 0,
        **lat_osc_args,
        **physics0_args
    }

    Dav = {
        run_v_mu: 0.001,
        run_t_min: 1,
        'theta_min_headcast': 37,
        'theta_max_headcast': 120,
        'theta_max_weathervane': 20,
        'ang_vel_weathervane': 60.0,
        'ang_vel_headcast': 240.0,
        'r_run2headcast': 0.148,
        'r_headcast2run': 2.0,
        'r_weathervane_stop': 2.0,
        'r_weathervane_resume': 1.0,
    }

    Sak = {
        'step_mu': 0.24,
        'step_std': 0.066,
        'initial_freq': 1.36,
        'turner_input_constant': 19.0,
        'attenuation_min': 0.2,
        'attenuation_max': 0.31,
        'max_vel_phase': 3.6,
        'stridechain_dist': c.bout_distros.run_count,
        # 'run_dist': c.bout_distros.run_dur,
        'pause_dist': c.bout_distros.pause_dur,
        "bend_correction_coef": 1.4,
        **lat_osc_args,
        **physics1_args
    }


    from lib.model.modules.locomotor import Sakagiannis2022, Levy_locomotor, Wystrach2016, Davies2015


    locos = {
       "Levy": [Levy_locomotor, Lev, False],
        "Levy+": [Levy_locomotor, Lev, True],
       "Wystrach": [Wystrach2016, Wys, False],
       "Wystrach+": [Wystrach2016, Wys, True],
       "Davies": [Davies2015, Dav, False],
        "Davies+": [Davies2015, Dav, True],
        "Sakagiannis": [Sakagiannis2022, Sak, False],
        "Sakagiannis+": [Sakagiannis2022, Sak, True],
        # "Sakagiannis++": [Sakagiannis2022, Sak2, False],
    }

    error_dict, loco_dict, bout_dict = run_locomotor_evaluation(d, locos, Nids=100, save_to='/home/panos/larvaworld_new/larvaworld/tests/metrics/model_comparison/100l')
    # error_tables(error_dict)
    # error_barplots(error_dict, normalization='minmax')
    # plot_trajectories(loco_dict)
    # plot_bouts(d)
    # plot_comparative_dispersion(loco_dict, return_fig=True)