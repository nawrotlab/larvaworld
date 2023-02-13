# Create composite figure
import os

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from larvaworld.lib import reg, aux, plot,util





def calibration_plot(save_to=None, files=None):
    from PIL import Image
    tick_params = {
        'axis': 'both',  # changes apply to the x-axis
        'which': 'both',  # both major and minor ticks are affected
        'bottom': False,  # ticks along the bottom edge are off
        'top': False,  # ticks along the top edge are off
        'labelbottom': False,  # labels along the bottom edge are off
        'labeltop': False,
        'labelleft': False,
        'labelright': False,
    }

    filename = 'calibration.pdf'
    fig = plt.figure(constrained_layout=True, figsize=(6 * 5, 2 * 5))
    gs = fig.add_gridspec(2, 6)
    interference = fig.add_subplot(gs[:, :2])
    bouts = fig.add_subplot(gs[0, 2:4])
    orient = fig.add_subplot(gs[0, 4:])
    angular = fig.add_subplot(gs[1, 3:])
    bend = fig.add_subplot(gs[1, 2])

    if save_to is None:
        save_to = '.'
    if files is None:
        filenames = [
            'interference/interference_orientation.png',
            'bouts/stridesNpauses_cdf_restricted_0.png',
            'stride/stride_orient_change.png',
            'turn/angular_pars_3.png',
            'stride/stride_bend_change.png'
        ]
        files = [f'{save_to}/{f}' for f in filenames]
    images = [Image.open(f) for f in files]
    axes = [interference, bouts, orient, angular, bend]
    for ax, im in zip(axes, images):
        ax.tick_params(**tick_params)
        ax.axis('off')
        ax.imshow(im, cmap=None, aspect=None)
    filepath = os.path.join(save_to, filename)
    plot.save_plot(fig, filepath, filename)
    return fig

@reg.funcs.graph('model summary')
def model_summary(mID, refID=None, refDataset=None, Nids=1, model_table=False, **kwargs):
    if refDataset is None:
        d = reg.loadRef(refID)
        d.load(step=False)
        refDataset = d
    refDataset.color = 'red'
    refDataset.config.color = 'red'
    e, c = refDataset.endpoint_data, refDataset.config

    dd = util.sim_model(mID=mID, refDataset=refDataset, dur=c.Nticks * c.dt / 60, dt=c.dt, Nids=Nids, color='blue',
                   dataset_id='model')

    if model_table:
        hh0 = 30
    else:
        hh0 = 0
    h, w = 67 + hh0, 74

    P = plot.GridPlot(name=f'{mID}_summary', width=w, height=h, scale=(0.25, 0.25), **kwargs)

    if model_table:
        P.plot(func='configuration', kws={'mID': mID}, h=hh0, w0=8, x0=True, y0=True)

    # valid = ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std']

    P.plot(func='module hists', kws={'mkey': 'crawler', 'mode': 'realistic', 'e': e, 'save_to': None},
           N=5, h=10, h0=hh0 + 3, share_h=True, dw=1, x0=True)

    shorts = ['sv', 'fov', 'foa', 'b']
    P.plot(func='stride cycle',
           kws={'datasets': [refDataset, dd], 'labels': ['experiment', dd.id], 'shorts': shorts, 'individuals': True,
                'save_to': None},
           N=len(shorts), w=29, h=32, h0=hh0 + 18, share_w=True, x0=True)

    ds = util.test_boutGens(**{'mID': mID, 'refDataset': refDataset})
    P.plot(func='epochs', kws={'datasets': ds, 'save_to': None},
           N=2, w=29, h0=hh0 + 56, share_h=True, dw=1, x0=True)

    P.plot(func='sample track', kws={'mID': mID, 'dur': 0.5, 'd': dd, 'save_to': None},
           N=5, w0=38, h0=hh0 + 18, share_w=True, dh=1)
    P.adjust((0.1, 0.95), (0.05, 0.98), 0.01, 0.2)
    P.annotate()
    return P.get()

@reg.funcs.graph('velocity definition')
def velocity_definition(d, save_to=None, save_as='vel_definition.pdf', component_vels=True, **kwargs):
    from larvaworld.lib.plot.metric import plot_segmentation_definition, plot_stride_variability

    if save_to is None:
        save_to = d.plot_dir

    h, w = 10, 22
    dh, dw = 2, 2
    h2, w2 = int(h / 2), int(w / 2)
    fig = plt.figure(constrained_layout=False, figsize=(w + dw - 3, h + dh - 3))

    gs = GridSpec(h + dh, w + dw, figure=fig)

    ''' Create the linear velocity figure'''
    ax1 = fig.add_subplot(gs[:, :w2])
    _ = plot_stride_variability(datasets=[d], fig=fig, axs=ax1, component_vels=component_vels)

    ''' Create the angular velocity figure'''
    ax2 = fig.add_subplot(gs[:h2, w2 + dw:])
    ax3 = fig.add_subplot(gs[h2 + dh:, w2 + dw:])
    _ = plot_segmentation_definition(datasets=[d], fig=fig, axs=[ax2, ax3])

    fig.text(0.01, 0.91, r'$\bf{A}$', fontsize=30)

    fig.text(0.5, 0.91, r'$\bf{B}$', fontsize=30)
    fig.text(0.5, 0.45, r'$\bf{C}$', fontsize=30)

    fig.subplots_adjust(hspace=0.1, wspace=0.5, bottom=0.1, top=0.9, left=0.07, right=0.95)
    fig.savefig(f'{save_to}/{save_as}', dpi=300)

@reg.funcs.graph('dispersal summary')
def dsp_summary(datasets, target=None, range=(0, 40), **kwargs):
    w, h = 54, 26
    P = plot.GridPlot(name=f'dsp_summary_{range}', width=w, height=h, scale=(0.4, 0.5), text_xy0=(0.05, 0.95), **kwargs)
    ds = [target] + datasets if target is not None else datasets
    Nds = len(ds)
    kws = {
        'datasets': ds,
        'save_to': None,
        'subfolder': None,
        'show': False

    }
    kws2 = {
        'dw': 0,
        'h': 8,
        'share_h': True
    }

    P.plot(func='trajectories', kws={'mode': 'origin', 'range': range, **kws}, N=Nds, x0=True, y0=True, **kws2)
    P.plot(func='dispersal', kws={'range': range, **kws}, N=1, w=16, h0=14, x0=True, **kws2)
    P.plot(func='crawl pars', kws={'pvalues': False, **kws}, N=3, w=30, w0=22, h0=14, **kws2)
    P.adjust((0.1, 0.95), (0.05, 0.9), 0.05, 0.1)
    P.annotate()
    return P.get()

@reg.funcs.graph('kinematic analysis')
def kinematic_analysis(datasets, **kwargs):
    w, h = 50, 28
    P = plot.GridPlot(name='kinematic_analysis', width=w, height=h, scale=(0.5, 0.5), text_xy0=(0.05, 0.94), **kwargs)

    kws = {
        'datasets': datasets,
        'save_to': None,
        'subfolder': None,
        'show': False

    }
    kws2 = {
        'dh': 4,
        'dw': 2,
        # 'h': 12,
        'w': int(w/2-2),

        # 'share_w': True,
    }

    kws1={
        'h': int(h/2-2),
        'x0': True,
        'share_h': True,
        **kws2
    }

    P.plot(func='fft multi',kws={**kws}, y0=True,N=1, **kws1)
    P.plot(func='epochs', kws={'plot_fits': ['powerlaw', 'exponential', 'lognormal', 'levy'], **kws},h0=int(h/2+2), N=2, **kws1)
    P.plot(func='stride cycle multi', kws={**kws}, N=2, h=h, w0=int(w/2+2),y0=True,share_w= True, **kws2)
    P.adjust((0.07, 0.95), (0.1, 0.9), 0.2, 0.1)
    P.annotate()
    return P.get()

@reg.funcs.graph('RvsS summary')
def RvsS_summary(entrylist, title, mdiff_df, **kwargs):
    h_mpl = 4
    w, h = 30, 60 + h_mpl * 2
    P = plot.GridPlot(name=f'RvsS_summary', width=w, height=h, scale=(0.7, 0.7), text_xy0=(0.05, 0.95), **kwargs)
    Nexps = len(entrylist)
    h1exp = int((h - h_mpl * 2) / Nexps)
    P.fig.text(x=0.5, y=0.98, s=title, size=35, weight='bold', horizontalalignment='center')

    P.plot(func='mpl', kws={'data': mdiff_df, 'font_size': 18}, w=w, x0=True, y0=True, h=h_mpl, w0=6, h0=0)
    ax_list = []
    for i, entry in enumerate(entrylist):
        h0 = i * h1exp + (i + 1) * 1 + h_mpl * 2
        axs = P.add(w=w, x0=True, h=h1exp - 2, h0=h0)
        P.plot(func=entry['plotID'], kws=entry['args'], axs=axs)
        axs.set_title(entry['name'], size=30, weight='bold', horizontalalignment='center', pad=15)
        ax_list.append(axs)
    P.adjust((0.1, 0.95), (0.05, 0.96), 0.05, 0.1)
    P.annotate()
    P.fig.align_ylabels(ax_list)
    return P.get()

@reg.funcs.graph('double-patch summary')
def DoublePatch_summary(datasets, title, mdiff_df,ks=None,name=None, **kwargs):
    Nmods = 2
    h_mpl = len(mdiff_df.index)
    hh_mpl = h_mpl + 4
    w, h = 32, 50 + hh_mpl
    if name is None:
        name = f'DoublePatch_summary',
    P = plot.GridPlot(name=name, width=w, height=h, scale=(0.8, 0.8), text_xy0=(0.05, 0.95), **kwargs)
    P.fig.text(x=0.5, y=0.98, s=title, size=35, weight='bold', horizontalalignment='center')
    P.plot(func='mpl', kws={'data': mdiff_df, 'font_size': 18}, w=w, x0=True, y0=True, h=h_mpl, w0=4 + int(Nmods / 2),
           h0=0)

    Nsubs = len(datasets)
    Ndds=Nmods*Nsubs
    h1exp = int(h - hh_mpl)
    h0=1 + hh_mpl
    ds=[]
    ls=[]

    for i, (subID, RnS) in enumerate(datasets.items()):
        if len(RnS)==1 :
            RnS=aux.flatten_list(RnS)
        ls+=[f'{subID}_{d.id}' for d in RnS]
        ds+=RnS
    kws1 = {
        'datasets': ds,
        'labels':ls,
        'save_to': None,
        'subfolder': None,
        'show': False,
        # 'title': False,

    }
    axs1 = P.add(w=w, x0=True, N=(3, 2), share_h=True, share_w=True, h=h1exp - 18, h0=h0, dh=3, dw=4)
    P.plot(func='double patch', kws={**kws1, 'title': False, 'ks':ks}, axs=axs1)
    P.fig.align_ylabels(axs1)
    axs2 = P.add(w=w, x0=True, N=(Nmods,Nsubs), share_h=True, share_w=True, h=16, h0=h-16,
                 dh=2, dw=1, cols_first=True)
    P.plot(func='trajectories', kws={**kws1, 'single_color': True}, axs=axs2)
    P.fig.align_ylabels(axs2)
    for ii, ax in enumerate(axs2):
        ax.yaxis.set_visible(True)
        ax.xaxis.set_visible(True)

    P.adjust((0.1, 0.95), (0.15, 0.9), 0.3, 0.2)
    P.annotate()
    return P.get()

@reg.funcs.graph('chemotaxis summary')
def chemo_summary(datasets, mdiff_df, title, **kwargs):
    Nmods = len(mdiff_df.columns)
    h_mpl = len(mdiff_df.index)
    hh_mpl = h_mpl + 4
    w, h = 30, 42 + hh_mpl
    P = plot.GridPlot(name=f'chemo_summary', width=w, height=h, scale=(0.7, 0.7), text_xy0=(0.05, 0.95), **kwargs)
    P.fig.text(x=0.5, y=0.98, s=title, size=35, weight='bold', horizontalalignment='center')
    P.plot(func='mpl', kws={'data': mdiff_df, 'font_size': 18}, w=w, x0=True, y0=True, h=h_mpl, w0=4 + int(Nmods / 2),
           h0=0)

    time_ks = ['c_odor1', 'dc_odor1']
    Nks = len(time_ks)
    Nexps = len(datasets)
    h1exp = int((h - hh_mpl) / Nexps)
    h1k = int(h1exp / (Nks + 1))
    for i, (exp, dds) in enumerate(datasets.items()):
        h0 = i * h1exp + (i + 1) * 1 + hh_mpl
        dds = aux.flatten_list(dds)
        Ndds = len(dds)
        kws1 = {
            'datasets': dds,
            'save_to': None,
            'subfolder': None,
            'show': False

        }
        axs = P.add(w=w, x0=True, N=Nks, share_w=True, dh=0, h=Nks * (h1k - 1), h0=h0)
        axs[0].set_title(exp, size=30, weight='bold', horizontalalignment='center', pad=15)
        P.plot(func='autoplot', kws={
            'ks': time_ks,
            'show_first': False,
            'individuals': False,
            'unit': 'min',
            **kws1
        }, axs=axs)
        P.plot(func='trajectories', kws={**kws1, 'single_color': True}, w=w, x0=True, N=Ndds, share_h=True, h=h1k - 2,
               h0=h0 + Nks * h1k)

    P.adjust((0.1, 0.95), (0.05, 0.95), 0.05, 0.1)
    P.annotate()
    return P.get()

@reg.funcs.graph('eval summary')
def result_summary(datasets, target, **kwargs):
    w, h = 50, 34
    P = plot.GridPlot(name=f'{target.id}_result_summary', width=w, height=h, scale=(0.5, 0.5), **kwargs)
    ds = [target] + datasets
    Nds = len(ds)
    kws = {
        'datasets': ds,
        'save_to': None,
        'subfolder': None,
        'show': False

    }
    kws2 = {
        'dw': 1,
        'h': 8,
        'share_h': True
    }

    dur = int(np.min([d.config.duration for d in ds]))
    P.plot(func='trajectories', kws={'mode': 'origin', 'range': (0, dur), **kws}, N=Nds, x0=True, y0=True, **kws2)
    P.plot(func='epochs', kws={'stridechain_duration': True, **kws}, N=2, w=18, h0=12, x0=True, **kws2)
    P.plot(func='epochs', kws={'turns': True, **kws}, N=2, w=18, h0=24, x0=True, **kws2)
    P.plot(func='crawl pars', kws={'pvalues': False, **kws}, N=3, w=28, w0=22, h0=12, **kws2)
    P.plot(func='angular pars', kws={'absolute': False, 'Npars': 3, **kws}, N=3, w=28, w0=22, h0=24, **kws2)

    P.adjust((0.1, 0.9), (0.1, 0.9), 0.1, 0.1)
    P.annotate()
    return P.get()

@reg.funcs.graph('sample track')
def test_model(mID=None, m=None, dur=2 / 3, dt=1 / 16, Nids=1, min_turn_amp=20, d=None, fig=None, axs=None, **kwargs):
    from larvaworld.lib.plot.traj import track_annotated
    if d is None:
        d = util.sim_model(mID=mID, m=m, dur=dur, dt=dt, Nids=Nids, enrichment=False)
    kws0 = aux.AttrDict({
        'datasets': [d],
        # 'labels' : [d],
    })
    s, e, c = d.step_data, d.endpoint_data, d.config
    Nticks = int(dur * 60 / dt)
    ss = s.xs(c.agent_ids[0], level='AgentID').loc[:Nticks]
    a_sv = ss[reg.getPar('sv')].values
    a_fov = ss[reg.getPar('fov')].values
    pars, labs = reg.getPar(['sv', 'c_CT', 'A_T', 'fov', 'b'], to_return=['d', 'symbol'])

    Nrows = len(pars)
    P = plot.Plot(name=f'{mID}_test', **kws0, **kwargs)
    P.build(Nrows, 1, figsize=(25, 5 * Nrows), sharex=True, axs=axs, fig=fig)
    kws1 = aux.AttrDict({
        'agent_idx': 0,
        'slice': (0, dur * 60),
        'dt': dt,
        'fig': P.fig,
        'show': False,
        **kws0
    })

    epochs = ['stride'] * 2 + ['turn'] * 3
    aas = [a_sv] * 2 + [a_fov] * 3
    a2s = [None, ss[pars[1]].values, ss[pars[2]].values, None, ss[pars[4]].values]
    extrs = [True, False, False, False, False]
    min_amps = [None] * 2 + [min_turn_amp] * 3

    for i, (p, l, ep, a, a2, extr, min_amp) in enumerate(zip(pars, labs, epochs, aas, a2s, extrs, min_amps)):
        track_annotated(epoch=ep, a=a, a2plot=a2, axs=P.axs[i], min_amp=min_amp, show_extrema=extr, ylab=l, **kws1)
        P.conf_ax(i, xvis=True if i == Nrows - 1 else False)
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.01, 0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()

@reg.funcs.graph('error summary')
def eval_summary(error_dict, evaluation, norm_mode='raw', eval_mode='pooled', **kwargs):
    label_dic = {
        '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
        'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}

    }
    labels = label_dic[eval_mode]

    w, h = 36, 56
    P = plot.GridPlot(name=f'{norm_mode}_{eval_mode}_error_summary', width=w, height=h, scale=(0.45, 0.45), **kwargs)

    P.plot(func='error barplot', kws={'error_dict': error_dict, 'evaluation': evaluation, 'labels': labels},
           N=2, share_w=True, dh=3, h=23, w=24, x0=True, y0=True)
    for i, (k, df) in enumerate(error_dict.items()):
        h0 = 28 + i * 14
        P.plot(func='error table', kws={'data': df, 'k': k, 'title' : labels[k]}, h=12, h0=h0, w=24, x0=True, N=1)
    P.adjust((0.1, 0.9), (0.05, 0.95), 0.1, 0.2)
    P.annotate()
    return P.get()
