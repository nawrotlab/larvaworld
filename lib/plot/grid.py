# Create composite figure
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from lib.conf.pars.pars import getPar
from lib.eval.evaluation import error_barplot
from lib.plot.base import GridPlot, BasePlot
from lib.plot.table import modelConfTable, error_table
from lib.plot.plot_datasets import module_endpoint_hists, plot_ang_pars, plot_crawl_pars, plot_dispersion
from lib.plot.plotting import stride_cycle, plot_stride_variability, plot_segmentation_definition, plot_trajectories, \
    plot_bouts, annotated_strideplot


def model_summary(refID, mID, Nids=1,model_table=True, **kwargs):
    from lib.conf.stored.conf import loadRef
    from lib.anal.fitting import test_boutGens
    from lib.eval.eval_aux import sim_model

    d = loadRef(refID)
    d.load(step=False, contour=False)
    d.id = 'experiment'
    d.config.id = 'experiment'
    d.color = 'red'
    d.config.color = 'red'
    e, c = d.endpoint_data, d.config

    dd = sim_model(mID=mID, refDataset=d, dur=c.Nticks * c.dt / 60, dt=c.dt, Nids=Nids, color='blue', dataset_id='model')

    if model_table :
        hh0 = 30
    else :
        hh0=0
    h, w = 67+hh0, 74


    P = GridPlot(name=f'{mID}_summary', width=w, height=h, scale=(0.25, 0.25), **kwargs)

    if model_table:
        P.plot(func=modelConfTable, kws={'mID': mID}, h=hh0,w0=8, x0=True, y0=True)

    valid = ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std']

    P.plot(func=module_endpoint_hists,kws={'module': 'crawler', 'valid': valid, 'e': e, 'save_to': None},
           N=len(valid), h=10, h0=hh0+3, share_h=True, dw=1, x0=True)

    shorts=['sv', 'fov', 'foa', 'b']
    P.plot(func=stride_cycle,kws={'datasets': [d, dd], 'shorts': shorts, 'individuals': True, 'save_to': None},
           N=len(shorts), w=29, h=32, h0=hh0+18, share_w=True, x0=True)

    P.plot(func=test_boutGens, kws={'mID': mID, 'refID': refID, 'save_to': None},
           N=2, w=29, h0=hh0+56, share_h=True, dw=1, x0=True)

    P.plot(func=test_model,kws={'mID': mID, 'dur': 0.5, 'd': dd, 'save_to': None},
           N=5, w0=38, h0=hh0+18, share_w=True, dh=1)
    P.adjust((0.1, 0.95), (0.05, 0.98), 0.01, 0.2)
    P.annotate()
    return P.get()


def combo_plot_vel_definition(d, save_to=None, save_as='vel_definition.pdf', component_vels=True):
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

def dsp_summary(datasets, target,range=(0,40), **kwargs):
    w, h = 54,26
    P = GridPlot(name=f'dsp_summary_{range}', width=w, height=h, scale=(0.4, 0.5), text_xy0=(0.05, 0.95), **kwargs)
    ds=[target]+datasets
    Nds=len(ds)
    kws = {
        'datasets': ds,
        'save_to': None,
        'subfolder' : None,
        'show': False

    }
    kws2= {
        'dw' : 0,
        'h' : 8,
        'share_h' :True
    }
    # plot_dispersion(range=(r0, r1), subfolder=None, **kws)

    P.plot(func=plot_trajectories, kws={'mode' : 'origin','range': range, **kws}, N=Nds,  x0=True, y0=True, **kws2)
    P.plot(func=plot_dispersion, kws={'range': range, **kws}, N=1, w=16, h0=14, x0=True, **kws2)
    # P.plot(func=plot_bouts, kws={'turns' : True,**kws}, N=2, w=18, h0=24,  x0=True, **kws2)

    P.plot(func=plot_crawl_pars, kws={'pvalues' : False,**kws}, N=3,w=30,w0=22, h0=14, **kws2)
    # P.plot(func=plot_ang_pars, kws={'absolute': False, 'Npars': 3, **kws}, N=3,  w=28, w0=22, h0=24, **kws2)

    P.adjust((0.1, 0.95), (0.05, 0.9), 0.05, 0.1)
    P.annotate()
    return P.get()

def result_summary(datasets, target, **kwargs):
    w, h = 50, 34
    P = GridPlot(name=f'{target.id}_result_summary', width=w, height=h, scale=(0.5, 0.5), **kwargs)
    ds=[target]+datasets
    Nds=len(ds)
    kws = {
        'datasets': ds,
        'save_to': None,
        'subfolder' : None,
        'show': False

    }
    kws2= {
        'dw' : 1,
        'h' : 8,
        'share_h' :True
    }

    dur=int(np.min([d.config.duration for d in ds]))
    # print([d.config.duration for d in ds])
    P.plot(func=plot_trajectories, kws={'mode' : 'origin','range': (0,dur), **kws}, N=Nds,  x0=True, y0=True, **kws2)
    P.plot(func=plot_bouts, kws={'stridechain_duration' : True, **kws}, N=2, w=18, h0=12, x0=True, **kws2)
    P.plot(func=plot_bouts, kws={'turns' : True,**kws}, N=2, w=18, h0=24,  x0=True, **kws2)

    P.plot(func=plot_crawl_pars, kws={'pvalues' : False,**kws}, N=3,w=28,w0=22, h0=12, **kws2)
    P.plot(func=plot_ang_pars, kws={'absolute': False, 'Npars': 3, **kws}, N=3,  w=28, w0=22, h0=24, **kws2)

    P.adjust((0.1, 0.9), (0.1, 0.9), 0.1, 0.1)
    P.annotate()
    return P.get()


def test_model(mID=None, m=None, dur=2 / 3, dt=1 / 16, Nids=1, min_turn_amp=20, d=None, fig=None, axs=None, **kwargs):
    if d is None:
        from lib.eval.eval_aux import sim_model
        d = sim_model(mID=mID, m=m, dur=dur, dt=dt, Nids=Nids, enrichment=False)
    s, e, c = d.step_data, d.endpoint_data, d.config

    Nticks = int(dur * 60 / dt)
    trange = np.arange(0, Nticks * dt, dt)
    ss = s.xs(c.agent_ids[0], level='AgentID').loc[:Nticks]

    pars, labs = getPar(['v', 'c_CT', 'Act_tur', 'fov', 'b'], to_return=['d', 'symbol'])

    Nrows = len(pars)
    P = BasePlot(name=f'{mID}_test', **kwargs)
    P.build(Nrows, 1, figsize=(25, 5 * Nrows), sharex=True, fig=fig, axs=axs)
    a_v = ss[getPar('v')].values
    a_fov = ss[getPar('fov')].values
    annotated_strideplot(a_v, dt, ax=P.axs[0])
    annotated_strideplot(a_v, dt, a2plot=ss[pars[1]].values, ax=P.axs[1], ylim=(0, 1), show_extrema=False)

    annotated_turnplot(a_fov, dt, a2plot=ss[pars[2]].values, ax=P.axs[2], min_amp=min_turn_amp)
    annotated_turnplot(a_fov, dt, ax=P.axs[3], min_amp=min_turn_amp)
    annotated_turnplot(a_fov, dt, a2plot=ss[pars[4]].values, ax=P.axs[4], min_amp=min_turn_amp)

    for i in range(Nrows):
        P.conf_ax(i, xlim=(0, trange[-1] + 10 * dt), ylab=labs[i], xlab='time (sec)',
                  xvis=True if i == Nrows - 1 else False)
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.01, 0.05)
    P.fig.align_ylabels(P.axs[:])
    return P.get()


def eval_summary(error_dict, evaluation, norm_mode='raw', eval_mode='pooled',**kwargs):

    label_dic = {
        '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
        'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}

    }
    labels=label_dic[eval_mode]


    w,h=36,56
    P = GridPlot(name=f'{norm_mode}_{eval_mode}_error_summary', width=w, height=h, scale=(0.45, 0.45), **kwargs)

    P.plot(func=error_barplot, kws={'error_dict': error_dict, 'evaluation' : evaluation,'labels' : labels},
           N=2,share_w=True, dh=3, h=23,w=24, x0=True, y0=True)
    for i, (k, df) in enumerate(error_dict.items()):
        h0 = 28 + i * 14
        P.plot(func=error_table, kws={'data': df,'k' : k,  'bbox': [0.5, 0, 1, 1]}, h=12,h0=h0,w=24, x0=True)
    P.adjust((0.1, 0.9), (0.05, 0.95), 0.1, 0.2)
    P.annotate()
    return P.get()
