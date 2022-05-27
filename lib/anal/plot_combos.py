# Create composite figure
# from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import matplotlib.pyplot as plt

from lib.anal.plot_aux import BasePlot


class GridPlot(BasePlot):
    def __init__(self, name, width, height, scale=(1, 1), **kwargs):
        super().__init__(name, **kwargs)
        ws, hs = scale
        self.width, self.height = width, height
        figsize = (int(width * ws), int(height * hs))
        self.fig = plt.figure(constrained_layout=False, figsize=figsize)
        self.grid = GridSpec(height, width, figure=self.fig)
        self.cur_w, self.cur_h = 0, 0
        self.cur_idx = 0

        def bf(Q):
            return fr'$\bf{{{Q.replace("$", "")}}}$'

        self.letters = [bf(Q) for Q in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']]

    def add_letter(self, x0, y0):
        Q = self.letters[self.cur_idx]
        self.fig.text(x0, y0, Q, fontsize=30)
        self.cur_idx += 1


    def add(self, N=1, w=None, h=None, w0=None, h0=None, dw=0, dh=0, share_w=False, share_h=False, letter=True, x0=None, y0=None):
        if w0 is None:
            w0 = self.cur_w
        if h0 is None:
            h0 = self.cur_h
        if letter :
            if x0 is None:
                x0 = w0 / self.width + 0.01
            if y0 is None:
                y0 = 1 - h0 / self.height - 0.01

            self.add_letter(x0, y0)
        if w is None:
            w = self.width - w0
        if h is None:
            h = self.height - h0

        if N == 1:
            ax = self.fig.add_subplot(self.grid[h0:h0 + h, w0:w0 + w])
            return ax
        else:
            if share_h:
                ww=int((w-(N-1)*dw)/N)
                axs = [self.fig.add_subplot(self.grid[h0:h0 + h, w0 + dw*i+ww * i:w0 + dw*i+ ww * (i + 1)]) for i in range(N)]
            elif share_w:
                hh = int((h-(N-1)*dh )/ N)
                axs = [self.fig.add_subplot(self.grid[h0+ dh*i + hh * i:h0+ dh*i + hh * (i + 1), w0:w0 + w]) for i in range(N)]
            return axs

    def plot(self, func, kws, axs=None, **kwargs):
        if axs is None:
            axs = self.add(**kwargs)
        _ = func(fig=self.fig, axs=axs, **kws)


def model_summary(refID, mID, Nids=1, **kwargs):
    from lib.conf.stored.conf import loadRef
    from lib.anal.plot_aux import modelConfTable, module_endpoint_hists, test_model
    from lib.anal.fitting import test_boutGens
    from lib.anal.eval_aux import sim_model
    from lib.anal.plotting import stride_cycle

    d = loadRef(refID)
    d.load(step=False, contour=False)
    d.id = 'experiment'
    d.config.id = 'experiment'
    d.color = 'red'
    d.config.color = 'red'
    e, c = d.endpoint_data, d.config

    dd = sim_model(mID=mID, dur=c.Nticks * c.dt / 60, dt=c.dt, Nids=Nids, color='blue', dataset_id='model')

    h, w = 83, 74

    P = GridPlot(name=f'{mID}_summary', width=w, height=h, scale=(0.25, 0.3), **kwargs)


    P.plot(func=modelConfTable, kws={'mID': mID}, h=30,w0=8, x0=0.01, y0=0.95)

    valid = ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std']

    P.plot(func=module_endpoint_hists,kws={'module': 'crawler', 'valid': valid, 'save_to': None, 'e': e},
           N=len(valid), h=10, h0=33, share_h=True, dw=1)

    P.plot(func=stride_cycle,kws={'datasets': [d, dd], 'shorts': ['sv', 'fov'], 'individuals': True, 'save_to': None},
           N=2, w=29, h=20, h0=48, share_w=True, y0=0.42)

    P.plot(func=test_boutGens, kws={'mID': mID, 'refID': refID, 'save_to': None},
           N=2, w=29, h0=73, share_h=True, y0=0.19, dw=1)

    P.plot(func=test_model,kws={'mID': mID, 'include_ang_suppression': True, 'save_to': None, 'dur': 2 / 3, 'd': dd},
           N=4, w0=38, h0=48, share_w=True, x0=0.45, y0=0.42, dh=1)
    P.adjust((0.07, 0.95), (0.05, 0.95), 0.01, 0.2)
    return P.get()


def combo_plot_vel_definition(d, save_to=None, save_as='vel_definition.pdf', component_vels=True):
    from lib.anal.plotting import plot_stride_variability, plot_segmentation_definition
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


def model_summary_old(refID, mID, Nids=1, save_to=None, save_as=None):
    from lib.conf.stored.conf import loadConf, kConfDict, loadRef, copyConf

    from lib.anal.plot_aux import modelConfTable, module_endpoint_hists, test_model
    from lib.anal.fitting import test_boutGens
    from matplotlib.gridspec import GridSpec
    from lib.anal.eval_aux import sim_model
    from lib.anal.plotting import stride_cycle

    d = loadRef(refID)
    d.load(step=False, contour=False)
    d.id = 'experiment'
    d.config.id = 'experiment'
    d.color = 'red'
    d.config.color = 'red'
    e, c = d.endpoint_data, d.config
    # s, e, c = d.step_data, d.endpoint_data, d.config

    dd = sim_model(mID=mID, dur=c.Nticks * c.dt / 60, dt=c.dt, Nids=Nids, color='blue', dataset_id='model')
    # Create composite figure
    h, w = 82, 40
    hs, ws = 0.3, 0.5
    figsize = (int(w * ws), int(h * hs))

    h11 = 30
    h20 = h11 + 3
    h21 = h20 + 10

    hl30 = h21 + 5
    hl31 = hl30 + 9
    hl40 = hl31  # +1
    hl41 = hl40 + 9
    hl50 = hl41 + 5
    hl51 = hl50 + 8

    hr30 = hl30
    hr31 = hl51
    Nr3 = 4
    hr3d = int((hr31 - hr30 - (Nr3 - 1)) / Nr3)
    # hr3d=

    # Nrows = 17
    Ncols = 4
    # hh = int(h / Nrows)
    ww = int(w / Ncols)

    # Nrows1 = 5
    # h1 = Nrows1 * hh
    # h21 = int((Nrows1 + 0.5) * hh)
    # h22 = int((Nrows1 + 2.5) * hh)
    # h3 = int((Nrows1 + 3.5) * hh)
    # h4 = int((Nrows1 + 6.0) * hh)

    # hhh = int(h4 - h3)
    # h5 = h3 + 2 * hhh
    # h6 = h3 + 3 * hhh

    w2 = int((Ncols / 2) * ww)
    w3 = int((Ncols / 2) * ww)
    w4 = int((Ncols / 2) * ww)

    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs = GridSpec(h, w, figure=fig)

    ax1 = fig.add_subplot(gs[:h11, 4:-2])

    # _=modelConfTable(confID=mID, save_as=f'{save_to}/{mID}.pdf', fig=fig, ax=ax1)
    _ = modelConfTable(mID=mID, fig=fig, axs=ax1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    valid = ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std']
    Nv = len(valid)
    wNv = int(w / Nv)
    axs2 = [fig.add_subplot(gs[h20:h21, wNv * i:wNv * (i + 1)]) for i in range(Nv)]
    _ = module_endpoint_hists(module='crawler', valid=valid, save_to=None,
                              e=e, Nbins=15, show_median=True, fig=fig, axs=axs2)

    ax3 = fig.add_subplot(gs[hl30:hl31, 1:2 * ww - 4])
    ax4 = fig.add_subplot(gs[hl40:hl41, 1:2 * ww - 4])
    ax34 = [ax3, ax4]
    _ = stride_cycle(datasets=[d, dd], shorts=['sv', 'fov'], individuals=True, save_to=None, fig=fig, axs=ax34)

    # _ = stride_cycle_solo(s, e, c, short='sv', absolute=False, pooled=False,save_to=None, fig=fig, axs=ax3)
    # _ = stride_cycle_solo(s, e, c, short='fov', pooled=False,save_to=None, fig=fig, axs=ax4)
    ax3.xaxis.set_visible(False)

    ax51 = fig.add_subplot(gs[hl50:hl51, 1:ww - 2])
    ax52 = fig.add_subplot(gs[hl50:hl51, ww - 1:2 * ww - 4])

    axs5 = [ax51, ax52]
    _ = test_boutGens(mID, refID, save_to=save_to, fig=fig, axs=axs5)

    axs6 = [fig.add_subplot(gs[hr30 + i * hr3d + i:hr30 + (i + 1) * hr3d + i, 2 * ww + 1:]) for i in range(Nr3)]
    _ = test_model(mID, save_to=save_to, include_ang_suppression=True, dur=2 / 3, fig=fig, axs=axs6, d=dd)

    x0 = 0.01
    x1 = 0.46
    y0 = 0.95
    y1 = 0.58
    y2 = 0.4
    y3 = 0.28
    y4 = 0.16

    fig.text(x0, y0, r'$\bf{A}$', fontsize=30)

    fig.text(x0, y1, r'$\bf{B}$', fontsize=30)
    fig.text(x0, y2, r'$\bf{C}$', fontsize=30)
    fig.text(x0, y3, r'$\bf{D}$', fontsize=30)
    fig.text(x0, y4, r'$\bf{E}$', fontsize=30)
    fig.text(x1, y2, r'$\bf{F}$', fontsize=30)

    fig.subplots_adjust(hspace=0.2, wspace=0.01, bottom=0.02, top=0.95, left=0.07, right=0.95)
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = f'{mID}_summary'
        filename = f'{save_to}/{save_as}.pdf'
        fig.savefig(filename, dpi=300)
    plt.close()
    return fig


if __name__ == '__main__':
    # cwd = os.getcwd()
    # print(cwd)
    # raise
    save_to = '/home/panos/Dropbox/Science/Images/my/Papers/02.locomotory_model/09.calibration/model_summaries/test'
    refID = 'None.150controls'
    mID = 'best_explorer'
    _ = model_summary(refID, mID, save_to=save_to, Nids=2, show=True)
