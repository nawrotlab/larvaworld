# Create composite figure
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec




def combo_plot_vel_definition(d, save_to=None,save_as='vel_definition.pdf', component_vels=True) :
    from lib.anal.plotting import plot_stride_variability, plot_segmentation_definition
    if save_to is None:
        save_to = d.plot_dir

    h,w=10,22
    dh,dw=2,2
    h2,w2=int(h/2),int(w/2)
    fig = plt.figure(constrained_layout=False, figsize=(w+dw-3,h+dh-3))


    gs = GridSpec(h+dh, w+dw, figure=fig)

    ''' Create the linear velocity figure'''
    ax1 = fig.add_subplot(gs[:, :w2])
    _=plot_stride_variability(datasets=[d], fig=fig, axs=ax1, component_vels=component_vels)

    ''' Create the angular velocity figure'''
    ax2 = fig.add_subplot(gs[:h2, w2 + dw:])
    ax3 = fig.add_subplot(gs[h2 + dh:, w2 + dw:])
    _=plot_segmentation_definition(datasets=[d], fig=fig, axs=[ax2,ax3])

    fig.text(0.01,0.91, r'$\bf{A}$', fontsize=30)

    fig.text(0.5,0.91, r'$\bf{B}$', fontsize=30)
    fig.text(0.5,0.45, r'$\bf{C}$', fontsize=30)

    fig.subplots_adjust(hspace=0.1, wspace=0.5, bottom=0.1, top=0.9, left=0.07, right=0.95)
    fig.savefig(f'{save_to}/{save_as}', dpi=300)

def average_model_summary(refID, mID, save_to=None,save_as='average_model') :
    from lib.conf.stored.conf import loadConf, kConfDict, loadRef, copyConf

    import matplotlib.pyplot as plt
    from lib.anal.plot_aux import modelConfTable, module_endpoint_hists, test_locomotor
    from lib.anal.plotting import stride_cycle_solo
    from lib.anal.fitting import test_boutGens
    from matplotlib.gridspec import GridSpec
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config
    # Create composite figure
    h, w = 82, 40
    hs, ws = 0.3, 0.5
    figsize = (int(w * ws), int(h * hs))

    h11=30
    h20=h11+3
    h21=h20+10

    hl30=h21+5
    hl31=hl30+8
    hl40=hl31+2
    hl41=hl40+8
    hl50=hl41+5
    hl51=hl50+8

    hr30=hl30
    hr31=hl51
    Nr3=4
    hr3d=int((hr31-hr30-(Nr3-1))/Nr3)
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
    _ = modelConfTable(confID=mID, fig=fig, ax=ax1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)

    valid = ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std']
    Nv = len(valid)
    wNv = int(w / Nv)
    axs2 = [fig.add_subplot(gs[h20:h21, wNv * i:wNv * (i + 1)]) for i in range(Nv)]
    _ = module_endpoint_hists(module='crawler', valid=valid,save_to=None,
                              e=e, Nbins=15, show_median=True, fig=fig, axs=axs2)

    ax3 = fig.add_subplot(gs[hl30:hl31, 1:2 * ww - 4])
    ax4 = fig.add_subplot(gs[hl40:hl41, 1:2 * ww - 4])
    _ = stride_cycle_solo(s, e, c, short='sv', absolute=False, pooled=False,save_to=None, fig=fig, axs=ax3)
    _ = stride_cycle_solo(s, e, c, short='fov', pooled=False,save_to=None, fig=fig, axs=ax4)
    ax3.xaxis.set_visible(False)

    ax51 = fig.add_subplot(gs[hl50:hl51, 1:ww-2])
    ax52 = fig.add_subplot(gs[hl50:hl51, ww-1:2 * ww - 4])

    axs5 = [ax51, ax52]
    _ = test_boutGens(mID, refID,save_to=None, fig=fig, axs=axs5)

    axs6 = [fig.add_subplot(gs[hr30+i*hr3d+i:hr30 + (i + 1) * hr3d+i, 2 * ww + 1:]) for i in range(Nr3)]
    _ = test_locomotor(mID,save_to=None,include_ang_suppression=True, dur=40, fig=fig, axs=axs6)

    x0 = 0.01
    x1 = 0.46
    y0 = 0.95
    y1 = 0.58
    y2 = 0.42
    y3 = 0.29
    y4 = 0.16

    fig.text(x0, y0, r'$\bf{A}$', fontsize=30)

    fig.text(x0, y1, r'$\bf{B}$', fontsize=30)
    fig.text(x0, y2, r'$\bf{C}$', fontsize=30)
    fig.text(x0, y3, r'$\bf{D}$', fontsize=30)
    fig.text(x0, y4, r'$\bf{E}$', fontsize=30)
    fig.text(x1, y2, r'$\bf{F}$', fontsize=30)

    fig.subplots_adjust(hspace=0.2, wspace=0.01, bottom=0.02, top=0.95, left=0.07, right=0.95)

    if save_to is None:
        save_to=d.plot_dir
    fig.savefig(f'{save_to}/{save_as}.pdf', dpi=300)


if __name__ == '__main__':
    import os

    # cwd = os.getcwd()
    # print(cwd)
    # raise
    save_to = '/home/panos/Dropbox/Science/Images/my/Papers/02.locomotory_model/09.calibration'
    refID = 'None.150controls'
    mID = '150l_explorer'
    _=average_model_summary(refID, mID, save_to=save_to)
