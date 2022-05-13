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
