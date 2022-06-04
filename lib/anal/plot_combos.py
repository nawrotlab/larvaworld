# Create composite figure
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
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
        # self.cur_idx = 0
        # self.text_x0, self.text_y0=0.05, 0.98
        # self.letters=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        # self.letter_dict={}
        # self.x0s, self.y0s=[],[]

    # def annotate(self, dx=-0.05, dy=0.005):
    #     for i, (ax, text) in enumerate(self.letter_dict.items()) :
    #
    #         X = self.text_x0 if ax in self.x0s else ax.get_position().x0+dx
    #         Y = self.text_y0 if ax in self.y0s else ax.get_position().y1+dy
    #         # if i in x0 :
    #         #     X=self.text_x0
    #         # if i in y0 :
    #         #     Y=self.text_y0
    #         self.fig.text(X, Y, text, size=30, weight='bold')


    def add(self, N=1, w=None, h=None, w0=None, h0=None, dw=0, dh=0, share_w=False, share_h=False, letter=True, x0=False, y0=False):


        if w0 is None:
            w0 = self.cur_w
        if h0 is None:
            h0 = self.cur_h

        if w is None:
            w = self.width - w0
        if h is None:
            h = self.height - h0

        if N == 1:
            axs = self.fig.add_subplot(self.grid[h0:h0 + h, w0:w0 + w])
            ax_letter=axs
            # if letter:
            #     self.letter_dict[axs]=self.letters[self.cur_idx]
            #     self.cur_idx += 1
            # return axs
        else:
            if share_h:
                ww=int((w-(N-1)*dw)/N)
                axs = [self.fig.add_subplot(self.grid[h0:h0 + h, w0 + dw*i+ww * i:w0 + dw*i+ ww * (i + 1)]) for i in range(N)]
            elif share_w:
                hh = int((h-(N-1)*dh )/ N)
                axs = [self.fig.add_subplot(self.grid[h0+ dh*i + hh * i:h0+ dh*i + hh * (i + 1), w0:w0 + w]) for i in range(N)]
            ax_letter = axs[0]
        self.add_letter(ax_letter,letter, x0=x0, y0=y0)
        return axs

    def plot(self, func, kws, axs=None, **kwargs):
        if axs is None:
            axs = self.add(**kwargs)
        _ = func(fig=self.fig, axs=axs, **kws)


def model_summary(refID, mID, Nids=1,model_table=True, **kwargs):
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

def dsp_summary(datasets, target,range=(0,40), **kwargs):
    from lib.anal.plotting import plot_trajectories,plot_crawl_pars,plot_dispersion
    w, h = 54,26
    P = GridPlot(name=f'dsp_summary_{range}',  width=w, height=h, scale=(0.4, 0.5),text_xy0=(0.05, 0.95), **kwargs)
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
    from lib.anal.plotting import plot_trajectories,plot_bouts,plot_crawl_pars,plot_ang_pars
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
    print([d.config.duration for d in ds])
    P.plot(func=plot_trajectories, kws={'mode' : 'origin','range': (0,dur), **kws}, N=Nds,  x0=True, y0=True, **kws2)
    P.plot(func=plot_bouts, kws={'stridechain_duration' : True, **kws}, N=2, w=18, h0=12, x0=True, **kws2)
    P.plot(func=plot_bouts, kws={'turns' : True,**kws}, N=2, w=18, h0=24,  x0=True, **kws2)

    P.plot(func=plot_crawl_pars, kws={'pvalues' : False,**kws}, N=3,w=28,w0=22, h0=12, **kws2)
    P.plot(func=plot_ang_pars, kws={'absolute': False, 'Npars': 3, **kws}, N=3,  w=28, w0=22, h0=24, **kws2)

    P.adjust((0.1, 0.9), (0.1, 0.9), 0.1, 0.1)
    P.annotate()
    return P.get()




if __name__ == '__main__':
    # cwd = os.getcwd()
    # print(cwd)
    # raise
    save_to = '/home/panos/Dropbox/Science/Images/my/Papers/02.locomotory_model/09.calibration/model_summaries/test99'
    refID = 'None.150controls'
    mID = 'NEU_PHI'

    # _=modelConfTable(mID, save_to=save_to, save_as=None)
    _ = model_summary(refID, mID, save_to=save_to, Nids=3, show=True, save_as='average_model3.pdf', model_table=False)