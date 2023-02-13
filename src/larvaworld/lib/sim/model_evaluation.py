import os
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


from larvaworld.lib import reg, aux, plot, util
from larvaworld.lib.sim.run_template import BaseRun



class EvalRun(BaseRun):
    def __init__(self,**kwargs):
        super().__init__(runtype = 'Eval',experiment='evaluation', **kwargs)


    def setup(self, refID=None,modelIDs=None, dataset_ids=None,offline=False,
                 norm_modes=['raw', 'minmax'], eval_modes=['pooled'],eval_metrics=None, N=5, dur=None,
                 enrichment=True,show=False):

        self.refID = refID
        self.modelIDs = modelIDs

        if dataset_ids is None:
            dataset_ids = self.modelIDs
        self.dataset_ids = dataset_ids
        self.eval_modes = eval_modes
        self.norm_modes = norm_modes
        self.offline = offline
        self.show = show
        self.figs = aux.AttrDict({'errors': {}, 'hist': {}, 'boxplot': {}, 'stride_cycle': {}, 'loco': {}, 'epochs': {},
                                  'models': {'table': {}, 'summary': {}}})




        self.N = N

        self.target = self.define_target(self.refID)
        self.evaluation, self.target_data = util.arrange_evaluation(self.target, eval_metrics)
        self.define_eval_args(self.evaluation)
        self.datasets = []
        self.error_dicts = {}
        self.error_plot_dir=f'{self.plot_dir}/errors'



        self.enrichment = enrichment
        self.exp_conf = self.prepare_exp_conf(dur=dur)


    def define_target(self, refID):
        target = reg.loadRef(refID)
        target.id = 'experiment'
        target.config.id = 'experiment'

        target.color = 'grey'
        target.config.color = 'grey'

        self.mID_colors = aux.N_colors(len(self.dataset_ids))
        self.model_colors = dict(zip(self.dataset_ids, self.mID_colors))

        return target


    def define_eval_args(self, ev):
        self.e_shorts = aux.flatten_list(ev['end']['shorts'].values.tolist())
        self.s_shorts = aux.flatten_list(ev['step']['shorts'].values.tolist())
        self.s_pars = aux.flatten_list(ev['step']['pars'].values.tolist())
        s_symbols = aux.flatten_list(ev['step']['symbols'].values.tolist())
        self.e_pars = aux.flatten_list(ev['end']['pars'].values.tolist())
        e_symbols = aux.flatten_list(ev['end']['symbols'].values.tolist())
        self.eval_symbols = aux.AttrDict(
            {'step': dict(zip(self.s_pars, s_symbols)), 'end': dict(zip(self.e_pars, e_symbols))})

        self.tor_durs, self.dsp_starts, self.dsp_stops = util.torsNdsps(self.s_pars + self.e_pars)


    def analyze(self):
        self.run_evaluation(self.target)
        self.plot_eval()

    def prepare_exp_conf(self, dur=None, video=False):
        exp = 'dispersion'
        exp_conf = reg.loadConf(id=exp, conftype='Exp')
        c = self.target.config
        if dur is None:
            dur = c.Nticks * c.dt / 60

        self.dur = dur

        exp_conf.sim_params.timestep=c.dt
        exp_conf.sim_params.duration=dur



        if self.enrichment is None:
            exp_conf.enrichment = None
        else:
            tor_durs, dsp_starts, dsp_stops = util.torsNdsps(self.s_pars + self.e_pars)
            exp_conf.enrichment.metric_definition.dispersion.dsp_starts = dsp_starts
            exp_conf.enrichment.metric_definition.dispersion.dsp_stops = dsp_stops
            exp_conf.enrichment.metric_definition.tortuosity.tor_durs = tor_durs

        if video:
            vis_kwargs = reg.get_null('visualization', mode='video', video_speed=60)
        else:
            vis_kwargs = reg.get_null('visualization', mode=None)

        kws = aux.AttrDict({
            'enrichment': exp_conf.enrichment,
            'vis_kwargs': vis_kwargs,
            'save_to': self.save_to,
            'store_data': self.store_data,
            'experiment': exp,
            'id': self.id,
            'sim_params': exp_conf.sim_params,
            'trials': {},
            'collections': ['pose'],
            'env_params': c.env_params,
            'larva_groups': reg.lgs(sample=self.refID, mIDs=self.modelIDs, ids=self.dataset_ids,
                                    cs=self.mID_colors,
                                    expand=True, N=self.N)})

        return kws

    def store(self):
        if self.store_data:
            aux.save_dict(self.error_dicts, f'{self.data_dir}/error_dicts.txt')
            reg.vprint(f'Results saved at {self.data_dir}')

    def run_evaluation(self, d, suf='fitted', min_size=20):
        print('Evaluating all models')
        for mode in self.eval_modes:
            k = f'{mode}_{suf}'
            self.error_dicts[k] = util.eval_fast(self.datasets, self.target_data, self.eval_symbols, mode=mode,
                                            min_size=min_size)
        self.error_dicts = aux.AttrDict(self.error_dicts)

    def plot_eval(self, suf='fitted'):
        for mode in self.eval_modes:
            k = f'{mode}_{suf}'
            self.figs.errors[k] = self.get_error_plots(self.error_dicts[k], mode)

    def get_error_plots(self, error_dict, mode='pooled'):
        GD = reg.graphs.dict
        label_dic = {
            '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
            'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}

        }
        labels = label_dic[mode]
        dic = {}
        for norm in self.norm_modes:
            error_dict0 = self.norm_error_dict(error_dict, mode=norm)
            df0 = pd.DataFrame.from_dict({k: df.mean(axis=1) for i, (k, df) in enumerate(error_dict0.items())})
            kws = {
                'save_to': f'{self.error_plot_dir}/{norm}',
                'show' : self.show
            }

            bars = {}
            tabs = {}


            for i, (k, df) in enumerate(error_dict0.items()):
                tabs[k] = GD['error table'](df, k, labels[k], **kws)
            tabs['mean'] = GD['error table'](df0, 'mean', 'average error', **kws)
            bars['full'] = GD['error barplot'](error_dict=error_dict0, evaluation=self.evaluation, labels=labels, **kws)

            # Summary figure with barplots and tables for both endpoint and timeseries metrics
            bars['summary'] = GD['error summary'](norm_mode=norm, eval_mode=mode, error_dict=error_dict0,
                                                  evaluation=self.evaluation, **kws)
            dic[norm] = {'tables': tabs, 'barplots': bars}
        return aux.AttrDict(dic)

    def norm_error_dict(self, error_dict, mode='raw'):
        dic = {}
        for k, df in error_dict.items():
            if mode == 'raw':
                df = df
            elif mode == 'minmax':
                df = pd.DataFrame(MinMaxScaler().fit(df).transform(df), index=df.index, columns=df.columns)
                # df = minmax(df)
            elif mode == 'std':
                df = pd.DataFrame(StandardScaler().fit(df).transform(df), index=df.index, columns=df.columns)
                # df = std_norm(df)
            dic[k] = df
        return aux.AttrDict(dic)

    def plot_models(self):
        GD = reg.graphs.dict
        save_to = self.plot_dir
        for mID in self.modelIDs:
            self.figs.models.table[mID] = GD['model table'](mID=mID, save_to=save_to, figsize=(14, 11))
            self.figs.models.summary[mID] = GD['model summary'](mID=mID, save_to=save_to, refID=self.refID)

    def plot_results(self, plots=['hists', 'trajectories', 'dispersion', 'bouts', 'fft', 'boxplots']):
        GD = reg.graphs.dict

        print('Generating comparative graphs')

        self.target.load(h5_ks=['epochs', 'angular', 'dspNtor'])
        ds = [self.target] + self.datasets

        kws = {
            'datasets': ds,
            'save_to': self.plot_dir,
            'show': self.show
        }
        kws2 = {
            'target': self.target,
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': self.show
        }
        self.figs.summary = GD['eval summary'](**kws2)

        self.figs.stride_cycle.norm = GD['stride cycle'](shorts=['sv', 'fov', 'rov', 'foa', 'b'], individuals=True,
                                                         **kws)
        if 'dispersion' in plots:
            for r0, r1 in itertools.product(self.dsp_starts, self.dsp_stops):
                k = f'dsp_{r0}_{r1}'
                fig1 = GD['dispersal'](range=(r0, r1), subfolder=None, **kws)
                fig2 = GD['trajectories'](name=f'traj_{r0}_{r1}', range=(r0, r1), subfolder=None, mode='origin', **kws)
                fig3 = GD['dispersal summary'](range=(r0, r1), **kws2)
                self.figs.loco[k] = aux.AttrDict({'plot': fig1, 'traj': fig2, 'summary': fig3})
        if 'bouts' in plots:
            self.figs.epochs.turn = GD['epochs'](turns=True, **kws)
            self.figs.epochs.runNpause = GD['epochs'](stridechain_duration=True, **kws)
        if 'fft' in plots:
            self.figs.loco.fft = GD['fft multi'](**kws)
        if 'hists' in plots:
            self.figs.hist.ang = GD['angular pars'](half_circles=False, absolute=False, Nbins=100, Npars=3,
                                                    include_rear=False, subfolder=None, **kws)
            self.figs.hist.crawl = GD['crawl pars'](subfolder=None, pvalues=False, **kws)
        if 'trajectories' in plots:
            self.figs.loco.trajectories = GD['trajectories'](subfolder=None, **kws)
        if 'boxplots' in plots:
            pass
            # self.figs.boxplot.end = self.plot_data(mode='end', type='box')
            # self.figs.boxplot.step = self.plot_data(mode='step', type='box')




    def preprocess(self,ds):
        Ddata, Edata = {}, {}
        for p, sh in zip(self.s_pars, self.s_shorts):
            Ddata[p] = {d.id: reg.par_dict.get(sh, d) for d in ds}
        for p, sh in zip(self.e_pars, self.e_shorts):
            Edata[p] = {d.id: reg.par_dict.get(sh, d) for d in ds}
        return aux.AttrDict({'step': Ddata, 'end': Edata})

    def plot_data(self, Nbins=None, mode='step', type='hist', in_mm=[]):
        self.sim_data = self.preprocess()
        if mode == 'step':
            if Nbins is None:
                Nbins = 100
            data0 = self.target_data.step
            pars = self.s_pars
            shorts = self.s_shorts
            symbols = self.eval_symbols.step
            sim_data = self.sim_data.step
        elif mode == 'end':
            if Nbins is None:
                Nbins = 20
            data0 = self.target_data.end
            pars = self.e_pars
            shorts = self.e_shorts
            symbols = self.eval_symbols.end
            sim_data = self.sim_data.end

        filename = f'{mode}_{type}'
        Nps = len(pars)
        Ncols = 4
        Nrows = int(np.ceil(Nps / Ncols))
        if type == 'hist':
            sharex = False
            sharey = True
        elif type == 'box':
            sharex = True
            sharey = False

        P = plot.AutoPlot(name=filename, subfolder=None, Nrows=Nrows, Ncols=Ncols, figsize=(5 * Ncols, 8 * Nrows),
                     sharex=sharex, sharey=sharey, show=self.show, save_to=self.plot_dir,
                     datasets=[self.target] + self.datasets)

        if type == 'hist':

            for i, (p, sym) in enumerate(symbols.items()):
                vs0 = data0[p].values
                ws0 = np.ones_like(vs0) / float(len(vs0))
                vmin, vmax = np.quantile(vs0, q=0.01), np.quantile(vs0, q=0.99)
                bins = np.linspace(vmin, vmax, Nbins)
                col0 = self.target.color
                _ = P.axs[i].hist(vs0, bins=bins, weights=ws0, label='experiment', color=col0, alpha=0.5)
                for id, df in sim_data[p].items():
                    vs = df.values
                    ws = np.ones_like(vs) / float(len(vs))
                    col = self.model_colors[id]
                    _ = P.axs[i].hist(vs, bins=bins, weights=ws, label=id, color=col,
                                      histtype='step', linewidth=3, facecolor=col, edgecolor=col, fill=False,
                                      alpha=0.5)
                P.axs[i].set_xlabel(sym, fontsize=20)
                if i % Ncols == 0:
                    P.axs[i].set_ylabel('probability', fontsize=20)
            P.axs[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            P.adjust(W=0.01, H=0.5)

        elif type == 'box':
            data = aux.concat_datasets(dict(zip(P.labels, P.datasets)), key=mode)
            palette = dict(zip(P.labels, P.colors))
            for sh in in_mm:
                data[reg.getPar(sh)] *= 1000

            for i, (p, sym) in enumerate(symbols.items()):
                kws = {
                    'x': "DatasetID",
                    'y': p,
                    'palette': P.colors,
                    'hue': None,
                    'data': data,
                    'ax': P.axs[i],
                    'width': 1.0,
                    'fliersize': 3,
                    'whis': 1.5,
                    'linewidth': None
                }
                g1 = sns.boxplot(**kws)
                plot.annotate_plot(show_ns=False, target_only=self.target.id, **kws)
                P.conf_ax(i, xticklabelrotation=30, ylab=sym, yMaxN=4, xvis=False if i < (Nrows - 1) * Ncols else True)

            P.adjust((0.1, 0.95), (0.15, 0.9), 0.5, 0.1)
            P.fig.align_ylabels(P.axs[:])
        return P.get()

    def simulate(self):
        self.setup(**self._setup_kwargs)
        c = self.target.config
        if self.offline:
            print(f'Simulating offline {len(self.dataset_ids)} models : {self.dataset_ids} with {self.N} larvae each')
            self.datasets += util.sim_models(mIDs=self.modelIDs, dur=self.dur, dt=c.dt, tor_durs=self.tor_durs,
                                        dsp_starts=self.dsp_starts, dsp_stops=self.dsp_stops,
                                        dataset_ids=self.dataset_ids,
                                        enrichment=self.enrichment,
                                        Nids=self.N, colors=list(self.model_colors.values()), env_params=c.env_params,
                                        refDataset=self.target, data_dir=self.data_dir)
        else:
            from larvaworld.lib.sim.single_run import ExpRun
            print(f'Simulating {len(self.dataset_ids)} models : {self.dataset_ids} with {self.N} larvae each')
            run = ExpRun(parameters = self.exp_conf)
            run.simulate()
            self.datasets += run.datasets

        if self.store_data:
            os.makedirs(self.data_dir, exist_ok=True)
            self.store()

        os.makedirs(self.plot_dir, exist_ok=True)
        self.analyze()
        return self.datasets


def eval_model_graphs(refID, mIDs, dIDs=None, id=None, save_to=None, N=10, enrichment=True, offline=False, dur=None,
                      **kwargs):
    if id is None:
        id = f'{len(mIDs)}mIDs'
    if dIDs is None:
        dIDs = mIDs
    if save_to is None:
        save_to = reg.datapath('evaluation', reg.loadConf('Ref',refID).dir)
    evrun = EvalRun(refID=refID, id=id, modelIDs=mIDs, dataset_ids=dIDs, N=N,dur=dur,
                    save_to=save_to,enrichment=enrichment, show=False, offline=offline, **kwargs)
    evrun.simulate()
    evrun.plot_models()
    evrun.plot_results()
    return evrun


def add_var_mIDs(refID, e=None, c=None, mID0s=None, mIDs=None, sample_ks=None):
    if e is None or c is None:
        d = reg.loadRef(refID)
        d.load(step=False)
        e, c = d.endpoint_data, d.config

    if mID0s is None:
        mID0s = list(c.modelConfs.average.keys())
    if mIDs is None:
        mIDs = [f'{mID0}_var' for mID0 in mID0s]
    if sample_ks is None:
        sample_ks = [
            'brain.crawler_params.stride_dst_mean',
            'brain.crawler_params.stride_dst_std',
            'brain.crawler_params.max_scaled_vel',
            'brain.crawler_params.max_vel_phase',
            'brain.crawler_params.initial_freq',
        ]
    kwargs = {k: 'sample' for k in sample_ks}
    entries = {}
    for mID0, mID in zip(mID0s, mIDs):
        m0 = reg.loadConf(id=mID0, conftype='Model').get_copy()
        m = m0.update_existingnestdict(kwargs)
        reg.saveConf(conf=m, id=mID, conftype='Model')
        entries[mID] = m
    return entries

def adapt_6mIDs(refID, e=None, c=None):
    if e is None or c is None:
        d = reg.loadRef(refID)
        d.load(step=False)
        e, c = d.endpoint_data, d.config

    fit_kws = {
        'eval_metrics': {
            'angular kinematics': ['b', 'fov', 'foa'],
            'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                     'dsp_0_40_max', 'dsp_0_60_max'],
            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
        },
        'cycle_curves': ['fov', 'foa', 'b']
    }

    fit_dict = util.GA_optimization(refID, fitness_target_kws=fit_kws)
    entries = {}
    mIDs = []
    for Tmod in ['NEU', 'SIN']:
        for Ifmod in ['PHI', 'SQ', 'DEF']:
            mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
            mID = f'{Ifmod}on{Tmod}'
            entry = reg.model.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                   space_mkeys=['turner', 'interference'],
                                   fit_dict=fit_dict)
            entries.update(entry)
            mIDs.append(mID)
    return entries, mIDs

def adapt_3modules(refID, e=None, c=None):
    if e is None or c is None:
        d = reg.loadRef(refID)
        d.load(step=False)
        e, c = d.endpoint_data, d.config

    fit_kws = {
        'eval_metrics': {
            'angular kinematics': ['b', 'fov', 'foa'],
            'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                     'dsp_0_40_max', 'dsp_0_60_max'],
            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
        },
        'cycle_curves': ['fov', 'foa', 'b']
    }

    fit_dict = util.GA_optimization(refID, fitness_target_kws=fit_kws)
    entries = {}
    mIDs = []
    for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
        for Tmod in ['NEU', 'SIN', 'CON']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'{Cmod}_{Tmod}_{Ifmod}_DEF'
                mID = f'{mID0}_fit'
                entry = reg.model.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                       space_mkeys=['crawler', 'turner', 'interference'],
                                       fit_dict=fit_dict)
                entries.update(entry)
                mIDs.append(mID)
    return entries, mIDs


def modelConf_analysis(d, avgVSvar=False, mods3=False):
    warnings.filterwarnings('ignore')
    c = d.config
    e = d.endpoint_data
    refID = c.refID
    if 'modelConfs' not in c.keys():
        c.modelConfs = aux.AttrDict({'average': {}, 'variable': {}, 'individual': {}, '3modules': {}})
    if avgVSvar:
        entries_avg, mIDs_avg = adapt_6mIDs(refID=c.refID, e=e, c=c)
        c.modelConfs.average = entries_avg

        reg.graphs.store_model_graphs(mIDs_avg, d.dir)
        eval_model_graphs(refID, mIDs=mIDs_avg, norm_modes=['raw', 'minmax'], id='6mIDs_avg', N=10)

        entries_var = add_var_mIDs(refID=c.refID, e=e, c=c,mID0s=mIDs_avg)
        mIDs_var = list(entries_var.keys())
        c.modelConfs.variable = entries_var
        eval_model_graphs(refID, mIDs=mIDs_var, norm_modes=['raw', 'minmax'], id='6mIDs_var', N=10)
        eval_model_graphs(refID, mIDs=mIDs_avg[:3] + mIDs_var[:3], norm_modes=['raw', 'minmax'], id='3mIDs_avgVSvar1',
                          N=10)
        eval_model_graphs(refID, mIDs=mIDs_avg[3:] + mIDs_var[3:], norm_modes=['raw', 'minmax'], id='3mIDs_avgVSvar2',
                          N=10)
    if mods3:
        entries_3m, mIDs_3m = adapt_3modules(refID=c.refID, e=e, c=c)
        c.modelConfs['3modules'] = entries_3m
        reg.graphs.store_model_graphs(mIDs_3m, d.dir)

        dIDs = ['NEU', 'SIN', 'CON']
        for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mIDs = [f'{Cmod}_{Tmod}_{Ifmod}_DEF_fit' for Tmod in dIDs]
                id = f'Tmod_variable_Cmod_{Cmod}_Ifmod_{Ifmod}'
                eval_model_graphs(mIDs=mIDs, dIDs=dIDs, norm_modes=['raw', 'minmax'], id=id, N=10)
    d.config = c
    d.save_config()


if __name__ == '__main__':
    refID = 'None.150controls'
    # mIDs = ['NEU_PHI', 'NEU_PHIx', 'PHIonSIN', 'PHIonSINx']
    mIDs = ['PHIonNEU', 'SQonNEU', 'PHIonSIN', 'SQonSIN']
    dataset_ids = mIDs
    # # dataset_ids = ['NEU mean', 'NEU var', 'SIN mean', 'SIN var']
    id = '4xee33e'
    #
    evrun = EvalRun(refID=refID, modelIDs=mIDs, dataset_ids=dataset_ids, offline=False)

    #
    # evrun.simulate(video=False)
    # evrun.eval()
    # evrun.plot_models()
    # evrun.plot_results()
