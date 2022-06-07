import itertools
import os
import time

import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt

from lib.anal.eval_aux import sim_dataset, enrich_dataset, arrange_evaluation, prepare_sim_dataset, \
    prepare_dataset, prepare_validation_dataset, torsNdsps, eval_fast, sim_models, RSS_dic
from lib.aux import naming as nam, dictsNlists as dNl
from lib.anal.plot_aux import modelConfTable, annotate_plot, AutoPlot
from lib.anal.plot_combos import model_summary, result_summary, dsp_summary
from lib.anal.plotting import plot_trajectories, plot_dispersion, plot_ang_pars, stride_cycle, plot_bouts, \
    plot_fft_multi, plot_crawl_pars
from lib.aux.colsNstr import N_colors, col_df


from lib.conf.base.dtypes import null_dict
from lib.conf.base.pars import RefParDict
from lib.conf.base.opt_par import getPar
from lib.conf.stored.conf import loadRef, expandConf, next_idx

par_dict = RefParDict().par_dict


class EvalRun :
    def __init__(self,refID,id=None,expVSsimIDs=False, eval_metrics=None,save_to=None,N=None,bout_annotation=True,
                 locomotor_models=None,modelIDs=None,dataset_ids=None, cross_validation=False,mode='load',
                 norm_modes = ['raw', 'minmax', 'std'],eval_modes=['pooled'],offline=False, store_data=True, show=False) :
        if id is None :
            id = f'evaluation_run_{next_idx("dispersion", "Eval")}'
        self.id=id
        if save_to is None:
            from lib.conf.base import paths
            save_to = paths.path("SIM")
        self.path=f'eval_runs'
        self.bout_annotation = bout_annotation
        self.save_to = save_to
        self.store_data = store_data
        self.label_dic = {
            '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
            'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}

        }

        self.refID=refID
        self.locomotor_models=locomotor_models
        self.modelIDs=modelIDs
        self.dataset_ids=dataset_ids
        if self.dataset_ids is None :
            self.dataset_ids = self.modelIDs
        self.expVSsimIDs=expVSsimIDs
        self.eval_modes=eval_modes
        self.norm_modes=norm_modes
        self.cross_validation=cross_validation
        self.offline=offline

        dir = f'{save_to}/{self.path}/{self.id}'

        self.define_paths(dir)



        self.loco_dict = {}
        self.figs = dNl.AttrDict.from_nested_dicts({'errors':{}, 'hist':{}, 'boxplot':{}, 'stride_cycle' : {}, 'loco' : {},'epochs' : {}, 'models' : {}})
        self.refDataset=loadRef(refID)
        self.refDataset.pooled_epochs = self.refDataset.load_pooled_epochs()
        self.N=N
        self.show=show
        self.target, self.target_val=self.define_target(self.refDataset,N)

        if mode=='load' :
            try :
                self.target_data = self.load_data('target_data')
                self.evaluation = self.load_data('evaluation')
                print('Loaded existing target data and evaluation metrics')


            except :
                self.evaluation, self.target_data = self.build_evaluation(eval_metrics)
                if self.store_data:
                    dNl.save_dict(self.target_data, self.dir_dict.target_data)
                    dNl.save_dict(self.evaluation, self.dir_dict.evaluation)
                print('Created novel target data and evaluation metrics')


        # if mode=='load' :
            try :
                from lib.stor.larva_dataset import LarvaDataset
                self.dataset_configs = self.load_data('dataset_configs')
                self.datasets = [LarvaDataset(**c,load_data=False) for id, c in self.dataset_configs.items()]
                print('Loaded existing datasets')
            except :
                self.datasets = []
                self.dataset_configs = {}


        # if mode=='load' :
            try :
                self.sim_data = self.load_data('sim_data')
            except :
                self.sim_data = None

        # if mode == 'load':
            try:
                self.error_dicts = self.load_data('error_dicts')

            except:
                self.error_dicts = {}

        self.define_eval_args(self.evaluation)


    def define_paths(self, dir):
        self.dir = dir
        self.data_dir = os.path.join(dir, 'data')
        self.plot_dir = os.path.join(dir, 'plots')
        self.error_dir = os.path.join(dir, 'errors')
        dir_dict = {
            'parent': self.dir,
            'data': self.data_dir,
            'models': os.path.join(dir, 'models'),
            'plot': self.plot_dir,
            'error': self.error_dir,
            **{f'error_{norm}' : os.path.join(self.error_dir, norm) for norm in self.norm_modes},
            'dataset_configs': os.path.join(self.data_dir, 'dataset_configs.txt'),
            'evaluation': os.path.join(self.error_dir, 'evaluation.txt'),
            'error_dicts': os.path.join(self.error_dir, 'error_dicts.txt'),
            'sim_data': os.path.join(self.data_dir, 'sim_data.txt'),
            'target_data': os.path.join(self.data_dir, 'target_data.txt'),
        }
        self.dir_dict = dNl.AttrDict.from_nested_dicts(dir_dict)
        for k in ['plot', 'data', 'error', 'models']+ [f'error_{norm}'  for norm in self.norm_modes]:
            os.makedirs(self.dir_dict[k], exist_ok=True)

    def define_target(self, d, N):
        target_val=None
        # pooled_epochs = d.load_pooled_epochs()

        if self.locomotor_models is not None and self.modelIDs is None :
            d.load(contour=False)
            s, e, c = prepare_dataset(d, N)
            target = dNl.AttrDict.from_nested_dicts(
                {'step_data': s, 'endpoint_data': e, 'config': c, 'pooled_epochs': d.pooled_epochs})
            if self.cross_validation and N <= d.config.N / 2:
                s_val, e_val, c_val = prepare_validation_dataset(d, N)
                target_val = dNl.AttrDict.from_nested_dicts(
                    {'step_data': s_val, 'endpoint_data': e_val, 'config': c_val, 'pooled_epochs': d.pooled_epochs})

            self.sim_mode='loco'
        elif self.locomotor_models is  None and self.modelIDs is not None:
            target = d
            target.id = 'experiment'
            target.config.id = 'experiment'

            target.color = 'grey'
            target.config.color = 'grey'
            self.sim_mode = 'model'
            self.model_colors = dict(zip(self.dataset_ids, N_colors(len(self.dataset_ids))))
        return target, target_val

    def build_evaluation(self, eval_metrics):
        # s, e, c = self.target.step_data, self.target.endpoint_data, self.target.config

        if eval_metrics is None:
            eval_metrics = {
                'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa', 'rov','roa','tur_fou'],
                'spatial displacement': ['cum_d', 'run_d', 'str_c_l', 'v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                         'dsp_0_40_max','dsp_0_60_max','str_N','tor5', 'tor20'],
                'temporal dynamics': ['fsv', 'ffov', 'run_t', 'pau_t', 'run_tr', 'pau_tr'],
                # 'stride cycle': ['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'],
                # 'epochs': ['run_t', 'pau_t'],
                # 'tortuosity': ['tor5', 'tor20']
            }

        temp, target_data= arrange_evaluation(self.target, eval_metrics)
        ev = {k: col_df(**dic) for k, dic in temp.items()}



        return ev, target_data

    def define_eval_args(self, ev):
        self.e_shorts = dNl.flatten_list(ev['end']['shorts'].values.tolist())
        self.s_shorts = dNl.flatten_list(ev['step']['shorts'].values.tolist())
        self.s_pars = dNl.flatten_list(ev['step']['pars'].values.tolist())
        s_symbols = dNl.flatten_list(ev['step']['symbols'].values.tolist())
        self.e_pars = dNl.flatten_list(ev['end']['pars'].values.tolist())
        e_symbols = dNl.flatten_list(ev['end']['symbols'].values.tolist())
        self.eval_symbols = dNl.AttrDict.from_nested_dicts(
            {'step': dict(zip(self.s_pars, s_symbols)), 'end': dict(zip(self.e_pars, e_symbols))})
        self.tor_durs, self.dsp_starts, self.dsp_stops = torsNdsps(self.s_pars + self.e_pars)

    def run(self,**kwargs):
        if self.sim_mode=='loco':
            self.sim_locomotors(**kwargs)
        elif self.sim_mode=='model':
            self.sim_models(**kwargs)
        # self.eval()

    def eval(self):
        self.changeIDs()
        self.sim_data = self.preprocess()
        self.run_evaluation(self.target, suf='fitted')
        self.plot_eval(suf='fitted')
        if self.target_val is not None :
            self.run_evaluation(self.target_val, suf='cross')
            self.plot_eval(suf='cross')

    def sim_locomotors(self, **kwargs):
        s, e, c = self.target.step_data, self.target.endpoint_data, self.target.config
        for ii, (loco_id, (func, conf, adapted, col)) in enumerate(self.locomotor_models.items()):
            print(f'Simulating model {loco_id} on {c.N} larvae')

            ee, cc = prepare_sim_dataset(e, c, loco_id, col)
            ss = sim_dataset(ee, cc, func, conf, adapted)
            pooled_epochs = enrich_dataset(ss, ee, cc, tor_durs=self.tor_durs, dsp_starts=self.dsp_starts, dsp_stops=self.dsp_stops)
            dd = dNl.AttrDict.from_nested_dicts({'id': loco_id, 'step_data': ss, 'endpoint_data': ee, 'config': cc, 'pooled_epochs': pooled_epochs})
            self.datasets.append(dd)


    def prepare_exp_conf(self,dur, video=False, **kwargs):
        exp='dispersion'
        exp_conf = expandConf(exp, 'Exp')
        c = self.target.config


        exp_conf.larva_groups = {dID: null_dict('LarvaGroup', sample=self.refID, model=expandConf(mID, 'Model'),
                                       default_color=self.model_colors[dID], distribution=null_dict('larva_distro', N=self.N)) for mID,dID in zip(self.modelIDs,self.dataset_ids)}
        exp_conf.env_params = c.env_params

        # if dur is None :
        #     dur = c.Nticks * c.dt / 60

        exp_conf.sim_params = null_dict('sim_params', timestep=c.dt, duration=dur,
                               path=self.path, sim_ID=self.id, store_data=self.store_data)
        if video:
            exp_conf.vis_kwargs = null_dict('visualization', mode='video', video_speed=60)
        else:
            exp_conf.vis_kwargs = null_dict('visualization', mode=None)
        exp_conf.save_to=self.save_to
        exp_conf.update(kwargs)
        # print(exp_conf.enrichment)
        # raise
        print(f'Preparing simulation {self.id} for {dur} minutes')
        return exp_conf

    def load_data(self, key='sim_data'):
        path = self.dir_dict[key]
        try:
            dic = dNl.load_dict(path, use_pickle=True)
            # print(f'{key} loaded')
            return dic
        except:
            raise(f'Stored {key} not found')
            # return None


    def store(self):
        # self.dataset_configs = {dd.id: dd.config for dd in self.datasets}
        if self.store_data:
            # dNl.save_dict(self.dataset_configs, self.dir_dict.dataset_configs)
            dNl.save_dict(self.sim_data, self.dir_dict.sim_data)
            dNl.save_dict(self.error_dicts, self.dir_dict.error_dicts)
            print(f'Results saved at {self.dir}')

    def run_evaluation(self, d,suf,min_size=20):
        print('Evaluating all models')
        for mode in self.eval_modes:
            k=f'{mode}_{suf}'
            self.error_dicts[k] = eval_fast(self.datasets, self.target_data, self.eval_symbols, mode=mode, min_size=min_size)

        self.error_dicts = dNl.AttrDict.from_nested_dicts(self.error_dicts)

        self.store()

    # def get_error_plots2(self,error_dict,mode='pooled',suf='fitted', **kwargs):
    #     label_dic = {
    #         '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
    #         'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}
    #
    #     }
    #     error_dir = f'{self.plot_dir}/errors'
    #     os.makedirs(error_dir, exist_ok=True)
    #     kws={
    #         'labels' :label_dic[mode],
    #         'save_to' :error_dir,
    #         'suf' : suf,
    #         'error_dict' : error_dict,
    #         **kwargs
    #     }
    #
    #     tabs=error_tables(**kws)
    #     bars={}
    #     for norm in self.norm_modes :
    #         fig,df=error_barplots(normalization=norm,evaluation=self.evaluation,**kws)
    #         bars[f'{suf}_{norm}']=fig
    #         tabs[f'{suf}_{norm}'] = error_table('average error', df, norm,suf=suf,save_to=error_dir, **kwargs)
    #     return dNl.AttrDict.from_nested_dicts({'tables': tabs, 'barplots': bars})

    def plot_eval(self, suf='fitted'):
        for mode in self.eval_modes:
            k=f'{mode}_{suf}'
            self.figs.errors[k] = self.get_error_plots(self.error_dicts[k], mode, show=self.show)

    def get_error_plots(self,error_dict,mode='pooled',**kwargs):

        labels=self.label_dic[mode]
        dic={}
        for norm in self.norm_modes :
            error_dict0= self.norm_error_dict(error_dict, mode=norm)
            df0 = pd.DataFrame.from_dict({k: df.mean(axis=1) for i, (k, df) in enumerate(error_dict0.items())})
            kws={
                'save_to' : self.dir_dict[f'error_{norm}'],
                **kwargs
            }

            bars = {}
            tabs = {}
            bars['summary'] = eval_summary(norm_mode=norm, eval_mode=mode, error_dict=error_dict0,
                                           evaluation=self.evaluation, **kws)

            for i, (k, df) in enumerate(error_dict0.items()):
                tabs[k] = error_table(df, k,labels[k],  **kws)
            tabs['mean'] = error_table(df0, 'mean','average error',  **kws)
            bars['full'] = error_barplot(error_dict=error_dict0, evaluation=self.evaluation, labels=labels, **kws)


            # Summary figure with barplots and tables for both endpoint and timeseries metrics
            # bars['summary'] = self.summary_plot(norm_mode=norm, eval_mode=mode, error_dict=error_dict0, **kws)

            dic[norm] = dNl.AttrDict.from_nested_dicts({'tables': tabs, 'barplots': bars})
        return dNl.AttrDict.from_nested_dicts(dic)
    #
    # def summary_plot(self, norm_mode='raw', eval_mode='pooled', suf='fitted',save_to=None,error_dict=None, **kwargs):
    #     from lib.anal.plot_combos import GridPlot
    #     save_to = self.dir_dict[f'error_{norm_mode}'] if save_to is None else save_to
    #     if error_dict is None :
    #         k = f'{eval_mode}_{suf}'
    #         error_dict=self.error_dicts[k]
    #     error_dict0 = self.norm_error_dict(error_dict, mode=norm_mode)
    #     labels=self.label_dic[eval_mode]
    #
    #
    #     w,h=36,52
    #     P = GridPlot(name=f'{norm_mode}_{eval_mode}_error_summary', width=w, height=h, scale=(0.45, 0.45),
    #                  save_to = save_to, **kwargs)
    #
    #     P.plot(func=error_barplot, kws={'error_dict': error_dict0, 'evaluation' : self.evaluation,
    #                                     'labels' : labels},N=2,share_w=True, dh=3, h=23,w=24, x0=True, y0=True)
    #     for i, (k, df) in enumerate(error_dict0.items()):
    #         P.plot(func=error_table, kws={'data': df, 'bbox': [0.4, 0, 1, 1]}, h=10,h0=28+i*12,w=24, x0=True)
    #     P.adjust((0.1, 0.9), (0.05, 0.95), 0.1, 0.2)
    #     P.annotate()
    #     return P.get()



    def norm_error_dict(self, error_dict, mode='raw'):
        from lib.anal.fitting import std_norm, minmax
        dic={}
        for k, df in error_dict.items():
            if mode == 'raw' :
                df=df
            elif mode == 'minmax':
                df = minmax(df)
            elif mode == 'std':
                df = std_norm(df)
            dic[k]=df
        return dNl.AttrDict.from_nested_dicts(dic)


    def plot_models(self):
        save_to=self.dir_dict.models
        self.figs.models.tables = dNl.AttrDict.from_nested_dicts(
            {mID: modelConfTable(mID, save_to=save_to, figsize=(14, 11)) for mID in self.modelIDs})
        self.figs.models.summaries = dNl.AttrDict.from_nested_dicts(
            {mID: model_summary(refID=self.refID, mID=mID, save_to=save_to) for mID in self.modelIDs})

    def plot_results(self, plots=['dispersion','hists','trajectories','boxplots','bouts','fft']):
        print('Generating comparative graphs')




        ds =[self.target] + self.datasets
        kws = {
            'datasets': ds ,
            'save_to': self.dir_dict.plot,
            'show': self.show
        }
        kws2 = {
            'target': self.target,
            'datasets': self.datasets,
            'save_to': self.dir_dict.plot,
            'show': self.show
        }


        # print([(d.id, d.config.N) for d in ds])
        if 'fft' in plots:
            self.figs.loco.fft = plot_fft_multi(**kws)
        if 'hists' in plots:
            self.figs.hist.step = self.plot_data(mode='step', type='hist')
            self.figs.hist.end = self.plot_data(mode='end', type='hist')
            self.figs.hist.ang = plot_ang_pars(half_circles=False, absolute=False, Nbins=100, Npars=5,include_rear=False, subfolder=None, **kws)
            self.figs.hist.crawl = plot_crawl_pars(subfolder=None, pvalues=False, **kws)
        if 'trajectories' in plots:
            self.figs.loco.trajectories = plot_trajectories(subfolder=None, **kws)
        if 'boxplots' in plots:
            self.figs.boxplot.end = self.plot_data(mode='end', type='box')
            # self.figs.boxplot.end = boxplots(shorts=self.e_shorts, key='end',annotation=True,show_ns=False,  target_only=evrun.target.id, **kws)
            self.figs.boxplot.step = self.plot_data(mode='step', type='box')
            # self.figs.boxplot.step = boxplots(shorts=self.s_shorts, key='step',annotation=True,show_ns=False, target_only=evrun.target.id, **kws)
        if 'bouts' in plots:
            self.figs.epochs.runNpause = plot_bouts(stridechain_duration=True, **kws)
            self.figs.epochs.turn = plot_bouts(turns=True, **kws)
        if 'dispersion' in plots:
            for r0, r1 in itertools.product(self.dsp_starts, self.dsp_stops):
                k=f'dsp_{r0}_{r1}'
                fig1=plot_dispersion(range=(r0, r1),subfolder=None, **kws)
                fig2=plot_trajectories(name=f'traj_{r0}_{r1}', range=(r0, r1),subfolder=None,mode='origin', **kws)
                fig3=dsp_summary(range=(r0, r1),**kws2)

                self.figs.loco[k] = dNl.AttrDict.from_nested_dicts({'plot': fig1, 'traj': fig2, 'summary' : fig3})

        self.figs.summary = result_summary(**kws2)
        self.figs.stride_cycle.norm = stride_cycle(shorts=['sv', 'fov', 'rov', 'foa', 'b'], individuals=True, **kws)

    def preprocess(self):
        Ddata,Edata = {},{}
        for p in self.s_pars:
            Ddata[p] = {d.id: d.step_data[p].dropna() for d in self.datasets}
        for p in self.e_pars:
            Edata[p] = {d.id: d.endpoint_data[p]  for d in self.datasets}
        sim_data = dNl.AttrDict.from_nested_dicts({'step': Ddata, 'end': Edata})
        return sim_data

    def preprocess2(self):
        Ddata,Edata = {},{}
        for p, sh in zip(self.s_pars, self.s_shorts):
            Ddata[p] = {d.id: par_dict.get(sh, d) for d in self.datasets}
        for p, sh in zip(self.e_pars, self.e_shorts):
            Edata[p] = {d.id: par_dict.get(sh, d) for d in self.datasets}
        sim_data = dNl.AttrDict.from_nested_dicts({'step': Ddata, 'end': Edata})
        return sim_data

    def plot_data(self, Nbins=None, mode='step', type='hist', in_mm = []):
        if mode=='step' :
            if Nbins is None :
                Nbins=100
            data0=self.target_data.step
            pars=self.s_pars
            shorts=self.s_shorts
            symbols=self.eval_symbols.step
            sim_data=self.sim_data.step
        elif mode=='end' :
            if Nbins is None :
                Nbins=20
            data0=self.target_data.end
            pars=self.e_pars
            shorts = self.e_shorts
            symbols=self.eval_symbols.end
            sim_data=self.sim_data.end

        filename = f'{mode}_{type}'
        Nps = len(pars)
        Ncols = 4
        Nrows = int(np.ceil(Nps / Ncols))
        if type == 'hist' :
            sharex = False
            sharey = True
        elif type == 'box' :
            sharex = True
            sharey = False

        P = AutoPlot(name=filename, subfolder=None, Nrows=Nrows, Ncols=Ncols, figsize=(5 * Ncols, 5 * Nrows),
                     sharex=sharex, sharey=sharey, show=self.show, save_to=self.plot_dir,
                     datasets=[self.target] + self.datasets)


        if type=='hist' :

            for i, (p, sym) in enumerate(symbols.items()):
                vs0=data0[p].values
                ws0 = np.ones_like(vs0) / float(len(vs0))
                vmin,vmax=np.quantile(vs0, q=0.01),np.quantile(vs0, q=0.99)
                bins = np.linspace(vmin,vmax, Nbins)
                col0=self.target.color
                _=P.axs[i].hist(vs0, bins=bins, weights=ws0, label='experiment', color=col0, alpha=0.5)
                for id,df in sim_data[p].items() :
                    vs=df.values
                    ws = np.ones_like(vs) / float(len(vs))
                    col= self.model_colors[id]
                    _=P.axs[i].hist(vs, bins=bins, weights=ws, label=id, color=col,
                                  histtype='step', linewidth=3,  facecolor=col, edgecolor=col, fill=False,
                                  alpha=0.5)
                P.axs[i].set_xlabel(sym, fontsize=20)
                if i%Ncols==0:
                    P.axs[i].set_ylabel('probability', fontsize=20)
            P.axs[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            P.adjust(W=0.01, H=0.5)

        elif type == 'box':
            from lib.anal.plot_aux import concat_datasets
            data=concat_datasets(P.datasets, key=mode)
            palette = dict(zip(P.labels, P.colors))
            for sh in in_mm:
                data[getPar(sh)] *= 1000

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
                annotate_plot(show_ns=False, target_only=self.target.id, **kws)
                # try:
                #     g1.get_legend().remove()
                # except:
                #     pass
                P.conf_ax(i, xticklabelrotation=30, ylab=sym, yMaxN=4,xvis=False if i < (Nrows - 1) * Ncols else True)

            P.adjust((0.1, 0.95), (0.15, 0.9), 0.5, 0.1)
            P.fig.align_ylabels(P.axs[:])
        return P.get()



    def changeIDs(self):
        if self.expVSsimIDs :
            self.target.id='experiment'
            self.target.config.id='experiment'
            if len(self.datasets) == 1:
                self.datasets[0].id='model'
                self.datasets[0].config.id='model'
            else :
                for i in range(len(self.datasets)) :
                    self.datasets[i].id = f'model {i+1}'
                    self.datasets[i].config.id = f'model {i+1}'

            self.model_colors = dict(zip([d.id for d in self.datasets], N_colors(len(self.datasets))))


    def add_dataset(self, d):
        self.datasets.append(d)

    def sim_models(self, dur=None,**kwargs):
        c=self.target.config
        if dur is None :
            dur = c.Nticks * c.dt / 60
        if self.N is None:
            self.N = 5

        if self.offline:
            print(f'Simulating offline {len(self.dataset_ids)} models : {self.dataset_ids} with {self.N} larvae each')
            self.datasets += sim_models(mIDs=self.modelIDs, dur=dur, dt=c.dt,tor_durs=self.tor_durs,dsp_starts=self.dsp_starts,dsp_stops=self.dsp_stops,
                                        dataset_ids=self.dataset_ids,bout_annotation=self.bout_annotation,
                               Nids=self.N, colors=list(self.model_colors.values()), env_params = c.env_params, refDataset=self.target, data_dir=self.data_dir)
        else :
            from lib.sim.single.single_run import SingleRun
            self.exp_conf = self.prepare_exp_conf(dur=dur, **kwargs)

            self.exp_conf.enrichment.metric_definition.dispersion.dsp_starts=self.dsp_starts
            self.exp_conf.enrichment.metric_definition.dispersion.dsp_stops=self.dsp_stops
            self.exp_conf.enrichment.metric_definition.tortuosity.tor_durs=self.tor_durs
            self.exp_conf.enrichment.bout_annotation= self.bout_annotation
            # self.exp_conf.enrichment = None

            print(f'Simulating {len(self.dataset_ids)} models : {self.dataset_ids} with {self.N} larvae each')
            run = SingleRun(**self.exp_conf)
            self.datasets += run.run()
        try :
            for dd in self.datasets :
                er=RSS_dic(dd, self.target)
                print(dd.id, er)
        except :
            pass
        self.dataset_configs = {dd.id: dd.config for dd in self.datasets}
        dNl.save_dict(self.dataset_configs, self.dir_dict.dataset_configs)


def error_table(data, k='',title = None, **kwargs):
    from lib.aux.combining import render_mpl_table
    data = np.round(data, 3).T
    figsize = ((data.shape[1] + 3) * 4, data.shape[0])
    fig = render_mpl_table(data, highlighted_cells='row_min', title=title, figsize=figsize,
                               adjust_kws={'left': 0.3, 'right': 0.95},
                               name=f'error_table_{k}', **kwargs)
    return fig


def error_barplot(error_dict, evaluation, axs=None, fig=None,labels=None,name='error_barplots',
                   titles=[r'$\bf{endpoint}$ $\bf{metrics}$', r'$\bf{timeseries}$ $\bf{metrics}$'], **kwargs):

    def build_legend(ax, eval_df) :
        h, l = ax.get_legend_handles_labels()
        empty = mpatches.Patch(color='none')
        counter = 0
        for g in eval_df.index:
            h.insert(counter, empty)
            l.insert(counter, eval_df['group_label'].loc[g])
            counter += (len(eval_df['shorts'].loc[g]) + 1)
        ax.legend(h, l, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=15)

    from lib.anal.plot_aux import BasePlot
    import matplotlib.patches as mpatches
    P = BasePlot(name=name, **kwargs)
    Nplots = len(error_dict)
    P.build(Nplots, 1, figsize=(20, Nplots * 6), sharex=False, fig=fig, axs=axs)
    P.adjust((0.07, 0.7), (0.05, 0.95), 0.05, 0.2)
    for ii, (k, eval_df) in enumerate(evaluation.items()):
        lab = labels[k] if labels is not None else k
        # ax = P.axs[ii] if axs is None else axs[ii]
        df = error_dict[k]
        color = dNl.flatten_list(eval_df['par_colors'].values.tolist())
        df = df[dNl.flatten_list(eval_df['symbols'].values.tolist())]
        df.plot(kind='bar', ax=P.axs[ii], ylabel=lab, rot=0, legend=False, color=color, width=0.6)
        build_legend(P.axs[ii], eval_df)
        P.conf_ax(ii,title=titles[ii],xlab='', yMaxN=4)
    return P.get()


def eval_summary(error_dict, evaluation, norm_mode='raw', eval_mode='pooled',**kwargs):
    from lib.anal.plot_combos import GridPlot

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
        P.plot(func=error_table, kws={'data': df,'k' : k,  'bbox': [0.4, 0, 1, 1]}, h=12,h0=h0,w=24, x0=True)
    P.adjust((0.1, 0.9), (0.05, 0.95), 0.1, 0.2)
    P.annotate()
    return P.get()



def plot_distros(datasets, distro_ps, save_to=None, show=False):
    ps2, ps2l = getPar(distro_ps, to_return=['d', 'lab'])
    Nps = len(distro_ps)
    Ncols = 4
    Nrows = int(np.ceil(Nps / Ncols))
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(5 * Ncols, 5 * Nrows), sharex=False, sharey=False)
    axs = axs.ravel()
    for i, (p, l) in enumerate(zip(ps2, ps2l)):
        vs = []
        for ii, d in enumerate(datasets):
            vs.append(d.step_data[p].dropna().abs().values)
        vvs = np.hstack(vs).flatten()
        bins = np.linspace(0, np.quantile(vvs, q=0.9), 40)
        for ii, d in enumerate(datasets):
            col = d.config.color
            weights = np.ones_like(vs[ii]) / float(len(vs[ii]))
            axs[i].hist(vs[ii], bins=bins, weights=weights, label=d.id, color=col, histtype='step', linewidth=3,
                        facecolor=col, edgecolor=col, fill=True, alpha=0.2)
        axs[i].set_title(l, fontsize=20)
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_distros.pdf', dpi=300)
    if show:
        plt.show()
    return fig


def plot_endpoint(datasets, end_ps, save_to=None, show=False):
    ps2, ps2l = getPar(end_ps, to_return=['d', 'lab'])
    Nps = len(end_ps)
    Ncols = 4
    Nrows = int(np.ceil(Nps / Ncols))
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(5 * Ncols, 5 * Nrows), sharex=False, sharey=False)
    axs = axs.ravel()
    for i, (p, l) in enumerate(zip(ps2, ps2l)):
        vs = []
        for ii, d in enumerate(datasets):
            vs.append(d.endpoint_data[p].dropna().values)
        vvs = np.hstack(vs).flatten()
        bins = np.linspace(np.min(vvs), np.max(vvs), 20)
        for ii, d in enumerate(datasets):
            col = d.config.color
            weights = np.ones_like(vs[ii]) / float(len(vs[ii]))
            _=axs[i].hist(vs[ii], bins=bins, weights=weights, label=d.id, color=col, histtype='step', linewidth=3,
                        facecolor=col, edgecolor=col, fill=True, alpha=0.2)
        axs[i].set_title(l, fontsize=20)
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_endpoint.pdf', dpi=300)
    if show:
        plt.show()
    plt.close()
    return fig


def compare2ref(ss, s=None, refID=None,
                shorts= ['b', 'fov', 'foa', 'tur_fou', 'tur_fov_max', 'v', 'a', 'run_d', 'run_t', 'pau_t', 'tor5', 'tor20'],pars=None):
    if s is None and refID is not None :
        d = loadRef(refID)
        d.load(contour=False)
        s, e, c = d.step_data, d.endpoint_data, d.config

    if pars is None and shorts is not None:
        pars = getPar(shorts)

    KSdic = {}
    for p, sh in zip(pars, shorts):
        if isinstance(s, dict):
            key = p if p in s.keys() else sh
            p_s = np.array(s[key])
            p_s = p_s[~np.isnan(p_s)]
        else:
            key = p if p in s.columns else sh
            p_s = s[key].dropna().values
        # else:
        #     continue
        if isinstance(ss, dict):
            key= p if p in ss.keys() else sh
            p_ss = np.array(ss[key])
            p_ss = p_ss[~np.isnan(p_ss)]
        else:
            key = p if p in ss.columns else sh
            p_ss = ss[key].dropna().values
        KSdic[p] = ks_2samp(p_s, p_ss)[0]
    return KSdic




if __name__ == '__main__':
    t0=time.time()
    from lib.anal.evaluation import EvalRun
    from lib.aux import dictsNlists as dNl

    # enr=null_dict('enrichment')
    # print(enr)
    # raise
    refID = 'None.150controls'
    # mIDs = ['NEU_PHI', 'NEU_PHIx', 'PHIonSIN', 'PHIonSINx']
    mIDs = ['PHIonNEU', 'SQonNEU', 'PHIonSIN', 'SQonSIN']
    dataset_ids = mIDs
    # dataset_ids = ['NEU mean', 'NEU var', 'SIN mean', 'SIN var']
    id = 'online_4models_test5'

    evrun = EvalRun(refID=refID, id=id,modelIDs=mIDs,dataset_ids=dataset_ids, N=5,
                    bout_annotation=True,show=False, offline=False)

    evrun.run(video=False)
    evrun.eval()
    # evrun.plot_results()
    t1 = time.time()
    print(int((t1-t0)*1000))

