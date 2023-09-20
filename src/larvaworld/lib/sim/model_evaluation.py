import os
import warnings

import param

from larvaworld.cli.argparser import update_larva_groups
from larvaworld.lib.param import NestedConf, PositiveInteger, class_generator
from larvaworld.lib.process.evaluation import DataEvaluation

warnings.simplefilter(action='ignore', category=FutureWarning)

import itertools
import numpy as np
import pandas as pd


from larvaworld.lib import reg, aux, plot, util


__all__ = [
    'EvalRun',
    'eval_model_graphs',
    'add_var_mIDs',
    'adapt_6mIDs',
    'adapt_3modules',
    'modelConf_analysis',
]

class EvalDataConf(DataEvaluation):

    def __init__(self,dataset=None, **kwargs):
        super().__init__(dataset =dataset, **kwargs)
        self.target.id = 'experiment'
        self.target.config.id = 'experiment'
        self.target.color = 'grey'
        self.target.config.color = 'grey'
        kwargs['dt'] = self.target.config.dt
        kwargs['duration'] = self.target.config.Nticks * kwargs['dt'] / 60




class EvalConf(EvalDataConf):
    modelIDs = reg.conf.Model.confID_selector(single=False)
    dataset_ids = param.List([], item_type=str, doc='The ids for the generated datasets')
    N = PositiveInteger(5, label='# agents/group', doc='Number of agents per model ID')


    def __init__(self,**kwargs):
        super().__init__(**kwargs)




class EvalRun(EvalConf, reg.generators.SimConfiguration):



    def __init__(self,show=False,enrichment=True,  **kwargs):
        '''
        Simulation mode 'Eval' compares/evaluates different models against a reference dataset obtained by a real or simulated experiment.


        Args:
            parameters: Dictionary of configuration parameters to be passed to the ABM model
            dataset: The stored dataset used as reference to evaluate the performance of the simulated models. If not specified it is retrieved using either the storage path (parameters.dir) or the respective unique reference ID (parameters.RefID)
            dur: Duration of the simulation. If not specifies defaults to the reference dataset duration.
            experiment: The type of experiment. Defaults to 'dispersion'
            **kwargs: Arguments passed to parent class
        '''
        super().__init__(runtype='Eval',**kwargs)


        self.show = show
        self.enrichment = enrichment

        self.figs = aux.AttrDict({'errors': {}, 'hist': {}, 'boxplot': {}, 'stride_cycle': {}, 'loco': {}, 'epochs': {},
                                  'models': {'table': {}, 'summary': {}}})


        self.error_plot_dir = f'{self.plot_dir}/errors'


    def simulate(self):
        conf = reg.conf.Exp.expand(self.experiment)
        conf.larva_groups = update_larva_groups(conf.larva_groups, N=self.N, mIDs=self.modelIDs, dIDs=self.dataset_ids,
                                                sample=self.refID)


        # mIDs = self.modelIDs
        # dIDs=self.dataset_ids
        Nm=len(self.modelIDs)
        kws={
            'dt': self.dt,
            'duration': self.duration,
        }
        c = self.target.config


        if self.offline is None:
            print(f'Simulating offline {Nm} models : {self.dataset_ids} with {self.N} larvae each')
            tor_durs = np.unique([int(ii[len('tortuosity') + 1:]) for ii in self.s_pars + self.e_pars if ii.startswith('tortuosity')])
            dsp = reg.getPar('dsp')
            dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in self.s_pars + self.e_pars if ii.startswith(f'{dsp}_')]
            dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
            dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()


            self.datasets = util.sim_models(mIDs=self.modelIDs, tor_durs=tor_durs,
                                        dsp_starts=dsp_starts, dsp_stops=dsp_stops,
                                        dataset_ids=self.dataset_ids,lgs=conf.larva_groups,
                                        enrichment=self.enrichment,
                                        Nids=self.N, env_params=c.env_params,
                                        refDataset=self.target, data_dir=self.data_dir, **kws)
        else:
            from larvaworld.lib.sim.single_run import ExpRun
            print(f'Simulating {Nm} models : {self.dataset_ids} with {self.N} larvae each')

            # conf.larva_groups=self.larva_groups
            if self.enrichment is None:
                conf.enrichment = None
            kws0 = aux.AttrDict({
                'dir': self.dir,
                'store_data': self.store_data,
                'experiment': self.experiment,
                'id': self.id,
                'offline': self.offline,
                'parameters': conf,
                **kws
            })
            run = ExpRun(**kws0)
            run.simulate()
            self.datasets = run.datasets
        self.analyze()
        if self.store_data:
            self.store()
        return self.datasets

    def get_error_plots(self, error_dict, mode='pooled'):
        GD = reg.graphs.dict
        label_dic = {
            '1:1': {'end': 'RSS error', 'step': r'median 1:1 distribution KS$_{D}$'},
            'pooled': {'end': 'Pooled endpoint values KS$_{D}$', 'step': 'Pooled distributions KS$_{D}$'}
        }
        labels = label_dic[mode]
        dic = aux.AttrDict()
        for norm in self.norm_modes:
            d = self.norm_error_dict(error_dict, mode=norm)
            df0 = pd.DataFrame.from_dict({k: df.mean(axis=1) for i, (k, df) in enumerate(d.items())})
            kws = {
                'save_to': f'{self.error_plot_dir}/{norm}',
                'show' : self.show
            }
            bars = {}
            tabs = {}
            for k, df in d.items():
                tabs[k] = GD['error table'](data=df, k=k, title=labels[k], **kws)
            tabs['mean'] = GD['error table'](data=df0, k='mean', title='average error', **kws)
            bars['full'] = GD['error barplot'](error_dict=d, evaluation=self.evaluation, labels=labels, **kws)
            # Summary figure with barplots and tables for both endpoint and timeseries metrics
            bars['summary'] = GD['error summary'](norm_mode=norm, eval_mode=mode, error_dict=d,
                                                  evaluation=self.evaluation, **kws)
            dic[norm] = {'tables': tabs, 'barplots': bars}
        return aux.AttrDict(dic)



    def analyze(self, **kwargs):
        print('Evaluating all models')
        os.makedirs(self.plot_dir, exist_ok=True)

        for mode in self.eval_modes:

            d = self.eval_datasets(self.datasets, mode=mode,**kwargs)
            self.figs.errors[mode] = self.get_error_plots(d, mode)
            self.error_dicts[mode] = d

    def store(self):
        aux.save_dict(self.error_dicts, f'{self.data_dir}/error_dicts.txt')
        reg.vprint(f'Results saved at {self.data_dir}')




    def plot_models(self):
        GD = reg.graphs.dict
        save_to = self.plot_dir
        for mID in self.modelIDs:
            self.figs.models.table[mID] = GD['model table'](mID=mID, save_to=save_to, figsize=(14, 11))
            self.figs.models.summary[mID] = GD['model summary'](mID=mID, save_to=save_to, refID=self.refID)

    def plot_results(self, plots=['hists', 'trajectories', 'dispersion', 'bouts', 'fft', 'boxplots']):
        GD = reg.graphs.dict


        self.target.load(h5_ks=['epochs', 'angular', 'dspNtor'])
        kws = {
            'datasets': [self.target] + self.datasets,
            'save_to': self.plot_dir,
            'show': self.show
        }
        kws1={
            'subfolder' : None,
            **kws
        }

        kws2 = {
            'target': self.target,
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': self.show
        }
        self.figs.summary = GD['eval summary'](**kws2)
        self.figs.stride_cycle.norm = GD['stride cycle'](shorts=['sv', 'fov', 'rov', 'foa', 'b'],
                                                         individuals=True,**kws)
        if 'dispersion' in plots:
            for r0, r1 in itertools.product(self.dsp_starts, self.dsp_stops):
                self.figs.loco[f'dsp_{r0}_{r1}'] = aux.AttrDict({
                    'plot': GD['dispersal'](range=(r0, r1), **kws1),
                    'traj': GD['trajectories'](name=f'traj_{r0}_{r1}', range=(r0, r1), mode='origin', **kws1),
                    'summary': GD['dispersal summary'](range=(r0, r1), **kws2)
                })
        if 'bouts' in plots:
            self.figs.epochs.turn = GD['epochs'](turns=True, **kws)
            self.figs.epochs.runNpause = GD['epochs'](stridechain_duration=True, **kws)
        if 'fft' in plots:
            self.figs.loco.fft = GD['fft multi'](**kws)
        if 'hists' in plots:
            self.figs.hist.ang = GD['angular pars'](half_circles=False, absolute=False, Nbins=100, Npars=3,
                                                    include_rear=False, **kws1)
            self.figs.hist.crawl = GD['crawl pars'](pvalues=False, **kws1)
        if 'trajectories' in plots:
            self.figs.loco.trajectories = GD['trajectories'](**kws1)
        if 'boxplots' in plots:
            pass
            # self.figs.boxplot.end = self.plot_data(mode='end', type='box')
            # self.figs.boxplot.step = self.plot_data(mode='step', type='box')








reg.gen.Eval=class_generator(EvalConf)


def eval_model_graphs(refID, mIDs, dIDs=None, id=None, dir=None, N=10,
                      **kwargs):
    if id is None:
        id = f'{len(mIDs)}mIDs'
    if dir is None:
        dir = f'{reg.conf.Ref.getID(refID)}/model/evaluation'

    parameters =reg.gen.Eval(refID=refID, modelIDs=mIDs,dataset_ids=dIDs,N=N).nestedConf

    evrun = EvalRun(parameters=parameters, id=id,
                    dir=dir, **kwargs)
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
            'brain.crawler_params.freq',
        ]
    kwargs = {k: 'sample' for k in sample_ks}
    entries = {}
    for mID0, mID in zip(mID0s, mIDs):
        m0 = reg.conf.Model.getID(mID0).get_copy()
        # m0 = reg.stored.getModel(mID0).get_copy()
        m = m0.update_existingnestdict(kwargs)
        reg.conf.Model.setID(mID, m)
        entries[mID] = m
    return entries

def adapt_6mIDs(refID, e=None, c=None):
    d = reg.loadRef(refID)
    if e is None or c is None:
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

    # fit_dict = GA_optimization(d = d, fit_kws=fit_kws)
    entries = {}
    mIDs = []
    for Tmod in ['NEU', 'SIN']:
        for Ifmod in ['PHI', 'SQ', 'DEF']:
            mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
            mID = f'{Ifmod}on{Tmod}'
            entry = reg.model.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                   space_mkeys=['turner', 'interference'],**fit_kws)
            entries.update(entry)
            mIDs.append(mID)
    return entries, mIDs

def adapt_3modules(refID, e=None, c=None):
    d = reg.loadRef(refID)
    if e is None or c is None:
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

    # fit_dict = GA_optimization(d = d, fit_kws=fit_kws)
    entries = {}
    mIDs = []
    for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
        for Tmod in ['NEU', 'SIN', 'CON']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'{Cmod}_{Tmod}_{Ifmod}_DEF'
                mID = f'{mID0}_fit'
                entry = reg.model.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                       space_mkeys=['crawler', 'turner', 'interference'],**fit_kws)
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
        eval_model_graphs(refID, mIDs=mIDs_avg, id='6mIDs_avg', N=10)

        entries_var = add_var_mIDs(refID=c.refID, e=e, c=c,mID0s=mIDs_avg)
        mIDs_var = list(entries_var.keys())
        c.modelConfs.variable = entries_var
        eval_model_graphs(refID, mIDs=mIDs_var,  id='6mIDs_var', N=10)
        eval_model_graphs(refID, mIDs=mIDs_avg[:3] + mIDs_var[:3], id='3mIDs_avgVSvar1',
                          N=10)
        eval_model_graphs(refID, mIDs=mIDs_avg[3:] + mIDs_var[3:], id='3mIDs_avgVSvar2',
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
                eval_model_graphs(mIDs=mIDs, dIDs=dIDs, id=id, N=10)
    d.config = c
    d.save_config()
