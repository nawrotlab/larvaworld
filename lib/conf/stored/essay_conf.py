import copy
import shutil

import pandas as pd


from lib.registry.pars import preg
from lib.sim.single.single_run import SingleRun
from lib.aux import dictsNlists as dNl, colsNstr as cNs, naming as nam
I=preg.init_dict

class Essay:
    def __init__(self, type, N=5, enrichment=I.base_enrich(), collections=['pose'], video=False, show=False,
                 **kwargs):
        if video:
            self.vis_kwargs = preg.get_null('visualization', mode='video', video_speed=60)
        else:
            self.vis_kwargs = preg.get_null('visualization', mode=None)
        self.N = N
        self.G = preg.graph_dict
        self.GT=preg.grouptype_dict
        self.CT=preg.conftype_dict
        self.M = preg.larva_conf_dict
        self.show = show
        self.type = type
        self.enrichment = enrichment
        self.collections = collections
        self.essay_id = f'{type}_{preg.next_idx(id=type, conftype="Essay")}'
        self.path = f'essays/{type}/{self.essay_id}/data'
        path = preg.path_dict["ESSAY"]
        self.full_path = f'{path}/{type}/{self.essay_id}/data'
        self.plot_dir = f'{path}/{type}/{self.essay_id}/plots'
        self.exp_dict = {}
        self.datasets = {}
        self.figs = {}
        self.results = {}

    def conf(self, exp, id, dur, lgs, env, **kwargs):
        sim = preg.get_null('sim_params', sim_ID=id, path=self.path, duration=dur)
        return preg.get_null('exp_conf', sim_params=sim, env_params=env, trials={},
                             larva_groups=lgs, experiment=exp, enrichment=self.enrichment,
                             collections=self.collections, **kwargs)

    def run(self):
        print(f'Running essay "{self.essay_id}"')
        for exp, cs in self.exp_dict.items():

            print(f'Running {len(cs)} versions of experiment {exp}')
            self.datasets[exp] = [SingleRun(**c, vis_kwargs=self.vis_kwargs).run() for c in cs]

        return self.datasets

    def anal(self):
        self.global_anal()
        # raise
        for exp, ds0 in self.datasets.items():
            if ds0 is not None and len(ds0) != 0 and all([d0 is not None for d0 in ds0]):
                self.analyze(exp=exp, ds0=ds0)

        shutil.rmtree(self.full_path, ignore_errors=True)
        return self.figs, self.results

    def analyze(self, exp, ds0):
        pass
        # return {}, None

    def global_anal(self):
        pass


class RvsS_Essay(Essay):
    def __init__(self, all_figs=False, N=1, **kwargs):
        super().__init__(type='RvsS', N=N, enrichment=I.enr_dict(proc=['spatial']),
                         collections=['pose', 'feeder', 'gut'], **kwargs)

        self.all_figs = all_figs
        self.qs = [1.0, 0.75, 0.5, 0.25, 0.15]
        self.hs = [0, 1, 2, 3, 4]
        self.durs = [10, 15, 20]
        self.dur = 5
        self.h_refeeding = 3
        self.dur_refeeding = 120
        self.dur_pathlength = 20

        self.substrates = ['Agar', 'Yeast']
        self.exp_dict = {
            **self.pathlength_exp(),
            **self.intake_exp(),
            **self.starvation_exp(),
            **self.quality_exp(),
            **self.refeeding_exp(),

        }

        self.mdiff_df, row_colors = self.M.diff_df(mIDs=['rover', 'sitter'])

    def RvsS_env(self, on_food=True):
        grid = preg.get_null('food_grid') if on_food else None
        return preg.get_null('env_conf',
                             arena=preg.get_null('arena', arena_shape='rectangular', arena_dims=(0.02, 0.02)),
                             food_params=preg.get_null('food_params', food_grid=grid),
                             )

    def pathlength_exp(self):
        from lib.conf.stored.exp_conf import RvsS_groups
        dur = self.dur_pathlength
        exp = 'PATHLENGTH'
        confs = []
        for n, nb in zip(self.substrates, [False, True]):
            kws = {
                'env': self.RvsS_env(on_food=nb),
                'lgs': RvsS_groups(expand=True, N=self.N),
                'id': f'{exp}_{n}_{dur}min',
                'dur': dur,
                'exp': exp
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def intake_exp(self):
        from lib.conf.stored.exp_conf import RvsS_groups
        exp = 'AD LIBITUM INTAKE'
        confs = []
        for dur in self.durs:
            kws = {
                'env': self.RvsS_env(on_food=True),
                'lgs': RvsS_groups(expand=True, N=self.N),
                'id': f'{exp}_{dur}min',
                'dur': dur,
                'exp': exp
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def starvation_exp(self):
        from lib.conf.stored.exp_conf import RvsS_groups
        exp = 'POST-STARVATION INTAKE'
        confs = []
        for h in self.hs:
            kws = {
                'env': self.RvsS_env(on_food=True),
                'lgs': RvsS_groups(expand=True, N=self.N, h_starved=h),
                'id': f'{exp}_{h}h_{self.dur}min',
                'dur': self.dur,
                'exp': exp
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def quality_exp(self):
        from lib.conf.stored.exp_conf import RvsS_groups
        exp = 'REARING-DEPENDENT INTAKE'
        confs = []
        for q in self.qs:
            kws = {
                'env': self.RvsS_env(on_food=True),
                'lgs': RvsS_groups(expand=True, N=self.N, q=q),
                'id': f'{exp}_{q}_{self.dur}min',
                'dur': self.dur,
                'exp': exp
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def refeeding_exp(self):
        from lib.conf.stored.exp_conf import RvsS_groups
        exp = 'REFEEDING AFTER 3h STARVED'
        h = self.h_refeeding
        dur = self.dur_refeeding
        kws = {
            'env': self.RvsS_env(on_food=True),
            'lgs': RvsS_groups(expand=True, N=self.N, h_starved=h),
            'id': f'{exp}_{h}h_{dur}min',
            'dur': dur,
            'exp': exp
        }
        return {exp: [self.conf(**kws)]}

    def get_entrylist(self, datasets, substrates, durs, qs, hs, G):
        entrylist = []
        pathlength_ls = dNl.flatten_list([[rf'{s} $for^{"R"}$', rf'{s} $for^{"S"}$'] for s in substrates])
        ls0 = [r'$for^{R}$', r'$for^{S}$']
        kws00 = {
            'leg_cols': ['black', 'white'],
            'markers': ['D', 's'],
        }
        for exp, dds in datasets.items():
            Ndds = len(dds)
            ds = dNl.flatten_list(dds)
            ls = pathlength_ls if exp == 'PATHLENGTH' else dNl.flatten_list([ls0] * Ndds)
            kws0 = {
                'datasets': ds,
                'labels': ls,
                'save_as': exp,
                **kws00
            }

            if exp == 'PATHLENGTH':
                kws = {
                    'xlabel': r'time on substrate $(min)$',
                    'scaled': False,
                    'unit': 'cm',
                    **kws0
                }
                plotID = 'pathlength'
            elif exp == 'AD LIBITUM INTAKE':
                kws = {
                    'xlabel': r'Time spent on food $(min)$',
                    'coupled_labels': durs,
                    'par_shorts': ['sf_am_V'],
                    **kws0
                }
                plotID = 'barplot'
            elif exp == 'POST-STARVATION INTAKE':
                kws = {
                    'xlabel': r'Food deprivation $(h)$',
                    'coupled_labels': hs,
                    'par_shorts': ['f_am'],
                    'ylabel': 'Food intake',
                    'scale': 1000,
                    **kws0}
                plotID = 'lineplot'
            elif exp == 'REARING-DEPENDENT INTAKE':
                kws = {
                    'xlabel': 'Food quality (%)',
                    'coupled_labels': [int(q * 100) for q in qs],
                    'par_shorts': ['sf_am_V'],
                    **kws0
                }
                plotID = 'barplot'
            elif exp == 'REFEEDING AFTER 3h STARVED':
                kws = {
                    'scaled': True,
                    'filt_amount': True,
                    **kws0
                }
                plotID = 'food intake (timeplot)'
            else:
                raise
            entry = G.entry(ID=plotID, title=exp, args=dNl.NestDict(kws))
            entrylist.append(entry)
        return entrylist

    def global_anal(self):
        self.entrylist = self.get_entrylist(datasets=self.datasets, substrates=self.substrates,
                                            durs=self.durs, qs=self.qs, hs=self.hs, G=self.G)
        kwargs = {'save_to': self.plot_dir,
                  'show': self.show}

        entry = self.G.entry('RvsS summary',
                             args={'entrylist': self.entrylist, 'title': f'ROVERS VS SITTERS ESSAY (N={self.N})',
                                   'mdiff_df' : self.mdiff_df})
        self.figs.update(self.G.eval0(entry, **kwargs))
        self.figs.update(self.G.eval(self.entrylist, **kwargs))

    def analyze(self, exp, ds0):
        if self.all_figs:
            entry = [e for e in self.entrylist if e['title'] == exp][0]
            kws = entry['args']
            RS_leg_cols = ['black', 'white']
            markers = ['D', 's']
            ls = [r'$for^{R}$', r'$for^{S}$']
            shorts = ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']
            pars = preg.getPar(shorts)

            #
            def dsNls(ds0, lls=None):
                if lls is None:
                    lls = dNl.flatten_list([ls] * len(ds0))
                dds = dNl.flatten_list(ds0)
                deb_dicts = dNl.flatten_list([d.load_dicts('deb') for d in dds])
                return {'datasets': dds,
                        'labels': lls,
                        'deb_dicts': deb_dicts,
                        'save_to': self.plot_dir,
                        'leg_cols': RS_leg_cols,
                        'markers': markers,
                        }

            if exp == 'PATHLENGTH':
                pass

            elif exp == 'AD LIBITUM INTAKE':
                kwargs = {**dsNls(ds0),
                          'coupled_labels': self.durs,
                          'xlabel': r'Time spent on food $(min)$'}
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = self.G.dict['barplot'](par_shorts=[s],
                                                                     save_as=f'2_AD_LIBITUM_{p}.pdf', **kwargs)

            elif exp == 'POST-STARVATION INTAKE':
                kwargs = {**dsNls(ds0),
                          'coupled_labels': self.hs,
                          'xlabel': r'Food deprivation $(h)$'}
                for ii in ['feeding']:
                    self.figs[ii] = self.G.dict['deb'](mode=ii, save_as=f'3_POST-STARVATION_{ii}.pdf',
                                                       include_egg=False,
                                                       label_epochs=False, **kwargs)
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = self.G.dict['lineplot'](par_shorts=[s],
                                                                      save_as=f'3_POST-STARVATION_{p}.pdf',
                                                                      **kwargs)

            elif exp == 'REARING-DEPENDENT INTAKE':
                kwargs = {**dsNls(ds0),
                          'coupled_labels': [int(q * 100) for q in self.qs],
                          'xlabel': 'Food quality (%)'
                          }
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = self.G.dict['barplot'](par_shorts=[s], save_as=f'4_REARING_{p}.pdf',
                                                                     **kwargs)

            elif exp == 'REFEEDING AFTER 3h STARVED':
                h = self.h_refeeding
                n = f'5_REFEEDING_after_{h}h_starvation_'
                kwargs = dsNls(ds0)
                self.figs[f'{exp} food-intake'] = self.G.dict['food intake (timeplot)'](scaled=True,
                                                                                        save_as=f'{n}scaled_intake.pdf',
                                                                                        **kwargs)
                self.figs[f'{exp} food-intake(filt)'] = self.G.dict['food intake (timeplot)'](scaled=True,
                                                                                              filt_amount=True,
                                                                                              save_as=f'{n}scaled_intake_filt.pdf',
                                                                                              **kwargs)
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = self.G.dict['timeplot'](par_shorts=[s], show_first=False,
                                                                      subfolder=None,
                                                                      save_as=f'{n}{p}.pdf', **kwargs)


class DoublePatch_Essay(Essay):
    def __init__(self, substrates=['sucrose', 'standard', 'cornmeal'], dur=5.0, arena_dims=(0.24, 0.24), patch_x=0.06,
                 patch_radius=0.025, **kwargs):
        super().__init__(type='DoublePatch', enrichment=I.enr_dict(proc=['spatial', 'angular', 'source'],
                                                                      bouts=['stride', 'pause', 'turn'],
                                                                      fits=False, interference=False, on_food=True),
                         collections=['pose', 'toucher', 'feeder', 'olfactor'], **kwargs)
        self.arena_dims = arena_dims
        self.patch_x = patch_x
        self.patch_radius = patch_radius
        self.substrates = substrates
        self.dur = dur
        self.exp_dict = self.time_ratio_exp()
        mIDs = ['rover', 'sitter']
        self.mdiff_df, row_colors = self.M.diff_df(mIDs=mIDs,
                                       ms=[self.M.loadConf(f'navigator_{mID}') for mID in mIDs])




    # def get_larvagroups_old(self, type='standard'):
    #     age = 72.0
    #     kws = {
    #         'distribution': preg.get_null('larva_distro', N=self.N, scale=(0.005, 0.005)),
    #         'life_history': preg.get_null('life_history', age=age,
    #                                       epochs={0: preg.get_null('epoch', start=0.0, stop=age,
    #                                                                substrate=preg.get_null('substrate'))}),
    #         'odor': preg.get_null('odor'),
    #         'sample': 'None.150controls'
    #     }
    #     lgs = {}
    #     mID0s = ['rover', 'sitter']
    #     mcols = ['blue', 'red']
    #     for mID0, mcol in zip(mID0s, mcols):
    #         mID = f'navigator_{mID0}'
    #         lgs[f'{type}_{mID0}'] = preg.get_null('LarvaGroup', default_color=mcol,
    #                                               model=self.M.loadConf(mID), **kws)
    #     return lgs

    def get_larvagroups(self, type='standard'):
        age = 72.0

        kws0 = {
            'kwdic': {
                'distribution': {'N': self.N, 'scale': (0.005, 0.005)},
                'life_history': {'age': age, 'epochs': self.GT.dict.epoch.entry(0, start=0.0, stop=age)},
            'odor':{}
            },
            'sample': 'None.150controls',
        }

        mcols = ['blue', 'red']
        mID0s = ['rover', 'sitter']
        lgs = {}
        for mID0, mcol in zip(mID0s, mcols):
            kws = {
                'default_color': mcol,
                'model': self.CT.dict.Model.loadConf(f'navigator_{mID0}'),
                **kws0
            }

            lgs.update(self.GT.dict.LarvaGroup.entry(id=f'{type}_{mID0}', **kws))
        return lgs

    def get_sources(self, type='standard'):
        o = preg.get_null('odor', odor_id='Odor', odor_intensity=2.0, odor_spread=0.0002)
        sus = {
            'Left_patch': preg.get_null('source', pos=(-self.patch_x, 0.0), default_color='green', group='Source',
                                        radius=self.patch_radius,
                                        amount=0.1, odor=o, type=type),
            'Right_patch': preg.get_null('source', pos=(self.patch_x, 0.0), default_color='green', group='Source',
                                         radius=self.patch_radius,
                                         amount=0.1, odor=o, type=type)
        }
        return sus

    def patch_env(self, type='standard'):
        sus = self.get_sources(type=type)

        conf = preg.get_null('env_conf',
                             arena=preg.get_null('arena', arena_shape='rectangular', arena_dims=self.arena_dims),
                             food_params={'source_groups': {},
                                          'food_grid': None,
                                          'source_units': sus},
                             odorscape={'odorscape': 'Gaussian'})

        return conf

    def time_ratio_exp(self, exp='double_patch'):
        confs = []
        for n in self.substrates:
            kws = {
                'env': self.patch_env(type=n),
                'lgs': self.get_larvagroups(type=n),
                'id': f'{exp}_{n}_{self.dur}min',
                'dur': self.dur,
                'exp': exp
            }
            confs.append(self.conf(**kws))
        return {exp: confs}

    def global_anal(self):
        kwargs = {
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': self.show,
            'title': f"DOUBLE PATCH ESSAY (N={self.N}, duration={self.dur}')",
            'mdiff_df': self.mdiff_df
        }
        # entry = self.G.entry('double-patch summary', args={})
        self.figs.update(self.G.eval0(entry=self.G.entry('double-patch summary', args={}), **kwargs))

    def analyze(self, exp, ds0):
        pass
        # kwargs = {'datasets': flatten_list(ds0),
        #           'save_to': self.plot_dir,
        #           # 'save_as': exp,
        #           'show': self.show}
        # entry = self.G.entry(ID='double patch',title=exp, args=kwargs)
        # self.figs[exp]=self.G.eval0(entry=entry)
        # self.figs.update(self.G.eval0(entry, **kwargs))
        # self.figs[exp] = fig
        # self.figs[exp] = self.G.dict['double patch'](**kwargs)


class Chemotaxis_Essay(Essay):
    def __init__(self, dur=5.0, gain=300.0,mode=1, **kwargs):
        super().__init__(type='Chemotaxis',
                         enrichment=I.enr_dict(proc=['spatial', 'angular', 'source'],
                                                  bouts=[], fits=False, interference=False, on_food=False),
                         collections=['pose', 'olfactor'], **kwargs)
        self.time_ks = ['c_odor1', 'dc_odor1']
        self.dur = dur
        self.gain = gain
        # self.mID0 = mID0
        if mode==1 :
            self.models = self.get_models1(gain)
        elif mode==2 :
            self.models = self.get_models2(gain)
        elif mode==3 :
            self.models = self.get_models3(gain)
        elif mode==4 :
            self.models = self.get_models4(gain)
        self.mdiff_df, row_colors = self.M.diff_df(mIDs=list(self.models.keys()), ms=[v.model for v in self.models.values()])
        self.exp_dict = self.chemo_exps(self.models)



    def get_models1(self, gain):
        mID0='RE_NEU_SQ_DEF_nav'
        o='brain.olfactor_params'
        mW = self.M.newConf(mID0=mID0, kwargs={f'{o}.odor_dict.Odor.mean': gain,
                                               f'{o}.perception': 'log'})
        mWlin = self.M.newConf(mID0=mID0, kwargs={f'{o}.odor_dict.Odor.mean': gain,
                                                  f'{o}.perception': 'linear'})
        mC = self.M.newConf(mID0=mID0, kwargs={f'{o}.odor_dict.Odor.mean': 0})
        mT = self.M.newConf(mID0=mID0, kwargs={f'{o}.odor_dict.Odor.mean': gain,
                                               f'{o}.perception': 'log',
                                               f'{o}.brute_force': True})
        mTlin = self.M.newConf(mID0=mID0, kwargs={f'{o}.odor_dict.Odor.mean': gain,
                                                  f'{o}.brute_force': True,
                                                  f'{o}.perception': 'linear'})

        T = 'Tastekin'
        W = 'Wystrach'
        models = {

            f'{T} (log)': {'model': mT, 'color': 'red'},
            f'{T} (lin)': {'model': mTlin, 'color': 'darkred'},
            f'{W} (log)': {'model': mW, 'color': 'lightgreen'},
            f'{W} (lin)': {'model': mWlin, 'color': 'darkgreen'},
            'controls': {'model': mC, 'color': 'magenta'},

        }
        return dNl.NestDict(models)

    def get_models2(self, gain):
        cols=cNs.N_colors(6)
        i=0
        models={}
        for Tmod in ['NEU', 'SIN'] :
            for Ifmod in ['PHI', 'SQ', 'DEF'] :
                mID0=f'RE_{Tmod}_{Ifmod}_DEF_nav'
                models[f'{Tmod}_{Ifmod}'] = {'model': self.M.newConf(mID0=mID0, kwargs={
                    f'brain.olfactor_params.brute_force': True,
                    f'brain.olfactor_params.odor_dict.Odor.mean': gain,
                    f'brain.interference_params.attenuation': 0.1,
                    f'brain.interference_params.attenuation_max': 0.0,
                }), 'color': cols[i]}
                i+=1
        return dNl.NestDict(models)

    def get_models3(self, gain):
        cols=cNs.N_colors(6)
        i=0
        models={}
        for Tmod in ['NEU', 'SIN'] :
            for Ifmod in ['PHI', 'SQ', 'DEF'] :
                mID0=f'RE_{Tmod}_{Ifmod}_DEF_nav'
                models[f'{Tmod}_{Ifmod}'] = {'model': self.M.newConf(mID0=mID0, kwargs={
                    f'brain.olfactor_params.perception': 'log',
                    f'brain.olfactor_params.decay_coef': 0.1,
                    f'brain.olfactor_params.odor_dict.Odor.mean': gain,
                    f'brain.interference_params.attenuation': 0.1,
                    f'brain.interference_params.attenuation_max': 0.9,
                }), 'color': cols[i]}
                i+=1

        return dNl.NestDict(models)

    def get_models4(self, gain):
        cols=cNs.N_colors(4)
        i=0
        models={}
        for Tmod in ['NEU', 'SIN'] :
            for Ifmod in ['PHI', 'SQ'] :
                mID0=f'RE_{Tmod}_{Ifmod}_DEF_var2_nav'
                models[f'{Tmod}_{Ifmod}'] = {'model': self.M.newConf(mID0=mID0, kwargs={
                    f'brain.olfactor_params.perception': 'log',
                    f'brain.olfactor_params.decay_coef': 0.1,
                    f'brain.olfactor_params.odor_dict.Odor.mean': gain,
                }), 'color': cols[i]}
                i+=1

        return dNl.NestDict(models)

    def chemo_exps(self,models):
        lg_kws = {
            'odor': preg.get_null('odor'),
            'sample': 'None.150controls'
        }
        kws = {
            'arena': preg.get_null('arena', arena_shape='rectangular', arena_dims=(0.1, 0.06)),
            'odorscape': {'odorscape': 'Gaussian'},
            'windscape': None,
            'thermoscape': None,
            'border_list': {},
        }

        exp1 = 'Orbiting behavior'
        kws1 = {
            'env': preg.get_null('env_conf',
                                 food_params={'source_groups': {},
                                              'food_grid': None,
                                              'source_units': {
                                                  'Source': preg.get_null('source', pos=(0.0, 0.0),
                                                                          group='Source',
                                                                          odor=preg.get_null('odor',
                                                                                             odor_id='Odor',
                                                                                             odor_intensity=2.0,
                                                                                             odor_spread=0.0002)
                                                                          ),
                                              }
                                              }, **kws),
            'lgs': {
                mID: preg.get_null('LarvaGroup',
                                   distribution=preg.get_null('larva_distro', N=self.N, mode='uniform'),
                                   default_color=dic['color'], model=dic['model'], **lg_kws) for mID, dic in
                models.items()},
            'id': f'{exp1}_exp',
            'dur': self.dur,
            'exp': exp1
        }

        exp2 = 'Up-gradient navigation'
        kws2 = {
            'env': preg.get_null('env_conf',
                                 food_params={'source_groups': {},
                                              'food_grid': None,
                                              'source_units': {
                                                  'Source': preg.get_null('source', pos=(0.04, 0.0),
                                                                          group='Source',
                                                                          odor=preg.get_null('odor',
                                                                                             odor_id='Odor',
                                                                                             odor_intensity=8.0,
                                                                                             odor_spread=0.0004)),
                                              }
                                              }, **kws),
            'lgs': {
                mID: preg.get_null('LarvaGroup',
                                   distribution=preg.get_null('larva_distro', N=self.N, mode='uniform',
                                                              loc=(-0.04, 0.0),
                                                              orientation_range=(-30.0, 30.0), scale=(0.005, 0.02)),
                                   default_color=dic['color'], model=dic['model'], **lg_kws) for mID, dic in
                models.items()},
            'id': f'{exp2}_exp',
            'dur': self.dur,
            'exp': exp2
        }

        return {exp1: [self.conf(**kws1)], exp2: [self.conf(**kws2)]}

    def analyze(self, exp, ds0):
        pass
        # kwargs = {'datasets': flatten_list(ds0),
        #           'save_to': self.plot_dir,
        #           'show': self.show}
        # entry_list = [
        #     self.G.entry('autoplot', args={
        #         'ks': self.time_ks,
        #         'show_first': False,
        #         'individuals': False,
        #         'subfolder': None,
        #         'unit': 'min',
        #         'name': f'{exp}_timeplot'
        #     }),
        #     self.G.entry('trajectories', args={
        #         'subfolder': None,
        #         'name': f'{exp}_trajectories',
        #     })
        # ]
        # self.figs[exp] = self.G.eval(entry_list, **kwargs)

    def global_anal(self):
        kwargs = {
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': self.show,
            'title': f'CHEMOTAXIS ESSAY (N={self.N})',
        }
        entry = self.G.entry('chemotaxis summary', args={'mdiff_df': self.mdiff_df})
        self.figs.update(self.G.eval0(entry, **kwargs))



def Essay_dict() :
    rover_sitter_essay = {
        'experiments': {
            'pathlength': {
                'exp_types': ['RvsS_off', 'RvsS_on'],
                'durations': [20, 20]
            },
            'intake': {
                'exp_types': ['RvsS_on'] * 3,
                'durations': [10, 15, 20]
            },
            'starvation': {
                'exp_types': [
                    'RvsS_on',
                    'RvsS_on_1h_prestarved',
                    'RvsS_on_2h_prestarved',
                    'RvsS_on_3h_prestarved',
                    'RvsS_on_4h_prestarved',
                ],
                'durations': [5] * 5
            },
            'quality': {
                'exp_types': [
                    'RvsS_on',
                    'RvsS_on_q75',
                    'RvsS_on_q50',
                    'RvsS_on_q25',
                    'RvsS_on_q15',
                ],
                'durations': [5] * 5
            },
            'refeeding': {
                'exp_types': [
                    'RvsS_on_3h_prestarved'
                ],
                'durations': [120]
            }
        },
        'exp_fig_folder': preg.path_dict["RvsS"]}
    d = {
    # 'roversVSsitters': rover_sitter_essay,
    # 'RvsS_essay': {}
}
    return d
if __name__ == "__main__":
    # E = RvsS_Essay(video=False, all_figs=False, show=False, N=1)
    # E = Chemotaxis_Essay(video=False, N=5, dur=5, mode=4)
    E = DoublePatch_Essay(video=True, N=5, dur=5)
    ds = E.run()
    figs, results = E.anal()
