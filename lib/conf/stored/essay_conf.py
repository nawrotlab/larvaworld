import copy
import shutil

from lib.aux.dictsNlists import flatten_list

from lib.conf.stored.exp_conf import RvsS_groups
from lib.registry.pars import preg
from lib.sim.single.single_run import SingleRun
from lib.aux import dictsNlists as dNl


class Essay:
    def __init__(self, type, enrichment=preg.base_enrich(), collections=['pose'], video=False, show=False, **kwargs):
        if video:
            self.vis_kwargs = preg.get_null('visualization', mode='video', video_speed=60)
        else:
            self.vis_kwargs = preg.get_null('visualization', mode=None)
        self.G = preg.graph_dict
        self.show = show
        self.type = type
        self.enrichment = enrichment
        self.collections = collections
        self.essay_id = f'{type}_{preg.next_idx(id=type, conftype="Essay")}'
        self.path = f'essays/{type}/{self.essay_id}/data'
        path = preg.path_dict["ESSAY"]
        # path=paths.path("ESSAY")
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
    def __init__(self, all_figs=False, **kwargs):
        super().__init__(type='RvsS', enrichment=preg.enr_dict(proc=['spatial']),
                         collections=['pose', 'feeder', 'gut'], **kwargs)
        self.qs = [1.0, 0.75, 0.5, 0.25, 0.15]
        self.hs = [0, 1, 2, 3, 4]
        self.durs = [10, 15, 20]
        self.dur = 5
        self.substrates = ['Agar', 'Yeast']
        self.exp_dict = {**self.intake_exp(), **self.starvation_exp(),
                         **self.quality_exp(), **self.refeeding_exp(), **self.pathlength_exp()}
        self.all_figs = all_figs

    def RvsS_env(self, on_food=True):
        grid = preg.get_null('food_grid') if on_food else None
        return preg.get_null('env_conf',
                             arena=preg.get_null('arena', arena_shape='rectangular', arena_dims=(0.02, 0.02)),
                             food_params=preg.get_null('food_params', food_grid=grid),
                             )

    def conf2(self, on_food=True, l_kws={}, **kwargs):
        return self.conf(env=self.RvsS_env(on_food=on_food), lgs=RvsS_groups(expand=True, **l_kws), **kwargs)

    def pathlength_exp(self, dur=20, exp='pathlength'):
        return {
            exp: [self.conf2(exp=exp, id=f'{exp}_{n}_{dur}min', dur=dur, on_food=nb) for n, nb in
                  zip(self.substrates, [False, True])]}

    def intake_exp(self, exp='intake'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{dur}min', dur=dur) for dur in self.durs]}

    def starvation_exp(self, exp='starvation'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{h}h_{self.dur}min', dur=self.dur, l_kws={'h_starved': h}) for h in
                      self.hs]}

    def quality_exp(self, exp='quality'):
        return {
            exp: [self.conf2(exp=exp, id=f'{exp}_{q}_{self.dur}min', dur=self.dur, l_kws={'q': q}) for q in self.qs]}

    def refeeding_exp(self, h=3, dur=120, exp='refeeding'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{h}h_{dur}min', dur=dur, l_kws={'h_starved': h}) for h in [h]]}

    def analyze(self, exp, ds0):
        RS_leg_cols = ['black', 'white']
        markers = ['D', 's']
        ls = [r'$for^{R}$', r'$for^{S}$']
        shorts = ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']
        pars = preg.getPar(shorts)

        def dsNls(ds0, lls=None):
            if lls is None:
                lls = flatten_list([ls] * len(ds0))
            dds = flatten_list(ds0)
            deb_dicts = flatten_list([d.load_dicts('deb') for d in dds])
            return {'datasets': dds,
                    'labels': lls,
                    'deb_dicts': deb_dicts,
                    'save_to': self.plot_dir,
                    'leg_cols': RS_leg_cols,
                    'markers': markers,
                    }

        if exp == 'pathlength':
            lls = flatten_list([[rf'{s} $for^{"R"}$', rf'{s} $for^{"S"}$'] for s in self.substrates])
            kwargs = {
                **dsNls(ds0, lls),
                'xlabel': r'time on substrate $(min)$',
            }
            self.figs[f'1_{exp}'] = preg.graph_dict['pathlength'](scaled=False, save_as=f'1_PATHLENGTH.pdf', unit='cm',
                                                                  **kwargs)

        elif exp == 'intake':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': self.durs,
                      'xlabel': r'Time spent on food $(min)$'}
            self.figs[f'2_{exp}'] = preg.graph_dict['barplot'](par_shorts=['sf_am_V'],
                                                               save_as=f'2_AD_LIBITUM_INTAKE.pdf', **kwargs)
            if self.all_figs:
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = preg.graph_dict['barplot'](par_shorts=[s],
                                                                         save_as=f'2_AD_LIBITUM_{p}.pdf', **kwargs)

        elif exp == 'starvation':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': self.hs,
                      'xlabel': r'Food deprivation $(h)$'}
            self.figs[f'3_{exp}'] = preg.graph_dict['lineplot'](par_shorts=['f_am_V'],
                                                                save_as='3_POST-STARVATION_INTAKE.pdf',
                                                                ylabel='Food intake', scale=1000, **kwargs)
            if self.all_figs:
                for ii in ['feeding']:
                    self.figs[ii] = preg.graph_dict['deb'](mode=ii, save_as=f'3_POST-STARVATION_{ii}.pdf',
                                                           include_egg=False,
                                                           label_epochs=False, **kwargs)
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = preg.graph_dict['lineplot'](par_shorts=[s],
                                                                          save_as=f'3_POST-STARVATION_{p}.pdf',
                                                                          **kwargs)

        elif exp == 'quality':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': [int(q * 100) for q in self.qs],
                      'xlabel': 'Food quality (%)'
                      }
            self.figs[f'4_{exp}'] = preg.graph_dict['barplot'](par_shorts=['sf_am_V'],
                                                               save_as='4_REARING-DEPENDENT_INTAKE.pdf', **kwargs)
            if self.all_figs:
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = preg.graph_dict['barplot'](par_shorts=[s], save_as=f'4_REARING_{p}.pdf',
                                                                         **kwargs)

        elif exp == 'refeeding':
            h = 3
            n = f'5_REFEEDING_after_{h}h_starvation_'
            kwargs = dsNls(ds0)
            self.figs[f'5_{exp}'] = preg.graph_dict['food intake (timeplot)'](scaled=True, filt_amount=True,
                                                                              save_as='5_REFEEDING_INTAKE.pdf',
                                                                              **kwargs)

            if self.all_figs:
                self.figs[f'{exp} food-intake'] = preg.graph_dict['food intake (timeplot)'](scaled=True,
                                                                                            save_as=f'{n}scaled_intake.pdf',
                                                                                            **kwargs)
                self.figs[f'{exp} food-intake(filt)'] = preg.graph_dict['food intake (timeplot)'](scaled=True,
                                                                                                  filt_amount=True,
                                                                                                  save_as=f'{n}scaled_intake_filt.pdf',
                                                                                                  **kwargs)
                for s, p in zip(shorts, pars):
                    self.figs[f'{exp} {p}'] = preg.graph_dict['timeplot'](par_shorts=[s], show_first=False,
                                                                          subfolder=None,
                                                                          save_as=f'{n}{p}.pdf', **kwargs)


class DoublePatch_Essay(Essay):
    def __init__(self, substrates=['sucrose', 'standard', 'cornmeal'], N=5, dur=5.0, **kwargs):
        super().__init__(type='DoublePatch', enrichment=preg.enr_dict(proc=['spatial', 'angular', 'source'],
                                                                      bouts=['stride', 'pause', 'turn'],
                                                                      fits=False, interference=False, on_food=True),
                         collections=['pose', 'toucher', 'feeder', 'olfactor'], **kwargs)
        self.substrates = substrates
        self.N = N
        self.dur = dur
        self.exp_dict = {**self.time_ratio_exp()}

    def patch_env(self, type='standard'):
        o = preg.get_null('odor', odor_id='Odor', odor_intensity=2.0, odor_spread=0.0002)
        sus = {
            'Left_patch': preg.get_null('source', pos=(-0.06, 0.0), default_color='green', group='Source', radius=0.025,
                                        amount=0.1, odor=o, type=type),
            'Right_patch': preg.get_null('source', pos=(0.06, 0.0), default_color='green', group='Source', radius=0.025,
                                         amount=0.1, odor=o, type=type)
        }

        conf = preg.get_null('env_conf',
                             arena=preg.get_null('arena', arena_shape='rectangular', arena_dims=(0.24, 0.24)),
                             food_params={'source_groups': {},
                                          'food_grid': None,
                                          'source_units': sus},
                             odorscape={'odorscape': 'Gaussian'})

        return conf

    def time_ratio_exp(self, exp='double_patch'):
        es = []
        for n in self.substrates:
            lgs = RvsS_groups(expand=True, age=72.0, N=self.N, navigator=True, pref=f'{n}_')

            e = self.conf(exp=exp, env=self.patch_env(type=n),
                          lgs=lgs,
                          dur=self.dur,
                          id=f'{exp}_{n}_{self.dur}min')
            es.append(e)

        return {exp: es}

    def analyze(self, exp, ds0):
        if exp == 'double_patch':
            kwargs = {'datasets': flatten_list(ds0),
                      'save_to': self.plot_dir,
                      'save_as': f'{exp}.pdf',
                      'show': self.show}
            self.figs[exp] = preg.graph_dict.dict['double patch'](**kwargs)


class Chemotaxis_Essay(Essay):
    def __init__(self, N=5, dur=5.0, gain=50.0, mID0='RE_NEU_PHI_DEF_nav', **kwargs):
        super().__init__(type='Chemotaxis', enrichment=preg.enr_dict(proc=['spatial', 'angular', 'source'],
                                                                     bouts=[], fits=False, interference=False,
                                                                     on_food=False),
                         collections=['pose', 'olfactor'], **kwargs)
        self.time_ks=['c_odor1', 'dc_odor1']
        self.N = N
        self.dur = dur
        self.gain = gain
        self.mID0 = mID0
        self.models = self.get_models(gain, mID0)
        self.exp_dict = self.chemo_exps()

    def get_models(self, gain, mID0):
        mW = preg.loadConf('Model', mID0)
        mW.brain.olfactor_params.odor_dict.Odor.mean = gain

        mC = dNl.NestDict(copy.deepcopy(mW))
        mC.brain.olfactor_params.odor_dict.Odor.mean = 0

        mT = dNl.NestDict(copy.deepcopy(mW))
        mT.brain.olfactor_params.brute_force = True

        models = {

            'Tastekin2018': {'model': mT, 'color': 'red'},
            'Wystrach2016': {'model': mW, 'color': 'green'},
            'controls': {'model': mC, 'color': 'blue'},

        }
        return models

    def chemo_exps(self):
        kws = {
            'arena': preg.get_null('arena', arena_shape='rectangular', arena_dims=(0.1, 0.06)),
            'odorscape': {'odorscape': 'Gaussian'},
        }

        env_mid = preg.get_null('env_conf',
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
                                             }, **kws)

        env_R = preg.get_null('env_conf',
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
                                           }, **kws)

        lgs_mid = {
            mID: preg.get_null('LarvaGroup',
                               distribution=preg.get_null('larva_distro', N=self.N, mode='uniform'),
                               default_color=dic['color'], model=dic['model']) for mID, dic in self.models.items()}

        lgs_R = {
            mID: preg.get_null('LarvaGroup',
                               distribution=preg.get_null('larva_distro', N=self.N, mode='uniform', loc=(-0.04, 0.0),
                                                          orientation_range=(-30.0, 30.0), scale=(0.005, 0.02)),
                               default_color=dic['color'], model=dic['model']) for mID, dic in self.models.items()}

        exp1 = 'Orbiting behavior'
        e1 = self.conf(exp=exp1, env=env_mid,
                       lgs=lgs_mid,
                       dur=self.dur,
                       id=f'{exp1}_exp')

        exp2 = 'Up-gradient navigation'
        e2 = self.conf(exp=exp2, env=env_R,
                       lgs=lgs_R,
                       dur=self.dur,
                       id=f'{exp2}_exp')

        return {exp1: [e1], exp2: [e2]}

    def analyze(self, exp, ds0):


        kwargs = {'datasets': flatten_list(ds0),
                  'save_to': self.plot_dir,
                  'show': self.show}
        entry_list = [
            self.G.entry('autoplot', args={
                'ks': self.time_ks,
                'show_first': False,
                'individuals': False,
                'subfolder': None,
                'unit': 'min',
                'name': f'{exp}_timeplot'
            }),
            self.G.entry('trajectories', args={
                'subfolder': None,
                'name': f'{exp}_trajectories',
            })
        ]
        self.figs[exp] = self.G.eval(entry_list, **kwargs)

    def global_anal(self):
        kwargs = {
            'datasets': self.datasets,
            'save_to': self.plot_dir,
            'show': self.show}
        entry = self.G.entry('chemotaxis summary', args={})
        self.figs['global'] = self.G.eval0(entry, **kwargs)


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

essay_dict = {
    'roversVSsitters': rover_sitter_essay,
    # 'RvsS_essay': {}
}

if __name__ == "__main__":
    # figs, results = RvsS_Essay(all_figs=False).run()
    E = Chemotaxis_Essay(video=False, N=3, dur=3, show=True)
    # E = DoublePatch_Essay(video=False, N=3, dur=3)
    ds = E.run()
    figs, results = E.anal()
