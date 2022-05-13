import shutil

from lib.anal.plotting import plot_pathlength, barplot, timeplot, plot_food_amount, lineplot, plot_debs, \
    boxplot_double_patch
from lib.aux.dictsNlists import flatten_list
from lib.conf.base.dtypes import enr_dict, null_dict, arena, base_enrich
from lib.conf.base.par import getPar
from lib.conf.stored.env_conf import env, f_pars, double_patches
from lib.conf.stored.exp_conf import RvsS_groups
from lib.conf.base import paths
from lib.conf.stored.conf import next_idx
from lib.sim.single.single_run import SingleRun


class Essay:
    def __init__(self, type, enrichment=base_enrich(), collections=['pose'], **kwargs):
        self.type = type
        self.enrichment = enrichment
        self.collections = collections
        self.essay_id = f'{type}_{next_idx(type, "Essay")}'
        self.path = f'essays/{type}/{self.essay_id}/data'
        self.full_path = f'{paths.path("ESSAY")}/{type}/{self.essay_id}/data'
        self.plot_dir = f'{paths.path("ESSAY")}/{type}/{self.essay_id}/plots'
        self.exp_dict = {}
        self.figs = {}
        self.results = {}

    def conf(self, exp, id, dur, lgs, env, **kwargs):
        sim = null_dict('sim_params', sim_ID=id, path=self.path, duration=dur)
        return null_dict('exp_conf', sim_params=sim, env_params=env, trials={},
                         larva_groups=lgs, experiment=exp, enrichment=self.enrichment,
                         collections=self.collections, **kwargs)

    def run(self):
        print(f'Essay {self.essay_id}')
        for exp, exp_confs in self.exp_dict.items():
            ds0 = [SingleRun(**c).run() for c in exp_confs]
            if ds0 is not None and len(ds0) != 0 and all([d0 is not None for d0 in ds0]):
                self.analyze(exp=exp, ds0=ds0)
        shutil.rmtree(self.full_path, ignore_errors=True)
        return self.figs, self.results

    def analyze(self, exp, ds0):
        pass
        # return {}, None


class RvsS_Essay(Essay):
    def __init__(self, all_figs=False, **kwargs):
        super().__init__(type='RvsS', enrichment=enr_dict(proc=['spatial']),
                         collections=['pose', 'feeder', 'gut'], **kwargs)
        self.qs=[1.0, 0.75, 0.5, 0.25, 0.15]
        self.hs=[0, 1, 2, 3, 4]
        self.durs=[10, 15, 20]
        self.dur=5
        self.substrates=['Agar', 'Yeast']
        self.exp_dict = {**self.intake_exp(), **self.starvation_exp(),
                         **self.quality_exp(), **self.refeeding_exp(),**self.pathlength_exp()}
        self.all_figs = all_figs

    def RvsS_env(self, on_food=True):
        grid = null_dict('food_grid') if on_food else None
        return null_dict('env_conf', arena=arena(0.02, 0.02),
                         food_params=null_dict('food_params', food_grid=grid),
                         )

    def conf2(self, on_food=True, l_kws={}, **kwargs):
        return self.conf(env=self.RvsS_env(on_food=on_food), lgs=RvsS_groups(expand=True, **l_kws), **kwargs)

    def pathlength_exp(self, dur=20, exp='pathlength'):
        return {
            exp: [self.conf2(exp=exp, id=f'{exp}_{n}_{dur}min', dur=dur, on_food=nb) for n, nb in
                  zip(self.substrates, [False, True])]}

    def intake_exp(self,  exp='intake'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{dur}min', dur=dur) for dur in self.durs]}

    def starvation_exp(self, exp='starvation'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{h}h_{self.dur}min', dur=self.dur, l_kws={'h_starved': h}) for h in self.hs]}

    def quality_exp(self, exp='quality'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{q}_{self.dur}min', dur=self.dur, l_kws={'q': q}) for q in self.qs]}

    def refeeding_exp(self, h=3, dur=120, exp='refeeding'):
        return {exp: [self.conf2(exp=exp, id=f'{exp}_{h}h_{dur}min', dur=dur, l_kws={'h_starved': h}) for h in [h]]}

    def analyze(self, exp, ds0):
        RS_leg_cols = ['black', 'white']
        markers = ['D', 's']
        ls = [r'$for^{R}$', r'$for^{S}$']
        shorts = ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']

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
            self.figs[f'1_{exp}'] = plot_pathlength(scaled=False, save_as=f'1_PATHLENGTH.pdf', unit='cm', **kwargs)

        elif exp == 'intake':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': self.durs,
                      'xlabel': r'Time spent on food $(min)$'}
            self.figs[f'2_{exp}'] = barplot(par_shorts=['sf_am_V'], save_as=f'2_AD_LIBITUM_INTAKE.pdf', **kwargs)
            if self.all_figs:
                for s in shorts:
                    p = getPar(s, to_return=['d'])[0]
                    self.figs[f'{exp} {p}'] = barplot(par_shorts=[s], save_as=f'2_AD_LIBITUM_{p}.pdf', **kwargs)

        elif exp == 'starvation':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': self.hs,
                      'xlabel': r'Food deprivation $(h)$'}
            self.figs[f'3_{exp}'] = lineplot(par_shorts=['f_am_V'], save_as='3_POST-STARVATION_INTAKE.pdf',
                                                 ylabel='Food intake', scale=1000, **kwargs)
            if self.all_figs:
                for ii in ['feeding']:
                    self.figs[ii] = plot_debs(mode=ii, save_as=f'3_POST-STARVATION_{ii}.pdf', include_egg=False,
                                              label_epochs=False, **kwargs)
                for s in shorts:
                    p = getPar(s, to_return=['d'])[0]
                    self.figs[f'{exp} {p}'] = lineplot(par_shorts=[s], save_as=f'3_POST-STARVATION_{p}.pdf',
                                                                 **kwargs)

        elif exp == 'quality':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': [int(q * 100) for q in self.qs],
                      'xlabel': 'Food quality (%)'
                      }
            self.figs[f'4_{exp}'] = barplot(par_shorts=['sf_am_V'], save_as='4_REARING-DEPENDENT_INTAKE.pdf', **kwargs)
            if self.all_figs:
                for s in shorts:
                    p = getPar(s, to_return=['d'])[0]
                    self.figs[f'{exp} {p}'] = barplot(par_shorts=[s], save_as=f'4_REARING_{p}.pdf', **kwargs)

        elif exp == 'refeeding':
            h = 3
            n = f'5_REFEEDING_after_{h}h_starvation_'
            kwargs = dsNls(ds0)
            self.figs[f'5_{exp}'] = plot_food_amount(scaled=True, filt_amount=True, save_as='5_REFEEDING_INTAKE.pdf',
                                                        **kwargs)

            if self.all_figs:
                self.figs[f'{exp} food-intake'] = plot_food_amount(scaled=True, save_as=f'{n}scaled_intake.pdf',**kwargs)
                self.figs[f'{exp} food-intake(filt)'] = plot_food_amount(scaled=True, filt_amount=True,
                                                                             save_as=f'{n}scaled_intake_filt.pdf',
                                                                             **kwargs)
                for s in shorts:
                    p = getPar(s, to_return=['d'])[0]
                    self.figs[f'{exp} {p}'] = timeplot(par_shorts=[s], show_first=False, subfolder=None,
                                                           save_as=f'{n}{p}.pdf', **kwargs)



class Patch_Essay(Essay):
    def __init__(self, substrates=['sucrose', 'standard', 'cornmeal'], N=5, dur=5.0, **kwargs):
        super().__init__(type='Patch', enrichment=enr_dict(proc=['spatial', 'angular', 'source'],
                                                           bouts=['stride', 'pause', 'turn'],
                                                           fits=False, on_food=True),
                         collections=['pose', 'toucher', 'feeder', 'olfactor'], **kwargs)
        self.substrates = substrates
        self.N = N
        self.dur = dur
        self.exp_dict = {**self.time_ratio_exp()}

    def patch_env(self, type='standard'):
        return env(arena(0.24, 0.24),
                   f_pars(su=double_patches(type)),
                   'G')

    def conf2(self, type='standard', l_kws={}, **kwargs):
        return self.conf(env=self.patch_env(type=type), lgs=RvsS_groups(expand=True, age=72.0, **l_kws), **kwargs)

    def time_ratio_exp(self, exp='double_patch'):
        return {
            exp: [self.conf2(exp=exp, id=f'{exp}_{n}_{self.dur}min', dur=self.dur, type=n,
                             l_kws={'N': self.N, 'navigator': True, 'pref': f'{n}_'}) for n in self.substrates]}

    def analyze(self, exp, ds0):
        if exp == 'double_patch':
            kwargs = {'datasets': flatten_list(ds0),
                      'save_to': self.plot_dir,
                      'save_as': f'{exp}.pdf',
                      'show': True}
            self.figs[exp] = boxplot_double_patch(**kwargs)



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
    'exp_fig_folder': paths.path('RvsS')}

essay_dict = {
    'roversVSsitters': rover_sitter_essay,
    # 'RvsS_essay': {}
}

if __name__ == "__main__":
    figs, results = RvsS_Essay(all_figs=False).run()
    # figs, results = Patch_Essay().run()
