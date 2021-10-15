import shutil
from lib.conf.base.dtypes import enrichment_dict, null_dict, arena
from lib.conf.stored.env_conf import env, f_pars, double_patches
from lib.conf.stored.exp_conf import RvsS_groups
from lib.conf.base import paths



def RvsS_essay(**kwargs) :

    from lib.conf.stored.conf import next_idx
    essay_id='RvsS'
    kws={
        'enrichment' : enrichment_dict(types=['spatial']),
        'collections' : ['pose', 'feeder', 'gut'],
        # 'experiment' : 'RvsS'
    }
    path0=f'{paths.path("ESSAY")}/{essay_id}/{essay_id}_{next_idx(essay_id, type="essay")}'
    path=f'{path0}/data'
    dur1, dur2, dur3=20,5,120
    # dur1, dur2, dur3=10,2,20

    def RvsS_env(on_food=True):
        grid = null_dict('food_grid') if on_food else None
        return null_dict('env_conf', arena=arena(0.02, 0.02),
                         food_params=null_dict('food_params', food_grid=grid),
                         )

    def sim(id, dur) :
        return null_dict('sim_params', sim_ID=id, path=path, duration=dur)

    def conf(exp, id, dur, on_food=True, l_kws={}):
        return null_dict('exp_conf', sim_params=sim(id, dur), env_params=RvsS_env(on_food=on_food),
                         larva_groups=RvsS_groups(**l_kws), experiment=exp, **kws)

    def pathlength_exp(dur=dur1, exp='pathlength') :
        return {exp:[conf(exp, f'{exp}_{n}_{dur}min',dur, on_food=nb) for n,nb in zip(['agar', 'yeast'], [False, True])]}

    def intake_exp(durs=[10, 15, 20], exp='intake') :
        return {exp:[conf(exp, f'{exp}_{dur}min',dur) for dur in durs]}

    def starvation_exp(hs = [0, 1, 2, 3, 4], dur=dur2, exp='starvation') :
        return {exp:[conf(exp, f'{exp}_{h}h_{dur}min',dur, l_kws={'h_starved' : h}) for h in hs]}

    def quality_exp(qs=[1.0, 0.75, 0.5, 0.25, 0.15], dur=dur2, exp='quality') :
        return {exp:[conf(exp, f'{exp}_{q}_{dur}min',dur, l_kws={'q' : q}) for q in qs]}

    def refeeding_exp(h=3, dur=dur3, exp='refeeding') :
        return {exp:[conf(exp, f'{exp}_{h}h_{dur}min',dur, l_kws={'h_starved' : h}) for h in [h]]}

    exps={**pathlength_exp(), **intake_exp(), **starvation_exp(), **quality_exp(), **refeeding_exp()}
    return exps, {'path' : path0, 'essay_type' : essay_id}

def run_RvsS_essay(**kwargs) :
    from lib.sim.single.analysis import essay_analysis
    from lib.sim.single.single_run import SingleRun
    exps, essay_kws = RvsS_essay(**kwargs)
    figs = {}
    results = {}
    for exp, exp_confs in exps.items():
        ds0 = [SingleRun(**c).run() for c in exp_confs]
        # ds0 = [run_sim(**c) for c in exp_confs]
        if ds0 is not None:
            fig_dict, res = essay_analysis(exp=exp, ds0=ds0, **essay_kws)
            figs.update(fig_dict)
            results[exp] = res
    shutil.rmtree(f'{essay_kws["path"]}/data')
    return figs, results

def double_patch_essay(substrates=['sucrose', 'standard', 'cornmeal'],N=5, dur=5.0, **kwargs):
    from lib.conf.stored.conf import next_idx
    essay_id = 'double_patch'
    kws = {
        'enrichment': enrichment_dict(types=['spatial','angular', 'source'], bouts=['stride', 'pause', 'turn'], fits=False, on_food=True),
        'collections': ['pose', 'toucher', 'feeder', 'olfactor'],
        # 'experiment' : 'RvsS'
    }
    path0 = f'{paths.path("ESSAY")}/{essay_id}/{essay_id}_{next_idx(essay_id, type="essay")}'
    path = f'{path0}/data'

    def double_patch_env(type='standard'):
        return env(arena(0.24, 0.24),
                        f_pars(su=double_patches(type)),
                        'G')

    def sim(id, dur):
        return null_dict('sim_params', sim_ID=id, path=path, duration=dur, store_data=False)

    def conf(exp, id, dur, type='standard', l_kws={}):
        lgs=RvsS_groups(expand=True, age=72.0, **l_kws)
        return null_dict('exp_conf', sim_params=sim(id, dur), env_params=double_patch_env(type=type),
                         larva_groups=lgs, experiment=exp,trials={}, **kws)

    def time_ratio_exp(dur=dur, exp='double_patch'):
        return {
            exp: [conf(exp, f'{exp}_{n}_{dur}min', dur, type=n, l_kws={'N':N, 'navigator' : True, 'pref' : f'{n}_'}) for n in substrates]}

    # def intake_exp(durs=[10, 15, 20], exp='intake'):
    #     return {exp: [conf(exp, f'{exp}_{dur}min', dur) for dur in durs]}
    #
    # def starvation_exp(hs=[0, 1, 2, 3, 4], dur=dur2, exp='starvation'):
    #     return {exp: [conf(exp, f'{exp}_{h}h_{dur}min', dur, l_kws={'h_starved': h}) for h in hs]}
    #
    # def quality_exp(qs=[1.0, 0.75, 0.5, 0.25, 0.15], dur=dur2, exp='quality'):
    #     return {exp: [conf(exp, f'{exp}_{q}_{dur}min', dur, l_kws={'q': q}) for q in qs]}
    #
    # def refeeding_exp(h=3, dur=dur3, exp='refeeding'):
    #     return {exp: [conf(exp, f'{exp}_{h}h_{dur}min', dur, l_kws={'h_starved': h}) for h in [h]]}

    exps = {**time_ratio_exp()}
    # exps = {**pathlength_exp(), **intake_exp(), **starvation_exp(), **quality_exp(), **refeeding_exp()}
    return exps, {'path': path0, 'essay_type': essay_id}


def run_double_patch_essay(**kwargs) :
    from lib.sim.single.analysis import essay_analysis
    from lib.sim.single.single_run import SingleRun
    exps, essay_kws = double_patch_essay(**kwargs)
    figs = {}
    results = {}
    for exp, exp_confs in exps.items():
        ds0 = [SingleRun(**c).run() for c in exp_confs]
        # ds0 = [run_sim(**c) for c in exp_confs]
        if ds0 is not None:
            fig_dict, res = essay_analysis(exp=exp, ds0=ds0, **essay_kws)
            figs.update(fig_dict)
            results[exp] = res
    shutil.rmtree(f'{essay_kws["path"]}/data')
    return figs, results

rover_sitter_essay = {
    'experiments':{
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
'exp_fig_folder' : paths.path('RvsS')}

essay_dict = {
    'roversVSsitters': rover_sitter_essay,
    'RvsS_essay' : {}
              }

if __name__ == "__main__":
    figs, results=run_double_patch_essay()
    print(results)


