import shutil

from lib.conf.conf import next_idx
from lib.conf.init_dtypes import enrichment_dict
from lib.conf.env_conf import *
from lib.sim.analysis import essay_analysis
from lib.sim.single_run import run_sim
from lib.stor import paths

def RvsS_essay(**kwargs) :
    essay_id='RvsS'
    kws={
        'enrichment' : enrichment_dict(types=['spatial']),
        'collections' : ['pose', 'feeder', 'gut'],
        # 'experiment' : 'RvsS'
    }
    path0=f'{paths.EssayFolder}/{essay_id}/{essay_id}_{next_idx(essay_id, type="essay")}'
    path=f'{path0}/data'
    dur1, dur2, dur3=20,5,120
    # dur1, dur2, dur3=10,2,20

    def sim(id, dur) :
        return null_dict('sim_params', sim_ID=id, path=path, duration=dur)

    def conf(exp, id, dur, env_kws={}):
        return null_dict('exp_conf', sim_params=sim(id, dur), env_params=RvsS_env(**env_kws, **kwargs),experiment=exp, **kws)

    def pathlength_exp(dur=dur1, exp='pathlength') :
        return {exp:[conf(exp, f'{exp}_{n}_{dur}min',dur, env_kws={'on_food' : nb}) for n,nb in zip(['agar', 'yeast'], [False, True])]}

    def intake_exp(durs=[10, 15, 20], exp='intake') :
        return {exp:[conf(exp, f'{exp}_{dur}min',dur) for dur in durs]}

    def starvation_exp(hs = [0, 1, 2, 3, 4], dur=dur2, exp='starvation') :
        return {exp:[conf(exp, f'{exp}_{h}h_{dur}min',dur, env_kws={'h_starved' : h}) for h in hs]}

    def quality_exp(qs=[1.0, 0.75, 0.5, 0.25, 0.15], dur=dur2, exp='quality') :
        return {exp:[conf(exp, f'{exp}_{q}_{dur}min',dur, env_kws={'q' : q}) for q in qs]}

    def refeeding_exp(h=3, dur=dur3, exp='refeeding') :
        return {exp:[conf(exp, f'{exp}_{h}h_{dur}min',dur, env_kws={'h_starved' : h}) for h in [h]]}

    exps={**pathlength_exp(), **intake_exp(), **starvation_exp(), **quality_exp(), **refeeding_exp()}
    return exps, {'path' : path0, 'essay_type' : essay_id}

def run_RvsS_essay(**kwargs) :
    exps, essay_kws = RvsS_essay(**kwargs)
    figs = {}
    results = {}
    for exp, exp_confs in exps.items():
        ds0 = [run_sim(**c) for c in exp_confs]
        if ds0 is not None:
            fig_dict, res = essay_analysis(exp=exp, ds0=ds0, **essay_kws)
            figs.update(fig_dict)
            results[exp] = res
    shutil.rmtree(f'{essay_kws["path"]}/data')
    return figs, results



rover_sitter_essay = {
    'experiments':{
    'pathlength': {
        'exp_types': ['rovers_sitters_on_agar', 'rovers_sitters_on_food'],
        'durations': [20, 20]
    },
    'intake': {
        'exp_types': ['rovers_sitters_on_food'] * 3,
        'durations': [10, 15, 20]
    },
    'starvation': {
        'exp_types': [
            'rovers_sitters_on_food',
            'rovers_sitters_on_food_1h_prestarved',
            'rovers_sitters_on_food_2h_prestarved',
            'rovers_sitters_on_food_3h_prestarved',
            'rovers_sitters_on_food_4h_prestarved',
        ],
        'durations': [5] * 5
    },
    'quality': {
        'exp_types': [
            'rovers_sitters_on_food',
            'rovers_sitters_on_food_q75',
            'rovers_sitters_on_food_q50',
            'rovers_sitters_on_food_q25',
            'rovers_sitters_on_food_q15',
        ],
        'durations': [5] * 5
    },
    'refeeding': {
        'exp_types': [
            'rovers_sitters_on_food_3h_prestarved'
        ],
        'durations': [120]
    }
},
'exp_fig_folder' : paths.RoverSitterFigFolder}

essay_dict = {
    'roversVSsitters': rover_sitter_essay,
    'RvsS_essay' : {}
              }

if __name__ == "__main__":
    figs, results=run_RvsS_essay()
    print(results)


