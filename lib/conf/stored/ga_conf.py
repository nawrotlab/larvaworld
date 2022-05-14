from types import BuiltinFunctionType, FunctionType
from typing import ClassVar

import numpy as np
from scipy.stats import ks_2samp

from lib.anal.plot_aux import plot_quantiles, BasePlot
from lib.aux.dictsNlists import flatten_dict, AttrDict
from lib.aux.xy_aux import eudi5x
from lib.conf.base.dtypes import ga_dict, null_dict
from lib.conf.base.par import ParDict, getPar
from lib.conf.stored.conf import expandConf, copyConf, saveConf, loadConf, next_idx, kConfDict
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot

from lib.ga.util.ga_engine import GAlauncher
from lib.process.aux import compute_interference_solo

ga_spaces = AttrDict.from_nested_dicts({
    'interference': ga_dict(name='interference', suf='brain.interference_params.', excluded=['feeder_phi_range']),
    'turner': ga_dict(name='turner', suf='brain.turner_params.',
                      excluded=['mode', 'noise', 'activation_noise', 'initial_amp', 'amp_range', 'initial_freq',
                                'freq_range', 'activation_range']),
    'physics': ga_dict(name='physics', suf='physics.', excluded=['ang_mode', 'ang_vel_coef', 'bend_correction_coef']),
    'sensorimotor': ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]),
    'olfactor': {**ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise']),
                 'brain.olfactor_params.odor_dict.Odor.mean': {'initial_value': 0.0, 'tooltip': 'Odor gain',
                                                               'dtype': float,
                                                               'name': 'Gain', 'min': -100.0, 'max': 1000.0}}
})


def interference_evaluation(robot, target_fov_curve):
    robot.finalize(eval_shorts=['b', 'fov', 'foa'],)
    fov_curve = robot.interference_curves(strict=False)['fov']
    if any(np.isnan(fov_curve)):
        return -np.inf
    else:
        return - np.nanmean(np.sqrt(np.nansum((fov_curve - target_fov_curve) ** 2)))


def distro_KS_evaluation(robot, eval_shorts, eval_labels, eval):
    robot.finalize(eval_shorts)
    ks = {}
    for p, lab in zip(eval_shorts, eval_labels):
        if robot.eval[p].shape[0] == 0:
            return -np.inf
        else:
            ks[lab] = ks_2samp(eval[p], robot.eval[p])[0]
            if np.isnan(ks[lab]):
                return -np.inf
    robot.genome.fitness_dict = ks
    return -np.mean(list(ks.values()))


def distro_KS_interference_evaluation(robot, eval_shorts, eval_labels, eval, target_fov_curve):
    r1 = int(distro_KS_evaluation(robot, eval_shorts, eval_labels, eval) * 10 ** 2)
    r2 = int(interference_evaluation(robot, target_fov_curve))
    return r1 + r2


def dst2source_evaluation(robot):
    f = robot.model.get_food()[0]
    dst = eudi5x(np.array(robot.trajectory), np.array(f.pos))
    return -np.mean(dst) / robot.cum_dst


def cum_dst(robot):
    return robot.cum_dst/robot.real_length


fitness_funcs = AttrDict.from_nested_dicts({
    'interference': interference_evaluation,
    'distro_KS': distro_KS_evaluation,
    'distro_KS_interference': distro_KS_interference_evaluation,
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})


def interference_plot(robots, generation_num, target_fov_curve, **kwargs):
    P = BasePlot(name=f'interference_generation_{generation_num}', **kwargs)
    P.build()
    Nbins = 64
    x = np.linspace(0, 2 * np.pi, Nbins)
    fov_curves = np.zeros([len(robots), Nbins]) * np.nan
    for i, robot in enumerate(robots):
        curves = robot.interference_curves(strict=False)
        fov_curves[i, :] = curves['fov']
    plot_quantiles(fov_curves, from_np=True, x=x, axis=P.axs[0], color_shading='red')
    fov_curve_mu = np.nanquantile(fov_curves, q=0.5, axis=0)
    RSS = int(np.nanmean(np.nansum((fov_curve_mu - target_fov_curve) ** 2)))
    P.axs[0].plot(x, target_fov_curve, color='black', linewidth=4)
    P.conf_ax(title=f'Generation {generation_num} - RSS : {RSS}', xlim=[0, 2 * np.pi])
    return P.get()


def distro_KS_plot(robots, generation_num, eval_shorts, eval_labels, eval, **kwargs):
    P = BasePlot(name=f'distro_KS_generation_{generation_num}', **kwargs)
    Nps = len(eval_shorts)
    P.build(1, Nps, figsize=(6 * Nps, 6), sharey=True)
    KS_dict = {}
    for i, (p, lab) in enumerate(zip(eval_shorts, eval_labels)):
        p0 = np.abs(eval[p])
        p1 = np.abs(np.concatenate([robot.eval[p] for robot in robots]))
        temp = np.concatenate([p0, p1])
        kws = {
            'bins': np.linspace(np.nanmin(temp), np.nanmax(temp), 50),
            'alpha': 0.7,
            'histtype': 'step',
            'linewidth': 3,
        }
        KS_dict[lab] = np.round(ks_2samp(p0, p1)[0], 2)
        P.axs[i].hist(p0, color='black', weights=np.ones_like(p0) / float(len(p0)), label='experiment', **kws)
        P.axs[i].hist(p1, color='red', weights=np.ones_like(p1) / float(len(p1)), label='simulation', **kws)
        P.conf_ax(i, xlab=lab, ylab='Probability' if i == 0 else None, leg_loc='upper right',
                  title=f'KS distance : {KS_dict[lab]}')
    KS_mean = np.nanmean(list(KS_dict.values()))
    title = f'Generation {generation_num} - mean KS distance : {np.round(KS_mean, 2)}'
    P.fig.suptitle(title)
    P.adjust(LR=(0.2, 0.9), BT=(0.2, 0.8))
    return P.get()


plot_funcs = AttrDict.from_nested_dicts({
    'interference': interference_plot,
    'distro_KS': distro_KS_plot,
})


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 2:
        return True
    else:
        return False


def ga_conf(name, spaceIDs, scene='no_boxes', refID=None, fit_kws={}, dt=0.1, dur=3, N=30, Nel=3, m0='Sakagiannis2022',
            m1=None, sel={}, build={},arena_size=None,
            envID=None, fitID=None, plotID=None, init='random', excl_func=None, robot_class=LarvaRobot, **kwargs):
    space_dict = {}
    for spaceID in spaceIDs:
        space_dict.update(ga_spaces[spaceID])

    build_kws = {
        'fitness_target_refID': refID,
        'fitness_target_kws': fit_kws,
        'base_model': m0,
        'bestConfID': m1,
        'exclude_func': excl_func,
        'init_mode': init,
        'robot_class': robot_class,
        'space_dict': space_dict,
    }

    kws = {'sim_params': null_dict('sim_params', duration=dur, timestep=dt),
           'scene': scene,
           'experiment': name,
           }

    if envID is not None:
        kws['env_params'] = expandConf(envID, 'Env')
        if arena_size is not None :
            kws['env_params'].arena.arena_dims=(arena_size, arena_size)
    if fitID is not None:
        build_kws['fitness_func'] = fitness_funcs[fitID]
    if plotID is not None:
        build_kws['plot_func'] = plot_funcs[plotID]

    kws['ga_select_kws'] = null_dict('ga_select_kws', Nagents=N, Nelits=Nel, **sel)
    kws['ga_build_kws'] = null_dict('ga_build_kws', **build_kws, **build)
    kws.update(kwargs)

    conf = null_dict('GAconf', **kws)
    return {name: conf}


# env = expandConf('mid_odor_gaussian', 'Env')
# env.arena.arena_dims = (0.2, 0.2)

ga_dic = AttrDict.from_nested_dicts({
    **ga_conf('interference', dt=1 / 16,dur=1, refID='None.40controls', m0='fitted_navigator',m1='fitted_navigator',
              fit_kws={'target_fov_curve': None},init='random',
              spaceIDs=['interference'], fitID='interference', plotID='interference',
              Nel=6, N=60, envID='arena_200mm'),
    **ga_conf('exploration', dur=1, dt=1 / 16, refID='None.40controls', m0='40l_explorer',
              m1='40l_explorer', fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'tur_fou', 'tur_t', 'tor5']},
              spaceIDs=['physics'], fitID='distro_KS', plotID='distro_KS',init='random',
              excl_func=bend_error_exclusion,
              Nel=2, N=20, envID='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='None.40controls', m0='average_explorer',m1='average_explorer*',
              fit_kws={'eval_shorts': ['b',  'tur_fou', 'tur_t', 'tor2'], 'target_fov_curve': None},excl_func=bend_error_exclusion,
              spaceIDs=['interference'], fitID='distro_KS_interference', plotID='distro_KS', init='model',
              Nel=2, N=10, envID='arena_200mm'),
    **ga_conf('chemorbit', dur=5, m0='Sakagiannis2022', m1='best_navigator',
              spaceIDs=['olfactor'], fitID='dst2source',
              Nel=5, N=50, envID='mid_odor_gaussian', arena_size=0.2),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              spaceIDs=['sensorimotor'], fitID='cum_dst', robot_class=ObstacleLarvaRobot,
              Nel=2, N=15, envID='dish', init='default',
              scene='obstacle_avoidance_700', arena_size=0.04)
})

#
# ga_dic2 = AttrDict.from_nested_dicts({
#     'interference': {
#     'ga_kws': {
#         'base_model': 'Sakagiannis2022*',
#         'space_dict': ga_spaces.interference,
#         'fitness_func': fitness_funcs.interference,
#         'fitness_target_refID': 'None.100controls',
#         'fitness_target_kws': {'target_fov_curve': None},
#         'plot_func': plot_funcs.interference,
#         'Nelits': 4,
#         'Nagents': 40,
#         # 'Pmutation': 0,
#         # 'selection_ratio': 1,
#         'max_Nticks': 400
#     },
#     'experiment': 'interference',
#     'env_params': expandConf('arena_200mm', 'Env'),
#     'scene_file': '../../ga/saved_scenes/no_boxes.txt'
# },
#     'exploration': {
#     'ga_kws': {
#         'base_model': 'Sakagiannis2022****',
#         'bestConfID': 'Sakagiannis2022****',
#         'space_dict': {**ga_spaces.interference, **ga_spaces.physics, **ga_spaces.turner},
#         'fitness_func': fitness_funcs.distro_KS,
#         'fitness_target_refID': 'None.100controls',
#         'fitness_target_kws': {'eval_shorts': ['b', 'bv', 'ba', 'tur_fou', 'tur_t', 'tor5']},
#         'exclude_func': bend_error_exclusion,
#         'plot_func': plot_funcs.distro_KS,
#         # 'init_mode': 'random',
#         'init_mode': 'base_model',
#         'Nelits': 2,
#         'Nagents': 10,
#         # 'Pmutation': 0.7,
#         # 'Cmutation': 0.5,
#         'max_dur': 0.8
#     },
#     'dt': 1/16,
#     'experiment': 'exploration',
#     'env_params': expandConf('arena_500mm', 'Env'),
#     'scene_file': '../../ga/saved_scenes/no_boxes.txt',
# },
#     'realism': {
#     'ga_kws': {
#         'base_model': 'Sakagiannis2022*',
#         # 'bestConfID': 'Sakagiannis2022**',
#         'space_dict': ga_spaces.physics,
#         'fitness_func': fitness_funcs.distro_KS_interference,
#         'fitness_target_refID': 'None.100controls',
#         'fitness_target_kws': {'eval_shorts': ['b', 'bv', 'ba', 'tur_fou', 'tur_t', 'tor5'],'target_fov_curve':None},
#         'plot_func': plot_funcs.interference,
#         'init_mode': 'base_model',
#         'Nelits': 2,
#         'Nagents': 10,
#         'Pmutation': 0.9,
#         'max_Nticks': 600
#     },
#     'experiment': 'realism',
#     'env_params': expandConf('arena_200mm', 'Env'),
#     'scene_file': '../../ga/saved_scenes/no_boxes.txt',
# },
#     'chemorbit': {
#     'ga_kws': {
#         'base_model': 'Sakagiannis2022*',
#         'bestConfID': 'Sakagiannis2022**',
#         'space_dict': ga_spaces.olfactor,
#         'fitness_func': fitness_funcs.dst2source,
#         'Nelits': 5,
#         'Nagents': 50,
#         'max_Nticks': 600
#     },
#     'experiment': 'chemorbit',
#     'env_params': env,
#     'scene_file': '../../ga/saved_scenes/no_boxes.txt',
# },
#     'obstacle_avoidance': {
#     'ga_kws': {
#         'base_model': 'Sakagiannis2022**',
#         'bestConfID': None,
#         'space_dict': ga_spaces.sensorimotor,
#         'fitness_func': fitness_funcs.cum_dst,
#         'robot_class': ObstacleLarvaRobot,
#         'plot_func': None,
#         'Nelits': 1,
#         'Nagents': 10,
#         'multicore': False,
#         'max_dur': 5
#     },
#     'experiment': 'obstacle_avoidance',
#     'env_params': expandConf('dish', 'Env'),
#     'scene_file': '../../ga/saved_scenes/obstacle_avoidance_600.txt',
# }
# })


if __name__ == '__main__':
    exp = 'interference'
    conf = loadConf(exp, 'Ga')
    #mID=conf.ga_build_kws.base_model
    # if type(mID) == str and mID in kConfDict('Model'):
    #     larva_pars = copyConf(mID, 'Model')
    #print(mID)
    # print(isinstance(conf.ga_build_kws.base_model, str))
    # GAlauncher(**conf)
