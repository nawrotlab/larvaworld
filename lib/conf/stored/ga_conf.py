import random

import numpy as np
from scipy.stats import ks_2samp
from unflatten import unflatten

from lib.anal.plot_aux import plot_quantiles, BasePlot
from lib.aux.dictsNlists import flatten_dict, AttrDict
from lib.aux.xy_aux import eudi5x, eudis5
from lib.conf.base.dtypes import ga_dict
from lib.conf.base.par import ParDict, getPar
from lib.conf.stored.conf import expandConf, copyConf, saveConf, loadConf, next_idx, kConfDict
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot

from lib.conf.stored.conf import loadRef
from lib.ga.util.ga_engine import GA_runner
from lib.process.aux import compute_interference_solo

ga_spaces={
'interference' : ga_dict(name='interference', suf='brain.interference_params.', excluded=['feeder_phi_range']),
'turner' : ga_dict(name='turner', suf='brain.turner_params.',excluded=['mode', 'noise', 'activation_noise', 'initial_amp', 'amp_range', 'initial_freq',
                                 'freq_range', 'activation_range']),
'physics' : ga_dict(name='physics', suf='physics.', excluded=['ang_mode', 'ang_vel_coef']),
'sensorimotor' : ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]),
'olfactor' : {**ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise']),
              'brain.olfactor_params.odor_dict.Odor.mean' : {'initial_value': 0.0, 'tooltip': 'Odor gain', 'dtype': float,
                                                        'name': 'Gain', 'min': -100.0, 'max': 1000.0}}
}

ga_spaces = AttrDict.from_nested_dicts(ga_spaces)
# interference_space = ga_dict(name='interference', suf='brain.interference_params.', excluded=['feeder_phi_range'])
# turner_space = ga_dict(name='turner', suf='brain.turner_params.',
#                        excluded=['mode', 'noise', 'activation_noise', 'initial_amp', 'amp_range', 'initial_freq',
#                                  'freq_range', 'activation_range'])
# physics_space = ga_dict(name='physics', suf='physics.', excluded=['ang_mode', 'ang_vel_coef'])
# sensorimotor_space = ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[])
# olf_dic = ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise'])
# olf_dic['brain.olfactor_params.odor_dict.Odor.mean'] = {'initial_value': 0.0, 'tooltip': 'Odor gain', 'dtype': float,
#
#                                                         'name': 'Gain', 'min': -100.0, 'max': 1000.0}

def interference_evaluation(robot, target_fov_curve):
    robot.finalize()
    curves = robot.interference_curves(strict=False)
    fov_curve = curves['fov']
    if any(np.isnan(fov_curve)):
        return -np.inf
    else:
        RSS = np.nanmean(np.sqrt(np.nansum((fov_curve - target_fov_curve) ** 2)))
        return -RSS

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
    return robot.cum_dst * 1000


fitness_funcs={
'interference' : interference_evaluation,
'distro_KS' : distro_KS_evaluation,
'distro_KS_interference' : distro_KS_interference_evaluation,
'dst2source' : dst2source_evaluation,
'cum_dst' : cum_dst,
# 'dst2source' : dst2source_evaluation,
}

fitness_funcs = AttrDict.from_nested_dicts(fitness_funcs)


def interference_plot(robots,generation_num, target_fov_curve, **kwargs):
    # g0 = robots[0].genome
    # Ngen = g0.generation_num
    P = BasePlot(name=f'GA_interference_generation_{generation_num}', **kwargs)
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
    # g0 = robots[0].genome
    # Ngen = g0.generation_num
    P = BasePlot(name=f'GA_realism_generation_{generation_num}', **kwargs)
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

plot_funcs={
'interference' : interference_plot,
'distro_KS' : distro_KS_plot,
}

plot_funcs = AttrDict.from_nested_dicts(plot_funcs)





interference_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        'space_dict': ga_spaces.interference,
        'fitness_func': fitness_funcs.interference,
        'fitness_target_refID': 'None.100controls',
        'fitness_target_kws': {'target_fov_curve': None},
        'plot_func': plot_funcs.interference,
        'Nelits': 4,
        'Nagents': 40,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_Nticks': 400
    },
    'experiment': 'interference',
    'env_params': expandConf('arena_200mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Interference parameter optimization',
    # 'seed': 1,
}


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 2:
        return True
    else:
        return False








exploration_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022****',
        'bestConfID': 'Sakagiannis2022****',
        'space_dict': {**ga_spaces.interference, **ga_spaces.physics, **ga_spaces.turner},
        # 'space_dict': physics_space,
        'fitness_func': fitness_funcs.distro_KS,
        'fitness_target_refID': 'None.100controls',
        'fitness_target_kws': {'eval_shorts': ['b', 'bv', 'ba', 'tur_fou', 'tur_t', 'tor5']},
        'exclude_func': bend_error_exclusion,
        'plot_func': plot_funcs.distro_KS,
        # 'init_mode': 'random',
        'init_mode': 'base_model',
        'Nelits': 2,
        'Nagents': 10,

        # 'Pmutation': 0.7,
        # 'Cmutation': 0.5,
        # 'selection_ratio': 1,
        'max_dur': 0.8
        # 'max_Nticks': 300
    },
    'dt': 1/16,
    'experiment': 'exploration',
    # 'show_screen': False,
    'env_params': expandConf('arena_500mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Realistic exploration',
    # 'seed': 1,
}




realism_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        # 'bestConfID': 'Sakagiannis2022**',
        # 'space_dict': {**physics_space, **interference_space},
        'space_dict': ga_spaces.physics,
        'fitness_func': fitness_funcs.distro_KS_interference,
        'fitness_target_refID': 'None.100controls',
        'fitness_target_kws': {'eval_shorts': ['b', 'bv', 'ba', 'tur_fou', 'tur_t', 'tor5'],'target_fov_curve':None},
        'plot_func': plot_funcs.interference,
        'init_mode': 'base_model',
        'Nelits': 2,
        'Nagents': 10,
        'Pmutation': 0.9,
        # 'selection_ratio': 1,
        'max_Nticks': 600
    },
    'experiment': 'realism',
    'env_params': expandConf('arena_200mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Realistic exploration',
    # 'seed': 1,
}




env = expandConf('mid_odor_gaussian', 'Env')
env.arena.arena_dims = (0.2, 0.2)

chemorbit_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        'bestConfID': 'Sakagiannis2022**',
        'space_dict': ga_spaces.olfactor,
        'fitness_func': fitness_funcs.dst2source,
        'plot_func': None,
        'Nelits': 5,
        'Nagents': 50,
        'multicore': False,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_Nticks': 600
    },
    'experiment': 'chemorbit',
    'env_params': env,
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Chemorbiting optimization',
    # 'seed': 1,
}





obstacle_avoidance_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022**',
        'bestConfID': None,
        'space_dict': ga_spaces.sensorimotor,
        'fitness_func': fitness_funcs.cum_dst,
        'robot_class': ObstacleLarvaRobot,
        'plot_func': None,
        'Nelits': 1,
        'Nagents': 10,
        'multicore': False,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_dur': 5
    },
    'experiment': 'obstacle_avoidance',
    'env_params': expandConf('dish', 'Env'),
    'scene_file': '../../ga/saved_scenes/obstacle_avoidance_600.txt',
    'caption': 'Obstacle avoidance GA',
    # 'seed': 1,
}

ga_dic = {
    'interference': interference_kws,
    'exploration': exploration_kws,
    'realism': realism_kws,
    'chemorbit': chemorbit_kws,
    'obstacle_avoidance': obstacle_avoidance_kws,
}

ga_dic = AttrDict.from_nested_dicts(ga_dic)

if __name__ == '__main__':
    exp = 'exploration'
    conf=loadConf(exp, 'Ga', use_pickle=True)
    GA_runner(**conf)
