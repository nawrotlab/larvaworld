import random

import numpy as np
from scipy.stats import ks_2samp
from unflatten import unflatten

from lib.anal.plot_aux import plot_quantiles
from lib.aux.dictsNlists import flatten_dict, AttrDict
from lib.aux.xy_aux import eudi5x, eudis5
from lib.conf.base.dtypes import ga_dict
from lib.conf.base.par import ParDict, getPar
from lib.conf.stored.conf import expandConf, copyConf, saveConf
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot

from lib.conf.stored.conf import loadRef
from lib.ga.util.ga_engine import GA_runner
from lib.process.aux import compute_interference_solo

d = loadRef('None.100controls')
c = d.config
target_curve = np.array(c.pooled_cycle_curves.fov)
target_sv_curve = np.array(c.pooled_cycle_curves.sv)


def interference_eval_func(robot, finalize=True):
    if finalize:
        robot.finalize()
    curves=robot.interference_curves(strict=False)
    fov_curve = curves['fov']
    if any(np.isnan(fov_curve)):
        return -np.inf
    else:
        RSS = np.nanmean(np.sqrt(np.nansum((fov_curve - target_curve) ** 2)))
        return -RSS


def interference_plot_func(robots):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    Nbins = 64
    x = np.linspace(0, 2 * np.pi, Nbins)
    fov_curves = np.zeros([len(robots), Nbins]) * np.nan
    for i, robot in enumerate(robots):
        curves=robot.interference_curves(strict=False)
        fov_curves[i, :] = curves['fov']
        # ax.plot(x,fov_curve, color='grey')
    plot_quantiles(fov_curves, from_np=True, x=x, axis=ax, color_shading='red')
    fov_curve_mu = np.nanquantile(fov_curves, q=0.5, axis=0)
    RSS = int(np.nanmean(np.nansum((fov_curve_mu - target_curve) ** 2)))
    ax.plot(x, target_curve, color='black', linewidth=4)
    ax.set_xlim([0, 2 * np.pi])
    g0 = robots[0].genome
    title = f'Generation {g0.generation_num} - RSS : {RSS}'
    ax.set_title(title)
    plt.show()
    print(title)
    print({k.split('.')[-1]: getattr(g0, k) for k in g0.space_dict.keys()})
    print()

    mmm1 = flatten_dict(copyConf('Sakagiannis2022', 'Model'))
    for k in g0.space_dict.keys():
        v = getattr(g0, k)
        # print_dic[k.split('.')[-1]] = v
        mmm1[k] = v
    mmm1 = AttrDict.from_nested_dicts(unflatten(mmm1))
    saveConf(conf=mmm1, id='Sakagiannis2022*', conf_type='Model')


interference_space = ga_dict(name='interference', suf='brain.interference_params.', excluded=['feeder_phi_range'])

interference_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        'space_dict': interference_space,
        'fitness_func': interference_eval_func,
        'plot_func': interference_plot_func,
        'Nelits': 4,
        'Nagents': 40,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_Nticks': 400
    },
    'env_params': expandConf('arena_200mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Interference parameter optimization',
    # 'seed': 1,
}


def target(shorts):
    refID = 'None.100controls'
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config
    dic = ParDict(mode='load').dict
    eval = {sh: s[dic[sh]['d']].dropna().values for sh in shorts}
    return eval


eval_shorts = ['b', 'fov', 'foa', 'tor5']
# eval_shorts = ['b', 'fov', 'foa', 'tur_fou', 'tur_fov_max', 'v', 'a', 'run_d', 'run_t', 'pau_t', 'tor5', 'tor20']
eval = target(eval_shorts)
xlabels, xlims, disps, xlabels0 = getPar(eval_shorts, to_return=['l', 'lim', 'd', 'lab'])


def realism_func(robot):
    robot.finalize(eval_shorts)
    return -np.nansum([ks_2samp(eval[p], robot.eval[p])[0] for p in eval_shorts])


def realism_plot(robots):
    Nps = len(eval_shorts)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, Nps, figsize=(5 * Nps, 5), sharey=True)

    axs = axs.ravel() if Nps > 1 else [axs]
    axs[0].set_ylabel('Probability')
    cumKS = 0
    for i, (p, lab) in enumerate(zip(eval_shorts, xlabels0)):
        p0 = np.abs(eval[p])
        p1 = np.abs(np.concatenate([robot.eval[p] for robot in robots]))
        temp = np.concatenate([p0, p1])
        kws = {
            'bins': np.linspace(np.nanmin(temp), np.nanmax(temp), 50),
            'alpha': 0.7,
            'histtype': 'step',
            'linewidth': 3,

        }
        axs[i].hist(p0, color='black', weights=np.ones_like(p0) / float(len(p0)), **kws)
        axs[i].hist(p1, color='red', weights=np.ones_like(p1) / float(len(p1)), **kws)
        axs[i].set_xlabel(lab)
        cumKS += ks_2samp(p0, p1)[0]
    g0 = robots[0].genome
    title = f'Generation {g0.generation_num} - cumKS : {np.round(cumKS, 2)}'
    fig.suptitle(title)
    # plt.sublots_adjust(bottom=0.2)
    plt.show()


physics_space = ga_dict(name='physics', suf='physics.', excluded=['ang_mode', 'ang_vel_coef'])

exploration_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        # 'bestConfID': 'Sakagiannis2022**',
        'space_dict': physics_space,
        'fitness_func': realism_func,
        'plot_func': realism_plot,
        'init_mode': 'base_model',
        'Nelits': 3,
        'Nagents': 30,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_Nticks': 300
    },
    'env_params': expandConf('arena_200mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Realistic exploration',
    # 'seed': 1,
}


def realism_x2_func(robot):
    r1 = int(realism_func(robot) * 10 ** 2)
    r2 = int(interference_eval_func(robot, finalize=False))
    # print(r1,r2)
    return r1 + r2


realism_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        # 'bestConfID': 'Sakagiannis2022**',
        'space_dict': physics_space,
        'fitness_func': realism_x2_func,
        'plot_func': interference_plot_func,
        'init_mode': 'base_model',
        'Nelits': 2,
        'Nagents': 10,
        'Pmutation': 0.9,
        # 'selection_ratio': 1,
        'max_Nticks': 600
    },
    'env_params': expandConf('arena_200mm', 'Env'),
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Realistic exploration',
    # 'seed': 1,
}


def dst2source_mu(robot):
    f = robot.model.get_food()[0]
    p = np.array(f.pos)
    dst = eudi5x(np.array(robot.trajectory), p)
    score = -np.mean(dst) / robot.cum_dst
    # dst = eudis5(robot.trajectory[-1], p)
    # score = -dst / robot.cum_dst
    return score


olf_dic = ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise'])
olf_dic['brain.olfactor_params.odor_dict.Odor.mean'] = {'initial_value': 0.0, 'tooltip': 'Odor gain', 'dtype': float,
                                                        'name': 'Gain', 'min': -100.0, 'max': 1000.0}

# odor_gain_dic=ga_dict(name='odor_gains', suf='brain.olfactor_params.odor_dict.Odor.', excluded=['unique_id', 'std'])
env = expandConf('mid_odor_gaussian', 'Env')
env.arena.arena_dims = (0.2, 0.2)

chemorbit_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022*',
        'bestConfID': 'Sakagiannis2022**',
        'space_dict': olf_dic,
        'fitness_func': dst2source_mu,
        'plot_func': None,
        'Nelits': 5,
        'Nagents': 50,
        'multicore': False,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_Nticks': 600
    },
    'env_params': env,
    'scene_file': '../../ga/saved_scenes/no_boxes.txt',
    'caption': 'Chemorbiting optimization',
    # 'seed': 1,
}


def cum_dst(robot):
    return robot.cum_dst*1000


obstacle_avoidance_kws = {
    'ga_kws': {
        'base_model': 'Sakagiannis2022**',
        'bestConfID': None,
        'space_dict': ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]),
        'fitness_func': cum_dst,
        'robot_class': ObstacleLarvaRobot,
        'plot_func': None,
        'Nelits': 1,
        'Nagents': 10,
        'multicore': False,
        # 'Pmutation': 0,
        # 'selection_ratio': 1,
        'max_dur': 5
    },
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

if __name__ == '__main__':
    # print(ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]).keys())
    # raise
    # GA_runner(**ga_dic['chemorbit'])
    GA_runner(**ga_dic['realism'])
    # GA_runner(**ga_dic['exploration'])
    # GA_runner(**ga_dic['obstacle_avoidance'])
