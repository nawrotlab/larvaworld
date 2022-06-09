import numpy as np
from scipy.stats import ks_2samp

from lib.anal.plot_aux import plot_quantiles, BasePlot
from lib.aux.dictsNlists import AttrDict
from lib.aux.xy_aux import eudi5x
from lib.conf.base.dtypes import ga_dict, null_dict
from lib.conf.stored.conf import expandConf
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot
from lib.anal.eval_aux import RSS
from lib.process.aux import detect_strides, mean_stride_curve, cycle_curve_dict

ga_spaces = AttrDict.from_nested_dicts({
    'interference': ga_dict(name='interference', suf='brain.interference_params.',
                            excluded=['feeder_phi_range','crawler_phi_range', 'mode', 'suppression_mode']),
    'turner': ga_dict(name='turner', suf='brain.turner_params.',only=['base_activation'],
                      excluded=['mode', 'noise', 'activation_noise', 'initial_amp', 'amp_range', 'initial_freq',
                                'freq_range', 'activation_range']),
    'physics': ga_dict(name='physics', suf='physics.',only=['torque_coef'],
                       excluded=['ang_mode', 'lin_damping', 'ang_vel_coef', 'bend_correction_coef']),
    'sensorimotor': ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]),
    'olfactor': {**ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise']),
                 'brain.olfactor_params.odor_dict.Odor.mean': {'initial_value': 0.0, 'tooltip': 'Odor gain',
                                                               'dtype': float,
                                                               'name': 'Gain', 'min': -100.0, 'max': 1000.0}}
})
def interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys, **kwargs):
    d1,d2=gdict['cycle_curves'],pooled_cycle_curves
    RSS_dic={sh:RSS(d1[sh][mode] ,np.array(d2[sh][mode])) for sh,mode in cycle_curve_keys.items()}
    return -np.mean(list(RSS_dic.values())),RSS_dic

def interference_evaluation2(robot, pooled_cycle_curves):
    robot_dic=robot.finalize(eval_shorts=['b', 'fov', 'foa', 'rov'])
    from lib.anal.eval_aux import RSS_dic, RSS
    dic=cycle_curve_dict(s=robot_dic,dt=robot.model.dt)
    error_dic = {}
    for sh, target_dic in pooled_cycle_curves.items():
        mode = 'abs' if sh == 'sv' else 'norm'
        error_dic[sh] = RSS(dic[sh][mode] ,np.array(target_dic[mode]))
    # print(error_dic)
    return -np.mean(list(error_dic.values()))


def distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval, **kwargs):
    ks_dic={s:ks_2samp(eval[s], gdict['eval'][s])[0] for s,l in zip(eval_shorts,eval_labels)}
    return -np.mean(list(ks_dic.values())),ks_dic

    # ks = {}
    # for p, lab in zip(eval_shorts, eval_labels):
    #     a=gdict['eval'][p]
    #     if a.shape[0] == 0:
    #         return -np.inf
    #     else:
    #         ks[lab] = ks_2samp(eval[p], a)[0]
    #         if np.isnan(ks[lab]):
    #             return -np.inf
    # # robot.genome.fitness_dict = ks
    # # print(ks)
    # return -np.mean(list(ks.values()))


def distro_KS_evaluation2(robot, eval_shorts, eval_labels, eval):
    # print(robot.unique_id)
    robot_dic=robot.finalize(eval_shorts)
    ks = {}
    for p, lab in zip(eval_shorts, eval_labels):
        if robot_dic[p].shape[0] == 0:
            return -np.inf
        else:
            ks[lab] = ks_2samp(eval[p], robot_dic[p])[0]
            if np.isnan(ks[lab]):
                return -np.inf
    robot.genome.fitness_dict = ks
    # print(ks)
    return -np.mean(list(ks.values()))

def distro_KS_interference_evaluation(gdict, eval_shorts, eval_labels, eval, pooled_cycle_curves, cycle_curve_keys):
    r1,ks_dic = distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval)
    r2,RSS_dic = interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys)
    dic=AttrDict.from_nested_dicts({'KS': ks_dic, 'RSS': RSS_dic})
    if np.isinf(r1) or np.isinf(r2):
        return -np.inf,dic
    else:
        # print(r1,r2)
        return r1 * 10 + r2,dic

def distro_KS_interference_evaluation2(robot, eval_shorts, eval_labels, eval, pooled_cycle_curves):
    r1 = distro_KS_evaluation(robot, eval_shorts, eval_labels, eval)
    r2 = interference_evaluation(robot, pooled_cycle_curves)
    if np.isinf(r1) or np.isinf(r2):
        return -np.inf
    else:
        # print(r1,r2)
        return r1 * 10 + r2

def dst2source_evaluation(gdict, source_xy):

    traj=gdict['step'][['x','y']].values
    dst=np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst=np.sum(dst)
    for label,pos in source_xy.items() :
        #f = pos
        dst2source = eudi5x(traj, np.array(pos))
        break
    return -np.mean(dst2source) / cum_dst, {}

def dst2source_evaluation2(robot):
    f = robot.model.get_food()[0]
    dst = eudi5x(np.array(robot.trajectory), np.array(f.pos))
    return -np.mean(dst) / robot.cum_dst, {}


def cum_dst(robot):
    return robot.cum_dst / robot.real_length


fitness_funcs = AttrDict.from_nested_dicts({
    'interference': interference_evaluation,
    'distro_KS': distro_KS_evaluation,
    'distro_KS_interference': distro_KS_interference_evaluation,
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})


def interference_plot(robots, generation_num, target_fov_curve, **kwargs):
    sh='fov'
    P = BasePlot(name=f'interference_generation_{generation_num}', **kwargs)
    P.build()
    Nbins = 64
    x = np.linspace(0, 2 * np.pi, Nbins)
    curves = np.zeros([len(robots), Nbins]) * np.nan
    for i, robot in enumerate(robots):
        robot_dic=robot.eval.dic
        strides = detect_strides(robot_dic.sv, robot.model.dt, return_runs=False, return_extrema=False)
        da = np.array([np.trapz(robot_dic.fov[s0:s1]) for ii, (s0, s1) in enumerate(strides)])
        curves[i, :] = mean_stride_curve(robot_dic[sh], strides,da)
    plot_quantiles(curves, from_np=True, x=x, axis=P.axs[0], color_shading='red')
    curve_mu = np.nanquantile(curves, q=0.5, axis=0)
    RSS = int(np.nanmean(np.nansum((curve_mu - target_fov_curve) ** 2)))
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
            m1=None, sel={}, build={}, arena_size=None,
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
        if arena_size is not None:
            kws['env_params'].arena.arena_dims = (arena_size, arena_size)
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
    **ga_conf('interference', dt=1 / 16, dur=3, refID='None.150controls', m0='phasic_explorer',
              m1='NEU_PHI',
              fit_kws={'pooled_cycle_curves': ['fov', 'rov', 'foa']}, init='model',
              spaceIDs=['interference','turner'], fitID='interference', plotID='interference',
              Nel=2, N=6, envID='arena_200mm'),
    **ga_conf('exploration', dur=0.5, dt=1 / 16, refID='None.150controls', m0='phasic_explorer',
              m1='NEU_PHI', fit_kws={'eval_shorts': ['b', 'bv', 'ba', 'tur_t', 'tur_fou', 'tor2', 'tor10']},
              spaceIDs=['interference','turner'], fitID='distro_KS', plotID='distro_KS', init='random',
              excl_func=bend_error_exclusion,
              Nel=2, N=10, envID='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='None.150controls', m0='NEU_PHI3', m1='NEU_PHI3',
              fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t'],
              # fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t', 'tor2', 'tor10'],
                       'pooled_cycle_curves': ['fov', 'foa','b', 'rov']},
              excl_func=bend_error_exclusion,
              spaceIDs=['interference', 'turner'], fitID='distro_KS_interference', plotID='distro_KS',
              init='model',
              Nel=2, N=10, envID='arena_200mm'),
    **ga_conf('chemorbit', dur=5, m0='navigator', m1='best_navigator',
              spaceIDs=['olfactor'], fitID='dst2source',fit_kws={'source_xy': None},
              Nel=5, N=50, envID='mid_odor_gaussian', arena_size=0.2),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              spaceIDs=['sensorimotor'], fitID='cum_dst', robot_class=ObstacleLarvaRobot,
              Nel=2, N=15, envID='dish', init='default',
              scene='obstacle_avoidance_700', arena_size=0.04)
})

if __name__ == '__main__':
    print(ga_spaces.interference)
    # print(ga_dict(name='physics', suf='physics.', excluded=None, only=['torque_coef','ang_damping','body_spring_k']))

