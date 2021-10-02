import os

import numpy as np
import pandas as pd
from scipy.stats import stats

import lib.aux.colsNstr as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes
from lib.anal.plotting import plot_spatiotemporal_variation, plot_bend2orientation_analysis, \
    plot_sliding_window_analysis, plot_marked_strides, plot_stride_distribution
from lib.anal.process.angular import compute_orientations, compute_spineangles, compute_angular_metrics
from lib.anal.process.basic import compute_extrema
from lib.anal.process.bouts import detect_contacting_chunks
from lib.aux.parsing import multiparse_dataset_by_sliding_window


def choose_velocity_flag(s=None, e=None, dt=None, Npoints=None, from_file=True, save_to=None, dataset=None, **kwargs):
    if all([k is None for k in [s, e, dt, Npoints]]):
        if dataset is not None:
            d = dataset
            s = d.step_data
            e = d.endpoint_data
            dt = d.dt
            Npoints = d.Npoints
        else:
            raise ValueError('No dataset provided')
    ids = s.index.unique('AgentID').values
    ps = nam.midline(Npoints, type='point')
    # Define all candidate velocities, their respective points and their key labels
    points = ['centroid'] + ps[1:] + ps
    vels = [nam.vel('centroid')] + nam.lin(nam.vel(ps[1:])) + nam.vel(ps)
    svels = nam.scal(vels)
    vels_minima = nam.min(vels)
    vels_maxima = nam.max(vels)
    svels_minima = nam.min(svels)
    svels_maxima = nam.max(svels)

    # self.compute_spatial_metrics(mode='full', is_last=True)
    # self.compute_orientations(mode='full', is_last=True)
    # self.compute_linear_metrics(mode='full', is_last=True)
    int = 0.3
    svel_max_thr = 0.1

    compute_extrema(s=s, dt=dt, parameters=svels, interval_in_sec=int,
                    threshold_in_std=None, abs_threshold=[np.inf, svel_max_thr])
    if not from_file:
        m_t_cvs = []
        m_s_cvs = []
        mean_crawl_ratios = []
        for sv, p, sv_min, sv_max in zip(svels, points, svels_minima, svels_maxima):
            detect_contacting_chunks(s=s, e=e, dt=dt, chunk='stride',
                                     track_point=p, mid_flag=sv_max, edge_flag=sv_min,
                                     vel_par=sv, chunk_dur_in_sec=None,  **kwargs)
            t_cvs = []
            s_cvs = []
            for id in ids:
                ss = s.xs(id, level='AgentID', drop_level=True)
                durs = ss['stride_dur'].dropna().values
                dsts = ss['scaled_stride_dst'].dropna().values
                t_cv = stats.variation(durs)
                s_cv = stats.variation(dsts)
                t_cvs.append(t_cv)
                s_cvs.append(s_cv)
            m_s_cvs.append(np.mean(s_cvs))
            m_t_cvs.append(np.mean(t_cvs))
            mean_crawl_ratios.append(e[nam.dur_ratio('stride')].mean())
        df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=svels,
                          columns=['spatial_cvs', 'temporal_cvs'])
        file_path = os.path.join(save_to, 'spatiotemporal_stride_cvs.csv')
        df.to_csv(file_path, index=True)
        print(f'Spatiotemporal cvs saved as {file_path}')
        a, b = np.min(mean_crawl_ratios), np.max(mean_crawl_ratios)
        mean_crawl_ratios = [10 + 100 * (c - a) / (b - a) for c in mean_crawl_ratios]
        if dataset is not None:
            plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=mean_crawl_ratios,
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}_sized.pdf')
            plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=[300 for c in mean_crawl_ratios],
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}.pdf')

    else:
        for flags, filename in zip([vels_minima, vels_maxima], ['velocity_minima_flags', 'velocity_maxima_flags']):
            m_s_cvs, m_t_cvs = compute_spatiotemporal_cvs(s, e, dt, flags=flags, points=points)
            df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=flags,
                              columns=['spatial_cvs', 'temporal_cvs'])
            file_path = os.path.join(save_to, f'{filename}.csv')
            df.to_csv(file_path, index=True)
            print(f'Spatiotemporal cvs saved as {file_path}')
            if dataset is not None:
                plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                              save_to=save_to, save_as=f'{filename}.pdf')


def choose_orientation_flag(s, segs, save_to=None):
    chunk = 'stride'
    ors = nam.orient(segs)
    stride_or = nam.orient(chunk)
    s_stride_or = s[stride_or].dropna().values
    s_ors_start = s[ors + [nam.start(chunk)]].dropna().values
    s_ors_stop = s[ors + [nam.stop(chunk)]].dropna().values
    # errors=np.zeros(len(ors))*np.nan
    rNps = np.zeros([len(ors), 4]) * np.nan
    # ps=np.zeros(len(ors))*np.nan
    for i, o in enumerate(ors):
        r1, p1 = stats.pearsonr(s_stride_or, s_ors_start[:, i])
        rNps[i, 0] = r1
        rNps[i, 1] = p1
        r2, p2 = stats.pearsonr(s_stride_or, s_ors_stop[:, i])
        rNps[i, 2] = r2
        rNps[i, 3] = p2
    #     errors[i]=np.sum(np.abs(np.diff(sigma[[o,stride_or]].dropna().values)))
    df = pd.DataFrame(np.round(rNps, 4), index=ors)
    df.columns = ['Pearson r (start)', 'p-value (start)', 'Pearson r (stop)', 'p-value (stop)']
    if save_to is not None:
        filename = f'{save_to}/choose_orientation.csv'
        df.to_csv(filename)
        print(f'Stride orientation prediction saved as {filename}!')


def compute_spatiotemporal_cvs(s, e, dt, flags, points):
    ids = s.index.unique('AgentID').values
    all_t_cvs = []
    all_s_cvs = []
    for id in ids:
        data = s.xs(id, level='AgentID', drop_level=True)
        l = e['length'].loc[id]
        t_cvs = []
        s_cvs = []
        for f, p in zip(flags, points):
            indexes = data[f].dropna().index.values
            t_cv = stats.variation(np.diff(indexes) * dt)
            t_cvs.append(t_cv)

            coords = np.array(data[nam.xy(p)].loc[indexes])
            dx = np.diff(coords[:, 0])
            dy = np.diff(coords[:, 1])
            d = np.sqrt(dx ** 2 + dy ** 2)
            scaled_d = d / l
            s_cv = stats.variation(scaled_d)
            s_cvs.append(s_cv)
            # print(v, temporal_std, spatial_std)
        all_t_cvs.append(t_cvs)
        all_s_cvs.append(s_cvs)
    m_t_cvs = np.mean(np.array(all_t_cvs), axis=0)
    m_s_cvs = np.mean(np.array(all_s_cvs), axis=0)
    return m_s_cvs, m_t_cvs


def choose_rotation_point(s=None, e=None, dt=None, Npoints=None, config=None, dataset=None):
    if all([k is None for k in [s, e, dt, Npoints]]):
        if dataset is not None:
            d = dataset
            s = d.step_data
            e = d.endpoint_data
            dt = d.dt
            Npoints = d.Npoints
            config = d.config
        else:
            raise ValueError('No dataset provided')
    points = nam.midline(Npoints, type='point')
    Nangles = np.clip(Npoints - 2, a_min=0, a_max=None)
    # angles = [f'angle{i}' for i in range(Nangles)]
    # Nsegs = np.clip(Npoints - 1, a_min=0, a_max=None)
    # segs = nam.midline(Nsegs, type='seg')
    compute_orientations(s,e, config)
    compute_spineangles(s, config, mode='full')
    compute_angular_metrics(s, config, mode='full')

    if dataset is not None:
        dataset.save()
        # best_combo = self.plot_bend2orientation_analysis(data=None)
        best_combo = plot_bend2orientation_analysis(dataset=dataset)
        front_body_ratio = len(best_combo) / Nangles
        dataset.two_segment_model(front_body_ratio=front_body_ratio)


def stride_max_flag_phase_analysis(dataset, agent_id=None, flag=None, par=None):
    d = dataset
    if d.step_data is None:
        d.load()
    s, e = d.step_data, d.endpoint_data
    if agent_id is None:
        agent_id = e.num_ticks.nlargest(1).index.values[0]
    data = s.xs(agent_id, level='AgentID', drop_level=False)
    if flag is None:
        flag = nam.scal(d.velocity)
    if par is None:
        par = nam.scal(d.distance)
    f = e[nam.freq(flag)].loc[agent_id]
    r = float((1 / f) / 2)
    multiparse_dataset_by_sliding_window(data=data, par=par, flag=nam.max(flag),
                                         radius_in_ticks=np.ceil(r / d.dt),
                                         description_to=f'{d.data_dir}/{par}_around_{flag}', condition='True',
                                         description_as=None, overwrite=True)
    optimal_flag_phase, mean_stride_dst = plot_sliding_window_analysis(dataset=d, parameter=par,
                                                                       flag=nam.max(flag),
                                                                       radius_in_sec=r)
    print(f'Optimal flag phase at {optimal_flag_phase} rad')
    print(f'Mean stride dst at optimum : {mean_stride_dst} (possibly scal)')
    d.stride_max_flag_phase = optimal_flag_phase
    d.config['stride_max_flag_phase'] = optimal_flag_phase
    d.save_config()


def stride_analysis(dataset, agent_id=None, flag=None, par_to_track=None, stride_max_flag_analysis=True):
    d = dataset
    if stride_max_flag_analysis:
        stride_max_flag_phase_analysis(dataset=d, agent_id=agent_id, flag=flag, par=par_to_track)
    plot_marked_strides(datasets=[d], agent_idx=0, agent_id=agent_id, slice=[0, 180])
    plot_stride_distribution(dataset=d, agent_id=agent_id, save_to=None)

# def multiparse_by_sliding_window(dataset, data, par, flag, radius_in_sec, condition='True',
#                                  description_as=None, overwrite=True):
#         multiparse_dataset_by_sliding_window(data=data, par=par, flag=flag, condition=condition,
#                                              radius_in_ticks=np.ceil(radius_in_sec / self.dt),
#                                              description_to=f'{self.data_dir}/{par}_around_{flag}',
#                                              description_as=description_as, overwrite=overwrite)
