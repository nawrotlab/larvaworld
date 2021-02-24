import shutil
from ast import literal_eval
from distutils.dir_util import copy_tree

from fitter import Fitter
from scipy.signal import argrelextrema, spectrogram
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import nan_euclidean_distances as dst

from lib.aux import functions as fun

from lib.anal.fitting import *
from lib.aux.parsing import parse_dataset, multiparse_dataset_by_sliding_window
from lib.anal.plotting import *
from lib.conf import mesa_space_in_mm
import lib.conf.env_modes as env
from lib.stor.datagroup import SimParConf
from lib.stor.paths import Ref_path, Ref_fits, RefFolder
from lib.model.envs._larvaworld import LarvaWorldReplay


class LarvaDataset:
    def __init__(self, dir, id='unnamed',
                 fr=16, Npoints=3, Ncontour=0,
                 par_conf=SimParConf, arena_pars=env.dish(0.1),
                 filtered_at=np.nan, rescaled_by=np.nan,
                 save_data_flag=True, load_data=True):
        self.par_config = par_conf
        self.save_data_flag = save_data_flag
        self.define_paths(dir)
        if os.path.exists(self.config_file_path):
            temp = pd.read_csv(self.config_file_path)
            self.config = {col: temp[col].values[0] for col in temp.columns.values}
            self.build_dirs()
            print(f'Loaded dataset {self.config["id"]} with existing configuration')
        else:
            self.config = {'id': id,
                           'fr': fr,
                           'filtered_at': filtered_at,
                           'rescaled_by': rescaled_by,
                           'Npoints': Npoints,
                           'Ncontour': Ncontour,
                           }
            self.config = {**self.config, **par_conf, **arena_pars}
            print(f'Initialized dataset {id} with new configuration')
        self.__dict__.update(self.config)
        self.arena_pars = {'arena_xdim': self.arena_xdim,
                           'arena_ydim': self.arena_ydim,
                           'arena_shape': self.arena_shape}
        self.dt = 1 / fr
        self.configure_body(Npoints=self.Npoints,Ncontour=self.Ncontour)
        self.define_linear_metrics(self.config)
        self.types_dict = self.build_types_dict()
        if load_data:
            try:
                self.load()
                print('Data loaded successfully from stored csv files.')
            except:
                print('Data not found. Load them manually.')
        # else:
        #     print('Data not loaded as requested.')

    ########################################
    ############# CALIBRATION ##############
    ########################################

    # Choose the velocity and spinepoint most suitable for crawling strides annotation

    def choose_velocity_flag(self, from_file=True, save_to=None):
        if self.step_data is None:
            self.load()
        ids = self.agent_ids
        # Define all candidate velocities, their respective points and their short labels
        points = ['centroid'] + self.points + self.points[1:]
        vels = [self.cent_vel] + self.points_vel + nam.lin(self.points_vel[1:])
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
        # self.add_min_max_flags(parameters=svels, interval_in_sec=int, is_last=False)
        self.add_min_max_flags(parameters=svels, interval_in_sec=int, threshold_in_std=None,
                               absolute_threshold=[np.inf, svel_max_thr], is_last=False)
        self.compute_dominant_frequencies(parameters=svels, freq_range=[0.7, 2.6], accepted_range=[0.7, 2.6])
        # raise
        if not from_file:
            m_t_cvs = []
            m_s_cvs = []
            mean_crawl_ratios = []
            for sv, p, sv_min, sv_max in zip(svels, points, svels_minima, svels_maxima):
                self.detect_contacting_chunks(chunk='stride', mid_flag=sv_max, edge_flag=sv_min,
                                              vel_par=sv, track_point=p, is_last=False)
                t_cvs = []
                s_cvs = []
                for id in ids:
                    s = self.step_data.xs(id, level='AgentID', drop_level=True)
                    durs = s['stride_dur'].dropna().values
                    dsts = s['scaled_stride_dst'].dropna().values
                    t_cv = st.variation(durs)
                    s_cv = st.variation(dsts)
                    t_cvs.append(t_cv)
                    s_cvs.append(s_cv)
                m_s_cvs.append(np.mean(s_cvs))
                m_t_cvs.append(np.mean(t_cvs))
                mean_crawl_ratios.append(self.endpoint_data[nam.dur_ratio('stride')].mean())
            # print(mean_crawl_ratios)
            df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=svels,
                              columns=['spatial_cvs', 'temporal_cvs'])
            file_path = os.path.join(self.data_dir, 'spatiotemoral_stride_cvs.csv')
            df.to_csv(file_path, index=True)
            print(f'Spatiotemporal cvs saved as {file_path}')
            a, b = np.min(mean_crawl_ratios), np.max(mean_crawl_ratios)
            mean_crawl_ratios = [10 + 100 * (c - a) / (b - a) for c in mean_crawl_ratios]
            plot_spatiotemporal_variation(dataset=self, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=mean_crawl_ratios,
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}_sized.pdf')
            plot_spatiotemporal_variation(dataset=self, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=[110 for c in mean_crawl_ratios],
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}.pdf')

        else:
            for flags, filename in zip([vels_minima, vels_maxima], ['velocity_minima_flags', 'velocity_maxima_flags']):
                m_s_cvs, m_t_cvs = self.compute_spatiotemporal_cvs(flags=flags, points=points)
                df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=flags,
                                  columns=['spatial_cvs', 'temporal_cvs'])
                file_path = os.path.join(self.data_dir, f'{filename}.csv')
                df.to_csv(file_path, index=True)
                print(f'Spatiotemporal cvs saved as {file_path}')

                plot_spatiotemporal_variation(dataset=self, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                              save_to=save_to, save_as=f'{filename}.pdf')

    def choose_orientation_flag(self, save_to=None):
        if self.step_data is None:
            self.load()
        if save_to is None:
            save_to = self.data_dir
        chunk = 'stride'
        s = self.step_data
        ors = nam.orient(self.segs)
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
        #     errors[i]=np.sum(np.abs(np.diff(s[[o,stride_or]].dropna().values)))
        df = pd.DataFrame(np.round(rNps, 4), index=ors)
        df.columns = ['Pearson r (start)', 'p-value (start)', 'Pearson r (stop)', 'p-value (stop)']
        filename = f'{save_to}/choose_orientation.csv'
        df.to_csv(filename)
        print(f'Stride orientation prediction saved as {filename}!')

    def compute_spatiotemporal_cvs(self, flags, points):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        all_t_cvs = []
        all_s_cvs = []
        for id in ids:
            data = s.xs(id, level='AgentID', drop_level=True)
            l = e['length'].loc[id]
            t_cvs = []
            s_cvs = []
            for f, p in zip(flags, points):
                indexes = data[f].dropna().index.values
                t_cv = st.variation(np.diff(indexes) * self.dt)
                t_cvs.append(t_cv)

                coords = np.array(data[nam.xy(p)].loc[indexes])
                dx = np.diff(coords[:, 0])
                dy = np.diff(coords[:, 1])
                d = np.sqrt(dx ** 2 + dy ** 2)
                scaled_d = d / l
                s_cv = st.variation(scaled_d)
                s_cvs.append(s_cv)
                # print(v, temporal_std, spatial_std)
            all_t_cvs.append(t_cvs)
            all_s_cvs.append(s_cvs)
        m_t_cvs = np.mean(np.array(all_t_cvs), axis=0)
        m_s_cvs = np.mean(np.array(all_s_cvs), axis=0)
        return m_s_cvs, m_t_cvs

    def choose_rotation_point(self):
        self.compute_orientations(mode='minimal', is_last=False)
        self.compute_spineangles(mode='full', is_last=False)
        self.compute_angular_metrics(mode='full', is_last=True)

        # best_combo = self.plot_bend2orientation_analysis(data=None)
        best_combo = plot_bend2orientation_analysis(dataset=self)
        front_body_ratio = len(best_combo) / self.Nangles
        self.two_segment_model(front_body_ratio=front_body_ratio)



    def apply_filter(self, pars, freq, N=1, inplace=False, refilter=False, is_last=True):
        if self.step_data is None:
            self.load()
        s=self.step_data
        if not refilter:
            if self.config['filtered_at'] is not None and not np.isnan(self.config['filtered_at']):
                prev_filter = self.config['filtered_at']
                print(
                    f'Dataset has already been filtered at {prev_filter} Hz. If you want to apply additional filter set refilter to True')
                return
        pars = [p for p in pars if p in s.columns]
        self.filtered_at = freq
        self.config['filtered_at'] = freq
        self.save_config()
        self.types_dict.update({col: float for col in nam.filt(pars)})
        print(f'Applying filter to all spatial parameters')
        fpars = nam.filt(pars)
        data = np.dstack(list(self.step_data[pars].groupby('AgentID').apply(pd.DataFrame.to_numpy)))
        f_array = fun.apply_filter_to_array_with_nans_multidim(data, freq=freq, fr=self.fr, N=N)
        if inplace == False:
            for j, p in enumerate(fpars):
                s[p] = f_array[:, j, :].flatten()
        else:
            for j, p in enumerate(pars):
                s[p] = f_array[:, j, :].flatten()
        if is_last:
            self.save()
        print('All parameters filtered')

    def interpolate_nans(self, pars, is_last=True):
        if self.step_data is None:
            self.load()
        for p in pars:
            try:
                for id in self.agent_ids:
                    d = fun.interpolate_nans(self.step_data[p].xs(id, level='AgentID', drop_level=True).values)
                    self.step_data.loc[(slice(None), id), p] = d
            except:
                pass
        if is_last:
            self.save()
        print('All parameters interpolated')

    def rescale(self, rescale_again=False, scale=1, is_last=True):
        if self.step_data is None:
            self.load()
        s,e =self.step_data, self.endpoint_data
        if not rescale_again:
            if self.config['rescaled_by'] is not None and not np.isnan(self.config['rescaled_by']):
                prev_scale = self.config['rescaled_by']
                print(f'Dataset already rescaled by {prev_scale}. If you want to rescale again set rescale_again to True')
                return
        print(f'Rescaling dataset by {scale}')
        dst_params = self.points_dst + [self.cent_dst] + self.segs
        vel_params = self.points_vel + [self.cent_vel]
        acc_params = self.points_acc + [self.cent_acc]
        other_params = ['spinelength']
        lin_pars = self.xy_pars + dst_params + vel_params + acc_params + other_params
        for p in lin_pars:
            try:
                s[p] = s[p].apply(lambda x: x * scale)
            except:
                pass
        try:
            e['length'] = e['length'].apply(lambda x: x * scale)
        except:
            pass
        self.rescaled_by = scale
        self.config['rescaled_by'] = scale
        if is_last:
            self.save()
        print(f'Dataset rescaled by {scale}.')

    def set_step_data(self, step_data):
        self.step_data = step_data
        self.agent_ids = self.step_data.index.unique('AgentID').values
        self.num_ticks = self.step_data.index.unique('Step').size
        self.starting_tick = self.step_data.index.unique('Step')[0]
        # self.save()

    def set_endpoint_data(self, endpoint_data):
        self.endpoint_data = endpoint_data

    def set_food_endpoint_data(self, food_endpoint_data):
        self.food_endpoint_data = food_endpoint_data

    def set_types_dict(self, types_dict):
        self.types_dict = types_dict

    def replace_outliers_with_nan(self, pars, stds=None, thresholds=None, additional_pars=None):
        if self.step_data is None:
            self.load()
        s=self.step_data
        for i, p in enumerate(pars):
            for id in self.agent_ids:
                k = s.loc[(slice(None), id), p]
                l = k.values
                if stds is not None:
                    m = k.mean()
                    s = k.std()
                    ind = np.abs(l - m) > stds * s
                if thresholds is not None:
                    low, high = thresholds[i]
                    if low is not None:
                        ind = l < low
                    if high is not None:
                        ind = l > high
                l[ind] = np.nan
                s.loc[(slice(None), id), p] = l
                if additional_pars is not None:
                    for apar in additional_pars:
                        ak = s.loc[(slice(None), id), apar]
                        al = ak.values
                        al[ind] = np.nan
                        s.loc[(slice(None), id), apar] = al

        self.save()
        print('All outliers replaced')

    def exclude_rows(self, flag_column, accepted_values=None, rejected_values=None, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        if accepted_values is not None:
            s.loc[s[flag_column] != accepted_values[0]] = np.nan
        if rejected_values is not None:
            s.loc[s[flag_column] == rejected_values[0]] = np.nan

        for id in self.agent_ids:
            e.loc[id, 'num_ticks'] = len(s.xs(id, level='AgentID', drop_level=True).dropna())
            e.loc[id, 'cum_dur'] = e.loc[id, 'num_ticks'] * self.dt

        if is_last:
            self.save()
        print(f'Rows excluded according to {flag_column}.')

    def drop_agents(self, agents, is_last=True):
        if self.step_data is None:
            self.load()
        self.step_data.drop(agents, level='AgentID', inplace=True)
        self.endpoint_data.drop(agents, inplace=True)
        if is_last:
            self.save()
        print(f'{len(agents)} agents dropped.')

    def drop_contour(self, is_last=True):
        if self.step_data is None:
            self.load()
        self.set_step_data(self.step_data.drop(columns=fun.flatten_list(self.contour_xy)))
        if is_last:
            self.save()
        print('Contour dropped.')

    def drop_step_pars(self, pars, is_last=True):
        if self.step_data is None:
            self.load()
        for p in pars:
            try:
                self.step_data.drop(columns=[p], inplace=True)
            except:
                pass
        self.set_step_data(self.step_data)
        if is_last:
            self.save()
            self.load()
        print(f'{len(pars)} parameters dropped. {len(self.step_data.columns)} remain.')
        # print(self.step_data.columns)

    def drop_unused_pars(self, is_last=True):
        vels = ['vel', nam.scal('vel')]
        lin = ['dst', 'vel', 'acc']
        lins = lin + nam.scal(lin) + nam.cum(['dst', nam.scal('dst')]) + nam.max(vels) + nam.min(vels)
        beh = ['stride', nam.chain('stride'), nam.non('stride'), 'pause', 'turn', 'Lturn', 'Rturn']
        behs = nam.start(beh) + nam.stop(beh) + nam.id(beh) + nam.dur(beh) + nam.length(beh)
        str = [nam.dst('stride'), nam.straight_dst('stride'), nam.orient('stride'), 'dispersion']
        strs = str + nam.scal(str)
        var = ['spinelength', 'ang_color', 'lin_color']
        vpars = lins + self.ang_pars + self.xy_pars + behs + strs + var

        self.drop_step_pars(pars=[p for p in self.step_data.columns.values if p not in vpars],
                            is_last=False)
        if is_last:
            self.save()
        print('Non simulated parameters dropped.')

    #####################################
    ####### ALIGNMENT/FIXATION ##########
    #####################################

    def align_trajectories(self, s=None, mode='origin', arena_dims=None, track_point=None, save_step_data=False):
        if s is None:
            s = self.step_data.copy(deep=True)
        ids = s.index.unique(level='AgentID').values
        if track_point is None:
            track_point = self.point
        xy_pars = nam.xy(track_point)
        if not set(xy_pars).issubset(s.columns):
            raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')
        all_xy_pars = self.points_xy + self.contour_xy + [self.cent_xy] + xy_pars
        all_xy_pars = [xy_pair for xy_pair in all_xy_pars if set(xy_pair).issubset(s.columns)]
        all_xy_pars = fun.group_list_by_n(np.unique(flatten_list(all_xy_pars)), 2)
        if mode == 'origin':
            print('Aligning trajectories to common origin')
            xy = [s[xy_pars].xs(id, level='AgentID').dropna().values[0] for id in ids]
        elif mode == 'arena':
            print('Centralizing trajectories in arena center')
            if arena_dims is not None:
                x0, y0 = arena_dims
            else:
                x0, y0 = self.arena_xdim * 1000, self.arena_ydim * 1000
            xy = [[x0 / 2, y0 / 2] for agent_id in ids]
        elif mode == 'center':
            print('Centralizing trajectories in trajectory center using min-max positions')
            xy_max = [s[xy_pars].xs(id, level='AgentID').max().values for id in ids]
            xy_min = [s[xy_pars].xs(id, level='AgentID').min().values for id in ids]
            xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]

        for id, p in zip(ids, xy):
            for x, y in all_xy_pars:
                s.loc[(slice(None), id), x] -= p[0]
                s.loc[(slice(None), id), y] -= p[1]

        if save_step_data:
            self.set_step_data(s)
            self.save(endpoint_data=False)
            print('Step data saved as requested!')
        return s

    def fix_point(self, point, secondary_point=None, s=None, arena_dims=None):
        if s is None:
            s = self.step_data.copy(deep=True)
        ids = s.index.unique(level='AgentID').values
        if len(ids) != 1:
            raise ValueError('Fixation only implemented for a single agent.')
        pars = [p for p in self.xy_pars if p in s.columns.values]
        if set(nam.xy(point)).issubset(s.columns):
            print(f'Fixing {point} to arena center')
            xy = [s[nam.xy(point)].xs(id, level='AgentID').copy(deep=True).values for id in ids]
            xy_start = [s[nam.xy(point)].xs(id, level='AgentID').copy(deep=True).dropna().values[0] for id in ids]
            bg_x = np.array([(p[:, 0] - start[0]) / arena_dims[0] for p, start in zip(xy, xy_start)])
            bg_y = np.array([(p[:, 1] - start[1]) / arena_dims[1] for p, start in zip(xy, xy_start)])
        else:
            raise ValueError(f" The requested {point} is not part of the dataset")
        for id, p in zip(ids, xy):
            for x, y in fun.group_list_by_n(pars, 2):
                s.loc[(slice(None), id), [x, y]] -= p

        if secondary_point is not None:
            if set(nam.xy(secondary_point)).issubset(s.columns):
                print(f'Fixing {secondary_point} as secondary point on vertical axis')
                xy_sec = [s[nam.xy(secondary_point)].xs(id, level='AgentID').copy(deep=True).values for id in ids]
                bg_a = np.array([np.arctan2(xy_sec[i][:, 1], xy_sec[i][:, 0]) - np.pi / 2 for i in range(len(xy_sec))])
            else:
                raise ValueError(f" The requested secondary {secondary_point} is not part of the dataset")

            for id, angle in zip(ids, bg_a):
                d = s[pars].xs(id, level='AgentID', drop_level=True).copy(deep=True).values
                s.loc[(slice(None), id), pars] = [fun.flatten_list(
                    fun.rotate_multiple_points(points=np.array(fun.group_list_by_n(d[i].tolist(), 2)),
                                               radians=a)) for i, a in enumerate(angle)]
        else:
            bg_a = np.array([np.zeros(len(bg_x[0])) for i in range(len(ids))])
        bg = [np.vstack((bg_x[i, :], bg_y[i, :], bg_a[i, :])) for i in range(len(ids))]

        # There is only a single larva so :
        bg = bg[0]
        print('Fixed-point dataset generated')
        return s, bg

    #####################################
    ############# STORAGE ###############
    #####################################

    def load(self, step_data=True, endpoint_data=True, food_endpoint_data=False):
        print(f'Loading data from {self.step_file_path}')
        # TODO Use this dict idea for annotation of parameters and for metric units
        # col_names = pd.read_csv(self.step_file_path, nrows=0).columns
        # types_dict = {'AgentID': str}
        # types_dict = {'Step': int, 'AgentID': str}

        # types_dict.update({col: float for col in col_names if col not in types_dict})
        # print(types_dict)
        if step_data:
            try:
                self.step_data = pd.read_csv(self.step_file_path, index_col=['Step', 'AgentID'], dtype=self.types_dict)
                print('Step data loaded according to types dictionary')
            except:
                self.step_data = pd.read_csv(self.step_file_path, index_col=['Step', 'AgentID'])
                print('Step data loaded independent of types dictionary')
            self.step_data.sort_index(level=['Step', 'AgentID'], inplace=True)
            self.agent_ids = self.step_data.index.unique('AgentID').values
            self.num_ticks = self.step_data.index.unique('Step').size
            self.starting_tick = self.step_data.index.unique('Step')[0]
            self.Nagents = len(self.agent_ids)
        if endpoint_data:
            self.endpoint_data = pd.read_csv(self.endpoint_file_path, index_col=0)
            self.endpoint_data.sort_index(inplace=True)
            self.Nagents = len(self.endpoint_data.index.values)
        if food_endpoint_data:
            self.food_endpoint_data = pd.read_csv(self.food_endpoint_file_path, index_col=0)
            self.food_endpoint_data.sort_index(inplace=True)

    def save(self, step_data=True, endpoint_data=True, food_endpoint_data=False):
        if self.save_data_flag == True:
            print('Saving data')
            self.build_dirs()
            if step_data:
                self.step_data.to_csv(self.step_file_path, index=True, header=True)
            if endpoint_data:
                self.endpoint_data.to_csv(self.endpoint_file_path, index=True, header=True)
            if food_endpoint_data:
                self.food_endpoint_data.to_csv(self.food_endpoint_file_path, index=True, header=True)
            self.save_config()

    def save_config(self):
        try:
            self.config['Nagents'] = self.Nagents
        except:
            pass
        dict = {k: [v] for k, v in self.config.items()}
        temp = pd.DataFrame.from_dict(dict)
        temp.to_csv(self.config_file_path, index=False, header=True)

    def save_agent(self, pars=None, header=True):
        if self.step_data is None:
            self.load()
        for i, agent in enumerate(self.agent_ids):
            if pars is not None:
                agent_data = self.step_data[pars].xs(agent, level='AgentID', drop_level=True)
            else:
                agent_data = self.step_data.xs(agent, level='AgentID', drop_level=True)
            dir_path = os.path.join(self.data_dir, 'single_larva_data')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            file_path = os.path.join(dir_path, f'{agent}.csv')
            agent_data.to_csv(file_path, index=True, header=header)
        print('All agent data saved.')

    def load_chunk_dataset(self, chunk=None, parameter=None):
        # if chunk_filename is None:
        #     chunk_filename = f'{parameter}_during_{chunk}.csv'
        # file_path = os.path.join(self.aux_dir, chunk_filename)
        if chunk == 'stride':
            file_path = file_path = f'{self.par_during_stride_dir}/{parameter}.csv'
        else:
            raise ValueError('Only stride chunks allowed')
        try:
            data = pd.read_csv(file_path, index_col=0)
            return data
        except:
            raise ValueError(f'No dataset at {file_path}')

    def load_dispersion_dataset(self, par='dispersion', filename=None, scaled=True):
        if filename is None:
            if scaled:
                p = nam.scal(par)
            else:
                p = par
            filename = f'{p}.csv'
        file_path = os.path.join(self.dispersion_dir, filename)
        try:
            data = pd.read_csv(file_path, index_col=0)
            return data
        except:
            raise ValueError(f'No dataset at {file_path}')

    def load_par_distro_dataset(self, par=None):
        file_path = f'{self.par_distro_dir}/{par}.csv'
        try:
            data = pd.read_csv(file_path, index_col=0)
            return data
        except:
            raise ValueError(f'No dataset at {file_path}')

    #####################################
    ############# PLOTS #################
    #####################################

    def plot_step_data(self, parameters, save_to=None, agent_ids=None, **kwargs):
        if save_to is None:
            save_to = self.plot_dir
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        if agent_ids is None:
            agent_ids = self.agent_ids
        plot_dataset(data=self.step_data, parameters=parameters, agent_ids=agent_ids, dt=self.dt, save_to=save_to,
                     **kwargs)

    def plot_angular_pars(self, bend_param=None, orientation_param=None, candidate_distributions=['t'],
                          chunk_only=None, num_sample=None, absolute=False):
        if bend_param is None:
            b = 'bend'
        else:
            b = bend_param
        if orientation_param is None:
            ho = 'front_orientation'
        else:
            ho = orientation_param
        temp = 'angular_pars'
        if absolute:
            temp = f'{temp}_abs'
        if chunk_only is None:
            plot_filename = f'{temp}.pdf'
        else:
            plot_filename = f'{temp}_during_{chunk_only}.pdf'
        bv = nam.vel(b)
        ba = nam.acc(b)
        hov = nam.vel(ho)
        hoa = nam.acc(ho)
        hca = f'turn_{nam.unwrap(ho)}'
        ang_pars = [b, bv, ba, hov, hoa, hca]

        self.fit_distribution(parameters=ang_pars, candidate_distributions=candidate_distributions, save_fits=True,
                              chunk_only=chunk_only, num_sample=num_sample, absolute=absolute)
        fit_angular_params(d=self, fit_filepath=self.fit_file_path, chunk_only=chunk_only, absolute=absolute,
                           save_to=None, save_as=plot_filename)

    def plot_bout_pars(self, num_sample=20, num_candidate_dist=5, time_to_fit=30):
        pars = [nam.length(nam.chain('stride')), nam.dur(nam.non('stride')),
                nam.dur_ratio('stride'), nam.dur_ratio('non_stride'),
                nam.num('stride'), nam.num('non_stride'),
                nam.dur('rest'), nam.dur('activity'),
                nam.dur_ratio('rest'), nam.dur_ratio('activity'),
                nam.num('rest'), nam.num('activity')]
        # pars=[self.duration_param('rest'), self.duration_param('activity'),
        #         self.duration_fraction_param('rest'), self.duration_fraction_param('activity'),]
        self.fit_distribution(parameters=pars, num_sample=num_sample, num_candidate_dist=num_candidate_dist,
                              time_to_fit=time_to_fit,
                              candidate_distributions=None, distributions=None, save_fits=True)

        fit_bout_params(d=self, fit_filepath=self.fit_file_path, save_to=None, save_as='bout_pars.pdf')

    def get_par_list(self, track_point=None, spinepoints=True, centroid=True, contours=True,
                     spineangle_params=['bend'], orientation_params=['front_orientation'],
                     chunk_params=['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']):
        if track_point is None:
            track_point = self.point
        if set(nam.xy(track_point)).issubset(self.step_data.columns):
            pos_xy_pars = nam.xy(track_point)
        else:
            pos_xy_pars = []
        if spinepoints == True and len(self.points_xy) >= 1:
            point_xy_pars = nam.xy(self.points, flat=True)
        else:
            point_xy_pars = []
        if centroid == True and len(self.cent_xy) >= 1:
            cent_xy_pars = self.cent_xy
        else:
            cent_xy_pars = []
        if contours == True and len(self.contour_xy) >= 1:
            contour_xy_pars = nam.xy(self.contour, flat=True)
        else:
            contour_xy_pars = []
        pars = cent_xy_pars + point_xy_pars + pos_xy_pars + contour_xy_pars + spineangle_params + orientation_params + chunk_params
        pars = np.unique(pars).tolist()
        return pars, pos_xy_pars

    def get_smaller_dataset(self, agent_ids=None, pars=None, time_range_in_ticks=None):
        if self.step_data is None:
            self.load()
        if agent_ids is None:
            agent_ids = self.agent_ids
        if pars is None:
            pars = self.step_data.columns.values.tolist()
        if time_range_in_ticks is None:
            s = self.step_data.loc[(slice(None), agent_ids), pars].copy(deep=True)
        else:
            a, b = time_range_in_ticks
            s = self.step_data.loc[(slice(a, b), agent_ids), pars].copy(deep=True)
        e = self.endpoint_data.loc[agent_ids]
        return s, e

    def visualize(self,
                  arena_pars=None,
                  env_params=None,
                  track_point=None,
                  spinepoints=True, centroid=True, contours=True,
                  dynamic_color=None,
                  agent_ids=None,
                  time_range_in_ticks=None,
                  align_mode=None, fix_point=None, secondary_fix_point=None,
                  save_to=None, save_as=None,
                  **kwargs):



        angle_pars = ['bend']
        or_pars = ['front_orientation'] + nam.orient(self.segs)
        chunk_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id']
        pars, pos_xy_pars = self.get_par_list(track_point=track_point, spinepoints=spinepoints, centroid=centroid,
                                              contours=contours,
                                              spineangle_params=angle_pars,
                                              orientation_params=or_pars,
                                              chunk_params=chunk_pars)

        if type(agent_ids) == list and all([type(i) == int for i in agent_ids]):
            agent_ids = [self.agent_ids[i] for i in agent_ids]
        if agent_ids is None:
            n0 = 'all'
        elif len(agent_ids) == 1:
            n0 = agent_ids[0]
        else:
            n0 = f'{len(agent_ids)}l'
        if dynamic_color is not None:
            pars.append(dynamic_color)
        pars = [p for p in pars if p in self.step_data.columns]
        s, e = self.get_smaller_dataset(agent_ids=agent_ids, pars=pars, time_range_in_ticks=time_range_in_ticks)

        if dynamic_color is not None:
            trajectory_colors = s[dynamic_color]
        else:
            trajectory_colors = None
        if env_params is None:
            if arena_pars is None:
                arena_pars = self.arena_pars
            env_params = {'arena_params': arena_pars,
                          'space_params': mesa_space_in_mm}
        arena_dims_in_m = env_params['arena_params']['arena_xdim'], env_params['arena_params']['arena_ydim']
        arena_dims = [i * 1000 for i in arena_dims_in_m]

        if align_mode is not None:
            s = self.align_trajectories(s=s, mode=align_mode, arena_dims=arena_dims, track_point=track_point)
            bg = None
            n1 = 'aligned'
        elif fix_point is not None:
            if type(fix_point) == int:
                if fix_point == -1:
                    fix_point = 'centroid'
                else:
                    fix_point = self.points[fix_point]
            if secondary_fix_point is not None:
                if type(secondary_fix_point) == int:
                    secondary_fix_point = self.points[secondary_fix_point]
            s, bg = self.fix_point(point=fix_point, secondary_point=secondary_fix_point, s=s,
                                   arena_dims=arena_dims)
            n1 = 'fixed'
        else:
            bg = None
            n1 = 'normal'
        replay_id = f'{n0}_{n1}'
        if save_as is None:
            save_as = replay_id
        if save_to is None:
            save_to = self.vis_dir

        Nsteps = len(s.index.unique('Step').values)
        replay_env = LarvaWorldReplay(id=replay_id, env_params=env_params,
                                      step_data=s, endpoint_data=e,
                                      Nsteps=Nsteps,
                                      dataset=self,
                                      pos_xy_pars=pos_xy_pars,
                                      background_motion=bg,
                                      dt=self.dt,
                                      trajectory_colors=trajectory_colors,
                                      save_to=save_to, media_name=save_as,
                                      **kwargs)

        replay_env.run()

        print('Visualization complete')

    #####################################
    ############# ENRICHMENT ############
    #####################################

    def compute_length(self, mode='full', is_last=True):
        if self.step_data is None:
            self.load()

        s,e = self.step_data, self.endpoint_data
        t=len(s)
        segs=self.segs
        Nsegs=len(segs)
        xy = s[nam.xy(self.points, flat=True)].values
        L = np.zeros([1, t]) * np.nan
        S = np.zeros([Nsegs, t]) * np.nan

        if mode == 'full':
            print(f'Computing lengths for {Nsegs} segments and total body length')
            for j in range(xy.shape[0]):
                for i, seg in enumerate(segs):
                    S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
                L[:, j] = np.nansum(S[:, j])
            for i, seg in enumerate(segs):
                s[seg] = S[i, :].flatten()
        elif mode == 'minimal':
            print(f'Computing body length')
            for j in range(xy.shape[0]):
                k = np.sum(np.diff(np.array(fun.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
                if not np.isnan(np.sum(k)):
                    sp_l = np.sum([np.sqrt(kk) for kk in k])
                else:
                    sp_l = np.nan
                L[:, j] = sp_l

        s['length'] = L.flatten()
        e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
        if is_last:
            self.save()
        print('All lengths computed.')

    def compute_centroid_from_contour(self, is_last=True):
        if self.step_data is None:
            self.load()
        s=self.step_data
        con_pars = nam.xy(self.contour, flat=True)
        if not set(con_pars).issubset(s.columns) or len(con_pars) == 0:
            print(f'No contour found. Not computing centroid')
        else:
            print(f'Computing centroid from {len(self.contour)} contourpoints')
            contour = s[con_pars].values
            Nconpoints = int(contour.shape[1] / 2)
            Nticks = contour.shape[0]
            contour = np.reshape(contour, (Nticks, Nconpoints, 2))
            c = np.zeros([Nticks, 2]) * np.nan
            for i in range(Nticks):
                c[i, :] = np.array(fun.compute_centroid(contour[i, :, :]))
            s[self.cent_xy[0]] = c[:, 0]
            s[self.cent_xy[1]] = c[:, 1]
        if is_last:
            self.save()
        print('Centroid coordinates computed.')

    def compute_length_and_centroid(self, recompute_length=False, recompute_centroid=False,
                                    drop_contour=True, is_last=True):
        if self.step_data is None:
            self.load()
        s,e=self.step_data, self.endpoint_data
        if 'length' in e.columns.values and not recompute_length:
            print('Length is already computed. If you want to recompute it, set recompute_length to True')
        else:
            self.compute_length(mode='minimal', is_last=False)
        if set(nam.xy('centroid')).issubset(s.columns.values) and not recompute_centroid:
            print('Centroid is already computed. If you want to recompute it, set recompute_centroid to True')
        else:
            self.compute_centroid_from_contour(is_last=False)
        if drop_contour:
            try:
                self.drop_contour(is_last=False)
            except:
                pass
        if is_last:
            self.save()

    def compute_spineangles(self, chunk_only=None, mode='full', is_last=True):
        if self.step_data is None:
            self.load()
        self.bend_angles = self.angles[:int(np.round(self.config['front_body_ratio'] * self.Nangles))]
        if chunk_only is None:
            s = self.step_data.copy(deep=False)
        else:
            print(f'Computation restricted to {chunk_only} chunks')
            s = self.step_data.loc[self.step_data[nam.id(chunk_only)].dropna().index.values].copy(deep=False)
        xy = [nam.xy(self.points[i]) for i in range(len(self.points))]
        if mode == 'full':
            angles = self.angles
        elif mode == 'minimal':
            angles = self.bend_angles
        N = len(angles)
        print(f'Computing {N} angles')
        xy_pars = fun.flatten_list([xy[i] for i in range(N + 2)])
        xy_ar = s[xy_pars].values
        Npoints = int(xy_ar.shape[1] / 2)
        Nticks = xy_ar.shape[0]
        xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))
        c = np.zeros([N, Nticks]) * np.nan
        for i in range(Nticks):
            c[:, i] = np.array([fun.angle(xy_ar[i, j + 2, :], xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(N)])
        for z, a in enumerate(angles):
            self.step_data[a] = c[z].T
        if is_last:
            self.save()
        print('All angles computed')

    def compute_bend(self, is_last=True):
        if self.step_data is None:
            self.load()
        s=self.step_data
        b_conf=self.config['bend']
        if b_conf is None:
            print('Bending angle not defined. Can not compute angles')
            return
        elif b_conf == 'from_vectors':
            print(f'Computing bending angle as the difference between front and rear orientations')
            s['bend'] = s.apply(lambda r: fun.angle_dif(r['front_orientation'], r['rear_orientation']), axis=1)
        elif b_conf == 'from_angles':
            self.compute_spineangles(mode='minimal', is_last=False)
            print(f'Computing bending angle as the sum of the first {len(self.bend_angles)} front angles')
            s['bend'] = s[self.bend_angles].sum(axis=1, min_count=1)

        if is_last:
            self.save()
        print('All bends computed')

    def compute_LR_bias(self, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        for id in ids:
            b = s['bend'].xs(id, level='AgentID', drop_level=True).dropna()
            bv = s[nam.vel('bend')].xs(id, level='AgentID',drop_level=True).dropna()
            e.loc[id, 'bend_mean'] = b.mean()
            e.loc[id, 'bend_vel_mean'] = bv.mean()
            e.loc[id, 'bend_std'] = b.std()
            e.loc[id, 'bend_vel_std'] = bv.std()

        if is_last:
            self.save()
        print('LR biases computed')

    def compute_orientations(self, mode='full', is_last=True):

        for key in ['front_vector_start', 'front_vector_stop', 'rear_vector_start', 'rear_vector_stop']:
            if self.config[key] is None:
                print('Front and rear vectors are not defined. Can not compute orientations')
                return
        else:
            f1, f2 = self.config['front_vector_start'], self.config['front_vector_stop']
            r1, r2 = self.config['rear_vector_start'], self.config['rear_vector_stop']

        if self.step_data is None:
            self.load()

        xy = [nam.xy(self.points[i]) for i in range(len(self.points))]
        s = self.step_data
        # s = self.step_data.copy(deep=False)
        print(f'Computing front and rear orientations')
        xy_pars = fun.flatten_list([xy[i] for i in [f2 - 1, f1 - 1, r2 - 1, r1 - 1]])
        xy_ar = s[xy_pars].values
        Npoints = int(xy_ar.shape[1] / 2)
        Nticks = xy_ar.shape[0]
        xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))

        c = np.zeros([2, Nticks]) * np.nan
        for i in range(Nticks):
            c[:, i] = np.array([fun.angle_to_x_axis(xy_ar[i, 2 * j, :], xy_ar[i, 2 * j + 1, :]) for j in range(2)])
        for z, a in enumerate(['front_orientation', 'rear_orientation']):
            s[a] = c[z].T
        if mode == 'full':
            N = len(self.segs)
            print(f'Computing additional orientations for {N} spinesegments')
            ors = nam.orient(self.segs)
            xy_pars = fun.flatten_list([xy[i] for i in range(N + 1)])
            xy_ar = s[xy_pars].values
            Npoints = int(xy_ar.shape[1] / 2)
            Nticks = xy_ar.shape[0]
            xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))
            c = np.zeros([N, Nticks]) * np.nan
            for i in range(Nticks):
                c[:, i] = np.array([fun.angle_to_x_axis(xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(N)])
            for z, a in enumerate(ors):
                s[a] = c[z].T
        if is_last:
            self.save()
        print('All orientations computed')

    def unwrap_orientations(self, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        pars = list(set([p for p in ['front_orientation', 'rear_orientation'] + nam.orient(
            self.segs) if p in s.columns.values]))
        for p in pars:
            for id in ids:
                ts = s.loc[(slice(None), id), p].values
                s.loc[(slice(None), id), nam.unwrap(p)] = fun.unwrap_deg(ts)
        if is_last:
            self.save()
        print('All orientations unwrapped')

    def compute_angular_metrics(self, mode='minimal', is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick
        self.unwrap_orientations(is_last=False)

        if mode == 'full':
            pars = self.angles + nam.orient(self.segs) + ['front_orientation',
                                                          'rear_orientation', 'bend']
        elif mode == 'minimal':
            pars = ['front_orientation', 'rear_orientation', 'bend']

        pars = [a for a in pars if a in s.columns]
        Npars = len(pars)

        print(f'Computing angular velocities and accelerations for {Npars} angular parameters')

        V = np.zeros([Nticks, Npars, Nids]) * np.nan
        A = np.zeros([Nticks, Npars, Nids]) * np.nan

        all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]

        vels = nam.vel(pars)
        accs = nam.acc(pars)

        for i, p in enumerate(pars):
            if nam.unwrap(p) in s.columns:
                p = nam.unwrap(p)
            for j, d in enumerate(all_d):
                angle = d[p].values
                avel = np.diff(angle) / self.dt
                aacc = np.diff(avel) / self.dt
                V[1:, i, j] = avel
                A[2:, i, j] = aacc
        for k, (v, a) in enumerate(zip(vels, accs)):
            s[v] = V[:, k, :].flatten()
            s[a] = A[:, k, :].flatten()
        if is_last:
            self.save()
        print('All angular parameters computed')

    def angular_analysis(self, recompute=False, mode='minimal', is_last=True):
        if self.step_data is None:
            self.load()
        if set(['front_orientation', 'rear_orientation', 'bend']).issubset(
                self.step_data.columns.values) and not recompute:
            print('Orientation and bend are already computed. If you want to recompute them, set recompute to True')
        else:
            self.compute_orientations(mode=mode, is_last=False)
            self.compute_bend(is_last=False)
        self.compute_angular_metrics(mode=mode, is_last=False)
        self.compute_LR_bias(is_last=False)
        if self.save_data_flag:
            b = 'bend'
            fo = 'front_orientation'
            ro = 'rear_orientation'
            bv, fov, rov = nam.vel([b, fo, ro])
            ba, foa, roa = nam.acc([b, fo, ro])
            self.create_par_distro_dataset([b, bv, ba, fov, foa, rov, roa])
        if is_last:
            self.save()
        print(f'Completed {mode} angular analysis.')

    def compute_spatial_metrics(self, mode='full', is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick
        dt = self.dt
        if 'length' in e.columns.values:
            lengths = e['length'].values
        else:
            lengths = None

        if mode == 'full':
            print(f'Computing distances, velocities and accelerations for {self.Npoints} points')
            points = self.points.copy()
            points += ['centroid']
        elif mode == 'minimal':
            print(f'Computing distances, velocities and accelerations for a single spinepoint')
            points = [self.point]

        points = np.unique(points).tolist()
        points = [p for p in points if set(nam.xy(p)).issubset(s.columns.values)]

        xy_params = self.raw_or_filtered_xy(points)
        xy_params = fun.group_list_by_n(xy_params, 2)

        all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        dsts = nam.dst(points)
        cum_dsts = nam.cum(dsts)
        vels = nam.vel(points)
        accs = nam.acc(points)

        for p, xy, dst, cum_dst, vel, acc in zip(points, xy_params, dsts, cum_dsts, vels, accs):
            D = np.zeros([Nticks, Nids]) * np.nan
            Dcum = np.zeros([Nticks, Nids]) * np.nan
            V = np.zeros([Nticks, Nids]) * np.nan
            A = np.zeros([Nticks, Nids]) * np.nan
            sD = np.zeros([Nticks, Nids]) * np.nan
            sDcum = np.zeros([Nticks, Nids]) * np.nan
            sV = np.zeros([Nticks, Nids]) * np.nan
            sA = np.zeros([Nticks, Nids]) * np.nan

            for i, data in enumerate(all_d):
                v, d = fun.compute_velocity(xy=data[xy].values, dt=dt, return_dst=True)
                a = np.diff(v) / dt
                cum_d = np.nancumsum(d)
                D[1:, i] = d
                Dcum[1:, i] = cum_d
                V[1:, i] = v
                A[2:, i] = a
                if lengths is not None:
                    l = lengths[i]
                    sD[1:, i] = d / l
                    sDcum[1:, i] = cum_d / l
                    sV[1:, i] = v / l
                    sA[2:, i] = a / l

            s[dst] = D.flatten()
            s[cum_dst] = Dcum.flatten()
            s[vel] = V.flatten()
            s[acc] = A.flatten()
            e[nam.cum(dst)] = Dcum[-1, :]

            if lengths is not None:
                s[nam.scal(dst)] = sD.flatten()
                s[nam.cum(nam.scal(dst))] = sDcum.flatten()
                s[nam.scal(vel)] = sV.flatten()
                s[nam.scal(acc)] = sA.flatten()
                e[nam.cum(nam.scal(dst))] = sDcum[-1, :]

        if is_last:
            self.save()
        print('All spatial parameters computed')

    def compute_linear_metrics(self, mode='full', is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids=len(ids)
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick
        dt = self.dt
        if 'length' in e.columns.values:
            lengths = e['length'].values
        else:
            lengths = None

        if mode == 'full':
            print(
                f'Computing linear distances, velocities and accelerations for {self.Npoints - 1} points')
            points = self.points[1:]
            orientations = nam.orient(self.segs)
        elif mode == 'minimal':
            if self.point == 'centroid' or self.point == self.points[0]:
                print('Defined point is either centroid or head. Orientation of front segment not defined.')
                return
            else:
                print(f'Computing linear distances, velocities and accelerations for a single spinepoint')
                points = [self.point]
                orientations = ['rear_orientation']

        if not set(orientations).issubset(s.columns):
            print('Required orientations not found. Component linear metrics not computed.')
            return

        xy_params = self.raw_or_filtered_xy(points)
        xy_params = fun.group_list_by_n(xy_params, 2)

        all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        dsts = nam.lin(nam.dst(points))
        cum_dsts = nam.cum(nam.lin(dsts))
        vels = nam.lin(nam.vel(points))
        accs = nam.lin(nam.acc(points))

        for p, xy, dst, cum_dst, vel, acc, orient in zip(points, xy_params, dsts, cum_dsts, vels, accs, orientations):
            D = np.zeros([Nticks, Nids]) * np.nan
            Dcum = np.zeros([Nticks, Nids]) * np.nan
            V = np.zeros([Nticks, Nids]) * np.nan
            A = np.zeros([Nticks, Nids]) * np.nan
            sD = np.zeros([Nticks, Nids]) * np.nan
            sDcum = np.zeros([Nticks, Nids]) * np.nan
            sV = np.zeros([Nticks, Nids]) * np.nan
            sA = np.zeros([Nticks, Nids]) * np.nan

            for i, data in enumerate(all_d):
                v, d = fun.compute_component_velocity(xy=data[xy].values, angles=data[orient].values, dt=dt,return_dst=True)
                a = np.diff(v) / dt
                cum_d = np.nancumsum(d)
                D[1:, i] = d
                Dcum[1:, i] = cum_d
                V[1:, i] = v
                A[2:, i] = a
                if lengths is not None:
                    l = lengths[i]
                    sD[1:, i] = d / l
                    sDcum[1:, i] = cum_d / l
                    sV[1:, i] = v / l
                    sA[2:, i] = a / l

            s[dst] = D.flatten()
            s[cum_dst] = Dcum.flatten()
            s[vel] = V.flatten()
            s[acc] = A.flatten()
            e[nam.cum(dst)] = Dcum[-1, :]

            if lengths is not None:
                s[nam.scal(dst)] = sD.flatten()
                s[nam.cum(nam.scal(dst))] = sDcum.flatten()
                s[nam.scal(vel)] = sV.flatten()
                s[nam.scal(acc)] = sA.flatten()
                e[nam.cum(nam.scal(dst))] = sDcum[-1, :]

        if is_last:
            self.save()
        print('All linear parameters computed')

    def store_global_linear_metrics(self, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        dic = {
            'x': nam.xy(self.point)[0],
            'y': nam.xy(self.point)[1],
            'dst': self.distance,
            'vel': self.velocity,
            'acc': self.acceleration,
            nam.scal('dst'): nam.scal(self.distance),
            nam.scal('vel'): nam.scal(self.velocity),
            nam.scal('acc'): nam.scal(self.acceleration),
            nam.cum('dst'): nam.cum(self.distance),
            nam.cum(nam.scal('dst')): nam.cum(nam.scal(self.distance))}
        for k, v in dic.items():
            try:
                s[k] = s[v]
            except:
                pass
        e[nam.cum('dst')] = e[nam.cum(self.distance)]
        e[nam.final('x')] = [s['x'].xs(id, level='AgentID').dropna().values[-1] for id in ids]
        e[nam.final('y')] = [s['y'].xs(id, level='AgentID').dropna().values[-1] for id in ids]
        e[nam.initial('x')] = [s['x'].xs(id, level='AgentID').dropna().values[0] for id in ids]
        e[nam.initial('y')] = [s['y'].xs(id, level='AgentID').dropna().values[0] for id in ids]
        e[nam.mean('vel')] = e[nam.cum(self.distance)] / e['cum_dur']
        try:
            e[nam.cum(nam.scal('dst'))] = e[nam.cum(nam.scal(self.distance))]
            e[nam.mean(nam.scal('vel'))] = e[nam.mean('vel')] / e['length']
        except:
            pass
        if is_last:
            self.save()

    def linear_analysis(self, mode='minimal', is_last=True):
        if self.step_data is None:
            self.load()

        # self.distance = nam.dst(self.point)
        self.compute_spatial_metrics(mode=mode, is_last=False)
        self.compute_linear_metrics(mode=mode, is_last=False)
        self.store_global_linear_metrics(is_last=False)
        if is_last:
            self.save()

    def compute_dispersion(self, recompute=False, starts=[0], is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        point = self.point
        for s0 in starts:
            if s0 == 0:
                p = 'dispersion'
            else:
                p = f'dispersion_{s0}'
            t0 = int(s0 / self.dt)
            p40 = f'40sec_{p}'
            fp, fp40 = nam.final([p, p40])
            mp, mp40 = nam.max([p, p40])
            mup = nam.mean(p)

            if set([mp, mp40]).issubset(e.columns.values) and not recompute:
                print(f'Dispersion starting at {s0} is already detected. If you want to recompute it, set recompute_dispersion to True')
                continue
            print(f'Computing dispersion starting at {s0} based on {point}')
            for id in ids:
                xy = s[nam.xy(point)].xs(id, level='AgentID', drop_level=True)
                try:
                    origin_xy = list(xy.dropna().values[t0])
                except:
                    print(f'No values to set origin point for {id}')
                    s.loc[(slice(None), id), p] = np.empty(len(xy)) * np.nan
                    continue
                d = dst(list(xy.values), [origin_xy])[:, 0]
                d[:t0] = np.nan
                s.loc[(slice(None), id), p] = d
                e.loc[id, mp] = np.nanmax(d)
                e.loc[id, mp40] = np.nanmax(d[:int(40 / self.dt)])
                e.loc[id, fp40] = d[int(40 / self.dt)]
                e.loc[id, mup] = np.nanmean(d)
                e.loc[id, fp] = s[p].xs(id, level='AgentID').dropna().values[-1]

                try:
                    l = e.loc[id, 'length']
                    s.loc[(slice(None), id), nam.scal(p)] = d / l
                    e.loc[id, nam.scal(mp)] = e.loc[id, mp] / l
                    e.loc[id, nam.scal(mp40)] = e.loc[id, mp40] / l
                    e.loc[id, nam.scal(fp40)] = e.loc[id, fp40] / l
                    e.loc[id, nam.scal(mup)] = e.loc[id, mup] / l
                    e.loc[id, nam.scal(fp)] = e.loc[id, fp] / l
                except:
                    pass
            self.create_dispersion_dataset(par=p, scaled=True)
            self.create_dispersion_dataset(par=p, scaled=False)
        if is_last:
            self.save()
        print('Dispersions computed')

    def compute_orientation_to_origin(self, origin=np.array([0, 0]), is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        p = self.point
        o = 'orientation_to_origin'
        fo='front_orientation'
        abs_o = nam.abs(o)
        final_o = nam.final(o)
        mean_abs_o = nam.mean(abs_o)

        print(f'Computing orientation to origin based on {p}')
        s[o] = s.apply(lambda r: fun.angle_sum(fun.angle_to_x_axis(r[nam.xy(p)].values,origin), r[fo]), axis=1)
        s[abs_o] = np.abs(s[o].values)
        for id in ids:
            e.loc[id, final_o] = s[o].xs(id, level='AgentID').dropna().values[-1]
            e.loc[id, mean_abs_o] = s[abs_o].xs(id, level='AgentID').dropna().mean()
        if is_last:
            self.save()
        print('Orientation to origin computed')

    def compute_dst_to_origin(self, origin=np.array([0, 0]), start_time_in_sec=0.0, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        p=self.point
        print(f'Computing distance to origin based on {p}')
        d = 'dst_to_origin'
        p_fin = nam.final(d)
        p_max = nam.max(d)
        p_mu = nam.mean(d)
        t0 = int(start_time_in_sec / self.dt)
        for id in ids:
            xy_data = s[nam.xy(p)].xs(id, level='AgentID', drop_level=True)
            d = dst(list(xy_data.values), [origin])[:, 0]
            s.loc[(slice(None), id), d] = d
            e.loc[id, p_max] = np.nanmax(d)
            e.loc[id, p_mu] = np.nanmean(d[t0:])
            e.loc[id, p_fin] = d[~np.isnan(d)][-1]
            try:
                l = e.loc[id, 'length']
                s.loc[(slice(None), id), nam.scal(d)] = d / l
                e.loc[id, nam.scal(p_max)] = e.loc[id, p_max] / l
                e.loc[id, nam.scal(p_mu)] = e.loc[id, p_mu] / l
                e.loc[id, nam.scal(p_fin)] = e.loc[id, p_fin] / l
            except:
                pass
        if is_last:
            self.save()
        print('Distance to origin computed')

    def add_min_max_flags(self, parameters, interval_in_sec, threshold_in_std=None, absolute_threshold=None,
                          is_last=True):
        if absolute_threshold is None:
            absolute_threshold = [+np.inf, -np.inf]
        if self.step_data is None:
            self.load()
        order = int(interval_in_sec / self.dt)

        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        Npars = len(parameters)
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick

        min_array = np.ones([Nticks, Npars, Nids]) * np.nan
        max_array = np.ones([Nticks, Npars, Nids]) * np.nan

        for i, p in enumerate(parameters):
            print(f'Calculating local extrema for {p}')
            p_min, p_max = nam.min(p), nam.max(p)
            s[p_min] = np.nan
            s[p_max] = np.nan
            d = s[p]
            std = d.std()
            mu = d.mean()
            if threshold_in_std is not None:
                thr_min = mu - threshold_in_std * std
                thr_max = mu + threshold_in_std * std
            else:
                thr_min, thr_max = absolute_threshold
            for j, id in enumerate(ids):
                df = d.xs(id, level='AgentID', drop_level=True)
                i_min = argrelextrema(df.values, np.less_equal, order=order)[0]
                i_max = argrelextrema(df.values, np.greater_equal, order=order)[0]
                i_min = i_min[df.loc[i_min + t0] < thr_min]
                i_max = i_max[df.loc[i_max + t0] > thr_max]

                min_array[i_min, i, j] = True
                max_array[i_max, i, j] = True

            s[p_min] = min_array[:, i, :].flatten()
            s[p_max] = max_array[:, i, :].flatten()
        if is_last:
            self.save()
        print('All local extrema flagged')

    def compute_dominant_frequencies(self, parameters, freq_range=None, accepted_range=None,
                                     compare_params=False, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        ids = self.agent_ids
        Nids = len(ids)
        Npars = len(parameters)
        V = np.zeros(Npars)
        F = np.ones((Npars, Nids)) * np.nan
        for i, p in enumerate(parameters):
            for j, id in enumerate(ids):
                d = s[p].xs(id, level='AgentID', drop_level=True)
                try:
                    f, t, Sxx = spectrogram(d, fs=1 / self.dt)
                    if freq_range:
                        f0, f1 = freq_range
                        rng = np.where((f >= f0) & (f <= f1))

                        # keep only frequencies of interest
                        f = f[rng]
                        Sxx = Sxx[rng, :][0]
                        max_Sxx = np.nanmax(Sxx)
                        V[i] += max_Sxx / np.nansum(Sxx)
                    max_freqs = f[np.where(Sxx == max_Sxx)[0]]
                    max_freq = max_freqs[int(len(max_freqs) / 2)]
                    if accepted_range:
                        if not accepted_range[0] < max_freq < accepted_range[1]:
                            print(f'Dominant frequency of {p} for {id} : {max_freq} outside the accepted_range')
                            max_freq = np.nan
                except:
                    max_freq = np.nan
                    print(f'Dominant frequency of {p} for {id} not found')
                F[i, j] = max_freq
        if compare_params:
            for i, p in enumerate(parameters):
                print(p, V[i])
            ind = np.argmax(V)
            best_p = parameters[ind]
            print(f'Best parameter : {best_p}')
            existing = fun.common_member(nam.freq(parameters), self.endpoint_data.columns.values)
            e.drop(columns=existing, inplace=True)
            e[nam.freq(best_p)] = F[ind]
        else:
            for i, p in enumerate(parameters):
                e[nam.freq(p)] = F[i]
        if is_last:
            self.save()
        print('All dominant frequencies computed')

    def compute_preference_index(self, arena_diameter_in_mm=None, return_num=False):
        if not hasattr(self, 'endpoint_data'):
            self.load(step_data=False)
        if arena_diameter_in_mm is None:
            arena_diameter_in_mm = self.arena_xdim * 1000
        r = 0.2 * arena_diameter_in_mm
        d = self.endpoint_data[nam.final('x')]
        N = d.count()
        N_l = d[d <= -r / 2].count()
        N_r = d[d >= +r / 2].count()
        N_m = d[(d <= +r / 2) & (d >= -r / 2)].count()
        pI = np.round((N_l - N_r) / N, 3)
        if return_num:
            return pI, N
        else:
            return pI

    #######################################
    ########## PARSING : GENERAL ##########
    #######################################

    def parse_around_flag(self, par, flag, radius_in_sec, offset_in_sec=0, condition='True', save_as=None):
        parse_dataset(data=self.step_data, par=par, flag=flag, condition=condition,
                      radius_in_ticks=np.ceil(radius_in_sec / self.dt), offset_in_ticks=int(offset_in_sec / self.dt),
                      save_as=save_as, save_to=self.data_dir)

    def multiparse_by_sliding_window(self, data, par, flag, radius_in_sec, condition='True',
                                     description_as=None, overwrite=True):
        multiparse_dataset_by_sliding_window(data=data, par=par, flag=flag, condition=condition,
                                             radius_in_ticks=np.ceil(radius_in_sec / self.dt),
                                             description_to=f'{self.data_dir}/{par}_around_{flag}',
                                             description_as=description_as, overwrite=overwrite)

    def create_par_distro_dataset(self, pars):
        if self.step_data is None:
            self.load()
        s = self.step_data
        pars_to_store = [p for p in pars if p in self.step_data.columns]
        filenames = [f'{p}.csv' for p in pars_to_store]
        for p, filename in zip(pars_to_store, filenames):
            p_data = s[p].dropna().reset_index(level=0, drop=True)
            p_data.sort_index(inplace=True)
            p_data.to_csv(f'{self.par_distro_dir}/{filename}', index=True, header=True)
            print(f'Dataset saved as {filename}')

    def create_chunk_dataset(self, chunk, pars, Npoints=32):
        if self.step_data is None:
            self.load()
        ids = self.agent_ids
        s = self.step_data
        pars_to_store = [p for p in pars if p in self.step_data.columns]

        if chunk == 'stride':
            filenames = [f'{p}.csv' for p in pars_to_store]
        else:
            raise ValueError('Only stride chunks allowed')

        all_data = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        all_starts = [d[d[nam.start(chunk)] == True].index.values.astype(int) for d in all_data]
        all_stops = [d[d[nam.stop(chunk)] == True].index.values.astype(int) for d in all_data]

        p_timeseries = [[] for p in pars_to_store]
        p_chunk_ids = [[] for p in pars_to_store]
        for id, d, starts, stops in zip(ids, all_data, all_starts, all_stops):
            for start, stop in zip(starts, stops):
                for i, p in enumerate(pars_to_store):
                    timeserie = d.loc[slice(start, stop), p].values
                    p_timeseries[i].append(timeserie)
                    p_chunk_ids[i].append(id)
        p_durations = [[len(i) for i in t] for t in p_timeseries]

        p_chunks = [[np.interp(x=np.linspace(0, 2 * np.pi, Npoints), xp=np.linspace(0, 2 * np.pi, dur), fp=ts, left=0,
                               right=0) for dur, ts in zip(durations, timeseries)] for durations, timeseries in
                    zip(p_durations, p_timeseries)]
        chunk_dfs = []
        for chunks, chunk_ids, filename in zip(p_chunks, p_chunk_ids, filenames):
            chunk_df = pd.DataFrame(np.array(chunks), index=chunk_ids, columns=np.arange(Npoints).tolist())
            chunk_df.to_csv(f'{self.par_during_stride_dir}/{filename}', index=True, header=True)
            chunk_dfs.append(chunk_df)
            print(f'Dataset saved as {filename}')
        return chunk_dfs

    def create_dispersion_dataset(self, par='dispersion', scaled=True):
        if self.step_data is None:
            self.load()

        if scaled:
            p = nam.scal(par)
        else:
            p = par
        filepath = f'{self.dispersion_dir}/{p}.csv'
        dsp = self.step_data[p]
        steps = self.step_data.index.unique('Step')
        Nticks = len(steps)
        dsp_ar = np.zeros([Nticks, 3]) * np.nan
        dsp_m = dsp.groupby(level='Step').quantile(q=0.5)
        dsp_u = dsp.groupby(level='Step').quantile(q=0.75)
        dsp_b = dsp.groupby(level='Step').quantile(q=0.25)
        # print(Nticks, len(dsp_m),len(dsp_u),len(dsp_b) )
        dsp_ar[:, 0] = dsp_m
        dsp_ar[:, 1] = dsp_u
        dsp_ar[:, 2] = dsp_b
        dsp_df = pd.DataFrame(dsp_ar, index=steps, columns=['median', 'upper', 'lower'])
        dsp_df.to_csv(filepath, index=True, header=True)
        print(f'Dataset saved as {filepath}')

    def compute_chunk_metrics(self, chunk_name, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        Nids = len(self.agent_ids)
        print(f'Computing chunk metrics for {Nids} agents')
        chunk_dur = nam.dur(chunk_name)
        e[nam.num(chunk_name)] = s[nam.stop(chunk_name)].groupby('AgentID').sum()
        e[nam.cum(chunk_dur)] = s[chunk_dur].groupby('AgentID').sum()
        e[nam.mean(chunk_dur)] = s[chunk_dur].groupby('AgentID').mean()
        e[nam.std(chunk_dur)] = s[chunk_dur].groupby('AgentID').std()
        e[nam.dur_ratio(chunk_name)] = e[nam.cum(chunk_dur)] / e['cum_dur']

        if is_last:
            self.save()
        print('Chunk metrics computed')

    def detect_contacting_chunks(self, chunk, track_point, mid_flag=None, edge_flag=None,
                                 control_pars=[], vel_par=None,
                                 chunk_dur_in_sec=None, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        Nticks = len(s.index.unique('Step'))
        ids = self.agent_ids
        Nids = len(ids)
        t0 = self.starting_tick

        chunk_start = nam.start(chunk)
        chunk_stop = nam.stop(chunk)
        chunk_id = nam.id(chunk)
        chunk_dur = nam.dur(chunk)
        chunk_contact = nam.contact(chunk)
        chunk_or = nam.orient(chunk)
        chunk_chain = nam.chain(chunk)
        chunk_chain_dur = nam.dur(chunk_chain)
        chunk_chain_length = nam.length(chunk_chain)
        chunk_dst = nam.dst(chunk)
        chunk_strdst = nam.straight_dst(chunk)
        track_xy = nam.xy(track_point)
        track_dst = nam.dst(track_point)

        if 'length' in e.columns.values:
            lengths = e['length'].values
            scaled_chunk_strdst = nam.scal(chunk_strdst)
            scaled_chunk_dst = nam.scal(chunk_dst)
        else:
            lengths = None

        params = [chunk_dst, chunk_strdst]
        control_pars += [track_dst] + track_xy
        self.types_dict.update({col: float for col in params + nam.scal(params)})

        if vel_par:
            freqs = e[nam.freq(vel_par)]
            mean_freq = freqs.mean()
            print(f'Replacing {freqs.isna().sum()} nan values with population mean frequency : {mean_freq}')
            freqs.fillna(value=mean_freq, inplace=True)
            chunk_dur_in_ticks = 1 / (freqs.values * self.dt)
            control_pars.append(vel_par)
            # chunk_dur_in_ticks = [11 for c in chunk_dur_in_ticks]
            # chunk_dur_in_ticks = (1 / (freqs.values * self.dt)).astype(int)
        elif chunk_dur_in_sec:
            chunk_dur_in_ticks = np.ones(Nids) * chunk_dur_in_sec / self.dt
        all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        all_mid_flag_ticks = [d[d[mid_flag] == True].index.values for d in all_d]
        all_edge_flag_ticks = [d[d[edge_flag] == True].index.values for d in all_d]
        all_valid_ticks = [d[control_pars].dropna().index.values for d in all_d]
        all_chunks = []
        for t, edges, mids, valid, d in zip(chunk_dur_in_ticks, all_edge_flag_ticks, all_mid_flag_ticks,
                                            all_valid_ticks, all_d):
            chunks = np.array([[a, b] for a, b in zip(edges[:-1], edges[1:]) if (b - a >= 0.6 * t)
                               and (b - a <= 1.6 * t)
                               and set(np.arange(a, b + 1)) <= set(valid)
                               and (any((m > a) and (m < b) for m in mids))
                               ])
            all_chunks.append(chunks)
        all_durs = []
        all_contacts = []
        for chunks in all_chunks:
            if len(chunks) > 0:
                durs = np.diff(chunks, axis=1)[:, 0] * self.dt
                contacts = [int(stop1 == start2) for (start1, stop1), (start2, stop2) in
                            zip(chunks[:-1, :], chunks[1:, :])] + [0]
            else:
                durs = []
                contacts = []
            all_durs.append(durs)
            all_contacts.append(contacts)
        start_array = np.zeros([Nticks, Nids]) * np.nan
        stop_array = np.zeros([Nticks, Nids]) * np.nan
        dur_array = np.zeros([Nticks, Nids]) * np.nan
        contact_array = np.zeros([Nticks, Nids]) * np.nan
        orientation_array = np.zeros([Nticks, Nids]) * np.nan
        id_array = np.zeros([Nticks, Nids]) * np.nan
        chunk_chain_length_array = np.zeros([Nticks, Nids]) * np.nan
        chunk_chain_dur_array = np.zeros([Nticks, Nids]) * np.nan
        dst_array = np.zeros([Nticks, Nids]) * np.nan
        straight_dst_array = np.zeros([Nticks, Nids]) * np.nan
        scaled_dst_array = np.zeros([Nticks, Nids]) * np.nan
        scaled_straight_dst_array = np.zeros([Nticks, Nids]) * np.nan

        arrays = [start_array, stop_array, dur_array, contact_array, id_array,
                  chunk_chain_length_array, chunk_chain_dur_array,
                  dst_array, straight_dst_array, scaled_dst_array, scaled_straight_dst_array, orientation_array]

        pars = [chunk_start, chunk_stop, chunk_dur, chunk_contact, chunk_id,
                chunk_chain_length, chunk_chain_dur,
                chunk_dst, chunk_strdst, scaled_chunk_dst, scaled_chunk_strdst, chunk_or]

        for i, (d, chunks, durs, contacts) in enumerate(zip(all_d, all_chunks, all_durs, all_contacts)):
            Nchunks = len(chunks)
            if Nchunks == 0:
                continue
            starts = chunks[:, 0]
            stops = chunks[:, 1]
            start_array[starts - t0, i] = True
            stop_array[stops - t0, i] = True
            dur_array[stops - t0, i] = durs
            contact_array[stops - t0, i] = contacts
            chain_counter = 0
            chain_dur_counter = 0
            dists = np.zeros((Nchunks, 3)) * np.nan
            for j, (start, stop, dur, contact) in enumerate(zip(starts, stops, durs, contacts)):
                id_array[start - t0:stop - t0, i] = j
                chain_counter += 1
                chain_dur_counter += dur
                if contact == 0:
                    chunk_chain_length_array[stop - t0, i] = chain_counter
                    chunk_chain_dur_array[stop - t0, i] = chain_dur_counter
                    chain_counter = 0
                    chain_dur_counter = 0
                dst = d.loc[slice(start, stop), track_dst].sum()
                xy = d.loc[slice(start, stop), track_xy].dropna().values
                straight_dst = euclidean(tuple(xy[-1]), tuple(xy[0]))
                orient = fun.angle_to_x_axis(xy[0], xy[-1])
                dists[j] = np.array([dst, straight_dst, orient])
            dst_array[stops - t0, i] = dists[:, 0]
            straight_dst_array[stops - t0, i] = dists[:, 1]
            orientation_array[stops - t0, i] = dists[:, 2]

            if lengths is not None:
                l = lengths[i]
                scaled_dst_array[stops - t0, i] = dists[:, 0] / l
                scaled_straight_dst_array[stops - t0, i] = dists[:, 1] / l
        for array, par in zip(arrays, pars):
            s[par] = array.flatten()

        e[nam.cum(chunk_dst)] = s[chunk_dst].groupby('AgentID').sum()
        e[nam.mean(chunk_dst)] = s[chunk_dst].groupby('AgentID').mean()
        e[nam.std(chunk_dst)] = s[chunk_dst].groupby('AgentID').std()
        e[nam.cum(chunk_strdst)] = s[chunk_strdst].groupby('AgentID').sum()
        e[nam.mean(chunk_strdst)] = s[chunk_strdst].groupby('AgentID').mean()
        e[nam.std(chunk_strdst)] = s[chunk_strdst].groupby('AgentID').std()
        e['stride_reoccurence_rate'] = 1 - 1 / s[chunk_chain_length].groupby('AgentID').mean()

        if 'length' in e.columns.values:
            e[nam.cum(scaled_chunk_dst)] = e[nam.cum(chunk_dst)] / e['length']
            e[nam.mean(scaled_chunk_dst)] = e[nam.mean(chunk_dst)] / e['length']
            e[nam.std(scaled_chunk_dst)] = e[nam.std(chunk_dst)] / e['length']
            e[nam.cum(scaled_chunk_strdst)] = e[nam.cum(chunk_strdst)] / e['length']
            e[nam.mean(scaled_chunk_strdst)] = e[nam.mean(chunk_strdst)] / e['length']
            e[nam.std(scaled_chunk_strdst)] = e[nam.std(chunk_strdst)] / e['length']
        self.compute_chunk_metrics(chunk, is_last=False)
        if self.save_data_flag:
            self.create_par_distro_dataset([chunk_chain_dur, chunk_chain_length])

        self.types_dict.update({col: bool for col in [chunk_start, chunk_stop]})
        self.types_dict.update({col: float for col in [chunk_dur]})
        self.types_dict.update({col: float for col in [chunk_id, chunk_chain_length, chunk_contact]})

        if is_last:
            self.save()
        print('All chunks-around-flag detected')

    def compute_chunk_overlap(self, base_chunk, overlapping_chunk, is_last=True):
        if self.step_data is None:
            self.load()
        s = self.step_data
        ids = self.agent_ids
        c0, c1 = base_chunk, overlapping_chunk
        print(f'Computing overlap of {c1} on {c0} chunks')
        p = nam.overlap_ratio(base_chunk=c0, overlapping_chunk=c1)
        s[p] = np.nan
        p0_id = nam.id(c0)
        p1_id = nam.id(c1)
        for id in ids:
            d = s.xs(id, level='AgentID', drop_level=True)
            inds = np.unique(d[p0_id].dropna().values)
            s0s = d[d[nam.stop(c0)] == True].index
            for i, s0 in zip(inds, s0s):
                d0 = d[d[p0_id] == i]
                d1 = d0[p1_id].dropna()
                s.loc[(s0, id), p] = len(d1) / len(d0)
        self.types_dict.update({p: float})
        if is_last:
            self.save()
        print('All chunk overlaps computed')
        return s[p].dropna().sum()

    def track_parameters_during_chunk(self, chunk, pars, mode='dif', is_last=True):
        if self.step_data is None:
            self.load()
        ids = self.agent_ids
        Nids = len(ids)
        s, e = self.step_data, self.endpoint_data
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick

        c0 = nam.start(chunk)
        c1 = nam.stop(chunk)

        all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
        all_0s = [d[d[c0] == True].index.values.astype(int) for d in all_d]
        all_1s = [d[d[c1] == True].index.values.astype(int) for d in all_d]

        if mode == 'dif':
            p_tracks = nam.chunk_track(chunk, pars)
        elif mode == 'max':
            p_tracks = nam.max(nam.chunk_track(chunk, pars))
        elif mode == 'min':
            p_tracks = nam.min(nam.chunk_track(chunk, pars))
        p0s = [f'{p}_at_{c0}' for p in pars]
        p1s = [f'{p}_at_{c1}' for p in pars]

        for p, p_track, p0, p1 in zip(pars, p_tracks, p0s, p1s):
            p_pars = [p_track, p0, p1]
            p_array = np.zeros([Nticks, Nids, len(p_pars)]) * np.nan
            ds = [d[p] for d in all_d]
            for k, (id, d, s0s, s1s) in enumerate(zip(ids, ds, all_0s, all_1s)):
                a0s = d[s0s].values
                a1s = d[s1s].values
                if mode == 'dif':
                    vs = a1s - a0s
                elif mode == 'max':
                    vs = [d[slice(s0, s1)].max() for s0, s1 in zip(s0s, s1s)]
                elif mode == 'min':
                    vs = [d[slice(s0, s1)].min() for s0, s1 in zip(s0s, s1s)]
                p_array[s1s - t0, k, 0] = vs
                p_array[s1s - t0, k, 1] = a0s
                p_array[s1s - t0, k, 2] = a1s
            for i, p in enumerate(p_pars):
                s[p] = p_array[:, :, i].flatten()
            e[nam.mean(p_track)] = s[p_track].groupby('AgentID').mean()
            e[nam.std(p_track)] = s[p_track].groupby('AgentID').std()

        if self.save_data_flag:
            self.create_par_distro_dataset(p_tracks + p0s + p1s)
        if is_last:
            self.save()
        print('All parameters tracked')

    def detect_chunks_on_condition(self, chunk_name, condition_param, chunk_only=None, max_value=np.inf,
                                   min_value=-np.inf, non_overlap_chunk=None,
                                   store_min=False, store_max=False,
                                   min_duration=0, is_last=True):
        if self.step_data is None:
            self.load()
        ids = self.agent_ids
        Nids = len(ids)
        s, e = self.step_data, self.endpoint_data
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick
        if chunk_only is not None:
            ss = s.loc[s[nam.id(chunk_only)].dropna().index]
        else:
            ss = s
        print(f'Detecting chunks-on-condition for {Nids} agents')
        c = chunk_name
        p_s0 = nam.start(c)
        p_s1 = nam.stop(c)
        p_id = nam.id(c)
        p_dur = nam.dur(c)
        p_max = nam.max(nam.chunk_track(c, condition_param))
        p_min = nam.min(nam.chunk_track(c, condition_param))

        if non_overlap_chunk is not None:
            non_ov_id = nam.id(non_overlap_chunk)
            data = [ss.xs(id, level='AgentID', drop_level=True) for id in ids]
            data = [d[d[non_ov_id].isna()] for d in data]
            data = [d[condition_param] for d in data]
        else:
            data = [ss[condition_param].xs(id, level='AgentID', drop_level=True) for id in ids]
        a_inds = [d[(d < max_value) & (d > min_value)].index for d in data]
        a_s0s = [t_inds[np.where(np.diff(t_inds, prepend=[-1]) != 1)[0]] for t_inds in a_inds]
        a_s1s = [t_inds[np.where(np.diff(t_inds, append=[np.inf]) != 1)[0]] for t_inds in a_inds]
        a_durs = [np.array([(s1 - s0) * self.dt for s0, s1 in zip(t_s0s, t_s1s)]) for t_s0s, t_s1s in zip(a_s0s, a_s1s)]
        inds = [np.where(t_durs >= min_duration) for t_durs in a_durs]
        durs = [t_durs[t_inds] for t_durs, t_inds in zip(a_durs, inds)]
        starts = [t_starts[t_inds] for t_starts, t_inds in zip(a_s0s, inds)]
        stops = [t_stops[t_inds] for t_stops, t_inds in zip(a_s1s, inds)]

        S0 = np.zeros([Nticks, Nids]) * np.nan
        S1 = np.zeros([Nticks, Nids]) * np.nan
        Dur = np.zeros([Nticks, Nids]) * np.nan
        Id = np.zeros([Nticks, Nids]) * np.nan
        Max = np.zeros([Nticks, Nids]) * np.nan
        Min = np.zeros([Nticks, Nids]) * np.nan

        for i, (id, ds, s0s, s1s) in enumerate(zip(ids, durs, starts, stops)):
            S0[s0s - t0, i] = True
            S1[s1s - t0, i] = True
            Dur[s1s - t0, i] = ds
            for j, (s0, s1) in enumerate(zip(s0s, s1s)):
                Id[s0 - t0:s1 - t0, i] = j
                if store_max:
                    Max[s1 - t0, i] = s.loc[(slice(s0, s1), id), condition_param].max()
                if store_min:
                    Min[s1 - t0, i] = s.loc[(slice(s0, s1), id), condition_param].min()

        arrays = [S0, S1, Dur, Id, Max, Min]
        pars = [p_s0, p_s1, p_dur, p_id, p_max, p_min]
        for a, p in zip(arrays, pars):
            a = a.flatten()
            s[p] = a

        self.compute_chunk_metrics(c, is_last=False)
        self.types_dict.update({col: bool for col in [p_s0, p_s1]})
        self.types_dict.update({col: float for col in [p_dur, p_id]})

        if is_last:
            self.save()
        print('All chunks-on-condition detected')

    def detect_non_chunks(self, chunk_name, non_chunk_name=None, guide_parameter=None,
                          min_duration=0, is_last=True):
        if self.step_data is None:
            self.load()
        s, e = self.step_data, self.endpoint_data
        min_dur = int(min_duration / self.dt)
        Nticks = len(s.index.unique('Step'))
        t0 = self.starting_tick
        ids = self.agent_ids
        Nids = len(ids)
        if guide_parameter is None:
            guide_parameter = self.velocity
        chunk_id = nam.id(chunk_name)
        if non_chunk_name is None:
            c = nam.non(chunk_name)
        else:
            c = non_chunk_name

        S0 = np.zeros([Nticks, Nids]) * np.nan
        S1 = np.zeros([Nticks, Nids]) * np.nan
        Dur = np.zeros([Nticks, Nids]) * np.nan
        Id = np.zeros([Nticks, Nids]) * np.nan

        p_s0 = nam.start(c)
        p_s1 = nam.stop(c)
        p_id = nam.id(c)
        p_dur = nam.dur(c)

        print(f'Detecting non-chunks for {Nids} agents')
        for j, id in enumerate(ids):
            d = s.xs(id, level='AgentID', drop_level=True)
            nonna = d[guide_parameter].dropna().index.values
            inval = d[chunk_id].dropna().index.values
            val = np.setdiff1d(nonna, inval, assume_unique=True)
            if len(val) == 0:
                print(f'No valid steps for {id}')
                continue
            s0s = np.sort([val[0]] + val[np.where(np.diff(val) > 1)[0] + 1].tolist())
            s1s = np.sort(val[np.where(np.diff(val) > 1)[0]].tolist() + [val[-1]])

            if len(s0s) != len(s1s):
                print('Number of start and stop indexes does not match')
                min = np.min([len(s0s), len(s1s)])
                s0s = s0s[:min]
                s1s = s1s[:min]
            v_s0s = []
            v_s1s = []
            for i, (s0, s1) in enumerate(zip(s0s, s1s)):
                if s0 > s1:
                    print(i, s0, s1)
                    print('Start index bigger than stop index.')
                    continue
                elif s1 - s0 <= min_dur:
                    continue
                elif d[guide_parameter].loc[slice(s0, s1)].isnull().values.any():
                    continue
                else:
                    v_s0s.append(s0)
                    v_s1s.append(s1)
            v_s0s = np.array(v_s0s)
            v_s1s = np.array(v_s1s)
            S0[v_s0s - t0, j] = True
            S1[v_s1s - t0, j] = True
            Dur[v_s1s - t0, j] = (v_s1s - v_s0s) * self.dt
            for k, (v_s0, v_s1) in enumerate(zip(v_s0s, v_s1s)):
                Id[v_s0 - t0:v_s1 - t0, j] = k

        pars = [p_s0, p_s1, p_dur, p_id]
        arrays = [S0, S1, Dur, Id]
        for p, a in zip(pars, arrays):
            s[p] = a.flatten()

        self.compute_chunk_metrics(c, is_last=False)
        self.types_dict.update({col: bool for col in [p_s0, p_s1]})
        self.types_dict.update({col: float for col in [p_dur, p_id]})

        if is_last:
            self.save()
        print('All non-chunks detected')

    def analyse_bouts(self, mode='stride', dur_max_in_std=None, dur_range=None):
        if mode == 'stride':
            p = nam.length(nam.chain('stride'))
            analyse_bouts(dataset=self, parameter=p, label='stridechains', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range, xlabel=r'$N_{strides}$',
                          save_as='stridechain_analysis.pdf', save_to=None)
            p = nam.dur(nam.chain('stride'))
            analyse_bouts(dataset=self, parameter=p, label='stridechains', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range,
                          save_as='stridechain_duration_analysis.pdf', save_to=None)
            p = nam.dur(nam.non('stride'))
            analyse_bouts(dataset=self, parameter=p, label='stride-free bouts', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range,
                          save_as='stride-free_bout_analysis.pdf', save_to=None)
        elif mode == 'rest':
            p = nam.dur('rest')
            analyse_bouts(dataset=self, parameter=p, label='rest bouts', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range,
                          save_as='rest_bout_analysis.pdf', save_to=None)
            p = nam.dur('activity')
            analyse_bouts(dataset=self, parameter=p, label='activity bouts', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range,
                          save_as='activity_bout_analysis.pdf', save_to=None)
        elif mode == 'pause':
            p = nam.dur('pause')
            analyse_bouts(dataset=self, parameter=p, label='pause bouts', scale_coef=1,
                          dur_max_in_std=dur_max_in_std, dur_range=dur_range,
                          save_as='pause_bout_analysis.pdf', save_to=None)

    #######################################
    ########## PARSING : PAUSES ##########
    #######################################

    def create_pause_dataset(self):
        pauses, pause_file_path = self.create_chunk_dataset('pause', pars=['bend', nam.vel('bend')])
        self.store_pause_datasets(filepath=pause_file_path)

    def load_pause_dataset(self, load_simulated=False):
        try:
            filenames = [f'pause_{n}_dataset.csv' for n in ['bends', 'bendvels']]
            paths = [os.path.join(self.data_dir, name) for name in filenames]
            exp_bends = pd.read_csv(paths[0], index_col=0).values
            exp_bendvels = pd.read_csv(paths[1], index_col=0).values
        except:
            raise ValueError('Experimental pauses not found')

        if load_simulated:
            try:
                filenames = [f'pause_best_{n}_dataset.csv' for n in ['bends', 'bendvels', 'acts']]
                paths = [os.path.join(self.data_dir, name) for name in filenames]
                sim_bends = pd.read_csv(paths[0], index_col=0).values
                sim_bendvels = pd.read_csv(paths[1], index_col=0).values
                sim_acts = pd.read_csv(paths[2], index_col=0).values
                return exp_bends, exp_bendvels, sim_bends, sim_bendvels, sim_acts
            except:
                raise ValueError('Simulated pauses not found')
        else:
            return exp_bends, exp_bendvels

    def pause_analysis(self, condition_param=None, max_value=0.1, min_value=-np.inf, min_duration=1, is_last=True):
        if self.step_data is None:
            self.load()
        if condition_param == None:
            condition_param = nam.scal(self.velocity)
        self.compute_orientations(is_last=False)
        self.compute_angular_metrics(is_last=False)
        self.compute_linear_metrics(mode='minimal', is_last=False)
        self.detect_chunks_on_condition(chunk_name='pause', condition_param=condition_param,
                                        max_value=max_value, min_value=min_value,
                                        min_duration=min_duration, is_last=False)
        self.compute_spineangles(chunk_only='pause', mode='full', is_last=False)
        self.compute_angular_metrics(is_last=True)
        #
        # best_combo = self.plot_bend2orientation_analysis(data=None)
        best_combo = plot_bend2orientation_analysis(dataset=self)
        front_body_ratio = len(best_combo) / self.Nangles
        self.two_segment_model(front_body_ratio=front_body_ratio)
        self.compute_bend(is_last=False)
        self.compute_angular_metrics()
        pauses, pause_file_path = self.create_chunk_dataset('pause', pars=['bend', nam.vel('bend')])
        self.store_pause_datasets(filepath=pause_file_path)
        plot_pauses(dataset=self, Npauses=10)
        if is_last:
            self.save()
        print('All pauses detected')

    def detect_bend_pauses(self, condition_param=None, max_value=30, min_value=-30, min_duration=0.5, is_last=True):
        if self.step_data is None:
            self.load()
        if condition_param == None:
            condition_param = nam.vel('front_orientation')
        # if max_value is None :
        #     data=self.step_data[condition_param].dropna().values
        #     data=data[data>=0.0]
        #     data=data[data<=0.5]
        #     plt.hist(data, bins=100)
        #     plt.show()
        #     minima, maxima=density_extrema(data, kernel_width=0.02, Nbins=100)
        #     max_value=minima[0]
        #     print(minima)
        #     print(f'Velocity threshold set at {max_value}')
        # raise
        self.compute_orientations(mode='minimal', is_last=False)
        self.compute_angular_metrics(mode='minimal', is_last=False)
        self.detect_chunks_on_condition(chunk_name='bend_pause', condition_param=condition_param,
                                        max_value=max_value, min_value=min_value,
                                        min_duration=min_duration, is_last=False)

        if is_last:
            self.save()
        print('All bend-pauses detected')

    def detect_pauses(self, recompute_pauses=False, stride_non_overlap=True,
                      condition_param=None, max_value=None, min_value=-np.inf, min_duration=0.1,
                      is_last=True):
        if self.step_data is None:
            self.load()
        chunk = 'pause'
        if nam.num(chunk) in self.endpoint_data.columns.values and not recompute_pauses:
            print('Pauses are already detected. If you want to recompute it, set recompute_pauses to True')
            return
        if max_value is None:
            max_value = self.config['scaled_vel_threshold']

        if condition_param == None:
            condition_param = nam.scal(self.velocity)

        if stride_non_overlap:
            non_overlap_chunk = 'stride'
        else:
            non_overlap_chunk = None
        self.detect_chunks_on_condition(chunk_name=chunk, condition_param=condition_param,
                                        max_value=max_value, min_value=min_value,
                                        non_overlap_chunk=non_overlap_chunk,
                                        min_duration=min_duration, is_last=False)
        if self.save_data_flag:
            self.create_par_distro_dataset([nam.dur(chunk)])
        if is_last:
            self.save()
        print('All crawl-pauses detected')

    #######################################
    ########## PARSING : STRIDES ##########
    #######################################

    def detect_strides(self, recompute=False, flag=None, track_point=None, mid_flag=None, use_edge_flag=True,
                       non_chunks=True, is_last=True):
        if self.step_data is None:
            self.load()
        c = 'stride'
        if nam.num(c) in self.endpoint_data.columns.values and not recompute:
            print('Strides are already detected. If you want to recompute it, set recompute to True')
            return
        sv_thr = self.config['scaled_vel_threshold']
        if track_point is None:
            track_point = self.point
        if flag is None:
            flag = nam.scal('vel')
        if mid_flag is None:
            mid_flag = nam.max(flag)
        if use_edge_flag is True:
            edge_flag = nam.min(flag)
        else:
            edge_flag = None

        pars_to_track = [p for p in
                         [nam.unwrap('front_orientation'), nam.unwrap('rear_orientation'), 'bend'] if
                         p in self.step_data.columns]

        self.add_min_max_flags(parameters=[flag], interval_in_sec=0.3, absolute_threshold=[np.inf, sv_thr],is_last=False)
        self.compute_dominant_frequencies(parameters=[flag], freq_range=[0.7, 2.5], accepted_range=[0.7, 2.5],is_last=False)
        self.detect_contacting_chunks(chunk=c, mid_flag=mid_flag, edge_flag=edge_flag,
                                      vel_par=flag, control_pars=pars_to_track,
                                      track_point=track_point, is_last=False)
        if non_chunks:
            self.detect_non_chunks(chunk_name=c, guide_parameter=flag, min_duration=0.0, is_last=False)
        self.track_parameters_during_chunk(chunk=c, pars=pars_to_track, is_last=False)

        if self.save_data_flag:
            self.create_chunk_dataset(c, pars=[flag, 'spinelength', nam.vel('front_orientation'),
                                                        nam.vel('rear_orientation'),
                                                        nam.vel('bend')])

        if is_last:
            self.save()
        print('All strides detected')

    def stride_max_flag_phase_analysis(self, agent_id=None, flag=None, par_to_track=None):
        if self.step_data is None:
            self.load()
        if agent_id is None:
            agent_id = self.endpoint_data.num_ticks.nlargest(1).index.values[0]
        data = self.step_data.xs(agent_id, level='AgentID', drop_level=False)
        if flag is None:
            flag = nam.scal(self.velocity)
        if par_to_track is None:
            par_to_track = nam.scal(self.distance)
        f = self.endpoint_data[nam.freq(flag)].loc[agent_id]
        r = float((1 / f) / 2)
        self.multiparse_by_sliding_window(data=data, par=par_to_track, flag=nam.max(flag),
                                          radius_in_sec=r)
        optimal_flag_phase, mean_stride_dst = plot_sliding_window_analysis(dataset=self, parameter=par_to_track,
                                                                           flag=nam.max(flag),
                                                                           radius_in_sec=r)
        print(f'Optimal flag phase at {optimal_flag_phase} rad')
        print(f'Mean stride dst at optimum : {mean_stride_dst} (possibly scal)')
        self.stride_max_flag_phase = optimal_flag_phase
        self.config['stride_max_flag_phase'] = optimal_flag_phase
        self.save_config()

    def stride_analysis(self, agent_id=None, flag=None, par_to_track=None, stride_max_flag_analysis=True):
        if stride_max_flag_analysis:
            self.stride_max_flag_phase_analysis(agent_id=agent_id, flag=flag, par_to_track=par_to_track)
        plot_strides(dataset=self, agent_id=agent_id, radius_in_sec=None, save_as='parsed_strides.pdf', save_to=None)
        plot_stride_distribution(dataset=self, agent_id=agent_id, save_to=None)

    #######################################
    ########## PARSING : TURNS ##########
    #######################################

    def detect_turns(self, recompute_turns=False, min_ang_vel=0,
                     condition_param=None, chunk_only=None, track_params=None, min_duration=None,
                     constant_bend_chunks=False, is_last=True):

        if self.step_data is None:
            self.load()
        s,e=self.step_data,self.endpoint_data
        c=condition_param
        p_Rt, p_Lt, p_t=nam.chunk_track('Rturn', c), nam.chunk_track('Lturn', c), nam.chunk_track('turn', c)
        if set(nam.num(['Lturn', 'Rturn'])).issubset(e.columns.values) and not recompute_turns:
            print('Turns are already detected. If you want to recompute it, set recompute_turns to True')
            return
        if chunk_only is not None:
            ss = s.loc[s[nam.id(chunk_only)].dropna().index]
        else:
            ss = s
        if min_duration is None:
            min_duration = self.dt
        if c is None:
            b = 'bend'
            bv = nam.vel(b)
            ho = 'front_orientation'
            hov = nam.vel(ho)
            c = hov
        if track_params is None:
            track_params = [nam.unwrap(ho)]
            # track_params = [b, unwrap(ho)]

        self.detect_chunks_on_condition(chunk_name='Lturn', chunk_only=chunk_only, condition_param=c,min_value=min_ang_vel,
                                        store_max=True, min_duration=min_duration, is_last=False)
        self.detect_chunks_on_condition(chunk_name='Rturn', chunk_only=chunk_only, condition_param=c,max_value=-min_ang_vel,
                                        store_min=True, min_duration=min_duration, is_last=False)
        self.track_parameters_during_chunk(chunk='Lturn', pars=track_params, is_last=False)
        self.track_parameters_during_chunk(chunk='Rturn', pars=track_params, is_last=False)

        if constant_bend_chunks:
            print('Additionally detecting constant bend chunks.')
            self.detect_chunks_on_condition(chunk_name='constant_bend', chunk_only=chunk_only, condition_param=bv,
                                            min_value=-min_ang_vel, max_value=min_ang_vel,
                                            min_duration=min_duration, is_last=False)

        s[nam.dur('turn')] = s[[nam.dur('Rturn'), nam.dur('Lturn')]].sum(axis=1, min_count=1)
        self.create_par_distro_dataset([nam.dur('turn')])
        for p in track_params:
            s[f'turn_{p}'] = ss[[nam.chunk_track(chunk_name='Rturn', params=p),
                                             nam.chunk_track(chunk_name='Lturn', params=p)]].sum(axis=1, min_count=1)
            self.create_par_distro_dataset([f'turn_{p}'])


        if is_last:
            self.save()
        print('All turns detected')

    #######################################
    ########## FIT DISTRIBUTIONS ##########
    #######################################

    def fit_distribution(self, parameters, num_sample=None, num_candidate_dist=10, time_to_fit=120,
                         candidate_distributions=None, distributions=None, save_fits=False,
                         chunk_only=None, absolute=False):
        if self.step_data is None or self.endpoint_data:
            self.load()
        if chunk_only is not None:
            s = self.step_data.loc[self.step_data[nam.id(chunk_only)].dropna().index]
        else:
            s = self.step_data
        all_dists = sorted([k for k in st._continuous_distns.__all__ if not (
            (k.startswith('rv_') or k.endswith('_gen') or (k == 'levy_stable') or (k == 'weibull_min')))])
        dists = []
        for k in all_dists:
            dist = getattr(st.distributions, k)
            if dist.shapes is None:
                dists.append(k)
            elif len(dist.shapes) <= 1:
                dists.append(k)
        results = []
        for i, p in enumerate(parameters):
            try:
                d = self.endpoint_data[p].dropna().values
            except:
                d = s[p].dropna().values
            if absolute:
                d = np.abs(d)
            if distributions is None:
                if candidate_distributions is None:
                    if num_sample is None:
                        ids = self.agent_ids
                    else:
                        ids = self.agent_ids[:num_sample]
                    try:
                        sample = s.loc[(slice(None), ids), p].dropna().values
                    except:
                        sample = self.endpoint_data.loc[ids, p].dropna().values
                    if absolute:
                        sample = np.abs(sample)
                    f = Fitter(sample)
                    f.distributions = dists
                    f.fit()
                    dists = f.summary(Nbest=num_candidate_dist).index.values
                else:
                    dists = candidate_distributions
                ff = Fitter(d)
                ff.distributions = dists
                ff.timeout = time_to_fit
                ff.fit()
                distribution = ff.get_best()
            else:
                distribution = distributions[i]
            name = list(distribution.keys())[0]
            args = list(distribution.values())[0]
            stat, pv = stats.kstest(d, name, args=args)
            print(
                f'Parameter {p} was fitted best by a {name} of args {args} with statistic {stat} and p-value {pv}')
            results.append((name, args, stat, pv))

        if save_fits:
            fits = [[p, nam, args, st, pv] for p, (nam, args, st, pv)
                    in zip(parameters, results)]
            fits_pd = pd.DataFrame(fits, columns=['parameter', 'dist_name', 'dist_args', 'statistic', 'p_value'])
            fits_pd = fits_pd.set_index('parameter')
            try:
                self.fit_data = pd.read_csv(self.fit_file_path, index_col=['parameter'])
                self.fit_data = fits_pd.combine_first(self.fit_data)
                print('Updated fits')
            except:
                self.fit_data = fits_pd
                print('Initialized fits')
            self.fit_data.to_csv(self.fit_file_path, index=True, header=True)
        return results

    def fit_dataset(self, target_dir, target_point=None, fit_filename=None,
                    angular_fit=True, endpoint_fit=True, bout_fit=True, crawl_fit=True,
                    absolute=False,
                    save_to=None):
        if save_to is None:
            save_to = self.comp_plot_dir
        if not os.path.exists(save_to):
            os.makedirs(save_to)

        dd = LarvaDataset(dir=target_dir, load_data=False)
        if fit_filename is None:
            file = dd.fit_file_path
        else:
            file = os.path.join(dd.data_dir, fit_filename)

        if angular_fit:
            ang_fits = fit_angular_params(d=self, fit_filepath=file, absolute=absolute,
                                          save_to=save_to, save_as='angular_fit.pdf')
        if endpoint_fit:
            end_fits = fit_endpoint_params(d=self, fit_filepath=file,
                                           save_to=save_to, save_as='endpoint_fit.pdf')
        if crawl_fit:
            crawl_fits = fit_crawl_params(d=self, target_point=target_point, fit_filepath=file,
                                          save_to=save_to,
                                          save_as='crawl_fit.pdf')
        if bout_fit:
            bout_fits = fit_bout_params(d=self, fit_filepath=file, save_to=save_to,
                                        save_as='bout_fit.pdf')

    def load_fits(self, filepath=None, selected_pars=None):
        if filepath is None:
            filepath = self.fit_file_path
        target = pd.read_csv(filepath, index_col='parameter')
        if selected_pars is not None:
            valid_pars = [p for p in selected_pars if p in target.index.values]
            target = target.loc[valid_pars]
        pars = target.index.values.tolist()
        dist_names = target['dist_name'].values.tolist()
        dist_args = target['dist_args'].values.tolist()
        dist_args = [tuple(float(s) for s in v.strip("()").split(",")) for v in dist_args]
        dists = [{k: v} for k, v in zip(dist_names, dist_args)]
        stats = target['statistic'].values.tolist()
        return pars, dists, stats

    def fit_distributions_from_file(self, filepath, selected_pars=None, save_fits=True):
        pars, dists, stats = self.load_fits(filepath=filepath, selected_pars=selected_pars)
        results = self.fit_distribution(parameters=pars, distributions=dists, save_fits=save_fits)
        global_fit = 0
        for s, (dist_name, dist_args, statistic, p_value) in zip(stats, results):
            global_fit += np.clip(statistic - s, a_min=0, a_max=np.inf)
        return global_fit

    def fit_geom_to_stridechains(self, is_last=True):
        if self.step_data is None:
            self.load()
        stridechains = self.step_data[nam.length(nam.chain('stride'))]
        # self.endpoint_data['stride_reoccurence_rate'] = 1 - 1 / stridechains.mean()
        mean, std = stridechains.mean(), stridechains.std()
        print(f'Mean and std of stride reoccurence rate among larvae : {mean}, {std}')
        p, sse = fun.fit_geom_distribution(stridechains.dropna().values)
        print(f'Stride reoccurence rate is {1 - p}')
        self.stride_reoccurence_rate = 1 - p
        self.stride_reoccurence_rate_sse = sse
        self.config['stride_reoccurence_rate'] = self.stride_reoccurence_rate
        self.config['stride_reoccurence_rate_sse'] = self.stride_reoccurence_rate_sse
        self.save_config()
        if is_last:
            self.save()
        print('Geometric distribution fitted to stridechains')

    def generate_traj_colors(self, is_last=True):
        if self.step_data is None:
            self.load()
        pars = [nam.scal(self.velocity), nam.vel('front_orientation')]
        edge_colors = [[(255, 0, 0), (0, 255, 0)], [(255, 0, 0), (0, 255, 0)]]
        labels = ['lin_color', 'ang_color']
        lims = [0.8, 300]
        for i in [0, 1]:
            try:
                (r1, b1, g1), (r2, b2, g2) = edge_colors[i]
                # self.step_data[labels[i]]=np.nan
                self.step_data[labels[i]] = self.step_data[pars[i]].apply(
                    lambda x: (np.round(r1 + (r2 - r1) * np.clip(np.abs(x) / lims[i], a_min=0, a_max=1), 3),
                               np.round(b1 + (b2 - b1) * np.clip(np.abs(x) / lims[i], a_min=0, a_max=1), 3),
                               np.round(g1 + (g2 - g1) * np.clip(np.abs(x) / lims[i], a_min=0, a_max=1), 3)))
            except:
                pass
        if is_last:
            self.save()

    def configure_body(self, Npoints, Ncontour):
        N, Nc = Npoints, Ncontour
        self.points = nam.midline(N, type='point')
        self.Nangles = np.clip(N - 2, a_min=0, a_max=None)
        self.angles = [f'angle{i}' for i in range(self.Nangles)]
        self.Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        self.segs = nam.midline(self.Nsegs, type='seg')

        self.points_xy = nam.xy(self.points)
        self.points_dst = nam.dst(self.points)
        self.points_vel = nam.vel(self.points)
        self.points_acc = nam.acc(self.points)
        self.point_lin_pars = self.points_dst + self.points_vel + self.points_acc

        self.angles_vel = nam.vel(self.angles)
        self.angles_acc = nam.acc(self.angles)
        self.angle_pars = self.angles + self.angles_vel + self.angles_acc

        self.contour = nam.contour(Nc)
        self.contour_xy = nam.xy(self.contour)

        self.cent_xy = nam.xy('centroid')
        self.cent_dst = nam.dst('centroid')
        self.cent_vel = nam.vel('centroid')
        self.cent_acc = nam.acc('centroid')
        self.cent_lin_pars = [self.cent_dst, self.cent_vel, self.cent_acc]

        ang = ['front_orientation', 'rear_orientation', 'bend']
        self.ang_pars = ang + nam.unwrap(ang) + nam.vel(ang) + nam.acc(ang)
        self.xy_pars = nam.xy(self.points + self.contour + ['centroid'], flat=True) + ['x', 'y']

    def build_types_dict(self):
        dic = {'Step': int, 'AgentID': str}
        dic.update({col: float for col in
                    nam.xy(self.points, flat=True) + self.point_lin_pars + self.angle_pars + self.segs + [
                        'dispersion', nam.scal('dispersion'),
                        'spinelength'] + self.ang_pars + nam.vel(
                        self.ang_pars) + nam.acc(self.ang_pars)})
        dic.update(
            {col: float for col in nam.scal(self.point_lin_pars + self.segs)})
        dic.update(
            {col: float for col in self.cent_xy + self.cent_lin_pars + nam.scal(
                self.cent_lin_pars)})
        dic.update({col: float for col in nam.xy(self.contour, flat=True)})
        dic.update({'collision_flag': float})
        return dic

    def define_paths(self, dir):
        self.dir = dir
        self.data_dir = os.path.join(dir, 'data')
        self.plot_dir = os.path.join(dir, 'plots')
        self.vis_dir = os.path.join(dir, 'visuals')
        self.aux_dir = os.path.join(dir, 'aux')
        self.par_distro_dir = os.path.join(self.aux_dir, 'par_distros')
        self.par_during_stride_dir = os.path.join(self.aux_dir, 'par_during_stride')
        self.dispersion_dir = os.path.join(self.aux_dir, 'dispersion')
        self.comp_plot_dir = os.path.join(self.plot_dir, 'comparative')
        self.dirs = [self.dir, self.data_dir, self.plot_dir, self.vis_dir, self.comp_plot_dir,
                     self.aux_dir, self.par_distro_dir, self.par_during_stride_dir, self.dispersion_dir]

        self.step_file_path = os.path.join(self.data_dir, 'step_data.csv')
        self.endpoint_file_path = os.path.join(self.data_dir, 'endpoint_data.csv')
        self.food_endpoint_file_path = os.path.join(self.data_dir, 'food_endpoint_data.csv')
        self.sim_pars_file_path = os.path.join(self.data_dir, 'sim_conf.txt')
        self.fit_file_path = os.path.join(self.data_dir, 'dataset_fit.csv')
        self.config_file_path = os.path.join(self.data_dir, 'dataset_conf.csv')

    def define_linear_metrics(self, config):
        if not np.isnan(config['point_idx']):
            self.point = self.points[config['point_idx'] - 1]
        else:
            self.point = 'centroid'
        self.distance = nam.dst(self.point)
        self.velocity = nam.vel(self.point)
        self.acceleration = nam.acc(self.point)
        if config['use_component_vel']:
            # self.distance = nam.lin(self.distance)
            self.velocity = nam.lin(self.velocity)
            self.acceleration = nam.lin(self.acceleration)

    def enrich(self, rescale_by=None, drop_collisions=False, interpolate_nans=False, filter_f=None,
               length_and_centroid=True, drop_contour=False, drop_unused_pars=False,
               drop_immobile=False, mode='minimal', dispersion_starts=[0],
               ang_analysis=True, lin_analysis=True, bout_annotation=['turn', 'stride', 'pause']):
        # print(self.config)
        # raise
        if rescale_by is not None:
            self.rescale(scale=rescale_by, is_last=False)
        if drop_collisions:
            self.exclude_rows(flag_column='collision_flag', accepted_values=[0], is_last=False)
        if interpolate_nans:
            self.interpolate_nans(pars=self.xy_pars, is_last=False)
        if filter_f is not None:
            self.apply_filter(pars=self.xy_pars, freq=filter_f, inplace=True, is_last=False)
        if length_and_centroid:
            self.compute_length_and_centroid(drop_contour=drop_contour, is_last=False)
        if ang_analysis:
            self.angular_analysis(recompute=False, mode=mode, is_last=False)
            if 'turn' in bout_annotation:
                self.detect_turns(recompute_turns=False, is_last=False)
        if lin_analysis:
            self.linear_analysis(mode=mode, is_last=False)
            self.compute_dispersion(recompute=False, starts=dispersion_starts, is_last=False)
            self.compute_tortuosity(is_last=False)
            if 'stride' in bout_annotation:
                self.detect_strides(recompute=False, is_last=False)
            if 'pause' in bout_annotation:
                self.detect_pauses(recompute_pauses=False, is_last=False)
        self.generate_traj_colors(is_last=False)

        if drop_immobile:
            self.drop_immobile_larvae(is_last=False)
        if drop_unused_pars:
            self.drop_unused_pars(is_last=False)
        self.save()

    def create_reference_dataset(self):
        if self.endpoint_data is None:
            self.load()
        # if not os.path.exists(RefFolder):
        #     os.makedirs(RefFolder)
        copy_tree(self.dir, RefFolder)
        e = self.endpoint_data
        pars = ['length', 'scaled_vel_freq',
                'stride_reoccurence_rate', 'scaled_stride_dst_mean', 'scaled_stride_dst_std']
        sample_pars = ['body_params.initial_length', 'neural_params.crawler_params.initial_freq',
                       'neural_params.intermitter_params.crawler_reoccurence_rate',
                       'neural_params.crawler_params.step_to_length_mu',
                       'neural_params.crawler_params.step_to_length_std'
                       ]
        v = e[pars].values
        v[:, 0] /= 1000
        df = pd.DataFrame(v, columns=sample_pars)
        df.to_csv(Ref_path)

        plot_stridesNpauses(datasets=[self], labels=['reference'],
                            stridechain_duration=False, pause_chunk='pause', time_unit='sec',
                            plot_fits='all', range='default',
                            save_to=None, save_as='stridesNpauses',
                            save_fits_to=None, save_fits_as=Ref_fits)
        print(f'Reference dataset saved.')

    def raw_or_filtered_xy(self, points):
        r = nam.xy(points, flat=True)
        f = nam.filt(r)
        if all(i in self.step_data.columns for i in f) :
            print('Using filtered xy coordinates')
            return f
        elif all(i in self.step_data.columns for i in r):
            print('Using raw xy coordinates')
            return r
        else:
            print('No xy coordinates exist. Not computing spatial metrics')
            return

    def drop_immobile_larvae(self, vel_threshold=0.1, is_last=True):
        self.compute_spatial_metrics(mode='minimal')
        D = self.step_data[nam.scal('velocity')]
        immobile_ids = []
        for id in self.agent_ids:
            d = D.xs(id, level='AgentIDs').dropna().values
            if len(d[d > vel_threshold]) == 0:
                immobile_ids.append(id)
        # e = self.endpoint_data
        # dsts = e[nam.cum(nam.dst(self.point))]
        # immobile_ids = dsts[dsts < min_dst].index.values
        print(f'{len(immobile_ids)} immobile larvae will be dropped')
        if len(immobile_ids) > 0:
            self.drop_agents(agents=immobile_ids, is_last=is_last)

    def get_par(self, par, endpoint_par=True):
        try:
            p_df = self.load_par_distro_dataset(par)
        except:
            if endpoint_par:
                if not hasattr(self, 'endpoint_data'):
                    self.load(step_data=False)
                p_df = self.endpoint_data[par]
            else:
                if not hasattr(self, 'step_data'):
                    self.load(endpoint_data=False)
                p_df = self.step_data[par]
        return p_df

    def delete(self):
        shutil.rmtree(self.dir)
        print('Dataset deleted')

    def compute_tortuosity(self, durs_in_sec=[2, 5, 10, 20], is_last=True):
        if self.endpoint_data is None:
            self.load(step_data=False)
        e=self.endpoint_data
        e['tortuosity'] = 1 - e['final_dispersion'] / e['cum_dst']
        durs = [int(self.fr * d) for d in durs_in_sec]
        Ndurs = len(durs)
        if Ndurs > 0:
            if self.step_data is None:
                self.load(endpoint_data=False)
            ids = self.agent_ids
            Nids = len(ids)
            s = self.step_data
            ds = [s[['x', 'y']].xs(id, level='AgentID') for id in ids]
            ds = [d.loc[d.first_valid_index(): d.last_valid_index()].values for d in ds]
            for j, r in enumerate(durs):
                par = f'tortuosity_{durs_in_sec[j]}'
                par_m, par_s = nam.mean(par), nam.std(par)
                T_m = np.ones(Nids) * np.nan
                T_s = np.ones(Nids) * np.nan
                for z, id in enumerate(ids):
                    si = ds[z]
                    u = len(si) % r
                    if u > 1:
                        si0 = si[:-u + 1]
                    else:
                        si0 = si[:-r + 1]
                    k = int(len(si0) / r)
                    T = []
                    for i in range(k):
                        t = si0[i * r:i * r + r + 1, :]
                        if np.isnan(t).any():
                            continue
                        else:
                            t_D = np.sum(np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1)))
                            t_L = np.sqrt(np.sum(np.array(t[-1, :] - t[0, :]) ** 2))
                            t_T = 1 - t_L / t_D
                            T.append(t_T)
                    T_m[z] = np.mean(T)
                    T_s[z] = np.std(T)
                e[par_m] = T_m
                e[par_s] = T_s

        print('Tortuosities computed')
        if is_last:
            self.save()

    def deb_analysis(self, is_last=True) :
        if self.step_data is None:
            self.load()
        s,e=self.step_data, self.endpoint_data
        e[nam.mean('deb_f')] = s['deb_f'].groupby('AgentID').mean()
        if is_last:
            self.save()

    def set_id(self, id):
        self.id=id
        self.config['id']=id
        self.save_config()

    def build_dirs(self):
        for i in self.dirs:
            if not os.path.exists(i):
                os.makedirs(i)

    def analysis(self):
        comparative_analysis(datasets=[self], labels=[self.id])

    def store_pause_datasets(self, filepath):
        d = pd.read_csv(filepath, index_col=['AgentID', 'Chunk'])
        pauses = []
        for i, row in d.iterrows():
            v = row.dropna().values[1:]
            v = [literal_eval(k) for k in v]
            v = np.array(v)
            pauses.append(v)
        B = fun.boolean_indexing([p[:, 0] for p in pauses], fillval=np.nan)
        BV = fun.boolean_indexing([p[:, 1] for p in pauses], fillval=np.nan)
        pds = [pd.DataFrame(data=B), pd.DataFrame(data=BV)]
        filenames = [f'pause_{n}_dataset.csv' for n in ['bends', 'bendvels']]
        for pdf, name in zip(pds, filenames):
            path = os.path.join(self.data_dir, name)
            pdf.to_csv(path, index=True, header=True)

    def split_dataset(self, larva_id_prefixes):
        if self.step_data is None:
            self.load()
        # s,e=self.step_data, self.endpoint_data
        new_ds=[]
        for f in larva_id_prefixes :
            new_id=f'{self.id}_{f}'
            new_dir = f'{self.dir}/../{new_id}'
            copy_tree(self.dir, new_dir)
            new_d=LarvaDataset(new_dir)
            new_d.set_id(new_id)

            invalid_ids=[id for id in self.agent_ids if not str.startswith(id, f)]
            new_d.drop_agents(invalid_ids)
            new_ds.append(new_d)
        print(f'Dataset {self.id} splitted in {[d.id for d in new_ds]}')
        return new_ds



