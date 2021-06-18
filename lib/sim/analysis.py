import numpy

from lib.anal.plotting import plot_endpoint_scatter, plot_turn_Dbearing, plot_turn_amp, plot_turns, plot_timeplot, \
    plot_navigation_index, plot_debs, plot_food_amount, plot_gut, plot_pathlength, plot_endpoint_params, barplot, \
    comparative_analysis, plot_marked_strides, plot_marked_turns, plot_chunk_Dorient2source, plot_distance_to_source
from lib.aux import functions
from lib.conf import dtype_dicts as dtypes
from lib.model.DEB.deb import deb_default
from lib.sim.single_run import load_reference_dataset
from lib.stor.larva_dataset import LarvaDataset


def sim_analysis(d: LarvaDataset, exp_type, show_output = False):
    if d is None:
        return
    s, e = d.step_data, d.endpoint_data
    fig_dict = {}
    results = {}
    if exp_type in ['patchy_food', 'uniform_food', 'food_grid']:
        # am = e['amount_eaten'].values
        # print(am)
        # cr,pr,fr=e['stride_dur_ratio'].values, e['pause_dur_ratio'].values, e['feed_dur_ratio'].values
        # print(cr+pr+fr)
        # cN, pN, fN = e['num_strides'].values, e['num_pauses'].values, e['num_feeds'].values
        # print(cN, pN, fN)
        # cum_sd, f_success=e['cum_scaled_dst'].values, e['feed_success_rate'].values
        # print(cum_sd, f_success)
        fig_dict['scatter_x4'] = plot_endpoint_scatter(datasets=[d], par_shorts=['cum_sd', 'f_am', 'str_tr', 'fee_tr'])
        fig_dict['scatter_x2'] = plot_endpoint_scatter(datasets=[d], par_shorts=['cum_sd', 'f_am'])

    elif exp_type in ['food_at_bottom']:
        ds = d.split_dataset(is_last=False, show_output = show_output)
        cc = {'datasets': ds,
              'save_to': d.plot_dir,
              'subfolder': None}

        fig_dict['orientation to center'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=None, par='orientation_to_center', **cc)
        fig_dict['bearing to 270deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=270, par=None, **cc)
        # fig_dict['bearing to -90deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=-90, par=None, **cc)
        # fig_dict['bearing to 90deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=90, par=None, **cc)
        # fig_dict['bearing to 0deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=0, par=None, **cc)
        # fig_dict['bearing to 180deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=180, par=None, **cc)
        # fig_dict['bearing to -180deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=-180, par=None, **cc)
        # fig_dict['bearing to 40deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=40, par=None, **cc)
        # fig_dict['bearing to -115deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=-115, par=None, **cc)
        fig_dict['bearing correction VS Y pos'] = plot_turn_amp(par_short='tur_y0', mode='hist', ref_angle=270, **cc)
        fig_dict['turn angle VS Y pos (hist)'] = plot_turn_amp(par_short='tur_y0', mode='hist',**cc)

        fig_dict['turn angle VS Y pos (scatter)'] = plot_turn_amp(par_short='tur_y0', mode='scatter', **cc)
        fig_dict['turn duration'] = plot_turn_amp(par_short='tur_t', mode='scatter', absolute=True, **cc)
        fig_dict['turn amplitude'] = plot_turns(**cc)
        fig_dict['Y position'] = plot_timeplot(['y'], show_first=False,legend_loc='lower left',  **cc)
        fig_dict['navigation index'] = plot_navigation_index(**cc)

        # for d in datasets :
        #     d.delete()

    elif exp_type in ['growth', 'rovers_sitters']:
        deb_model = deb_default(epochs=d.config['epochs'], substrate_quality=d.config['substrate_quality'])

        if exp_type == 'rovers_sitters':
            roversVSsitters = True
            ds = d.split_dataset(groups=['Sitter', 'Rover'], show_output = show_output)
            labels = ['Sitters', 'Rovers']
        else:
            roversVSsitters = False
            ds = [d]
            labels = [d.id]

        deb_dicts = list(d.load_deb_dicts().values()) + [deb_model]
        # print(d.load_deb_dicts().values())
        # deb_dicts = [deb_dict(d, id, epochs=epochs) for id in d.agent_ids] + [deb_model]
        c = {'save_to': d.plot_dir,
             'roversVSsitters': roversVSsitters}
        c1 = {'deb_dicts': deb_dicts[:-1],
              'sim_only': True}


        for m in ['feeding','reserve_density', 'fs', 'assimilation', 'food_ratio_1','food_ratio_2','food_mass_1','food_mass_2']:
            # for m in ['f', 'hunger', 'minimal', 'full', 'complete']:
            # for t in ['hours', 'seconds']:
            # print(m)
            for t in ['hours']:
                save_as = f'{m}_in_{t}.pdf'
                fig_dict[f'{m} ({t})'] = plot_debs(save_as=save_as, mode=m, time_unit=t, **c, **c1)

        for m in ['energy','growth',  'full']:
            save_as = f'{m}_vs_model.pdf'
            fig_dict[f'{m} vs model'] = plot_debs(deb_dicts=deb_dicts, save_as=save_as, mode=m, **c)

        if exp_type == 'rovers_sitters':
            cc = {'datasets': ds,
                  'labels': labels,
                  'save_to': d.plot_dir}
            fig_dict['faeces ratio'] = plot_timeplot(['f_out_r'], show_first=False, legend_loc='upper left', **cc)
            fig_dict['faeces amount'] = plot_timeplot(['f_out'], show_first=False, legend_loc='upper left', **cc)
            fig_dict['food absorption efficiency'] = plot_timeplot(['abs_r'], show_first=False, legend_loc='upper left', **cc)
            fig_dict['food absorbed'] = plot_timeplot(['f_ab'], show_first=False, legend_loc='upper left', **cc)
            fig_dict['food intake (timeplot)'] = plot_timeplot(['f_am'], show_first=False, legend_loc='upper left', **cc)
            fig_dict['food intake'] = plot_food_amount(**cc)
            fig_dict['food intake (filt)'] = plot_food_amount(filt_amount=True, **cc)
            fig_dict['gut occupancy'] = plot_gut(**cc)
            fig_dict['pathlength'] = plot_pathlength(scaled=False, **cc)
            fig_dict['endpoint'] = plot_endpoint_params(mode='deb', **cc)
            try:
                fig_dict['food intake (barplot)'] = barplot(par_shorts=['f_am'], **cc)
            except:
                pass


    elif exp_type == 'dispersion':

        target_dataset = load_reference_dataset(dataset_id=d.config['sample_dataset'])
        ds = [d, target_dataset]
        labels = ['simulated', 'empirical']

        # targeted_analysis(ds)
        dic0 = comparative_analysis(datasets=ds, labels=labels, simVSexp=True, save_to=None)
        fig_dict.update(dic0)
        dic1 = plot_marked_strides(dataset=d, agent_ids=d.agent_ids[:3], title=' ', slices=[[10, 50], [60, 100]])
        fig_dict.update(dic1)
        dic2 = plot_marked_turns(dataset=d, agent_ids=d.agent_ids[:3], min_turn_angle=20)
        fig_dict.update(dic2)




    elif exp_type in ['chemotaxis_approach', 'chemotaxis_local', 'chemotaxis_diffusion']:
        if exp_type in ['chemotaxis_local', 'chemotaxis_diffusion']:
            for chunk in ['turn', 'stride', 'pause']:
                for dur in [0.0, 0.5, 1.0]:
                    fig_dict[f'{chunk}_Dorient2source_min_{dur}_sec'] = plot_chunk_Dorient2source(datasets=[d],
                                                                                                  chunk=chunk,
                                                                                                  source=(0.0, 0.0),
                                                                                                  min_dur=dur)
            # fig_dict['turn_Dorient2center'] = plot_turn_Dorient2center(datasets=[d], labels=[d.id])
        for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']:
            fig_dict[p] = plot_timeplot([p], datasets=[d])
        dic = plot_distance_to_source(dataset=d, exp_type=exp_type)
        fig_dict.update(dic)
        vis_kwargs = dtypes.get_dict('visualization', mode='image', image_mode='final', show_display=False,
                                     random_colors=True, trajectories=True, trajectory_dt=0,
                                     visible_clock=False, visible_scale=False, media_name='single_trajectory')
        d.visualize(agent_ids=[d.agent_ids[0]], vis_kwargs=vis_kwargs)
    elif exp_type in ['odor_pref_test', 'odor_pref_train', 'odor_pref_test_on_food']:
        ind = d.compute_preference_index()
        print(f'Preference for left odor : {np.round(ind, 3)}')
        results['PI'] = ind

    if exp_type in ['odor_preference_RL', 'odor_pref_train']:
        fig_dict['best_gains_table'] = plot_timeplot(['g_odor1', 'g_odor2'], datasets=[d], show_first=False,
                                                     table='best_gains')
        fig_dict['reward_table'] = plot_timeplot(['cum_reward'], datasets=[d], show_first=False, table='best_gains')
        fig_dict['olfactor_decay_table'] = plot_timeplot(['D_olf'], datasets=[d], show_first=False, table='best_gains')
        # fig_dict['best_gains'] = plot_timeplot(['g_odor1', 'g_odor2'], datasets=[d], show_first=False)
        # fig_dict['best_gains_inds'] = plot_timeplot(['g_odor1', 'g_odor2'], datasets=[d], show_first=False, individuals=True)
    elif exp_type == 'chemotaxis_RL':
        fig_dict['best_gains_table'] = plot_timeplot(['g_odor1'], datasets=[d], show_first=False, table='best_gains')
        fig_dict['reward_table'] = plot_timeplot(['cum_reward'], datasets=[d], show_first=False, table='best_gains')
        fig_dict['olfactor_decay_table'] = plot_timeplot(['D_olf'], datasets=[d], show_first=False, table='best_gains')
        fig_dict['olfactor_decay_table_inds'] = plot_timeplot(['D_olf'], datasets=[d], show_first=False,
                                                              table='best_gains', individuals=True)
        # fig_dict['best_gains'] = plot_timeplot(['g_odor1', 'g_odor2'], datasets=[d], show_first=False)
    elif exp_type == 'realistic_imitation':
        d.save_agent(pars=fun.flatten_list(d.points_xy) + fun.flatten_list(d.contour_xy), header=True)

    print(f'    Analysis complete!')
    return fig_dict, results