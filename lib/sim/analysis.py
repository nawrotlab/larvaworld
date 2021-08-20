import numpy as np

from lib.anal.plotting import plot_endpoint_scatter, plot_turn_Dbearing, plot_turn_amp, plot_turns, plot_timeplot, \
    plot_navigation_index, plot_debs, plot_food_amount, plot_gut, plot_pathlength, plot_endpoint_params, barplot, \
    comparative_analysis, plot_marked_turns, plot_chunk_Dorient2source, plot_marked_strides, targeted_analysis, \
    plot_stridesNpauses, plot_ang_pars, plot_interference
from lib.aux import functions as fun
from lib.conf import dtype_dicts as dtypes
from lib.model.DEB.deb import deb_default
from lib.sim.single_run import load_reference_dataset
from lib.stor.larva_dataset import LarvaDataset


def sim_analysis(d: LarvaDataset, exp_type, show_output=False):
    ccc = {'show': False}
    if d is None:
        return
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

        # fig_dict['angular'] = plot_ang_pars(datasets=[d], **ccc)
        # fig_dict['bouts'] = plot_stridesNpauses(datasets=[d], plot_fits=None, only_fit_one=False, test_detection=True,
        #                                         **ccc)

        fig_dict['scatter_x4'] = plot_endpoint_scatter(datasets=[d], keys=['cum_sd', 'f_am', 'str_tr', 'pau_tr'], **ccc)
        fig_dict['scatter_x2'] = plot_endpoint_scatter(datasets=[d], keys=['cum_sd', 'f_am'], **ccc)

    elif exp_type in ['food_at_bottom']:
        ds = d.split_dataset(is_last=False, show_output=show_output)
        cc = {'datasets': ds,
              'save_to': d.plot_dir,
              'subfolder': None,
              **ccc}

        fig_dict['bearing correction VS Y pos'] = plot_turn_amp(par_short='tur_y0', mode='hist', ref_angle=270, **cc)
        fig_dict['turn angle VS Y pos (hist)'] = plot_turn_amp(par_short='tur_y0', mode='hist', **cc)
        fig_dict['turn angle VS Y pos (scatter)'] = plot_turn_amp(par_short='tur_y0', mode='scatter', **cc)
        fig_dict['turn duration'] = plot_turn_amp(par_short='tur_t', mode='scatter', absolute=True, **cc)
        fig_dict['turn amplitude'] = plot_turns(**cc)
        fig_dict['Y position'] = plot_timeplot(['y'], show_first=False, legend_loc='lower left', **cc)
        fig_dict['navigation index'] = plot_navigation_index(**cc)
        fig_dict['orientation to center'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=None, **cc)
        fig_dict['bearing to 270deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=270, par=None, **cc)

    elif exp_type in ['growth', 'rovers_sitters']:
        deb_model = deb_default(epochs=d.config['epochs'], substrate_quality=d.config['substrate_quality'])
        if exp_type == 'rovers_sitters':
            roversVSsitters = True
            ds = d.split_dataset(groups=['Sitter', 'Rover'], show_output=show_output)
            labels = ['Sitters', 'Rovers']
        else:
            roversVSsitters = False
            ds = [d]
            labels = [d.id]

        deb_dicts = d.load_deb_dicts() + [deb_model]
        c = {'save_to': d.plot_dir,
             'roversVSsitters': roversVSsitters}
        c1 = {'deb_dicts': deb_dicts[:-1],
              'sim_only': True}

        for m in ['feeding', 'reserve_density', 'fs', 'assimilation', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
                  'food_mass_2']:
            for t in ['hours']:
                save_as = f'{m}_in_{t}.pdf'
                fig_dict[f'{m} ({t})'] = plot_debs(save_as=save_as, mode=m, time_unit=t, **c, **c1, **ccc)

        for m in ['energy', 'growth', 'full']:
            save_as = f'{m}_vs_model.pdf'
            fig_dict[f'{m} vs model'] = plot_debs(deb_dicts=deb_dicts, save_as=save_as, mode=m, **c, **ccc)

        if exp_type == 'rovers_sitters':
            cc = {'datasets': ds,
                  'labels': labels,
                  'save_to': d.plot_dir,
                  **ccc}
            cc1 = {'show_first': False, 'legend_loc': 'upper_left', **cc}

            fig_dict['faeces ratio'] = plot_timeplot(['f_out_r'], **cc1)
            fig_dict['faeces amount'] = plot_timeplot(['f_out'], **cc1)
            fig_dict['food absorption efficiency'] = plot_timeplot(['abs_r'], **cc1)
            fig_dict['food absorbed'] = plot_timeplot(['f_ab'], **cc1)
            fig_dict['food intake (timeplot)'] = plot_timeplot(['f_am'], **cc1)

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
        dic0 = comparative_analysis(datasets=ds, labels=labels, simVSexp=True, save_to=None, **ccc)
        fig_dict.update(dic0)
        dic1 = {f'marked_strides_idx_0_slice_{s0}-{s1}': plot_marked_strides(datasets=[d], agent_idx=0,
                                                                             slice=[s0, s1], **ccc) for (s0, s1) in
                [(10, 50), (60, 100)]}
        # dic1 = plot_marked_strides(dataset=d, agent_ids=d.agent_ids[:3], title=' ', slices=[[10, 50], [60, 100]])
        fig_dict.update(dic1)
        dic2 = plot_marked_turns(dataset=d, agent_ids=d.agent_ids[:3], min_turn_angle=20, **ccc)
        fig_dict.update(dic2)

    elif exp_type in ['chemotaxis_approach', 'chemotaxis_local', 'chemotaxis_diffusion']:
        if exp_type in ['chemotaxis_local', 'chemotaxis_diffusion']:
            ps = ['o_cent', 'sd_cent', 'd_cent']
            source = (0.0, 0.0)
        elif exp_type in ['chemotaxis_approach']:
            ps = ['o_chem', 'sd_chem', 'd_chem']
            source = (0.04, 0.0)
        for p in ps:
            fig_dict[p] = plot_timeplot([p], datasets=[d], show_first=True, **ccc)
        for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']:
            fig_dict[p] = plot_timeplot([p], datasets=[d], **ccc)
        for chunk in ['turn', 'stride', 'pause']:
            for dur in [0.0, 0.5, 1.0]:
                try:
                    fig_dict[f'{chunk}_bearing2source_min_{dur}_sec'] = plot_chunk_Dorient2source(datasets=[d],
                                                                                                  chunk=chunk,
                                                                                                  source=source,
                                                                                                  min_dur=dur, **ccc)
                except:
                    pass
        vis_kwargs = dtypes.get_dict('visualization', mode='image', image_mode='final', show_display=False,
                                     random_colors=True, trajectories=True,
                                     visible_clock=False, visible_scale=False, media_name='single_trajectory')
        d.visualize(agent_ids=[d.agent_ids[0]], vis_kwargs=vis_kwargs)

    if 'odor_pref' in exp_type:
        ind = d.compute_preference_index()
        print(f'Preference for left odor : {np.round(ind, 3)}')
        results['PI'] = ind

    if exp_type in ['odor_pref_RL', 'chemotaxis_RL']:
        c = {
            'datasets': [d],
            'show_first': False,
            'table': 'best_gains',
            **ccc
        }



        g_keys = ['g_odor1'] if exp_type == 'chemotaxis_RL' else ['g_odor1', 'g_odor2']
        fig_dict['best_gains_table'] = plot_timeplot(g_keys, save_as='best_gains.pdf', **c)
        fig_dict['olfactor_decay_table'] = plot_timeplot(['D_olf'], save_as='olfactor_decay.pdf', **c)
        fig_dict['olfactor_decay_table_inds'] = plot_timeplot(['D_olf'], save_as='olfactor_decay_inds.pdf',
                                                              individuals=True, **c)
        fig_dict['reward_table'] = plot_timeplot(['cum_reward'], save_as='reward.pdf', **c)
    elif exp_type == 'realistic_imitation':
        d.save_agent(pars=fun.flatten_list(d.points_xy) + fun.flatten_list(d.contour_xy), header=True)
    if exp_type == 'dish':
        targeted_analysis([d])
        fig_dict = {f'stride_track_idx_0_in_{s0}-{s1}': plot_marked_strides(datasets=[d], agent_idx=0,
                                                                            slice=[s0, s1], **ccc) for (s0, s1) in
                    [(0, 60)]}

    print(f'    Analysis complete!')
    return fig_dict, results
