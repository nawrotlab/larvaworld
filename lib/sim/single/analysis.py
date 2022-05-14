import itertools
import warnings
from typing import List

import numpy as np

from lib.aux.combining import combine_pdfs
from lib.aux.dictsNlists import flatten_list, unique_list
from lib.anal.comparing import ExpFitter
from lib.anal.plotting import plot_turn_Dbearing, plot_turn_amp, plot_turns, timeplot, \
    plot_navigation_index, plot_debs, plot_food_amount, plot_pathlength, plot_endpoint_params, barplot, \
    plot_chunk_Dorient2source, plot_marked_strides, lineplot, plot_stridesNpauses, \
    plot_interference, plot_dispersion, plot_stride_Dbend, plot_stride_Dorient, plot_ang_pars, calibration_plot, \
    plot_crawl_pars, boxplot, boxplot_double_patch, plot_2pars, plot_nengo_network, plot_marked_turns, \
    plot_sample_tracks
from lib.conf.stored.conf import loadConf
from lib.conf.base.dtypes import null_dict
from lib.conf.base.par import getPar
from lib.model.DEB.deb import deb_default
from lib.conf.base import paths
from lib.stor.larva_dataset import LarvaDataset
import lib.aux.naming as nam


def sim_analysis(ds: List[LarvaDataset], exp_type, show=False, delete_datasets=False):
    if ds is None:
        return
    # if not type(ds) == list:
    #     ds = [ds]
    d = ds[0]
    ccc = {'show': show,
           'save_to': d.config['parent_plot_dir']}
    cc = {'datasets': ds,
          'subfolder': None,
          **ccc}
    figs = {}
    results = {}

    if 'tactile' in exp_type:
        tact_kws={'unit' : 'min',**cc}
        figs['time ratio on food (final)'] = plot_endpoint_params(par_shorts=['on_food_tr'], **cc)
        figs['time ratio on food'] = timeplot(['on_food_tr'], **tact_kws)
        figs['time on food'] = timeplot(['cum_f_det'], **tact_kws)
        figs['turner input'] = timeplot(['A_tur'], show_first=True, **tact_kws)
        figs['turner output'] = timeplot(['Act_tur'], show_first=True, **tact_kws)
        figs['tactile activation'] = timeplot(['A_touch'], show_first=True, **tact_kws)

    if 'RvsS' in exp_type:
        figs.update(**intake_analysis(**cc))

    if exp_type in ['growth', 'RvsS']:
        figs.update(**deb_analysis(**ccc))

    if 'RL' in exp_type:
        c = {
            'show_first': False,
            'table': 'best_gains',
            **cc
        }

        g_keys = ['g_odor1'] if exp_type == 'chemotaxis_RL' else ['g_odor1', 'g_odor2']
        figs['best_gains_table'] = timeplot(g_keys, save_as='best_gains.pdf', **c)
        figs['olfactor_decay_table'] = timeplot(['D_olf'], save_as='olfactor_decay.pdf', **c)
        figs['olfactor_decay_table_inds'] = timeplot(['D_olf'], save_as='olfactor_decay_inds.pdf',
                                                     individuals=True, **c)
        figs['reward_table'] = timeplot(['cum_reward'], save_as='reward.pdf', **c)
    elif exp_type == 'realistic_imitation':
        d.save_agents(pars=flatten_list(d.points_xy) + flatten_list(d.contour_xy))
    if exp_type == 'dish':
        # targeted_analysis(ds)
        figs = {f'stride_track_idx_0_in_{s0}-{s1}': plot_marked_strides(agent_idx=0,
                                                                        slice=[s0, s1], **cc) for (s0, s1) in
                [(0, 60)]}
    if exp_type == 'imitation':
        f = ExpFitter(d.config['sample'])
        results['sample_fit'] = f.compare(d, save_to_config=True)
        print(results['sample_fit'])

    if exp_type in ['food_at_bottom']:
        figs.update(**foraging_analysis(d.config['sources'], **cc))

    if 'anemo' in exp_type:
        for group in ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V', 'wind_effect_on_Fr'] :
            figs[group] = plot_nengo_network(group,same_plot=True if group=='anemotaxis' else False, **cc)
        figs['anemotaxis'] = timeplot(['anemotaxis'], show_first=False, **cc)
        figs['final anemotaxis'] = plot_endpoint_params(par_shorts=['anemotaxis'], **cc)

        figs['wind activation VS bearing to wind'] = plot_2pars(['o_wind','A_wind'], **cc)
        figs['wind activation'] = timeplot(['A_wind'], show_first=False, **cc)
        figs['anemotaxis VS bearing to wind'] = plot_2pars(['anemotaxis','o_wind'], **cc)
        figs['bearing to wind direction'] = timeplot(['o_wind'], show_first=False, **cc)





    if 'dispersion' in exp_type:
        samples = unique_list([d.config['sample'] for d in ds])
        targets = [LarvaDataset(loadConf(sd, 'Ref')['dir']) for sd in samples]
        figs0 = comparative_analysis(datasets=ds + targets, **ccc)
        figs.update(figs0)

    if 'chemo' in exp_type:
        for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']:
            figs[p] = timeplot([p], **cc)
        figs['turns'] = plot_turns(**cc)
        figs['ang_pars'] = plot_ang_pars(Npars=5,**cc)
        figs.update(**source_analysis(d.config['source_xy'], **cc))

        vis_kwargs = null_dict('visualization', mode='image', image_mode='final', show_display=False,
                               random_colors=True, trails=True,
                               visible_clock=False, visible_scale=False, media_name='single_trajectory')
        d.visualize(agent_ids=[d.agent_ids[0]], vis_kwargs=vis_kwargs)
    if delete_datasets:
        for d in ds:
            d.delete()
    print(f'    Analysis complete!')
    return figs, results


def intake_analysis(**kwargs):
    kwargs0 = {'show_first': False, 'legend_loc': 'upper left', **kwargs}
    figs = {}
    figs['faeces ratio'] = timeplot(['sf_faeces_M'], **kwargs0)
    figs['faeces amount'] = timeplot(['f_faeces_M'], **kwargs0)
    figs['food absorption efficiency'] = timeplot(['sf_abs_M'], **kwargs0)
    figs['food absorbed'] = timeplot(['f_abs_M'], **kwargs0)
    figs['food intake (timeplot)'] = timeplot(['f_am'], **kwargs0)

    figs['food intake'] = plot_food_amount(**kwargs)
    figs['food intake (filt)'] = plot_food_amount(filt_amount=True, **kwargs)
    # figs['gut occupancy'] = plot_gut(**kwargs)
    figs['pathlength'] = plot_pathlength(scaled=False, **kwargs)
    # figs['endpoint'] = plot_endpoint_params(mode='deb', **kwargs)
    try:
        figs['food intake (barplot)'] = barplot(par_shorts=['f_am'], **kwargs)
    except:
        pass
    return figs


def source_analysis(source_xy, **kwargs):
    figs = {}
    for n, pos in source_xy.items():
        for p in [nam.bearing2(n), nam.dst2(n), nam.scal(nam.dst2(n))]:
            figs[p] = timeplot(pars=[p], **kwargs)

        for chunk in ['turn', 'stride', 'pause']:
            for dur in [0.0, 0.5, 1.0]:
                try:
                    figs[f'{chunk}_bearing2_{n}_min_{dur}_sec'] = plot_chunk_Dorient2source(chunk=chunk,
                                                                                            source_ID=n,
                                                                                            min_dur=dur, **kwargs)
                except:
                    pass
    return figs


def foraging_analysis(sources, **kwargs):
    figs = {}
    for n, pos in sources.items():
        figs[f'bearing to {n}'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=None, source_ID=n, **kwargs)
        figs['bearing to 270deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=270, source_ID=n, **kwargs)
    figs['bearing correction VS Y pos'] = plot_turn_amp(par_short='tur_y0', mode='hist', ref_angle=270, **kwargs)
    figs['turn angle VS Y pos (hist)'] = plot_turn_amp(par_short='tur_y0', mode='hist', **kwargs)
    figs['turn angle VS Y pos (scatter)'] = plot_turn_amp(par_short='tur_y0', mode='scatter', **kwargs)
    figs['turn duration'] = plot_turn_amp(par_short='tur_t', mode='scatter', absolute=True, **kwargs)
    # figs['turn amplitude'] = TurnPlot(**kwargs).get()
    figs['turn amplitude'] = plot_turns(**kwargs)
    figs['Y position'] = timeplot(['y'], legend_loc='lower left', **kwargs)
    figs['navigation index'] = plot_navigation_index(**kwargs)

    return figs


def double_patch_analysis(sources, **kwargs):
    figs = {}
    figs['time_inside_patches'] = boxplot(par_short='on_food_tr',
                                          xlabel='substrate', **kwargs)
    # figs['bearing correction VS Y pos'] = plot_turn_amp(par_short='tur_y0', mode='hist', ref_angle=270, **kwargs)
    # figs['turn angle VS Y pos (hist)'] = plot_turn_amp(par_short='tur_y0', mode='hist', **kwargs)
    # figs['turn angle VS Y pos (scatter)'] = plot_turn_amp(par_short='tur_y0', mode='scatter', **kwargs)
    # figs['turn duration'] = plot_turn_amp(par_short='tur_t', mode='scatter', absolute=True, **kwargs)
    # # figs['turn amplitude'] = TurnPlot(**kwargs).get()
    # figs['turn amplitude'] = plot_turns(**kwargs)
    # figs['Y position'] = timeplot(['y'], legend_loc='lower left', **kwargs)
    # figs['navigation index'] = plot_navigation_index(**kwargs)
    # for n, pos in sources.items():
    #     figs[f'bearing to {n}'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=None,source_ID=n, **kwargs)
    #     figs['bearing to 270deg'] = plot_turn_Dbearing(min_angle=5.0, ref_angle=270, source_ID=n, **kwargs)
    return figs


def essay_analysis(essay_type, exp, ds0, all_figs=False, path=None):
    if path is None:
        parent_dir = f'essays/{essay_type}/global_test'
        plot_dir = f'{paths.path("SIM")}/{parent_dir}/plots'
    else:
        plot_dir = f'{path}/plots'
    ccc = {'show': False}
    if len(ds0) == 0 or any([d0 is None for d0 in ds0]):
        return {}, {}
    figs = {}
    results = {}

    if essay_type in ['roversVSsitters', 'RvsS']:
        RS_leg_cols = ['black', 'white']
        markers = ['D', 's']
        ls = [r'$for^{R}$', r'$for^{S}$']
        shorts = ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']

        def dsNls(ds0, lls=None):
            if lls is None:
                lls = flatten_list([ls] * len(ds0))
            dds = flatten_list(ds0)
            deb_dicts = [d.load_dicts('deb') for d in dds]
            # for d in ds0:
            #     ds, debs = split_rovers_sitters(d)
            #     dds += ds
            #     deb_dicts += debs

            return {'datasets': dds,
                    'labels': lls,
                    'deb_dicts': deb_dicts,
                    'save_to': plot_dir,
                    'leg_cols': RS_leg_cols,
                    'markers': markers,
                    **ccc
                    }

        if exp == 'pathlength':
            lls = flatten_list([[rf'{s} $for^{"R"}$', rf'{s} $for^{"S"}$'] for s in ['Agar', 'Yeast']])
            kwargs = {
                **dsNls(ds0, lls),
                'xlabel': r'time on substrate_type $(min)$',
            }
            figs['1_pathlength'] = plot_pathlength(scaled=False, save_as=f'1_PATHLENGTH.pdf', unit='cm', **kwargs)

        elif exp == 'intake':
            kwargs = {**dsNls(ds0),
                      'coupled_labels': [10, 15, 20],
                      'xlabel': r'Time spent on food $(min)$'}
            figs['2_intake'] = barplot(par_shorts=['sf_am_V'], save_as=f'2_AD_LIBITUM_INTAKE.pdf', **kwargs)
            if all_figs:
                for s in shorts:
                    p = getPar(s)
                    figs[f'intake {p}'] = barplot(par_shorts=[s], save_as=f'2_AD_LIBITUM_{p}.pdf', **kwargs)

        elif exp == 'starvation':
            hs = [0, 1, 2, 3, 4]
            kwargs = {**dsNls(ds0),
                      'coupled_labels': hs,
                      'xlabel': r'Food deprivation $(h)$'}
            figs['3_starvation'] = lineplot(par_shorts=['f_am_V'], save_as='3_POST-STARVATION_INTAKE.pdf',
                                            ylabel='Food intake', scale=1000, **kwargs)
            if all_figs:
                for ii in ['feeding']:
                    figs[ii] = plot_debs(mode=ii, save_as=f'3_POST-STARVATION_{ii}.pdf', include_egg=False,
                                         label_epochs=False, **kwargs)
                for s in shorts:
                    p = getPar(s)
                    figs[f'post-starvation {p}'] = lineplot(par_shorts=[s], save_as=f'3_POST-STARVATION_{p}.pdf',
                                                            **kwargs)

        elif exp == 'quality':
            qs = [1.0, 0.75, 0.5, 0.25, 0.15]
            qs_labels = [int(q * 100) for q in qs]
            kwargs = {**dsNls(ds0),
                      'coupled_labels': qs_labels,
                      'xlabel': 'Food quality (%)'
                      }
            figs['4_quality'] = barplot(par_shorts=['sf_am_V'], save_as='4_REARING-DEPENDENT_INTAKE.pdf', **kwargs)
            if all_figs:
                for s in shorts:
                    p = getPar(s)
                    figs[f'rearing-quality {p}'] = barplot(par_shorts=[s], save_as=f'4_REARING_{p}.pdf', **kwargs)

        elif exp == 'refeeding':
            h = 3
            n = f'5_REFEEDING_after_{h}h_starvation_'
            kwargs = dsNls(ds0)
            figs['5_refeeding'] = plot_food_amount(scaled=True, filt_amount=True, save_as='5_REFEEDING_INTAKE.pdf',
                                                   **kwargs)

            if all_figs:
                figs[f'refeeding food-intake'] = plot_food_amount(scaled=True, save_as=f'{n}scaled_intake.pdf',
                                                                  **kwargs)
                figs[f'refeeding food-intake(filt)'] = plot_food_amount(scaled=True, filt_amount=True,
                                                                        save_as=f'{n}scaled_intake_filt.pdf', **kwargs)
                for s in shorts:
                    p = getPar(s)
                    figs[f'refeeding {p}'] = timeplot(par_shorts=[s], show_first=False, subfolder=None,
                                                      save_as=f'{n}{p}.pdf', **kwargs)
        # for d in kwargs['datasets'] :
        #     d.delete()

    if essay_type in ['double_patch']:
        if exp == 'double_patch':
            kwargs = {'datasets': flatten_list(ds0),
                      'save_to' : plot_dir,
                      'save_as' : 'double_patch.pdf',
                      # 'pair_ids': ['sucrose', 'standard', 'cornmeal'],
                      # 'common_ids': ['Rover', 'Sitter'],
                      # 'xlabel': 'substrate',
                      'show': True,
                      # 'complex_colors' : True,
                      # 'pair_colors': dict(zip(['sucrose', 'standard', 'cornmeal'], ['green', 'orange', 'magenta'])),
                      # 'common_color_prefs': dict(zip(['Rover', 'Sitter'], ['dark', 'light'])),
                      }
            figs['double_patch'] = boxplot_double_patch(**kwargs)

    print(f'    Analysis complete!')
    return figs, results


def comparative_analysis(datasets, labels=None, simVSexp=False, save_to=None, **kwargs):
    figs = {}
    warnings.filterwarnings('ignore')
    if save_to is None:
        save_to = datasets[0].dir_dict.comp_plot
    if labels is None:
        labels = [d.id for d in datasets]
    cc = {'datasets': datasets,
          'labels': labels,
          'save_to': save_to}
    figs['stridesNpauses'] = plot_stridesNpauses(**cc, plot_fits='best',**kwargs)
    for m in ['minimal', 'tiny']:
        figs[f'endpoint_{m}'] = plot_endpoint_params(**cc, mode=m, **kwargs)
    for m in ['orientation', 'bend', 'spinelength']:
        for agent_idx in [None, 0, 1]:
            i = '' if agent_idx is None else f'_{agent_idx}'
            try:
                figs[f'interference_{m}{i}'] = plot_interference(**cc, mode=m, agent_idx=agent_idx, **kwargs)
            except:
                pass
    for scaled in [True, False]:
        for fig_cols in [1, 2]:
            for r0, r1 in itertools.product([0, 20], [40, 80, 120, 160, 200]):
                s = 'scaled_' if scaled else ''
                l = f'{s}dispersion_{r0}->{r1}_{fig_cols}'
                try:
                    figs[l] = plot_dispersion(**cc, scaled=scaled, fig_cols=fig_cols, range=(r0, r1), **kwargs)
                except:
                    pass

    try:
        figs['stride_Dbend'] = plot_stride_Dbend(**cc, show_text=False, **kwargs)
    except:
        pass
    try:
        figs['stride_Dorient'] = plot_stride_Dorient(**cc, absolute=True, **kwargs)
    except:
        pass
    try:
        figs['ang_pars'] = plot_ang_pars(**cc, absolute=True, include_turns=False, Npars=3,**kwargs)
    except:
        pass
    try:
        figs['calibration'] = calibration_plot(save_to=save_to, **kwargs)
    except:
        pass
    figs['crawl_pars'] = plot_crawl_pars(**cc, **kwargs)
    figs['turns'] = plot_turns(**cc, **kwargs)
    figs['turn_duration'] = plot_turn_amp(**cc, **kwargs)
    combine_pdfs(file_dir=save_to)
    return figs


def targeted_analysis(datasets, labels=None, save_to=None, pref='', show=False, **kwargs):
    # with fun.suppress_stdout():
    if save_to is None:
        save_to = datasets[0].dir_dict.comp_plot
    if labels is None:
        labels = [d.id for d in datasets]
    anal_kws = {'datasets': datasets,
                'labels': labels,
                'save_to': save_to,
                'subfolder': None,
                'show': show,
                **kwargs}


    for k in ['best'] :
        plot_stridesNpauses(**anal_kws, plot_fits=k, save_as=f'bouts{pref}.pdf', save_fits_as=f'bout_fits{pref}.csv')
    raise
    plot_endpoint_params(**anal_kws, mode='tiny')
    plot_sample_tracks(**anal_kws, slice=[0, 60])
    plot_ang_pars(**anal_kws, Npars=3, save_as=f'ang_pars{pref}.pdf', save_fits_as=f'ang_pars_ttest{pref}.csv')
    plot_marked_turns(dataset=datasets[0], slices=[(0, 180)], **kwargs)
    plot_marked_strides(**anal_kws, agent_idx=1, slice=[0, 180], save_as=f'sample_tracks{pref}.pdf')


    plot_endpoint_params(**anal_kws, mode='stride_def', save_as=f'stride_pars{pref}.pdf',
                         save_fits_as=f'stride_pars_ttest{pref}.csv')
    plot_turns(**anal_kws)
    plot_interference(**anal_kws, mode='orientation', save_as=f'interference{pref}.pdf')
    plot_crawl_pars(**anal_kws, save_as=f'crawl_pars{pref}.pdf', save_fits_as=f'crawl_pars_ttest{pref}.csv')

    plot_endpoint_params(**anal_kws, mode='result', save_as=f'results{pref}.pdf')
    plot_endpoint_params(**anal_kws, mode='reorientation', save_as=f'reorientation{pref}.pdf')
    plot_endpoint_params(**anal_kws, mode='tortuosity', save_as=f'tortuosity{pref}.pdf')
    plot_dispersion(**anal_kws, scaled=True, fig_cols=2, range=(0, 80), ymax=18, save_as=f'dispersion{pref}.pdf')


def deb_analysis(datasets,**kwargs) :
    figs={}
    deb_model = deb_default(**datasets[0].config['life_history'])
    deb_dicts = flatten_list([d.load_dicts('deb') for d in datasets])
    kws = {'roversVSsitters': True,
         'datasets':datasets,
         **kwargs}

    for m in ['energy', 'growth', 'full']:
        save_as = f'{m}_vs_model.pdf'
        figs[f'DEB.{m} vs model'] = plot_debs(deb_dicts=deb_dicts+ [deb_model], save_as=save_as, mode=m, **kws)
    for m in ['feeding', 'reserve_density', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
              'food_mass_2', 'hunger', 'EEB','fs']:
        for t in ['hours']:
            try :
                save_as = f'{m}_in_{t}.pdf'
                figs[f'FEED.{m} ({t})'] = plot_debs(deb_dicts=deb_dicts,sim_only=True,save_as=save_as, mode=m, time_unit=t, **kws)
            except :
                pass

    return figs