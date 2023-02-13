

from larvaworld.lib import reg, aux


def entry(ID, name=None, **kwargs):
    return reg.graphs.entry(ID, name=name, **kwargs)

analysis_dict = aux.AttrDict({
    'tactile': [
        entry('endpoint pars (hist)','time ratio on food (final)',ks=['on_food_tr']),
        entry('timeplot', 'time ratio on food',ks=['on_food_tr'],  unit='min'),
        entry('timeplot', 'time on food',ks=['cum_f_det'],  unit='min'),
        entry('timeplot', 'turner input',ks=['A_tur'],  unit='min', show_first=True),
        entry('timeplot', 'turner output',ks=['Act_tur'],  unit='min', show_first=True),
        entry('timeplot', 'tactile activation',ks=['A_touch'],  unit='min', show_first=True),
        entry('ethogram'),
    ],
    'chemo': [
        # autotime(['sv', 'fov', 'b', 'a']),
        entry('autoplot', ks=['c_odor1', 'dc_odor1', 'A_olf', 'A_T', 'I_T']),
        entry('trajectories'),
        # entry('turn amplitude'),
        # entry('angular pars', Npars=5),

    ],
    'intake': [
        # 'deb_analysis',
        # *[time(p) for p in ['sf_faeces_M', 'f_faeces_M', 'sf_abs_M', 'f_abs_M', 'f_am']],
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('food intake (timeplot)', 'food intake (filtered)', filt_amount=True),
        entry('pathlength', scaled=False),
entry('barplot', name='food intake (barplot)', ks=['f_am']),
        entry('ethogram')

    ],
    'anemotaxis': [

*[entry('nengo', name=p, group=p, same_plot=True if p == 'anemotaxis' else False)for p in
          ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V',
           'wind_effect_on_Fr']],
        *[entry('timeplot', ks=[p]) for p in ['A_wind', 'anemotaxis']],
        # *[scat(p) for p in [['o_wind', 'A_wind'], ['anemotaxis', 'o_wind']]],
entry('endpoint pars (hist)', name='final anemotaxis', ks=['anemotaxis'])

    ],
    'thermo': [
        entry('trajectories'),
entry('autoplot', ks=['temp_W', 'dtemp_W', 'temp_C', 'dtemp_C', 'A_therm'], show_first=True, individuals=False)
    ],
    'puff': [

        # entry('trajectories'),
        # entry('ethogram', add_samples=False),
        entry('pathlength', scaled=False),
        *[entry('timeplot', ks=[p], absolute=True) for p in ['fov', 'foa']],
        # *[time(p, abs=True) for p in ['fov', 'foa','b', 'bv', 'ba']],
        *[entry('timeplot', ks=[p]) for p in ['sv', 'sa']],
        # *[time(p) for p in ['sv', 'sa', 'v', 'a']],
    ],
    'RL': [
        entry('timeplot', 'olfactor_decay_table', ks=['D_olf'], table='best_gains'),
        entry('timeplot', 'olfactor_decay_table_inds',ks=['D_olf'],  table='best_gains',
             individuals=True),
        entry('timeplot', 'reward_table', ks=['cum_reward'], table='best_gains'),
        entry('timeplot', 'best_gains_table',ks=['g_odor1'], table='best_gains'),
        entry('timeplot', 'best_gains_table_x2',ks=['g_odor1', 'g_odor2'],  table='best_gains'),
    ],
    'patch': [entry('timeplot', 'Y position', ks=['y'], legend_loc='lower left'),
              entry('navigation index'),
              entry('turn amplitude'),
              entry('turn duration'),
              entry('turn amplitude VS Y pos', 'turn angle VS Y pos (scatter)', mode='scatter'),
              entry('turn amplitude VS Y pos', 'turn angle VS Y pos (hist)', mode='hist'),
              entry('turn amplitude VS Y pos', 'bearing correction VS Y pos', mode='hist', ref_angle=270),
              ],
    'survival': [
        # 'foraging_list',
        entry('timeplot', 'time ratio on food', ks=['on_food_tr'], unit='min'),
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('pathlength', scaled=False)

    ],
    'deb': [
        *[entry('deb',  name = f'DEB.{m} (hours)',sim_only = False, mode=m, save_as=f"{m}_in_hours.pdf") for m in ['energy', 'growth', 'full']],
        *[entry('deb',  name = f'FEED.{m} (hours)',sim_only = True, mode=m, save_as=f"{m}_in_hours.pdf") for m in
          ['feeding', 'reserve_density', 'assimilation', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
           'food_mass_2', 'hunger', 'EEB', 'fs']],
    ],
    'endpoint': [

entry('boxplot (simple)', ks=['l', 'str_N', 'dsp_0_40_max', 'run_tr', 'fv', 'ffov', 'v_mu', 'sv_mu', 'tor5_mu', 'tor5_std',
                'tor20_mu', 'tor20_std']),
entry('boxplot (simple)', ks=['l', 'fv', 'v_mu', 'run_tr']),
        entry('crawl pars')
    ],
    'distro': [
        entry('distros', mode='box'),
        entry('distros', mode='hist'),
        entry('angular pars', Npars=5)
    ],

    'dsp': [

        entry('dispersal', range=(0, 40)),
        # entry('dispersal', range=(0, 60)),
        entry('dispersal summary', range=(0, 40)),
        # entry('dispersal summary', range=(0, 60)),
    ],
    'general': [
        entry('ethogram', add_samples=False),
        entry('pathlength', scaled=False),
        # entry('navigation index'),
        entry('epochs', stridechain_duration=True),

    ],
    'stride': [
        entry('stride cycle'),
        entry('stride cycle', individuals=True),
    ],
    'traj': [
        entry('trajectories', mode='default', unit='mm'),
        entry('trajectories', name='aligned2origin', mode='origin', unit='mm', single_color=True),
    ],
    'track': [
        entry('stride track'),
        entry('turn track'),
    ]
})

