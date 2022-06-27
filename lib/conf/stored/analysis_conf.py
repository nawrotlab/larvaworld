import lib.aux.dictsNlists as dNl
import lib.aux.naming as nam
from lib.plot.dict import graph_dict
from lib.registry.pars import preg


def entry(ID, title=None, **kwargs):
    return graph_dict.entry(ID, title=title, **kwargs)


def time(short=None, par=None, title=None, u='sec', f1=False, abs=False, **kwargs):
    if title is None:
        title = par if par is not None else preg.getPar(short)
        # name =f'{short} timeplot'
    args = {
        'par_shorts': [short] if short is not None else [],
        'pars': [par] if par is not None else [],
        # 'show_first': f1,
        'unit': u,
        'absolute': abs,
        **kwargs
    }
    return entry('timeplot', title=title, **args)


def autotime(ks, title=None, u='sec', f1=True, ind=True, **kwargs):
    if title is None:
        title = f'autoplot_x{len(ks)}'
        # name =f'{short} timeplot'
    args = {
        'ks': ks,
        # 'show_first': f1,
        'unit': u,
        'show_first': f1,
        'individuals': ind,
        **kwargs
    }
    return entry('autoplot', title=title, **args)


def end(shorts=None, title=None, **kwargs):
    if title is None:
        title = f'endpoint plot'
    args = {
        'par_shorts': shorts,
        # 'show_first': f1,
        # 'unit': u,
        **kwargs
    }
    return entry('endpoint pars (hist)', title=title, **args)


def bar(short, title=None, **kwargs):
    if title is None:
        title = preg.getPar(short)
        # name =f'{short} timeplot'
    args = {
        'par_shorts': [short],
        **kwargs
    }
    return entry('barplot', title=title, **args)


def nengo(group, title=None, **kwargs):
    if title is None:
        title = group
    args = {
        'group': group,
        **kwargs
    }
    return entry('nengo', title=title, **args)


def deb(mode, title=None, u='hours', pref='FEED', **kwargs):
    if title is None:
        if pref == 'FEED':
            title = f'FEED.{mode} ({u})'
            sim_only = True
        elif pref == 'DEB':
            title = f'DEB.{mode} vs model'
            sim_only = False
    args = {
        'mode': mode,
        'save_as': f'{mode}_in_{u}.pdf',
        'time_unit': u,
        'sim_only': sim_only,
        **kwargs
    }
    return entry('deb', title=title, **args)


def source_anal_list(sources, **kwargs):
    d0 = []
    for n, pos in sources.items():
        for ref_angle, title in zip([None, 270], [f'bearing to {n}', 'bearing to 270deg']):
            d0.append(entry('bearing/turn', title, min_angle=5.0, ref_angle=ref_angle, source_ID=n, **kwargs))

        d0 += [time(par=p, **kwargs) for p in [nam.bearing2(n), nam.dst2(n), nam.scal(nam.dst2(n))]],

        for chunk in ['stride', 'pause', 'Lturn', 'Rturn']:
            for dur in [0.0, 0.5, 1.0]:
                title = f'{chunk}_bearing2_{n}_min_{dur}_sec'
                d0.append(entry('bearing to source/epoch', title, min_dur=dur, chunk=chunk, source_ID=n, **kwargs))
    return d0


analysis_dict = dNl.NestDict({
    'tactile': [
        end(['on_food_tr'], 'time ratio on food (final)'),
        time('on_food_tr', 'time ratio on food', u='min'),
        time('cum_f_det', 'time on food', u='min'),
        time('A_tur', 'turner input', u='min', f1=True),
        time('Act_tur', 'turner output', u='min', f1=True),
        time('A_touch', 'tactile activation', u='min', f1=True),
        entry('ethogram'),
    ],
    'chemo': [

        autotime(['sv', 'fov', 'b', 'x', 'a']),
        entry('trajectories'),
        # *[time(p) for p in ['c_odor1']],
        *[time(p) for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']],
        # 'source_anal_list',
        entry('turn amplitude'),
        entry('angular pars', Npars=5),
        entry('ethogram'),

    ],
    'intake': [
        # 'deb_analysis',
        # *[time(p) for p in ['sf_faeces_M', 'f_faeces_M', 'sf_abs_M', 'f_abs_M', 'f_am']],
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('food intake (timeplot)', 'food intake (filtered)', filt_amount=True),
        entry('pathlength', scaled=False),
        bar('f_am', 'food intake (barplot)'),
        entry('ethogram')

    ],
    'anemotaxis': [
        *[nengo(p, same_plot=True if p == 'anemotaxis' else False) for p in
          ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V',
           'wind_effect_on_Fr']],
        *[time(p) for p in ['A_wind', 'anemotaxis']],
        # *[scat(p) for p in [['o_wind', 'A_wind'], ['anemotaxis', 'o_wind']]],
        end(['anemotaxis'], 'final anemotaxis')

    ],
    'puff': [

        # *[nengo(p, same_plot=True if p == 'anemotaxis' else False) for p in
        #           ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V',
        #            'wind_effect_on_Fr']],
        # entry('ethogram', add_samples=True),
        entry('ethogram', add_samples=False),
        # *[time(p) for p in ['A_wind', 'anemotaxis', 'o_wind']],
        *[time(p, abs=True) for p in ['fov', 'foa']],
        # *[time(p, abs=True) for p in ['fov', 'foa','b', 'bv', 'ba']],
        *[time(p) for p in ['sv', 'sa']],
        # *[time(p) for p in ['sv', 'sa', 'v', 'a']],
    ],
    'RL': [
        time('D_olf', 'olfactor_decay_table', save_as='olfactor_decay.pdf', table='best_gains'),
        time('D_olf', 'olfactor_decay_table_inds', save_as='olfactor_decay_inds.pdf', table='best_gains',
             individuals=True),
        time('cum_reward', 'reward_table', save_as='reward.pdf', table='best_gains'),
        time('g_odor1', 'best_gains_table', save_as='best_gains.pdf', table='best_gains'),
        time(*['g_odor1', 'g_odor2'], 'best_gains_table_x2', save_as='best_gains_x2.pdf', table='best_gains'),
    ],
    'patch': [time('y', 'Y position', legend_loc='lower left'),
              entry('navigation index'),
              entry('turn amplitude'),
              entry('turn duration'),
              entry('turn amplitude VS Y pos', 'turn angle VS Y pos (scatter)', mode='scatter'),
              entry('turn amplitude VS Y pos', 'turn angle VS Y pos (hist)', mode='hist'),
              entry('turn amplitude VS Y pos', 'bearing correction VS Y pos', mode='hist', ref_angle=270),
              ],
    'survival': [
        # 'foraging_list',
        time('on_food_tr', 'time ratio on food', u='min'),
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('pathlength', scaled=False)

    ],
    'deb': [
        *[deb(m, pref='DEB') for m in ['energy', 'growth', 'full']],
        *[deb(m, pref='FEED') for m in
          ['feeding', 'reserve_density', 'assimilation', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
           'food_mass_2', 'hunger', 'EEB', 'fs']],
    ]
})
