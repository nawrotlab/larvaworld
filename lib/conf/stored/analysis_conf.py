import numpy as np
from lib.conf.base.dtypes import null_dict
from lib.conf.base.par import getPar


# tact_kws={'unit' : 'min',**cc}
#         figs['time ratio on food (final)'] = plot_endpoint_params(par_shorts=['on_food_tr'], **cc)
#         figs['time ratio on food'] = timeplot(['on_food_tr'], **tact_kws)
#         figs['time on food'] = timeplot(['cum_f_det'], **tact_kws)
#         figs['turner input'] = timeplot(['A_tur'], show_first=True, **tact_kws)
#         figs['turner output'] = timeplot(['Act_tur'], show_first=True, **tact_kws)
#         figs['tactile activation'] = timeplot(['A_touch'], show_first=True, **tact_kws)

def entry(plotID, title=None, **kwargs):
    if title is None:
        title = plotID
    return {'title': title, 'plotID': plotID, 'args': kwargs}


def time(short, title=None, u='sec', f1=False, **kwargs):
    if title is None:
        title = getPar(short, to_return=['d'])[0]
        # name =f'{short} timeplot'
    args = {
        'par_shorts': [short],
        'show_first': f1,
        'unit': u,
        **kwargs
    }
    return {'title': title, 'plotID': 'timeplot', 'args': args}


def end(shorts=None, title=None, **kwargs):
    if title is None:
        title = f'endpoint plot'
        # title =getPar(short, to_return=['d'])[0]
    args = {
        'par_shorts': shorts,
        # 'show_first': f1,
        # 'unit': u,
        **kwargs
    }
    return {'title': title, 'plotID': 'endpoint params', 'args': args}

    if 'chemo' in exp_type:
        for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']:
            figs[p] = timeplot([p], **cc)
        figs['turns'] = plot_turns(**cc)
        figs['ang_pars'] = plot_ang_pars(Npars=5, **cc)
        figs.update(**source_analysis(d.config['sources'], **cc))


def bar(short, title=None, **kwargs):
    if title is None:
        title = getPar(short, to_return=['d'])[0]
        # name =f'{short} timeplot'
    args = {
        'par_shorts': [short],
        **kwargs
    }
    return {'title': title, 'plotID': 'barplot', 'args': args}


def scat(shorts, title=None, **kwargs):
    if title is None:
        d1, d2 = getPar(shorts, to_return=['d'])[0]
        title = f'{d1} VS {d2}'
    args = {
        'shorts': shorts,
        **kwargs
    }
    return {'title': title, 'plotID': 'scatter', 'args': args}


def nengo(group, title=None, **kwargs):
    if title is None:
        title = group
    args = {
        'group': group,
        **kwargs
    }
    return {'title': title, 'plotID': 'nengo', 'args': args}


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
    return {'title': title, 'plotID': 'deb', 'args': args}


analysis_dict = {
    'tactile': [
        end(['on_food_tr'], 'time ratio on food (final)'),
        time('on_food_tr', 'time ratio on food', u='min'),
        time('cum_f_det', 'time on food', u='min'),
        time('A_tur', 'turner input', u='min', f1=True),
        time('Act_tur', 'turner output', u='min', f1=True),
        time('A_touch', 'tactile activation', u='min', f1=True)
    ],
    'chemo': [
        *[time(p) for p in ['c_odor1', 'dc_odor1', 'A_olf', 'A_tur', 'Act_tur']],
        entry('turn amplitude'),
        entry('angular pars', Npars=5),
        'source_analysis'
    ],
    'intake': [
        *[time(p) for p in ['sf_faeces_M', 'f_faeces_M', 'sf_abs_M', 'f_abs_M', 'f_am']],
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('food intake (timeplot)', 'food intake (filtered)', filt_amount=True),
        entry('pathlength', scaled=False),
        bar('f_am', 'food intake (barplot)'),
        'deb_analysis'
    ],
    'anemotaxis': [
        *[nengo(p, same_plot=True if p == 'anemotaxis' else False) for p in
          ['anemotaxis', 'frequency', 'interference', 'velocity', 'crawler', 'turner', 'wind_effect_on_V',
           'wind_effect_on_Fr']],
        *[time(p) for p in ['A_wind', 'anemotaxis', 'o_wind']],
        *[scat(p) for p in [['o_wind', 'A_wind'], ['anemotaxis', 'o_wind']]],
        end(['anemotaxis'], 'final anemotaxis'),
    ],
    'RL': [
        time('D_olf', 'olfactor_decay_table', save_as='olfactor_decay.pdf', table='best_gains'),
        time('D_olf', 'olfactor_decay_table_inds', save_as='olfactor_decay_inds.pdf', table='best_gains',
             individuals=True),
        time('cum_reward', 'reward_table', save_as='reward.pdf', table='best_gains'),
        time('g_odor1', 'best_gains_table', save_as='best_gains.pdf', table='best_gains'),
        time(*['g_odor1', 'g_odor2'], 'best_gains_table_x2', save_as='best_gains_x2.pdf', table='best_gains'),
    ],
    'survival': [
        entry('foraging'),
        time('on_food_tr', 'time ratio on food', u='min'),
        entry('food intake (timeplot)', 'food intake (raw)'),
        entry('pathlength', scaled=False)

    ]
    # 'DEB' : [
    #     *[deb(m, pref='DEB') for m in ['energy', 'growth', 'full']],
    #     *[deb(m, pref='FEED') for m in ['feeding', 'reserve_density', 'assimilation', 'food_ratio_1', 'food_ratio_2', 'food_mass_1',
    #               'food_mass_2', 'hunger', 'EEB','fs']],
    # ]

}
