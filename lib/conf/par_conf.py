import copy
from typing import Tuple, Type

import lib.aux.naming as nam
import numpy as np
import pandas as pd
import shelve

import lib.aux.functions as fun
from lib.aux.collecting import step_database
import lib.stor.paths as paths


def base(method, input, **kwargs):
    if type(input) == str:
        return method(input, **kwargs)
    elif type(input) == list:
        return [method(i, **kwargs) for i in input]


def bar(p):
    return rf'$\bar{{{p.replace("$", "")}}}$'


def wave(p):
    return rf'$\~{{{p.replace("$", "")}}}$'


def sub(p, q):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}$'


def sup(p, q):
    return rf'${{{p.replace("$", "")}}}^{{{q}}}$'


def subsup(p, q, z):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}^{{{z}}}$'


def hat(p):
    return f'$\hat{{{p.replace("$", "")}}}$'


def ast(p):
    return f'${p.replace("$", "")}^{{*}}$'


def th(p):
    return fr'$\theta_{{{p.replace("$", "")}}}$'


def hat_th(p):
    return fr'$\hat{{\theta}}_{{{p}}}$'


def dot(p):
    return fr'$\dot{{{p.replace("$", "")}}}$'


def ddot(p):
    return fr'$\ddot{{{p.replace("$", "")}}}$'


def dot_th(p):
    return fr'$\dot{{\theta}}_{{{p}}}$'


def ddot_th(p):
    return fr'$\ddot{{\theta}}_{{{p}}}$'


def dot_hat_th(p):
    return fr'$\dot{{\hat{{\theta}}}}_{{{p}}}$'


def ddot_hat_th(p):
    return fr'$\ddot{{\hat{{\theta}}}}_{{{p}}}$'


def lin(p):
    return fr'${{{p.replace("$", "")}}}_{{l}}$'


def get_lambda(attr):
    return lambda a: getattr(a, attr)


def set_ParDb():
    b, fo, ro = 'bend', 'front_orientation', 'rear_orientation'
    bv, fov, rov = nam.vel([b, fo, ro])
    ba, foa, roa = nam.acc([b, fo, ro])
    fou, rou = nam.unwrap([fo, ro])
    d, v, a = 'dst', 'vel', 'acc'
    sd, sv, sa = nam.scal([d, v, a])
    ld, lv, la = nam.lin([d, v, a])
    sld, slv, sla = nam.scal([ld, lv, la])
    std = nam.straight_dst(d)
    sstd = nam.scal(std)
    fv, fsv = nam.freq([v, sv])
    cum_d, cum_sd = nam.cum([d, sd])

    srd, pau, tur, fee = 'stride', 'pause', 'turn', 'feed'
    chunks = [srd, pau, tur, fee]
    srd_t, pau_t, tur_t, fee_t = nam.dur(chunks)
    srd_tr, pau_tr, tur_tr, fee_tr = nam.dur_ratio(chunks)
    srd_N, pau_N, tur_N, fee_N = nam.num(chunks)
    srd_d = nam.dst(srd)
    srd_sd = nam.scal(srd_d)
    dsp = 'dispersion'
    dsp40 = f'40sec_{dsp}'
    sdsp, sdsp40 = nam.scal([dsp, dsp40])
    f_dsp, f_dsp40, f_sdsp, f_sdsp40 = nam.final([dsp, dsp40, sdsp, sdsp40])
    mu_dsp, mu_dsp40, mu_sdsp, mu_sdsp40 = nam.mean([dsp, dsp40, sdsp, sdsp40])
    max_dsp, max_dsp40, max_sdsp, max_sdsp40 = nam.max([dsp, dsp40, sdsp, sdsp40])
    srd_fo, srd_ro, srd_b = nam.chunk_track(srd, [fou, rou, b])

    l_angle = 'angle $(deg)$'
    l_angvel = 'angular velocity $(deg/sec)$'
    l_angacc = 'angular acceleration, $(deg^2/sec)$'
    l_time = 'time $(sec)$'
    l_time_ratio = 'time ratio $(-)$'
    l_freq = 'frequency $(Hz)$'
    l_dst = 'distance $(mm)$'
    l_body_length = 'body length $(mm)$'
    l_vel = 'velocity $(mm/sec)$'
    l_acc = 'acceleration $(mm/sec^2)$'
    l_sc_dst = 'scaled distance $(-)$'
    l_sc_vel = 'scaled velocity $(sec^{-1})$'
    l_sc_acc = 'scaled acceleration $(sec^{-2})$'
    l_num = 'counts $(#)$'
    l_mass = 'mass $(mg)$'

    sc_unit_dict = {l_dst: l_sc_dst,
                    l_vel: l_sc_vel,
                    l_acc: l_sc_acc}

    def generate_entries(bases, types):
        entry_dict = {
            'stride': [nam.chunk_track, {'chunk_name': 'stride'}, 'str_', 'pre', sub, {'q': 'str'}],
            'dur': [nam.dur, {}, '_t', 'suf', sub, {'p': 't'}],
            'dur_ratio': [nam.dur_ratio, {}, '_tr', 'suf', sub, {'p': 'r'}],
            'lin': [nam.lin, {}, 'l', 'pre', sub, {'q': 'l'}],
            'mean': [nam.mean, {}, '_mu', 'suf', bar, {}],
            'std': [nam.std, {}, '_std', 'suf', wave, {}],
            'max': [nam.max, {}, '_max', 'suf', sub, {'q': 'max'}],
            'fin': [nam.final, {}, '_fin', 'suf', sub, {'q': 'fin'}],
            'scal': [nam.scal, {}, 'sigma', 'pre', ast, {}]}
        entries = []
        for base in bases:
            for type in types:
                fn, sn, sym, esym, u = base
                conf = entry_dict[type]
                if type == 'dur':
                    nu = l_time
                elif type == 'dur_ratio':
                    nu = l_time_ratio
                elif type == 'scal':
                    nu = sc_unit_dict[u]
                else:
                    nu = u
                if conf[3] == 'suf':
                    nsn = f'{sn}{conf[2]}'
                elif conf[3] == 'pre':
                    nsn = f'{conf[2]}{sn}'
                try:
                    nsym, nesym = conf[4](p=sym, **conf[5]), conf[4](p=esym, **conf[5])
                except:
                    nsym, nesym = conf[4](q=sym, **conf[5]), conf[4](q=esym, **conf[5])
                try:
                    nfn = conf[0](fn, **conf[1])
                except:
                    nfn = conf[0](params=fn, **conf[1])
                entries.append([nfn, nsn, nsym, nesym, nu])

        return np.array(entries)

    cols = ['par', 'shortcut', 'symbol', 'exp_symbol', 'unit']

    temp_ang = [[b, 'b', 'b'],
                [fo, 'fo', r'or_{f}'],
                [ro, 'ro', r'or_{r}'],
                ]
    ang_ar = []
    for (fn, sn, sym) in temp_ang:
        ang_ar.append([fn, sn, th(sym), hat_th(sym), l_angle])
        ang_ar.append([nam.vel(fn), f'{sn}v', dot_th(sym), dot_hat_th(sym), l_angvel])
        ang_ar.append([nam.acc(fn), f'{sn}a', ddot_th(sym), ddot_hat_th(sym), l_angacc])
    ang_ar = np.array(ang_ar)

    lin_ar = np.array([
        [d, 'd', 'd', hat('d'), l_dst],
        [ld, 'ld', sub('d', 'l'), sub(hat('d'), 'l'), l_dst],
        [v, 'v', 'v', hat('v'), l_vel],
        [a, 'a', dot('v'), dot(hat('v')), l_acc],
        [lv, 'lv', sub('v', 'l'), sub(hat('v'), 'l'), l_vel],
        [la, 'la', sub(dot('v'), 'l'), sub(dot(hat('v')), 'l'), l_acc],
        [cum_d, 'cum_d', sub('d', 'cum'), sub(hat('d'), 'cum'), l_dst],
        ['dst_to_center', 'd_cent', sub('d', 'cent'), sub(hat('d'), 'cent'), l_dst],
        ['dst_to_chemotax_odor', 'd_chem', sub('d', 'chem'), sub(hat('d'), 'chem'), l_dst],
        # [d_, 'cum_d', sub('d', 'cum'), sub(hat('d'), 'cum'), l_dst],
        [fv, 'fv', sub('f', 'v'), sub(hat('f'), 'v'), l_freq]

    ])

    sc_lin_ar = np.array([
        [sd, 'sd', sup('d', '*'), sup(hat('d'), '*'), l_sc_dst],
        [sld, 'sld', subsup('d', 'l', '*'), subsup(hat('d'), 'l', '*'), l_sc_dst],
        [sv, 'sv', sup('v', '*'), sup(hat('v'), '*'), l_sc_vel],
        [sa, 'sa', sup(dot('v'), '*'), sup(dot(hat('v')), '*'), l_sc_acc],
        [slv, 'slv', subsup('v', 'l', '*'), subsup(hat('v'), 'l', '*'), l_sc_vel],
        [sla, 'sla', subsup(dot('v'), 'l', '*'), subsup(dot(hat('v')), 'l', '*'), l_sc_acc],
        [cum_sd, 'cum_sd', subsup('d', 'cum', '*'), subsup(hat('d'), 'cum', '*'), l_sc_dst],
        [fsv, 'fsv', subsup('f', 'v', '*'), subsup(hat('f'), 'v', '*'), l_freq],
    ])

    temp_chunk = [['str', 'stride'],
                  ['non_str', 'non_stride'],
                  ['pau', 'pause'],
                  ['tur', 'turn'],
                  ['Ltur', 'Lturn'],
                  ['Rtur', 'Rturn'],
                  ['fee', 'feed'],
                  ['chn', 'stridechain'],

                  ]
    chunk_ar = []
    for (suf, cn) in temp_chunk:
        chunk_ar.append([nam.dst(cn), f'{suf}_d', sub('d', suf), sub(hat('d'), suf), l_dst])
        chunk_ar.append(
            [nam.scal(nam.dst(cn)), f'{suf}_sd', subsup('d', suf, '*'), subsup(hat('d'), suf, '*'), l_sc_dst])
        chunk_ar.append(
            [nam.straight_dst(cn), f'{suf}_std', subsup('d', suf, 'st'), subsup(hat('d'), suf, 'st'), l_dst])
        chunk_ar.append(
            [nam.scal(nam.straight_dst(cn)), f'{suf}_sstd', subsup('d', suf, 'st*'), subsup(hat('d'), suf, 'st*'),
             l_sc_dst])
        chunk_ar.append([nam.dur(cn), f'{suf}_t', sub('t', cn), sub(hat('t'), cn), l_time])
        chunk_ar.append([f'{cn}_y', f'{suf}_y', sub('y', cn), sub(hat('y'), cn), l_dst])
        chunk_ar.append([f'{cn}_x', f'{suf}_x', sub('x', cn), sub(hat('x'), cn), l_dst])
        chunk_ar.append([f'y_at_{cn}_start', f'{suf}_y0', sub('y0', cn), sub(hat('y0'), cn), l_dst])
        chunk_ar.append([f'y_at_{cn}_stop', f'{suf}_y1', sub('y1', cn), sub(hat('y1'), cn), l_dst])
        chunk_ar.append([f'x_at_{cn}_start', f'{suf}_x0', sub('x0', cn), sub(hat('x0'), cn), l_dst])
        chunk_ar.append([f'x_at_{cn}_stop', f'{suf}_x1', sub('x1', cn), sub(hat('x1'), cn), l_dst])
        chunk_ar.append([f'{fou}_at_{cn}_start', f'{suf}_fo0', sub('fo0', cn), sub(hat('fo0'), cn), l_angle])
        chunk_ar.append([f'{fou}_at_{cn}_stop', f'{suf}_fo1', sub('fo1', cn), sub(hat('fo1'), cn), l_angle])
        chunk_ar.append(
            [nam.mean(nam.dur(cn)), f'{suf}_t_mu', sub(bar('t'), cn), sub(bar(hat('t')), cn), l_time])
        chunk_ar.append(
            [nam.std(nam.dur(cn)), f'{suf}_t_std', sub(wave('t'), cn), sub(wave(hat('t')), cn), l_time])
        chunk_ar.append(
            [nam.cum(nam.dur(cn)), f'cum_{suf}_t', subsup('t', cn, 'cum'), subsup(hat('t'), cn, 'cum'),
             l_time])
        chunk_ar.append(
            [nam.max(nam.dur(cn)), f'{suf}_t_max', subsup('t', cn, 'm'), subsup(hat('t'), cn, 'm'), l_time])
        chunk_ar.append(
            [nam.start(cn), f'{suf}0', subsup('t', cn, 0), subsup(hat('t'), cn, 0), l_time])
        chunk_ar.append(
            [nam.stop(cn), f'{suf}1', subsup('t', cn, 1), subsup(hat('t'), cn, 1), l_time])
        chunk_ar.append(
            [nam.length(cn), f'{suf}_l', sub(cn, 'l'), sub(hat(cn), 'l'), l_num])
        chunk_ar.append(
            [nam.id(cn), f'{suf}_id', sub(cn, 'id'), sub(hat(cn), 'id'), l_num])
        chunk_ar.append([nam.dur_ratio(cn), f'{suf}_tr', sub('r', cn), sub(hat('r'), cn), l_time_ratio])
        chunk_ar.append([nam.num(cn), f'{suf}_N', sub('N', f'{cn}sigma'), sub(hat('N'), f'{cn}sigma'), f'# {cn}sigma'])
    chunk_ar = np.array(chunk_ar)

    temp_dsp = [[dsp, 'disp', 'disp', hat('disp')],
                [dsp40, 'disp40', sup('disp', 40), sup(hat('disp'), 40)]]

    dsp_ar = []
    for (fn, sn, sym, esym) in temp_dsp:
        dsp_ar.append([fn, sn, sym, esym, l_dst])
        dsp_ar.append([nam.scal(fn), f'sigma{sn}', sup(sym, '*'), sup(esym, '*'), l_sc_dst])
        dsp_ar.append([nam.mean(fn), f'{sn}_mu', bar(sym), bar(esym), l_dst])
        dsp_ar.append([nam.scal(nam.mean(fn)), f'sigma{sn}_mu', sup(bar(sym), '*'), sup(bar(esym), '*'), l_sc_dst])
        dsp_ar.append([nam.max(fn), f'{sn}_max', sub(sym, 'max'), sub(esym, 'max'), l_dst])
        dsp_ar.append(
            [nam.scal(nam.max(fn)), f'sigma{sn}_max', subsup(sym, 'max', '*'), subsup(esym, 'max', '*'), l_sc_dst])
        dsp_ar.append([nam.final(fn), f'{sn}_fin', sub(sym, 'fin'), sub(esym, 'fin'), l_dst])
        dsp_ar.append([nam.scal(nam.final(fn)), f'sigma{sn}_fin', subsup(sym, 'fin', '*'), subsup(esym, 'fin', '*'),
                       l_sc_dst])

    dsp_ar = np.array(dsp_ar)

    par_ar = np.array([
        ['cum_dur', 'cum_t', sub('t', 'cum'), sub(hat('t'), 'cum'), l_time],
        ['length', 'l_mu', bar('l'), bar(hat('l')), l_body_length],
        ['stride_reoccurence_rate', 'str_rr', sub('str', 'rr'), sub(hat('str'), 'rr'), '-'],
        ['length', 'l', 'l', hat('l'), l_body_length],
        ['amount_eaten', 'f_am', sub('m', 'feed'), sub(hat('m'), 'feed'), 'food intake (mg)'],
        ['amount_absorbed', 'f_ab', sub('m', 'absorbed'), sub(hat('m'), 'absorbed'), 'food absorbed (mg)'],
        ['amount_faeces', 'f_out', sub('m', 'faeces'), sub(hat('m'), 'faeces'), 'faeces (mg)'],
        ['max_V_bite', 'f_am_max', subsup('m', 'feed', 'm'), subsup(hat('m'), 'feed', 'm'), 'max food intake (mg)'],
        ['mass', 'm', 'm', hat('m'), l_mass],
        ['hunger', 'hunger', 'hunger', hat('hunger'), f'hunger (-)'],
        ['reserve_density', 'reserve_density', 'reserve_density', hat('reserve_density'), f'reserve density (-)'],
        ['puppation_buffer', 'puppation_buffer', 'puppation_buffer', hat('puppation_buffer'), f'puppation buffer (-)'],
        ['deb_f', 'deb_f', sub('f', 'deb'), sub(hat('f'), 'deb'), f'functional response (-)'],
        ['deb_f_deviation', 'deb_f_dev', sub('f', 'deb'), sub(hat('f'), 'deb'), f'functional response deviation (-)'],
        ['deb_f_mean', 'deb_f_mu', sub(bar('f'), 'deb'), sub(hat(bar('f')), 'deb'), f'functional response (-)'],
        ['Nlarvae', 'lar_N', sub('N', 'larvae'), sub(hat('N'), 'larvae'), f'# larvae'],
    ])

    orient_ar = np.array([[f'turn_{fou}', 'tur_fo', r'$\theta_{turn}$', r'$\hat{\theta}_{turn}$', l_angle],
                          [f'Lturn_{fou}', 'Ltur_fo', r'$\theta_{Lturn}$', r'$\hat{\theta}_{Lturn}$', l_angle],
                          [f'Rturn_{fou}', 'Rtur_fo', r'$\theta_{Rturn}$', r'$\hat{\theta}_{Rturn}$', l_angle],
                          [srd_fo, 'str_fo', r'$\Delta{\theta}_{or_{f}}$', r'$\Delta{\hat{\theta}}_{or_{f}}$', l_angle],
                          [srd_ro, 'str_ro', r'$\Delta{\theta}_{or_{r}}$', r'$\Delta{\hat{\theta}}_{or_{r}}$', l_angle],
                          [srd_b, 'str_b', r'$\Delta{\theta}_{b}$', r'$\Delta{\hat{\theta}}_{b}$', l_angle]
                          ])

    temp_tor = []
    for i in [2, 5, 10, 20]:
        fn = f'tortuosity_{i}'
        sn = f'tor{i}'
        sym = sup('tor', i)
        esym = sup(hat('tor'), i)
        u = '-'
        temp_tor.append([fn, sn, sym, esym, u])
    tor_ar = generate_entries(bases=temp_tor, types=['mean', 'std']).tolist()
    tor_ar.append(['tortuosity', 'tor', 'tor', hat('tor'), '-'])
    tor_ar = np.array(tor_ar)
    random_ar1 = generate_entries(bases=lin_ar[:-1, :].tolist(), types=['mean', 'std', 'fin'])
    sc_random_ar1 = generate_entries(bases=random_ar1.tolist(), types=['scal'])

    srd_sc_random_ar1 = generate_entries(bases=sc_random_ar1.tolist(), types=['stride'])
    random_ar2 = generate_entries(bases=chunk_ar[:5, :].tolist(), types=['mean', 'std'])
    # sc_random_ar2 = generate_entries(bases=random_ar2[:4, :].tolist(), types=['scal'])
    random_ar3 = generate_entries(bases=ang_ar.tolist(), types=['mean', 'std'])
    random_ar4 = generate_entries(bases=orient_ar.tolist(), types=['mean', 'std'])
    random_ar5 = generate_entries(bases=sc_lin_ar.tolist(), types=['mean', 'std'])
    sc_chunk_ar = generate_entries(bases=random_ar5.tolist(), types=['stride'])
    par_ar = np.vstack([par_ar,
                        ang_ar,
                        lin_ar,
                        sc_lin_ar,
                        # sc_random_ar2,
                        chunk_ar,
                        random_ar2,
                        dsp_ar,
                        tor_ar,
                        orient_ar,
                        random_ar1,
                        sc_random_ar1,
                        sc_chunk_ar,

                        srd_sc_random_ar1,
                        random_ar3,
                        random_ar4,
                        random_ar5])

    ind_col = 1
    sel_cols = [x for x in range(par_ar.shape[1]) if x != ind_col]
    par_db = pd.DataFrame(data=par_ar[:, sel_cols], index=par_ar[:, ind_col],
                          columns=[c for i, c in enumerate(cols) if i != ind_col])
    par_db.index.name = cols[ind_col]

    par_db = par_db[~par_db.index.duplicated(keep='first')]
    par_db['unit'].loc['str_tr'] = '% time crawling'
    par_db['unit'].loc['non_str_tr'] = '% time not crawling'
    par_db['unit'].loc['pau_tr'] = '% time pausing'
    par_db['unit'].loc['fee_tr'] = '% time feeding'
    par_db['unit'].loc['tur_tr'] = '% time turning'
    par_db['unit'].loc['Ltur_tr'] = '% time turning left'
    par_db['unit'].loc['Rtur_tr'] = '% time turning right'

    par_db['unit'].loc['cum_sd'] = 'scaled pathlength'
    par_db['unit'].loc['cum_d'] = 'pathlength $(mm)$'

    par_db['unit'].loc['b'] = 'bend angle $(deg)$'
    par_db['unit'].loc['bv'] = 'bending velocity $(deg/sec)$'
    par_db['unit'].loc['ba'] = 'bending acceleration $(deg^2/sec)$'
    par_db['unit'].loc['fo'] = 'orientation angle $(deg)$'
    par_db['unit'].loc['ro'] = 'rear orientation angle $(deg)$'
    par_db['unit'].loc['fov'] = 'orientation velocity $(deg/sec)$'
    par_db['unit'].loc['foa'] = 'orientation acceleration $(deg^2/sec)$'
    par_db['unit'].loc['rov'] = 'rear orientation velocity $(deg/sec)$'
    par_db['unit'].loc['roa'] = 'rear orientation acceleration $(deg^2/sec)$'

    par_db['unit'].loc['str_fo'] = r'$\Delta\theta_{or}$ over strides $(deg)$'
    par_db['unit'].loc['str_ro'] = r'$\Delta\theta_{or_{r}}$ over strides $(deg)$'
    par_db['unit'].loc['tur_fo'] = r'$\Delta\theta_{or}$ over turns $(deg)$'
    par_db['unit'].loc['tur_ro'] = r'$\Delta\theta_{or_{r}}$ over turns $(deg)$'

    par_db['unit'].loc['fee_N'] = '# feeding events'

    par_db.loc['sf_am'] = {'par': 'scaled_amount_eaten',
                           'symbol': '${m^{*}}_{feed}$',
                           'exp_symbol': '${\hat{m^{*}}}_{feed}$',
                           'unit': 'food intake as % larval mass',
                           # 'collect' : None
                           }

    par_db.loc['g_odor1'] = {'par': 'first_odor_best_gain',
                             'symbol': '${G}_{odor_{1}}$',
                             'exp_symbol': '${\hat{G}_{odor_{1}}$',
                             'unit': 'Gain G(t)',
                             # 'collect' : 'first_odor_concentration'
                             }

    par_db.loc['g_odor2'] = {'par': 'second_odor_best_gain',
                             'symbol': '${G}_{odor_{2}}$',
                             'exp_symbol': '${\hat{G}_{odor_{2}}$',
                             'unit': 'Gain G(t)',
                             # 'collect' : 'first_odor_concentration'
                             }

    par_db.loc['cum_reward'] = {'par': 'cum_reward',
                             'symbol': '${R}_{cum}$',
                             'exp_symbol': '${\hat{R}_{cum}$',
                             'unit': 'Reward R(t)',
                             # 'collect' : 'first_odor_concentration'
                             }

    par_db.loc['D_olf'] = {'par': 'best_olfactor_decay',
                             'symbol': '${D}_{olf}$',
                             'exp_symbol': '${\hat{D}_{olf}$',
                             'unit': 'Olfactor decay coeeficient',
                             # 'collect' : 'first_odor_concentration'
                             }

    par_db.loc['c_odor1'] = {'par': 'first_odor_concentration',
                             'symbol': '${C}_{odor_{1}}$',
                             'exp_symbol': '${\hat{C}_{odor_{1}}$',
                             'unit': 'Concentration C(t), $\mu$M',
                             # 'collect' : 'first_odor_concentration'
                             }

    par_db.loc['dc_odor1'] = {'par': 'first_odor_concentration_change',
                              'symbol': '$\delta{C}_{odor_{1}}$',
                              'exp_symbol': '$\delta{\hat{C}_{odor_{1}}$',
                              'unit': 'Concentration change dC(t), $-$',
                              # 'collect' : 'first_odor_concentration'
                              }

    par_db.loc['A_olf'] = {'par': 'olfactory_activation',
                           'symbol': '$A_{olf}$',
                           'exp_symbol': '$\hat{A}_{olf}$',
                           'unit': 'Olfactory activation',
                           # 'collect' : 'olfactory_activation'
                           }

    par_db.loc['A_tur'] = {'par': 'turner_activation',
                           'symbol': '$A_{tur}$',
                           'exp_symbol': '$\hat{A}_{tur}$',
                           'unit': 'Turner activation',
                           # 'collect' : 'turner_activation'
                           }

    par_db.loc['y'] = {'par': 'y',
                           'symbol': '$y$',
                           'exp_symbol': '$\hat{y}$',
                           'unit': 'Y position $(mm)$',
                           # 'collect' : 'turner_activation'
                           }

    par_db.loc['x'] = {'par': 'x',
                       'symbol': '$x$',
                       'exp_symbol': '$\hat{x}$',
                       'unit': 'X position $(mm)$',
                       # 'collect' : 'turner_activation'
                       }

    # par_db.loc['deb_f_deviation'] = {'par': 'turner_activation',
    #                        'symbol': '$A_{tur}$',
    #                        'exp_symbol': '$\hat{A}_{tur}$',
    #                        'unit': 'Turner activation',
    #                        # 'collect' : 'turner_activation'
    #                        }

    par_db.loc['Act_tur'] = {'par': 'turner_activity',
                             'symbol': '$Act_{tur}$',
                             'exp_symbol': '$\hat{Act}_{tur}$',
                             'unit': 'Turner activity',
                             # 'collect' : 'ang_activity'
                             }

    par_db.loc['fo2cen'] = {'par': 'orientation_to_center',
                             'symbol': r'$\theta_{or_{cen}}$',
                             'exp_symbol': '$\hat{theta}_{or_{cen}}$',
                             'unit': 'orientation angle $(deg)$',
                             # 'lim': [-180, 180],
                             # 'collect' : 'ang_activity'
                             }

    par_db.loc['abs_r'] = {'par': 'food_absorption_efficiency',
                            'symbol': r'$r_{absorption}$',
                            'exp_symbol': '$\hat{r}_{absorption}$',
                            'unit': 'food absorption ratio $(-)$',
                            # 'lim': [-180, 180],
                            # 'collect' : 'ang_activity'
                            }

    par_db.loc['f_out_r'] = {'par': 'faeces_ratio',
                           'symbol': r'$r_{faeces}$',
                           'exp_symbol': '$\hat{r}_{faeces}$',
                           'unit': 'faeces ratio $(-)$',
                           # 'lim': [-180, 180],
                           # 'collect' : 'ang_activity'
                           }

    par_db['lim'] = None
    par_db['lim'].loc['f_out_r'] = [0, 1]
    par_db['lim'].loc['abs_r'] = [0, 1]
    par_db['lim'].loc['fo2cen'] = [-180, 180]
    par_db['lim'].loc['b'] = [-180, 180]
    par_db['lim'].loc['fo'] = [0, 360]
    par_db['lim'].loc['ro'] = [0, 360]
    par_db['lim'].loc['fov'] = [-300, 300]
    par_db['lim'].loc['rov'] = [-300, 300]

    # par_db['lim'].loc['f_am'] = [0.0, 10 ** -5]
    par_db['lim'].loc['hunger'] = [0.0, 1.0]
    par_db['lim'].loc['puppation_buffer'] = [0.0, 1.0]
    par_db['lim'].loc['reserve_density'] = [0.0, 2.0]
    par_db['lim'].loc['deb_f'] = [0.0, 2.0]

    par_db['lim'].loc['g_odor1'] = [-500.0, 500.0]
    par_db['lim'].loc['g_odor2'] = [-500.0, 500.0]
    par_db['lim'].loc['c_odor1'] = [0.0, 8.0]
    par_db['lim'].loc['dc_odor1'] = [-0.05, 0.05]
    par_db['lim'].loc['A_olf'] = [-1.0, 1.0]
    par_db['lim'].loc['A_tur'] = [10.0, 40.0]
    par_db['lim'].loc['Act_tur'] = [-20.0, 20.0]
    par_db['lim'].loc['str_sd_std'] = [0.0, 0.15]
    par_db['lim'].loc['str_sstd_std'] = [0.0, 0.2]
    # par_db['lim'].loc['l_mu'] = [2.5, 4.5]
    par_db['lim'].loc['fsv'] = [1.0, 2.5]
    par_db['lim'].loc['str_sd_mu'] = [0.1, 0.3]
    par_db['lim'].loc['str_sd_std'] = [0.0, 0.1]
    par_db['lim'].loc['str_tr'] = [0.3, 1.0]
    par_db['lim'].loc['pau_tr'] = [0.0, 0.5]
    par_db['lim'].loc['sv_mu'] = [0.05, 0.35]
    par_db['lim'].loc['tor2_mu'] = [0.0, 0.5]
    par_db['lim'].loc['tor5_mu'] = [0.0, 0.5]
    par_db['lim'].loc['tor10_mu'] = [0.0, 0.5]
    par_db['lim'].loc['tor20_mu'] = [0.0, 0.5]
    par_db['lim'].loc['str_fo_mu'] = [-5.0, 5.0]
    par_db['lim'].loc['str_fo_std'] = [10.0, 40.0]
    par_db['lim'].loc['tur_fo_mu'] = [5.0, 25.0]
    par_db['lim'].loc['tur_fo_std'] = [10.0, 40.0]
    par_db['lim'].loc['b_mu'] = [-5.0, 5.0]
    par_db['lim'].loc['b_std'] = [10.0, 30.0]
    par_db['lim'].loc['pau_t_mu'] = [0.0, 2.0]

    to_drop1 = [f'{c}_l' for c in ['non_str', 'pau', 'str', 'tur', 'Ltur', 'Rtur', 'fee']]
    to_drop2 = fun.flatten_list([[f'{c}_{d}' for d in ['sstd', 'std', 'sd', 'd']] for c in ['non_str', 'fee']])
    to_drop = to_drop1 + to_drop2
    par_db.drop(labels=to_drop, inplace=True)

    par_db['collect'] = None
    for k, v in step_database.items():
        par_db['collect'].loc[par_db['par'] == k] = v

    par_db['disp_name']=par_db['par']
    disp_dict={
        'sv' : r'velocity$_{scaled}$',
        'sv_mu' : r'mean velocity$_{scaled}$',
        'v' : 'velocity',
        'l_mu' : 'body length',
        'fsv' : 'crawl frequency',
        'str_sd_mu' : r'stridestep$_{scaled}$ mean',
        'str_sd_std' : r'stridestep$_{scaled}$ std',
        'b' : 'body bend',
        'b_mu' : 'body bend mean',
        'b_std' : 'body bend std',
        'bv': 'bend velocity',
        'bv_mu': 'bend velocity mean',
        'bv_std': 'bend velocity std',
        'fov': 'turn velocity',
        'fov_mu': 'turn velocity mean',
        'fov_std': 'turn velocity std',
        'str_fo_mu': r'stride $\Delta_{or}$ mean',
        'str_fo_std': r'stride $\Delta_{or}$ std',
        'tur_fo_mu': 'turn angle mean',
        'tur_fo_std': 'turn angle std',
        'pau_t_mu': 'pause duration mean',
        'pau_t_std': 'pause duration std',
        'str_tr': 'crawl time ratio',
        'pau_tr': 'pause time ratio',
        'Ltur_tr': 'Lturn time ratio',
        'Rtur_tr': 'Rturn time ratio',
        'tur_tr': 'turn time ratio',
        'sdisp40_fin': r'final dispersion$_{scaled}$ 40 sec',
        'disp40_fin': r'final dispersion 40 sec',
        **{f'tor{ii}' : rf'tortuosity$_{{{ii} sec}}$' for ii in [2,5,10,20]},
        **{f'tor{ii}_mu' : rf'tortuosity$_{{{ii} sec}}$ mean' for ii in [2,5,10,20]},
        **{f'tor{ii}_std' : rf'tortuosity$_{{{ii} sec}}$ std' for ii in [2,5,10,20]},
    }
    for kk,vv in disp_dict.items() :
        par_db['disp_name'].loc[kk] = vv

    par_db = set_dtype(par_db)
    par_db = set_collect_from(par_db)

    par_db.to_csv(paths.ParDb_path, index=True, header=True)

    return par_db


def set_dtype(par_db):
    db = par_db.to_dict('index')
    for k in db.keys():
        if k in [
            'non_str0', 'non_str1',
            'str0', 'str1',
            'pau0', 'pau1',
            'tur0', 'tur1',
            'Ltur0', 'Ltur1',
            'Rtur0', 'Rtur1',
            'fee0', 'fee1',
            'chn0', 'chn1',
        ]:
            db[k]['dtype'] = bool
        elif str(k).endswith('id'):
            db[k]['dtype'] = str
        elif str(k).endswith('N'):
            db[k]['dtype'] = int
        elif 'counts' in db[k]['unit']:
            db[k]['dtype'] = int
        elif str(k).endswith('mu') or str(k).endswith('std'):
            db[k]['dtype'] = float
        elif 'fo' in k or 'ro' in k or 'disp' in k:
            db[k]['dtype'] = float
        else:
            db[k]['dtype'] = float
        # elif 'disp' in k or 'ro' in k :
        # elif 'disp' in k or 'ro' in k :
        #     print(k)
    par_db = pd.DataFrame.from_dict(db, orient='index')
    return par_db


def set_collect_from(par_db):
    from lib.model._agent import LarvaworldAgent
    db = par_db.to_dict('index')
    for k in db.keys():
        if db[k]['par'] in step_database:
            db[k]['collect_from'] = LarvaworldAgent
        else:
            db[k]['collect_from'] = None
            # db[k]['collect_from'] = 'Unknown'
            # print(k)
    par_db = pd.DataFrame.from_dict(db, orient='index')
    return par_db


def load_ParDb():
    import lib.gui.gui_lib as gui
    df = pd.read_csv(paths.ParDb_path, index_col=0)
    df['lim'] = [gui.retrieve_value(v, Tuple[float, float]) for v in df['lim'].values]
    df['dtype'] = [gui.retrieve_value(v, Type) for v in df['dtype'].values]
    return df


def set_ParShelve(par_db):
    # ATTENTION : This must NOT be the laded par_db but the one just created. Othrwise everything is float!
    temp = copy.deepcopy(par_db)
    with shelve.open(paths.ParShelve_path) as db0:
        for k, v in temp.to_dict('index').items():
            #     if 'LarvaworldAgent' in v['collect_from'] :
            #         v['collect_from']=LarvaworldAgent
            #     elif 'Larvaworld' in v['collect_from'] :
            #         v['collect_from']=LarvaWorld
            db0[k] = v
    db0.close()


def get_par_dict(short=None, par=None, retrieve_from='shelve'):
    dic=None
    if retrieve_from == 'shelve':
        db = shelve.open(paths.ParShelve_path)
    elif retrieve_from == 'par_db':
        db = load_ParDb().to_dict('index')
    if short is not None:
        if short not in list(db.keys()):
            raise ValueError(f'Parameter shortcut {short} does not exist in parameter database')
        dic = db[short]
    elif par is not None:
        for k in db.keys():
            if db[k]['par'] == par:
                dic = db[k]
    else:
        raise ValueError('Either the shortcut or the parameter name must be provided.')
    if retrieve_from == 'shelve':
        db.close()
    return dic


def par_dict_lists(shorts=None, pars=None, retrieve_from='shelve',to_return=['par', 'symbol', 'unit', 'lim']):
    if shorts is not None:
        par_dicts = [get_par_dict(short=short, retrieve_from=retrieve_from) for short in shorts]
    elif pars is not None:
        par_dicts = [get_par_dict(par=par, retrieve_from=retrieve_from) for par in pars]
    else:
        raise ValueError('Either the shortcut_defaults or the parameter names must be provided.')
    r = []
    for p in to_return:
        r.append([d[p] for d in par_dicts])
    return r


def par_in_db(short=None, par=None):
    res = False
    db = shelve.open(paths.ParShelve_path)
    if short is not None:
        if short in list(db.keys()):
            res = True
    elif par is not None:
        for k in db.keys():
            if db[k]['par'] == par:
                res = True
    db.close()
    return res


def get_runtime_pars():
    return fun.unique_list([p for p in list(step_database.keys()) if par_in_db(par=p)])


# print(get_par_dict(par='orientation_to_center'))
# print(random_ar2)
# print('c_odor1' in par_db.index.to_list())

# print(par_db['par'].loc[par_db['collect'].isin([None])].index.tolist())


if __name__ == '__main__':
    # Use this to update the database
    par_db = set_ParDb()
    set_ParShelve(par_db)
    # print(mode(get_par('c_odor1')['dtype']))
    # print(get_par_dict(short='fov'))
    # print(par_db.loc['g_odor1'])
    print('final_dst_to_chemotax_odor' in list(step_database.keys()))
    print(par_in_db(par='final_dst_to_chemotax_odor'))
    # print(get_par_dict(par='cum_dst', retrieve_from='par_db'))
    print(par_db.loc['d_chem_fin'])