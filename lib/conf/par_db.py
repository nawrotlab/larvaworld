import lib.aux.naming as nam
import numpy as np
import pandas as pd

from lib.aux.collecting import step_database
from lib.stor.paths import ParDb_path


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
            'scal': [nam.scal, {}, 's', 'pre', ast, {}]}
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
        [fv, 'fv', sub('f', 'v'), sub(hat('f'), 'v'), l_freq],
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
        chunk_ar.append([nam.num(cn), f'{suf}_N', sub('N', f'{cn}s'), sub(hat('N'), f'{cn}s'), f'# {cn}s'])
    chunk_ar = np.array(chunk_ar)

    temp_dsp = [[dsp, 'disp', 'disp', hat('disp')],
                [dsp40, 'disp40', sup('disp', 40), sup(hat('disp'), 40)]]

    dsp_ar = []
    for (fn, sn, sym, esym) in temp_dsp:
        dsp_ar.append([fn, sn, sym, esym, l_dst])
        dsp_ar.append([nam.scal(fn), f's{sn}', sup(sym, '*'), sup(esym, '*'), l_sc_dst])
        dsp_ar.append([nam.mean(fn), f'{sn}_mu', bar(sym), bar(esym), l_dst])
        dsp_ar.append([nam.scal(nam.mean(fn)), f's{sn}_mu', sup(bar(sym), '*'), sup(bar(esym), '*'), l_sc_dst])
        dsp_ar.append([nam.max(fn), f'{sn}_max', sub(sym, 'max'), sub(esym, 'max'), l_dst])
        dsp_ar.append(
            [nam.scal(nam.max(fn)), f's{sn}_max', subsup(sym, 'max', '*'), subsup(esym, 'max', '*'), l_sc_dst])
        dsp_ar.append([nam.final(fn), f'{sn}_fin', sub(sym, 'fin'), sub(esym, 'fin'), l_dst])
        dsp_ar.append([nam.scal(nam.final(fn)), f's{sn}_fin', subsup(sym, 'fin', '*'), subsup(esym, 'fin', '*'),
                       l_sc_dst])

    dsp_ar = np.array(dsp_ar)

    par_ar = np.array([
        ['cum_dur', 'cum_t', sub('t', 'cum'), sub(hat('t'), 'cum'), l_time],
        ['length', 'l_mu', bar('l'), bar(hat('l')), l_body_length],
        ['stride_reoccurence_rate', 'str_rr', sub('str', 'rr'), sub(hat('str'), 'rr'), '-'],
        ['length', 'l', 'l', hat('l'), l_body_length],
        ['amount_eaten', 'f_am', sub('m', 'feed'), sub(hat('m'), 'feed'), l_mass],
        ['max_feed_amount', 'f_am_max', subsup('m', 'feed', 'm'), subsup(hat('m'), 'feed', 'm'), l_mass],
        ['mass', 'm', 'm', hat('m'), l_mass],
        ['hunger', 'hunger', 'hunger', hat('hunger'), f'hunger (-)'],
        ['reserve_density', 'reserve_density', 'reserve_density', hat('reserve_density'), f'reserve density (-)'],
        ['puppation_buffer', 'puppation_buffer', 'puppation_buffer', hat('puppation_buffer'), f'puppation buffer (-)'],
        ['deb_f', 'deb_f', sub('f', 'deb'), sub(hat('f'), 'deb'), f'functional response (-)'],
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
    random_ar1 = generate_entries(bases=lin_ar[:-1, :].tolist(), types=['mean', 'std'])
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

    par_db.loc['c_odor1'] = {'par': 'first_odor_concentration',
                             'symbol': '${C}_{odor_{1}}$',
                             'exp_symbol': '${\hat{C}_{odor_{1}}$',
                             'unit': 'Concentration C(t), $\mu$M',
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

    par_db.loc['Act_tur'] = {'par': 'turner_activity',
                             'symbol': '$Act_{tur}$',
                             'exp_symbol': '$\hat{Act}_{tur}$',
                             'unit': 'Turner activity',
                             # 'collect' : 'ang_activity'
                             }

    par_db['lim'] = None
    par_db['lim'].loc['b'] = [-180, 180]
    par_db['lim'].loc['fo'] = [0, 360]
    par_db['lim'].loc['ro'] = [0, 360]
    par_db['lim'].loc['fov'] = [-300, 300]
    par_db['lim'].loc['rov'] = [-300, 300]

    par_db['lim'].loc['f_am'] = [0.0, 10 ** -5]
    par_db['lim'].loc['hunger'] = [0.0, 1.0]
    par_db['lim'].loc['puppation_buffer'] = [0.0, 1.0]
    par_db['lim'].loc['reserve_density'] = [0.0, 2.0]
    par_db['lim'].loc['deb_f'] = [0.0, 2.0]

    par_db['collect'] = None
    for k, v in step_database.items():
        par_db['collect'].loc[par_db['par'] == k] = v

    # par_db['type'] = None
    # par_db['type'].loc['str0'] = bool
    # par_db['type'].loc['str1'] = bool

    par_db.to_csv(ParDb_path, index=True, header=True)


def load_ParDb():
    df = pd.read_csv(ParDb_path, index_col=0)
    return df


par_db = load_ParDb()


# print(par_db.loc['f_am'])
# print(random_ar2)
# print('c_odor1' in par_db.index.to_list())
# print(par_db['par'].loc[par_db['collect'].isin([None])])
# print(par_db['par'].loc[par_db['collect'].isin([None])].index.tolist())


