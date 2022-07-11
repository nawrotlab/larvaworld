import lib.aux.dictsNlists as dNl
# from lib.ga.util.genome import Genome

from lib.registry.pars import preg

def arrange_fitness(fitness_func, fitness_target_refID, fitness_target_kws,dt, source_xy=None):
    cycle_ks, eval_kNps = None, None
    ks = []
    robot_dict = dNl.NestDict()
    if fitness_target_refID is not None:
        d = preg.loadRef(fitness_target_refID)
        if 'eval_shorts' in fitness_target_kws.keys():
            shs = fitness_target_kws['eval_shorts']

            eval_pars, eval_lims, eval_labels = preg.getPar(shs, to_return=['d', 'lim', 'lab'])
            fitness_target_kws['eval'] = {sh: d.get_par(p, key='distro').dropna().values for p, sh in
                                          zip(eval_pars, shs)}
            ks += shs
            eval_kNps={sh: p for p, sh in zip(eval_pars, shs)}
            robot_dict.eval = {sh: [] for p, sh in zip(eval_pars, shs)}
            fitness_target_kws['eval_labels'] = eval_labels
        if 'pooled_cycle_curves' in fitness_target_kws.keys():
            curves = d.config.pooled_cycle_curves
            shorts = fitness_target_kws['pooled_cycle_curves']
            cycle_ks = shorts
            ks += shorts
            dic = {}
            for sh in shorts:
                dic[sh] = 'abs' if sh == 'sv' else 'norm'

            fitness_target_kws['cycle_curve_keys'] = dic
            fitness_target_kws['pooled_cycle_curves'] = {sh: curves[sh] for sh in shorts}
            robot_dict.cycle_curves = {sh: [] for sh in shorts}

        fitness_target = d
    else:
        fitness_target = None
    if 'source_xy' in fitness_target_kws.keys():
        fitness_target_kws['source_xy'] = source_xy
    robot_dict.step=None
    ks = dNl.unique_list(ks)

    def robot_func(ss) :
        gdict = dNl.NestDict()
        gdict.step = ss
        if cycle_ks:
            from lib.process.aux import cycle_curve_dict
            gdict.cycle_curves = cycle_curve_dict(s=ss, dt=dt, shs=cycle_ks)
        if eval_kNps:
            gdict.eval = {sh: ss[p].dropna().values for sh, p in eval_kNps.items()}
        return gdict
    # dic0 = self.fit_dict.robot_dict
    # cycle_ks, eval_ks = None, None
    # ks = []
    # if 'eval' in robot_dict.keys():
    #     eval_ks = fitness_target_kws['eval_shorts']
    #     ks += eval_ks
    # if 'cycle_curves' in robot_dict.keys():
    #     cycle_ks = list(fitness_target_kws['pooled_cycle_curves'].keys())
    #     ks += cycle_ks
    # ks = dNl.unique_list(ks)
    return dNl.NestDict({'func': fitness_func, 'target_refID': fitness_target_refID,
                         'keys' : ks, 'robot_func' : robot_func,
                         # 'keys' : {'eval' : eval_kNps, 'cycle':cycle_ks, 'all':ks},
                         'target_kws': fitness_target_kws, 'target': fitness_target, 'robot_dict': robot_dict})