


def confReset_funcs(k):
    from lib.conf.stored import aux_conf, data_conf, batch_conf, exp_conf, env_conf, essay_conf, ga_conf, larva_conf
    from lib.aux import naming as nam, dictsNlists as dNl
    d = dNl.NestDict({
        'Ref': data_conf.Ref_dict,
        'Model': larva_conf.Model_dict,
        'ModelGroup': larva_conf.ModelGroup_dict,
        'Env': env_conf.Env_dict,
        'Exp': exp_conf.Exp_dict,
        'ExpGroup': exp_conf.ExpGroup_dict,
        'Essay': essay_conf.Essay_dict,
        'Batch': batch_conf.Batch_dict,
        'Ga': ga_conf.Ga_dict,
        'Tracker': data_conf.Tracker_dict,
        'Group': data_conf.Group_dict,
        'Trial': aux_conf.Trial_dict,
        'Life': aux_conf.Life_dict,
        'Body': aux_conf.Body_dict
    })
    return d[k]


if __name__ == '__main__':


    from lib.aux.data_aux import update_mdict
    import lib.aux.dictsNlists as dNl
    from lib.registry.pars import preg
    M=preg.larva_conf_dict
    from lib.registry.par_tree import tree_dict
    CT=preg.conftype_dict
    a=CT.dict['Ga']
    m=a.mdict
    g={}
    g[0] =a.gConf(**{})

    g[1]=a.expandConf(conf=g[0])

    md=a.expand_mdict(a.mdict)

    g[2]=a.gConf(m0=md)
    g[3] = a.expandConf(conf=g[2])

    g[4]=preg.get_null('GAconf')

    g[5]=a.expandConf(conf=g[4])



    for i,gg in g.items():
        print(i , gg.ga_build_kws)
    # dNl.dicprint(g)

    # tree=tree_dict(g)
    # print(tree)

    raise
#     from lib.conf.stored.conf import kConfDict
#
#     print(preg.get_null('trials'))
#     print('''cccc''')
#     raise
#     # a=kConfDict('Model')
#     # b=CT.dict['Model'].ConfIDs
#     # print(len(a), len(b))
#     # raise
#     # print(CT.dict['Model'].ConfIDs)
#
#     # m=preg.larva_conf_dict.loadConf('RE_NEU_PHI_DEF_nav')
#     # print(m)
#     # raise


    # g1 = a.gConf()
#     # a.mdict = a.expand_mdict(a.mdict)
#     # g2 = a.gConf()
#     # dNl.dicsprint([g1, g2])
#     #
#     # print(g1.env_params)
#     # print(g2.env_params)
#     # raise
    kws={}
#     # kws={'ga_build_kws' :preg.get_null('ga_build_kws',space_mkeys=['turner', 'crawler'])}
#     # kws={'ga_build_kws.space_mkeys':['turner', 'crawler']}
#


    # print(a.mdict.larva_groups.v)
    # print(a.mdict.trials.v)
    # print(a.mdict.env_params.keys())
    # raise
    m=preg.get_null('exp_conf')
    # m=a.expandConf(conf=m)

    # m=a.gConf(**kws)
    print(m)
    raise

#

    gs=[m,mm]
    for g in gs:

        print(g.env_params.arena)
        print(g.larva_groups)
#     print(mm.env_params.arena)
#     # dNl.dicsprint([m,mm])
#     #
#     #
    raise
#
#     from lib.sim.ga.ga_launcher import GAlauncher
#     GA=GAlauncher(**m)
#     best_genome = GA.run()
#     #
#     # GA=GAlauncher(**mm)
#     # best_genome = GA.run()
#
#
#     raise
#     fmm=dNl.flatten_dict(mm)
#
#     fm=dNl.flatten_dict(m)
#     fmdict = dNl.flatten_dict(a.mdict)
#     d=a.loadDict()
#     eval = {}
#     for id, conf in d.items():
#         # eval[id] = update_mdict(a.mdict, conf)
#         print(id,'.......................')
#         fconf=dNl.flatten_dict(conf)
#         # feval=dNl.flatten_dict(eval[id])
#
#         for d, p in fconf.items():
#
#
#             try:
#
#                 if p==fm[d] and p==fmm[d] :
#                     pass
#                 else:
#                     print()
#                     print('........', d)
#                     print(p, fm[d], fmm[d])
#             except:
#                 pass
#
#         # break
#     # print(eval)
#
#     # m=M.generate_configuration(a.mdict)
#     #
#     # dNl.dicprint(m)
#     # print(m)
#     # print(a.eval)
#     # dd = a.reset_func()
#     # d = a.loadDict()
#     # #
#     # N0, N1 = len(d), len(dd)
#     # print(N0, N1)
#     # d.update(dd)
#     # #
#     # Ncur = len(d)
#     # Nnew = Ncur - N0
#     # Nup = N1 - Nnew
#     # print(f'{a.k}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing')
#     # print(a.ConfID_entry(default='exploration'))
#     # raise
#     #
#     # print(confReset_funcs(k='Ga'))