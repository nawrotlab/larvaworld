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


