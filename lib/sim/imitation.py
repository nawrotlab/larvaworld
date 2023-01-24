from lib import reg


def imitation_exp(sample, model='explorer', idx=0, N=None, duration=None, imitation=True, **kwargs):
    sample_conf = reg.loadConf(id=sample, conftype='Ref')

    base_larva = reg.expandConf(id=model, conftype='Model')
    if imitation:
        exp = 'imitation'
        larva_groups = {
            'ImitationGroup': reg.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                                        imitation=True,
                                        distribution={'N': N})}
    else:
        exp = 'evaluation'
        larva_groups = {
            sample: reg.get_null('LarvaGroup', sample=sample, model=base_larva, default_color='blue',
                              imitation=False,
                              distribution={'N': N})}
    id = sample_conf.id

    if duration is None:
        duration = sample_conf.duration / 60
    sim_params = reg.get_null('sim_params', timestep=1 / sample_conf['fr'], duration=duration)
    env_params = sample_conf.env_params
    exp_conf = reg.get_null('Exp', sim_params=sim_params, env_params=env_params, larva_groups=larva_groups,
                         trials={}, enrichment=reg.par.base_enrich())
    exp_conf['experiment'] = exp
    exp_conf.update(**kwargs)
    return exp_conf
