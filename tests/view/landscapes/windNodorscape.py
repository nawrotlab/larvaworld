import numpy as np

from larvaworld.lib import reg, aux
from larvaworld.lib.sim.single_run import ExpRun

exp='Wind&Odorscape visualization'



def get_conf(odor_mode, puff_mode, wind_mode):
    media_name = f'{odor_mode}_{puff_mode}_air-puffs_variable_wind_{wind_mode}'

    if odor_mode == 'diffusion_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Diffusion', grid_dims=(41, 41), gaussian_sigma=(0.95, 0.95),
                                  evap_const=0.9)
        oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
        oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
    elif odor_mode == 'gaussian_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Gaussian')
        oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
        oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
    else:
        raise ValueError('Not implemented')
    sus = {
        'Source_L': reg.get_null('source', default_color='blue', group='Source', radius=0.003, amount=0.0, odor=oL,
                                  pos=(0.0, 0.0)),
    }

    Npuffs = 100
    if puff_mode=='single' :
        puffs = {i: reg.get_null('air_puff', duration=2, speed=40, direction=i / Npuffs * 2 * np.pi, start_time=5 + 10 * i)
                 for i in range(Npuffs)}
        wind_speed = 0.0
    elif puff_mode == 'repetitive':
        puffs = {'puff_group': reg.get_null('air_puff', duration=2, speed=40, direction=np.pi, start_time=5, N=Npuffs,
                                             interval=5.0)}
        wind_speed = 0.0
    elif puff_mode == 'no':
        puffs = {}
        wind_speed = 30.0
    else:
        raise ValueError('Not implemented')

    windscape = reg.get_null('windscape', wind_direction=-np.pi / 2, wind_speed=wind_speed, puffs=puffs)

    conf = {'parameters': aux.AttrDict({
        'sim_params': reg.get_null('sim_params', duration=2.0),
        'env_params': reg.get_null('Env',
                               arena=reg.get_null('arena', shape='rectangular', dims=(0.3, 0.3)),
                               food_params={'source_groups': {},
                                            'food_grid': None,
                                            'source_units': sus},
                               odorscape=odorscape,
                               windscape=windscape),
        'larva_groups': {},
        'trials': {},
        'collections': None,
        'experiment': exp,
    }),
        'screen_kws': {
            'vis_kwargs': reg.get_null('visualization', mode='video', video_speed=10, media_name=media_name)},
        'save_to': '.'
    }
    return conf



for odor_mode in ['gaussian_odorscape', 'diffusion_odorscape'] :
    for puff_mode in ['single', 'repetitive', 'no'] :
        for wind_mode in ['direction', 'speed', 'no'] :

            conf = get_conf(odor_mode, puff_mode, wind_mode)
            env = ExpRun(**conf)
            env.setup(**env._setup_kwargs)
            env.odor_layers['Odor_L'].visible = True
            env.odor_aura = True
            env.windscape.visible = True
            while env.t <= env.Nsteps:
                if wind_mode == 'direction':
                    env.windscape.set_wind_direction((env.t / 10 / np.pi) % (2 * np.pi))
                if wind_mode == 'speed':
                    env.windscape.wind_speed = env.t % 100
                env.t += 1
                env.step()
            env.end()
