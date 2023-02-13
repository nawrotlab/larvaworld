import numpy as np

from larvaworld.lib import reg, aux
from larvaworld.lib.sim.single_run import ExpRun

exp='Odorscape visualization'


def oG(c=1, id='Odor'):
    return reg.get_null('odor', odor_id=id, odor_intensity=2.0 * c, odor_spread=0.0002 * np.sqrt(c))


def oD(c=1, id='Odor'):
    return reg.get_null('odor', odor_id=id, odor_intensity=300.0 * c, odor_spread=0.1 * np.sqrt(c))


def get_conf(media_name):
    if media_name == 'diffusion_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Diffusion', gaussian_sigma=(0.95, 0.5),evap_const=0.9)
        oR = oD(id='Odor_R')
        oL = oD(id='Odor_L')
        # oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
        # oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
    elif media_name == 'gaussian_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Gaussian')
        oR = oG(id='Odor_R')
        oL = oG(id='Odor_L')
        # oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
        # oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
    else :
        raise ValueError ('Not implemented')
    sus = {
        'Source_L': reg.get_null('source', default_color='blue', group='Source', radius=0.003, amount=0.0, odor=oL,
                                  pos=(-0.01, 0.0)),
        'Source_R': reg.get_null('source', default_color='cyan', group='Source', radius=0.003, amount=0.0, odor=oR,
                                  pos=(0.01, 0.0)),
    }

    conf= {
        'parameters' : aux.AttrDict({
        'sim_params': reg.get_null('sim_params',duration=1),
        'env_params': reg.get_null('Env',
                                   food_params={'source_groups': {},
                                                'food_grid': None,
                                                'source_units': sus},
                                   odorscape=odorscape),
        'larva_groups': {},
        'trials': {},
        'collections': None,
        'experiment': exp,
    }),
    'screen_kws' : {'vis_kwargs': reg.get_null('visualization', mode='video', video_speed=10, media_name=media_name)},
    'save_to' : '.'}

    return conf

for media_name in ['gaussian_odorscape', 'diffusion_odorscape'] :
    conf=get_conf(media_name)
    env = ExpRun(**conf)
    env.setup(**env._setup_kwargs)
    env.odor_layers['Odor_R'].visible = True
    while env.t <= env.Nsteps:
        env.t += 1
        env.step()
    env.end()

