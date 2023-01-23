
from lib import reg, aux
from lib.model.envs.world_runnable import Larvaworld

exp='Odorscape visualization'


def get_conf(media_name):
    if media_name == 'diffusion_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Diffusion', grid_dims=(51, 51), gaussian_sigma=(0.95, 0.5),
                                  evap_const=0.9)
        oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
        oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
    elif media_name == 'gaussian_odorscape':
        odorscape = reg.get_null('odorscape', odorscape='Gaussian')
        oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
        oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
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
        'sim_params': reg.get_null('sim_params', sim_ID=exp, duration=1.0, store_data=False),
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
    env = Larvaworld(**conf)
    env.setup(**env._setup_kwargs)
    env.odor_layers['Odor_R'].visible = True
    while env.t <= env.Nsteps:
        env.t += 1
        env.step()
    env.end()

