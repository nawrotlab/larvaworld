
from lib import reg
from lib.model.envs.world_sim import WorldSim

N = 500
# mode = 'D'
mode='G'
if mode == 'D':
    media_name = 'diffusion_odorscape'
    odorscape = reg.get_null('odorscape', odorscape='Diffusion', grid_dims=(51, 51), gaussian_sigma=(0.95, 0.5),
                              evap_const=0.9)
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
elif mode == 'G':
    media_name = 'gaussian_odorscape'
    odorscape = reg.get_null('odorscape', odorscape='Gaussian')
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
sus = {
    'Source_L': reg.get_null('source', default_color='blue', group='Source', radius=0.003, amount=0.0, odor=oL,
                              pos=(-0.01, 0.0)),
    'Source_R': reg.get_null('source', default_color='cyan', group='Source', radius=0.003, amount=0.0, odor=oR,
                              pos=(0.01, 0.0)),
}
env_params = reg.get_null('Env',
                           food_params={'source_groups': {},
                                        'food_grid': None,
                                        'source_units': sus},
                           odorscape=odorscape)
env = WorldSim(env_params=env_params, Nsteps=N,
               vis_kwargs=reg.get_null('visualization', mode='video', video_speed=10, media_name=media_name))
env.odor_layers['Odor_R'].visible = True
env.run()
