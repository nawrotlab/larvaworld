import numpy as np

from lib import reg
from lib.model.envs.world_sim import WorldSim

test_direction = False
test_speed = False
test_single_puffs = False
test_repetitive_puffs = True

N = 1000
mode = 'D'
# mode='G'
if mode == 'D':
    odorscape = reg.get_null('odorscape', odorscape='Diffusion', grid_dims=(41, 41), gaussian_sigma=(0.95, 0.95),
                              evap_const=0.9)
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
    # oR = preg.oD(id='Odor_R')
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
elif mode == 'G':
    odorscape = reg.get_null('odorscape', odorscape='Gaussian')
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
sus = {
    'Source_L': reg.get_null('source', default_color='blue', group='Source', radius=0.003, amount=0.0, odor=oL,
                              pos=(0.0, 0.0)),
}

Npuffs = 100
if test_single_puffs:
    puffs = {i: reg.get_null('air_puff', duration=2, speed=40, direction=i / Npuffs * 2 * np.pi, start_time=5 + 10 * i)
             for i in range(Npuffs)}
    wind_speed = 0.0
elif test_repetitive_puffs:
    puffs = {'puff_group': reg.get_null('air_puff', duration=2, speed=40, direction=np.pi, start_time=5, N=Npuffs,
                                         interval=5.0)}
    wind_speed = 0.0
else:
    puffs = {}
    wind_speed = 30.0
windscape = reg.get_null('windscape', wind_direction=-np.pi / 2, wind_speed=wind_speed, puffs=puffs)
env_params = reg.get_null('Env',
                           arena=reg.get_null('arena', arena_shape='rectangular', arena_dims=(0.3, 0.3)),
                           food_params={'source_groups': {},
                                        'food_grid': None,
                                        'source_units': sus},
                           odorscape=odorscape,
                           windscape=windscape)
env = WorldSim(env_params=env_params, Nsteps=N,
               vis_kwargs=reg.get_null('visualization', mode='video', video_speed=5, media_name='windNodorscape'))
env.odor_layers['Odor_L'].visible = True
env.odor_aura = True
env.windscape.visible = True
env.is_running = True
while env.is_running and env.Nticks < env.Nsteps:
    if test_direction:
        env.windscape.set_wind_direction((env.Nticks / 10 / np.pi) % (2 * np.pi))
    if test_speed:
        env.windscape.wind_speed = env.Nticks % 100
    env.step()
    env.progress_bar.update(env.Nticks)
    env.screen_manager.step(env.Nticks)
