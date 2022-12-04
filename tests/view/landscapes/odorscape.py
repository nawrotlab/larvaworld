
from lib.registry import reg
from lib.model.envs.world_sim import WorldSim

N = 500
mode = 'D'
# mode='G'
# odorscape=null_dict('odorscape')
if mode == 'D':
    media_name = 'diffusion_odorscape'
    odorscape = reg.get_null('odorscape', odorscape='Diffusion', grid_dims=(51, 51), gaussian_sigma=(0.95, 0.5),
                              evap_const=0.9)
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=300.0, odor_spread=0.1)
    # oR = preg.oD(id='Odor_R')
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=300.0, odor_spread=0.1)
    # oL = preg.oD(id='Odor_L')
    # env_params = null_dict('env_conf', food_params=f_pars(su=su(pos=(0.0, 0.0), o=oD())), odorscape=odorscape)
elif mode == 'G':
    media_name = 'gaussian_odorscape'
    odorscape = reg.get_null('odorscape', odorscape='Gaussian')
    oR = reg.get_null('odor', odor_id='Odor_R', odor_intensity=2.0, odor_spread=0.0002)
    # oR = preg.oG(id='Odor_R')
    oL = reg.get_null('odor', odor_id='Odor_L', odor_intensity=2.0, odor_spread=0.0002)
    # oL = preg.oG(id='Odor_L')
sus = {
    'Source_L': reg.get_null('source', default_color='blue', group='Source', radius=0.003, amount=0.0, odor=oL,
                              pos=(-0.01, 0.0)),
    'Source_R': reg.get_null('source', default_color='cyan', group='Source', radius=0.003, amount=0.0, odor=oR,
                              pos=(0.01, 0.0)),
    # **su(id='Source_R', pos=(0.01, 0.0), o=oR, c='cyan'),
    # **su(id='Source_L', pos=(-0.01, 0.0), o=oL, c='blue'),
}
env_params = reg.get_null('env_conf',
                           food_params={'source_groups': {},
                                        'food_grid': None,
                                        'source_units': sus},
                           # food_params=f_pars(su=sus),
                           odorscape=odorscape)
# env_params=null_dict('env_conf', odorscape=odorscape, food_params=f_pars(su=su(pos=(0.0, 0.0), o=oG(2, id='Odor'))))
# env_params=null_dict('env_conf', windscape=windscape, border_list={'Border' : null_dict('Border', points=[(-0.03,0.02), (0.03,0.02)])})
env = WorldSim(env_params=env_params, Nsteps=N,
               vis_kwargs=reg.get_null('visualization', mode='video', video_speed=10, media_name=media_name))
env.odor_layers['Odor_R'].visible = True
# env.windscape.visible=True
env.is_running = True
while env.is_running and env.Nticks < env.Nsteps:
    # if test_direction :
    #     env.windscape.set_wind_direction((env.Nticks/10/np.pi)%(2*np.pi))
    # if test_speed :
    #     env.windscape.wind_speed=env.Nticks%100
    env.step()
    env.progress_bar.update(env.Nticks)
    env.render(env.Nticks)
