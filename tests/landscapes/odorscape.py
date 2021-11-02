import numpy as np
from lib.conf.base.dtypes import null_dict, oG, oD
from lib.conf.stored.env_conf import f_pars, su
from lib.model.envs._larvaworld import LarvaWorld
from lib.model.envs._larvaworld_sim import LarvaWorldSim

test_direction=True
test_speed=False
test_single_puffs=False
test_repetitive_puffs=False
# test_mode='direction'

N=1000
mode='D'
# mode='G'
# odorscape=null_dict('odorscape')
if mode=='D' :
    o = null_dict('odorscape', odorscape='Diffusion', grid_dims=(51,51), gaussian_sigma=(0.95,0.95), evap_const=0.9)
    env_params = null_dict('env_conf', food_params=f_pars(su=su(pos=(0.0, 0.0), o=oD())), odorscape=o)
elif mode== 'G' :
    o = null_dict('odorscape', odorscape='Gaussian')
    env_params=null_dict('env_conf', food_params=f_pars(su=su(pos=(0.0, 0.0), o=oG())), odorscape=o)
# env_params=null_dict('env_conf', odorscape=odorscape, food_params=f_pars(su=su(pos=(0.0, 0.0), o=oG(2, id='Odor'))))
# env_params=null_dict('env_conf', windscape=windscape, border_list={'Border' : null_dict('Border', points=[(-0.03,0.02), (0.03,0.02)])})
env=LarvaWorldSim(env_params=env_params, Nsteps=N, vis_kwargs=null_dict('visualization', mode='video', video_speed=10, media_name='odorscape'))
env.odor_layers['Odor'].visible=True
# env.windscape.visible=True
env.is_running=True
while env.is_running and env.Nticks < env.Nsteps:
    # if test_direction :
    #     env.windscape.set_wind_direction((env.Nticks/10/np.pi)%(2*np.pi))
    # if test_speed :
    #     env.windscape.wind_speed=env.Nticks%100
    env.step()
    env.progress_bar.update(env.Nticks)
    env.render(env.Nticks)
