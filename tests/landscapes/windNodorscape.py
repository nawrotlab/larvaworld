import numpy as np
from lib.conf.base.dtypes import null_dict, oG, oD, arena
from lib.conf.stored.env_conf import f_pars, su
from lib.model.envs._larvaworld import LarvaWorld
from lib.model.envs._larvaworld_sim import LarvaWorldSim

test_direction=True
test_speed=False
test_single_puffs=False
test_repetitive_puffs=False

N=1000
mode='D'
# mode='G'
# odorscape=null_dict('odorscape')
if mode=='D' :
    odorscape = null_dict('odorscape', odorscape='Diffusion', grid_dims=(41,41), gaussian_sigma=(0.95,0.95), evap_const=0.9)
    oR = oD(id='Odor_R')
    oL = oD(id='Odor_L')
    # env_params = null_dict('env_conf', food_params=f_pars(su=su(pos=(0.0, 0.0), o=oD())), odorscape=odorscape)
elif mode== 'G' :
    odorscape = null_dict('odorscape', odorscape='Gaussian')
    oR = oG(id='Odor_R')
    oL = oG(id='Odor_L')
sus={
    # **su(id='Source_R', pos=(0.08, -0.12), o=oR, c='cyan'),
    **su(id='Source_L', pos=(-0.1, 0.05), o=oL, c='blue'),
     }

Npuffs=100
if test_single_puffs :
    puffs={i:null_dict('air_puff', duration=2, speed=40, direction=i/Npuffs*2*np.pi, start_time=5+10*i) for i in range(Npuffs)}
    wind_speed = 0.0
elif test_repetitive_puffs :
    puffs= {'puff_group':null_dict('air_puff', duration=2, speed=40, direction=np.pi, start_time=5, N=Npuffs, interval=5.0)}
    wind_speed = 0.0
else :
    puffs={}
    wind_speed = 30.0
windscape=null_dict('windscape', wind_direction=-np.pi/2, wind_speed=wind_speed, puffs=puffs)
env_params=null_dict('env_conf',arena=arena(0.3,0.3), food_params=f_pars(su=sus), odorscape=odorscape, windscape=windscape)
# env_params=null_dict('env_conf', odorscape=odorscape, food_params=f_pars(su=su(pos=(0.0, 0.0), o=oG(2, id='Odor'))))
# env_params=null_dict('env_conf', windscape=windscape, border_list={'Border' : null_dict('Border', points=[(-0.03,0.02), (0.03,0.02)])})
env=LarvaWorldSim(env_params=env_params, Nsteps=N, vis_kwargs=null_dict('visualization', mode='video', video_speed=60, media_name='odorscape'))
env.odor_layers['Odor_L'].visible=True
env.odor_aura=True
env.windscape.visible=True
env.is_running=True
while env.is_running and env.Nticks < env.Nsteps:
    if test_direction :
        env.windscape.set_wind_direction((env.Nticks/10/np.pi)%(2*np.pi))
    if test_speed :
        env.windscape.wind_speed=env.Nticks%100
    env.step()
    env.progress_bar.update(env.Nticks)
    env.render(env.Nticks)
