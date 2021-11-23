import numpy as np
from lib.conf.base.dtypes import null_dict
from lib.model.envs._larvaworld import LarvaWorld
from lib.model.envs._larvaworld_sim import LarvaWorldSim

test_direction=False
test_speed=False
test_single_puffs=False
test_repetitive_puffs=True
# test_mode='direction'

N=1000
Npuffs=10
if test_single_puffs :
    puffs={i:null_dict('air_puff', duration=5, speed=50, direction=i/Npuffs*2*np.pi, start_time=5+10*i) for i in range(Npuffs)}
    wind_speed = 0.0
elif test_repetitive_puffs :
    puffs= {'puff_group':null_dict('air_puff', duration=5, speed=50, direction=np.pi, start_time=5, N=Npuffs, interval=10.0)}
    wind_speed = 0.0
else :
    puffs={}
    wind_speed = 10.0
windscape=null_dict('windscape', wind_direction=0.0, wind_speed=wind_speed, puffs=puffs)
env_params=null_dict('env_conf', windscape=windscape, border_list={'Border' : null_dict('Border', points=[(-0.03,0.02), (0.03,0.02)])})
env=LarvaWorldSim(env_params=env_params, Nsteps=N, vis_kwargs=null_dict('visualization', mode='video', video_speed=10, media_name='windscape'))

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
