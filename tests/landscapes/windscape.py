import numpy as np
from lib.conf.base.dtypes import null_dict
from lib.model.envs._larvaworld import LarvaWorld

test_direction=False
test_speed=False
test_puffs=True
# test_mode='direction'

N=1000
windscape=null_dict('windscape', wind_direction=0.0, wind_speed=0.0)
env_params=null_dict('env_conf', windscape=windscape, border_list={'Border' : null_dict('Border', points=[(-0.03,0.02), (0.03,0.02)])})
env=LarvaWorld(env_params=env_params, Nsteps=N, vis_kwargs=null_dict('visualization', mode='video', video_speed=1, media_name='windscape'))
if test_puffs :
    env.windscape.add_puff(duration=5, speed=50, direction=np.pi, start_time=10)
    env.windscape.add_puff(duration=5, speed=50, direction=np.pi/2, start_time=20)
    env.windscape.add_puff(duration=5, speed=50, direction=-np.pi, start_time=30)
    env.windscape.add_puff(duration=5, speed=50, direction=3/2*np.pi, start_time=40)
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
