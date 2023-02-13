import numpy as np

from larvaworld.lib import reg, aux
from larvaworld.lib.sim.single_run import ExpRun

exp='Windscape visualization'

def get_conf(puff_mode, wind_mode):
    media_name = f'{puff_mode}_air-puffs_variable_wind_{wind_mode}'

    Npuffs=10
    if puff_mode=='single' :
        puffs={i:reg.get_null('air_puff', duration=5, speed=50, direction=i/Npuffs*2*np.pi, start_time=5+10*i) for i in range(Npuffs)}
        wind_speed = 0.0
    elif puff_mode=='repetitive' :
        puffs= {'puff_group':reg.get_null('air_puff', duration=5, speed=50, direction=np.pi/4, start_time=5, N=Npuffs, interval=10.0)}
        wind_speed = 0.0
    elif puff_mode == 'no':
        puffs={}
        wind_speed = 10.0
    else :
        raise ValueError ('Not implemented')
    windscape=reg.get_null('windscape', wind_direction=0.0, wind_speed=wind_speed, puffs=puffs)

    conf= {'parameters' : aux.AttrDict({
        'sim_params': reg.get_null('sim_params', duration=2.0),
        'env_params': reg.get_null('Env', windscape=windscape,
                                   border_list={'Border': reg.get_null('Border', points=[(-0.03, 0.02), (0.03, 0.02)])}),
        'larva_groups': {},
        'trials': {},
        'collections': None,
        'experiment': exp,
    }),
        'screen_kws': {'vis_kwargs': reg.get_null('visualization', mode='video', video_speed=10, media_name=media_name)},
        'save_to': '.'
    }
    return conf


for puff_mode in ['single', 'repetitive', 'no'] :
    for wind_mode in ['direction', 'speed', 'no'] :
        conf=get_conf(puff_mode, wind_mode)
        env = ExpRun(**conf)
        env.setup(**env._setup_kwargs)
        env.windscape.visible=True

        while env.t <= env.Nsteps:
            if wind_mode=='direction':
                env.windscape.set_wind_direction((env.t/10/np.pi)%(2*np.pi))
            if wind_mode=='speed' :
                env.windscape.wind_speed=env.t%100
            env.t += 1
            env.step()
        env.end()


