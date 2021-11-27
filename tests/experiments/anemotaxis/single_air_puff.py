import itertools

import nengo_gui
from lib.anal.argparsers import update_exp_conf
from lib.anal.plotting import plot_ethogram
from lib.conf.base.dtypes import null_dict
from lib.conf.stored.conf import expandConf
from lib.sim.single.single_run import SingleRun

video=False
analysis=True
mini=True
N=10
torque_coef=0.41
target='freq_space_search'
if mini:
    dur = 1.5
    start = 30
else:
    dur = 2.5
    start = 55

exp = 'single_puff'
d = {}
d['sim_params'] = null_dict('sim_params', Box2D=False, duration=dur)

w={'puffs': {'Puff': {'N': 1, 'duration': 30.0, 'start_time': start, 'speed': 100}}}
for id, args in w['puffs'].items():
    w['puffs'][id] = null_dict('air_puff', **args)

if video :
    vis_kwargs = null_dict('visualization', mode='video', video_speed=60)
else :
    vis_kwargs = null_dict('visualization', mode=None)

for wws in list(itertools.combinations_with_replacement([0, 5, 10],4)) :
    ws={
    'hunch_lin': wws[0],
    'hunch_ang': wws[1],
    'bend_lin': wws[2],
    'bend_ang': wws[3],
    }
    exp_conf = update_exp_conf(exp, d, N)
    exp_conf.larva_groups.Larva.model.brain.windsensor_params.weights=ws
    exp_conf.env_params.windscape = null_dict('windscape', **w)
    exp_conf.larva_groups.Larva.model.physics=null_dict('physics', torque_coef=torque_coef)
    run = SingleRun(**exp_conf, vis_kwargs=vis_kwargs)
    ds=run.run()
    if analysis :
        # plot_ethogram(save_to=f'./single_air_puff/{target}/{str(wws)}', datasets=ds, subfolder=None)
        # plot_ethogram(save_to=f'./single_air_puff/{exp_conf.sim_params.sim_ID}', datasets=ds)
        run.analyze(save_to=f'./single_air_puff/{target}/{str(wws)}', show=False)