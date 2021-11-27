import nengo_gui
from lib.anal.argparsers import update_exp_conf
from lib.conf.base.dtypes import null_dict
from lib.conf.stored.conf import expandConf
from lib.sim.single.single_run import SingleRun

use_nengo_gui=True
video=False

# mode='locomotion'
mode='anemotaxis'
# mode='chemotaxis'
# mode='feeding'
# mode='foraging'
# mode='odor_preference'
windsensor=False
if mode=='locomotion' :
    exp='nengo_dish'

    m='nengo_explorer'
elif mode=='anemotaxis' :
    exp='anemotaxis'
    windsensor=True
    m = 'nengo_explorer'
elif mode=='chemotaxis' :
    exp='chemotaxis'
    m = 'nengo_navigator'
elif mode=='feeding' :
    exp='food_grid'
    m = 'nengo_feeder'
elif mode=='foraging' :
    exp='patchy_food'
    m = 'nengo_forager'
elif mode=='odor_preference' :
    exp='PItest_off'
    m = 'nengo_navigator_x2'


N=1
d={}
d['sim_params']=null_dict('sim_params', Box2D=False)
exp_conf = update_exp_conf(exp, d, N)

exp_conf.larva_groups.Larva.model=expandConf(m, 'Model')

exp_conf.larva_groups.Larva.model.brain.modules.windsensor=windsensor
if video :
    vis_kwargs = null_dict('visualization', mode='video', video_speed=60)
else :
    vis_kwargs = null_dict('visualization', mode=None)
run = SingleRun(**exp_conf, vis_kwargs=vis_kwargs)
model=run.env.get_flies()[0].brain




if use_nengo_gui :
    nengo_gui.GUI(__file__).start()
else :
    ds=run.run()
    run.analyze(save_to='./test_run', show=False)