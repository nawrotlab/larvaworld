
# print(kConfDict('Ref'))
# raise
# refID = 'None.Tane_test'
from lib.registry.pars import preg

refID = 'exploration.3controls'
d = preg.loadRef(refID)
d.visualize(s0=None, e0=None, vis_kwargs=None, agent_ids=None, save_to=None, time_range=None,
            draw_Nsegs=None, env_params=None, track_point=None, dynamic_color=None, use_background=False,
            transposition=None, fix_point=None, fix_segment=None)
