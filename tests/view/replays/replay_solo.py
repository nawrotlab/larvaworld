# refID = 'None.Tane_test'
from lib.registry.pars import preg

refID = 'exploration.50controls'
d = preg.loadRef(refID)
d.visualize_single(id=0, close_view=True, fix_point=6, fix_segment=-1, save_to='./media',
                         draw_Nsegs=None)
