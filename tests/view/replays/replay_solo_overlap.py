from lib.registry.pars import preg

refID = 'exploration.dish_0'
d = preg.loadRef(refID)
d.visualize_single(id=5, close_view=True, fix_point=6, fix_segment=-1, save_to='./media',
                         draw_Nsegs=None, vis_kwargs=preg.get_null('visualization', mode='image', image_mode='overlap', media_name='overlap',
                                                               draw_contour=False))
