from lib.anal.plotting import plot_ethogram, timeplot, plot_2pars, plot_stridesNpauses
from lib.conf.stored.conf import loadRef, kConfDict

# print(kConfDict('Ref'))
refID='Puff.Starved'
d=loadRef(refID)
# d.load()
# s,e,c=d.step_data,d.endpoint_data,d.config
# c['source_xy']={}
# d.step_data,d.endpoint_data = annotate(s=s,e=e,config=c,stride=False,pause=True,turn=False,fits=True, recompute=True, is_last=True, min_dur=0.1)
# d.save()
# d.save_config(add_reference=True, refID=refID)
# raise
# plot_2pars(shorts=['foa', 'a'], datasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['foa', 'sa'], datasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fov', 'v'], datasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fov', 'sv'], datasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fo', 'fov'], datasetes=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fo', 'bv'], datasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fo', 'ba'], daetasets=[d], show=True, larva_legend=False)
# plot_2pars(shorts=['fo', 'foa'], datasets=[d], show=True, larva_legend=False)
# timeplot(par_shorts=['v'], datasets=[d], show=True)
# timeplot(par_shorts=['sv'], datasets=[d], show=True)
# timeplot(par_shorts=['a'], datasets=[d], show=True)
# timeplot(par_shorts=['sa'], datasets=[d], show=True)
# timeplot(par_shorts=['b'], datasets=[d], show=True, absolute=True)
# timeplot(par_shorts=['x'], datasets=[d], show=True)
# timeplot(par_shorts=['y'], datasets=[d], show=True)
# timeplot(par_shorts=['ba'], datasets=[d], show=True, absolute=True)
# timeplot(par_shorts=['fov'], datasets=[d], show=True, absolute=True)
# timeplot(par_shorts=['tor2'], datasets=[d], show=True)
# timeplot(par_shorts=['fov', 'foa'], datasets=[d], show=True, absolute=True)
# plot_ethogram(datasets=[d], show=True, add_samples=True)
plot_stridesNpauses(datasets=[d],plot_fits='best', show=True, range='broad')