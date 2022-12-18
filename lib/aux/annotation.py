from lib.aux.fitting import fit_epochs, get_bout_distros
from lib.aux import dictsNlists as dNl, naming as nam


def register_bout_distros(c,e):
    from lib.model.modules.intermitter import get_EEB_poly1d
    try:
        c['intermitter'] = {
            nam.freq('crawl'): e[nam.freq(nam.scal(nam.vel('')))].mean(),
            nam.freq('feed'): e[nam.freq('feed')].mean() if nam.freq('feed') in e.columns else 2.0,
            'dt': c.dt,
            'crawl_bouts': True,
            'feed_bouts': True,
            'stridechain_dist': c.bout_distros.run_count,
            'pause_dist': c.bout_distros.pause_dur,
            'run_dist': c.bout_distros.run_dur,
            'feeder_reoccurence_rate': None,
        }
        c['EEB_poly1d'] = get_EEB_poly1d(**c['intermitter']).c.tolist()
    except :
        pass




def annotate(d, interference=True, on_food=True, store=True, **kwargs) :
    from lib.process import aux, patch
    s, e ,c= d.step_data, d.endpoint_data,d.config

    d.chunk_dicts = aux.comp_chunk_dicts(s, e, c, store=store)

    aux.turn_mode_annotation(e, d.chunk_dicts)
    patch.comp_patch(s, e, c)
    if interference:
        d.cycle_curves = aux.compute_interference(s=s, e=e, c=c, chunk_dicts=d.chunk_dicts)
        d.grouped_epochs = dNl.group_epoch_dicts(d.chunk_dicts)
        d.pooled_epochs = fit_epochs(d.grouped_epochs)
        c.bout_distros = get_bout_distros(d.pooled_epochs)
        register_bout_distros(c, e)
    if on_food:
        patch.comp_on_food(s, e, c)