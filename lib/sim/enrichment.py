from lib.stor.larva_dataset import LarvaDataset
from lib.stor import paths


def sim_enrichment(d: LarvaDataset, experiment):
    cc = {'show_output': False, 'is_last': False}
    if experiment in ['rovers_sitters']:
        pass
    elif experiment == 'dish':
        d.enrich( **cc)
    elif experiment == 'focus':
        d.process(types=['angular'], **cc)
        d.detect_bouts(bouts=['turn'], **cc)
    elif experiment == 'dispersion':
        d.enrich(**cc)
    # elif experiment == 'odor_pref_test':
    #     d.enrich(**cc)
    elif experiment in ['chemotaxis_local', 'chemotaxis_diffusion']:
        d.enrich(source=(0.0, 0.0), **cc)
    elif experiment in ['chemotaxis_approach']:
        d.enrich(source=(0.04, 0.0), **cc)
    elif experiment in ['food_at_bottom', 'food_grid']:
        d.enrich(source=(0.0, 0.0), **cc)
    print(f'    Dataset enriched!')
    return d