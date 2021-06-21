from lib.stor.larva_dataset import LarvaDataset


def sim_enrichment(d: LarvaDataset, experiment):
    cc = {'show_output': False, 'is_last': False}
    if experiment in ['rovers_sitters']:
        pass
    elif experiment == 'dish':
        d.preprocess(**cc)
        d.process(**cc)
        d.detect_bouts(**cc)
    elif experiment == 'focus':
        d.process(types=['angular'], **cc)
        d.detect_bouts(bouts=['turn'], **cc)
    elif experiment == 'dispersion':
        d.enrich(**cc)
    elif experiment in ['chemotaxis_local', 'chemotaxis_diffusion']:
        d.enrich(source=(0.0, 0.0), **cc)
    elif experiment in ['chemotaxis_approach']:
        d.enrich(source=(0.04, 0.0), **cc)
    elif experiment in ['food_at_bottom']:
        d.process(**cc)
        d.detect_bouts(bouts=['turn'], **cc)
    print(f'    Dataset enriched!')
    return d