from lib.stor.larva_dataset import LarvaDataset
from lib.stor import paths


def sim_enrichment(d: LarvaDataset, experiment):
    cc = {'show_output': False, 'is_last': False}
    if experiment in ['rovers_sitters']:
        pass
    elif experiment == 'dish':
        if not paths.new_format :
            d.process(**cc)
        else :
            d.process(types=['angular'],**cc)
        d.detect_bouts(**cc)
        # print(p for p in d.endpoint_data.columns if p.startswith('t'))
    elif experiment == 'focus':
        d.process(types=['angular'], **cc)
        d.detect_bouts(bouts=['turn'], **cc)
    elif experiment == 'dispersion':
        d.enrich(**cc)
    elif experiment in ['chemotaxis_local', 'chemotaxis_diffusion']:
        if not paths.new_format :
            d.enrich(source=(0.0, 0.0), **cc)
        else :
            d.detect_bouts(**cc)
    elif experiment in ['chemotaxis_approach']:
        if not paths.new_format:
            d.enrich(source=(0.04, 0.0), **cc)
        else :
            d.detect_bouts(**cc)
    elif experiment in ['food_at_bottom']:
        if not paths.new_format:
            d.process(**cc)
        d.detect_bouts(bouts=['turn'], **cc)
    print(f'    Dataset enriched!')
    return d