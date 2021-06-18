from lib.stor.larva_dataset import LarvaDataset


def sim_enrichment(d: LarvaDataset, experiment):
    cc = {'show_output': False, 'is_last': False}
    d.build_dirs()

    # if experiment in ['growth']:
    #     pass

    if experiment in ['rovers_sitters']:
    # if experiment in ['growth', 'rovers_sitters']:
        d.deb_analysis(**cc)
    elif experiment == 'focus':
        d.angular_analysis(**cc)
        d.detect_turns(**cc)
    elif experiment == 'dispersion':
        d.enrich(length_and_centroid=False, **cc)
    elif experiment in ['chemotaxis_local', 'chemotaxis_diffusion']:
        d.enrich(length_and_centroid=False, source_location=(0.0, 0.0), **cc)
        # d.linear_analysis(is_last=False)
        # d.angular_analysis(is_last=False)
        # d.detect_strides(is_last=False)
        # d.detect_pauses(is_last=False)
        # d.detect_turns(is_last=False)
        # for chunk in ['turn', 'stride', 'pause']:
        #     d.compute_chunk_bearing2source(chunk=chunk, source=(0.0, 0.0), is_last=False)
    elif experiment in ['food_at_bottom']:
        d.linear_analysis(**cc)
        d.angular_analysis(**cc)
        d.detect_turns(**cc)
    print(f'    Dataset enriched!')
    return d