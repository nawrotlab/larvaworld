from larvaworld.lib import reg, aux, plot

import matplotlib.pyplot as plt


def test_plots() :
    refIDs=['AttP240.Fed', 'AttP240.Deprived', 'AttP240.Starved']
    figs={}
    ds=reg.loadRefDs(refIDs, h5_ks=['angular', 'midline', 'epochs'])
    kws={'datasets':ds, 'show':False, 'save_to': f'{reg.ROOT_DIR}/tests/plots',
         'subfolder': None}


    for k in ['boxplot (simple)', 'epochs', 'fft multi', 'dispersal summary',
              'kinematic analysis', 'angular pars', 'crawl pars', 'stride cycle',
              'stride cycle multi', 'ethogram', 'Y pos', 'dispersal', 'pathlength',
              'trajectories'] :
        figs[k] = reg.graphs.run(k, save_as=k,**kws)
        assert isinstance(figs[k], plt.Figure)
        plt.close(figs[k])
