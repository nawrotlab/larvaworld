import os

from larvaworld.lib import reg, aux, plot

import matplotlib.pyplot as plt


def test_plots() :

    gIDs=reg.conf.Ref.RefGroupIDs
    gID = gIDs[1]
    dcol=reg.loadRefGroup(gID, load=True, h5_ks=['angular', 'midline', 'epochs'])
    assert dcol.dir is not None
    assert os.path.exists(dcol.dir)
    print(dcol.labels)
    # raise
    # refIDs=['AttP240.Fed', 'AttP240.Deprived', 'AttP240.Starved']
    figs={}
    # ds=[reg.loadRef(id, load=True, h5_ks=['angular', 'midline', 'epochs']) for id in refIDs]
    kws={'save_to': f'{reg.ROOT_DIR}/tests/plots', 'show':False}


    graphIDs= ['endpoint box', 'epochs', 'fft multi','powerspectrum', 'dispersal summary',
              'kinematic analysis', 'angular pars', 'crawl pars', 'stride cycle',
              'stride cycle multi', 'ethogram', 'dispersal', 'pathlength',
              'trajectories']

    figs = dcol.plot(ids=graphIDs[3:4],**kws)
    for k in graphIDs:
        assert isinstance(figs[k], plt.Figure)
        plt.close(figs[k])
