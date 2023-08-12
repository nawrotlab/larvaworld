import os
import matplotlib.pyplot as plt

from larvaworld.lib import reg, aux, plot




def test_plots() :
    gIDs=reg.conf.Ref.RefGroupIDs
    gID = gIDs[1]
    dcol=reg.loadRefGroup(gID)
    assert dcol.dir is not None
    assert os.path.exists(dcol.dir)
    kws={'save_to': f'{reg.ROOT_DIR}/../../tests/plots', 'show':False}


    graphIDs= ['endpoint box', 'epochs', 'fft multi', 'dispersal summary',
              'kinematic analysis', 'angular pars', 'crawl pars', 'stride cycle',
              'stride cycle multi', 'ethogram', 'dispersal', 'pathlength',
              'trajectories'][2:3]

    figs = dcol.plot(ids=graphIDs,**kws)
    for k in graphIDs:
        assert isinstance(figs[k], plt.Figure)
        plt.close(figs[k])
