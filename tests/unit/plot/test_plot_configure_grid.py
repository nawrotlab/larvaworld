import itertools

import numpy as np

from larvaworld.lib import util
from larvaworld.lib.plot import configure_subplot_grid


def test_subplot_grid_configuration():
    vs = [None] + np.arange(20).tolist()
    for idx in itertools.product(vs, vs, vs):
        if None not in idx or 0 in idx:
            continue
        N, Ncols, Nrows = idx
        if N is not None:
            if (Ncols is not None and Ncols > N) or (Nrows is not None and Nrows > N):
                continue
        kws = util.AttrDict(configure_subplot_grid(N=N, Ncols=Ncols, Nrows=Nrows))
        if Ncols:
            assert kws.ncols == Ncols
        if Nrows:
            assert kws.nrows == Nrows
        if N:
            assert kws.ncols * kws.nrows >= N
            if Ncols is None:
                assert (kws.ncols - 1) * kws.nrows <= N
            if Nrows is None:
                assert (kws.nrows - 1) * kws.ncols <= N
