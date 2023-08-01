import numpy as np
import os

from larvaworld.lib import aux

def test_angular_funcs() :
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    p3 = (1, 1)
    a1=30
    a2=45

    pps=aux.rotate_points_around_point([p1,p2], np.pi / 2, pmid)
    assert  np.round(pps[0][0])==p2[0]
    assert  np.round(pps[1][0])==p3[0]