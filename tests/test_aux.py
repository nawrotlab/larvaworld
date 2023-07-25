import numpy as np
import os

from larvaworld.lib import aux

def test_angular_funcs() :
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    a1=30
    a2=45

    x,y=aux.rotate_point_around_point(p1, np.pi / 2, pmid)
    assert  np.round(x,2),np.round(y,2)== p2