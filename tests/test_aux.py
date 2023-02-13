import numpy as np

from larvaworld.lib import aux

def test_angular_funcs() :
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    a1=30
    a2=45

    assert aux.angle_from_3points(p1, pmid, p2) == 90
    assert aux.angle_to_x_axis(p1, p2) == 90
    assert aux.angle_dif(a1, a2) == -15
    x,y=aux.rotate_point_around_point(p1, np.pi / 2, pmid)
    assert  np.round(x,2),np.round(y,2)== p2