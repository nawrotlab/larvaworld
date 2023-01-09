import numpy as np

from lib import aux

def test_angle() :
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    a1=30
    a2=45

    assert aux.angle_from_3points(p1, pmid, p2) == 90
    assert aux.angle_to_x_axis(p1, p2) == 90
    assert aux.angle_dif(a1, a2) == -15
    assert aux.rotate_point_around_point(p1, np.deg2rad(a2), pmid) == (-np.sqrt(2), 0)