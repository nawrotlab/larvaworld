import numpy as np
import pytest

from larvaworld.lib import aux
from larvaworld.lib.param import OrientedPoint


def test_oriented_point() :
    P=OrientedPoint(pos=(2,2), orientation=np.pi/2)

    # Single point
    p00, p01=(-1,1), (1,1)
    assert P.translate(p00)== pytest.approx(p01)

    # List of points
    p10,p11=[(-1,1), (1,1)],[(1,1),(1,3)]
    assert P.translate(p10)== pytest.approx(p11)

    # 2D Array of points
    p10, p11 = np.array([(-1., 1.), (1., 1.)]), [(1, 1), (1, 3)]
    assert P.translate(p10) == pytest.approx(p11)