import numpy as np

from lib import aux


def test_angles_between_vectors():
    """
    Test function for the angles_between_vectors function
    """

    # Angle relative to x-axis
    # Test case 1
    xy_front = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    xy_mid = None
    xy_rear = None
    a = aux.angles_between_vectors(xy_front, xy_mid, xy_rear)
    assert np.allclose(a, [0, 90, 180, -90], rtol=1e-9, atol=1e-9)

    # Test case 2
    xy_front = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    xy_mid = np.array([[1, 0]] * 4)
    xy_rear = None

    a = aux.angles_between_vectors(xy_front, xy_mid, xy_rear)
    assert np.allclose(a, [0, 135, 180, -135], rtol=1e-9, atol=1e-9)

    # Angle relative to rear vector

    # Test case 1: test with vectors in all directions
    xy_front = np.array([[1, 0], [0, 1], [-1, 0], [0, 1]])
    xy_mid = None
    xy_rear = np.array([[0, 1], [-1, 0], [-1, -1], [1, -1]])
    a = aux.angles_between_vectors(xy_front, xy_mid, xy_rear)
    expected_angles = np.array([90, 90, 135, -45])
    assert np.allclose(a, expected_angles), "Test case 1 failed: angles are not calculated correctly."

    # Test case 2: test with changing rotation point
    N = 5
    xy_front = np.array([[1, 0]] * N)
    xy_rear = np.array([[-1, 0]] * N)
    xy_mid = np.array([np.zeros(N), np.linspace(-1, 1, N)]).T
    a = aux.angles_between_vectors(xy_front, xy_mid, xy_rear)
    expected_angles = np.array([90, 0.29516724 * 180, 0., -0.29516724 * 180, -90])
    assert np.allclose(a, expected_angles), "Test case 1 failed: angles are not calculated correctly."


def test_angular_funcs() :
    p1 = (-1, -1)
    pmid = (0, 0)
    p2 = (-1, 1)
    a1=30
    a2=45
    assert aux.angle_dif(a1, a2) == -15
    x,y=aux.rotate_point_around_point(p1, np.pi / 2, pmid)
    assert  np.round(x,2),np.round(y,2)== p2


# This function is used to test the "wrap_angle_to_0" function

def test_wrap_angle_to_0():
    # Test case 1: input angle is 0
    assert aux.wrap_angle_to_0(0) == 0

    # Test case 2: input angle is less than 0
    assert aux.wrap_angle_to_0(-45, True) == -45

    # Test case 3: input angle is smaller than -180
    assert aux.wrap_angle_to_0(-320, True) == 40

    # Test case 4: input angle is greater than 180
    assert aux.wrap_angle_to_0(270, True) == -90