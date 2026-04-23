import numpy as np
import panel as pn
import pytest

from larvaworld.lib.param import NestedConf, OrientedPoint, TrackerOps


def test_oriented_point():
    P = OrientedPoint(pos=(2, 2), orientation=np.pi / 2)

    # Single point
    p00, p01 = (-1, 1), (1, 1)
    assert P.translate(p00) == pytest.approx(p01)

    # List of points
    p10, p11 = [(-1, 1), (1, 1)], [(1, 1), (1, 3)]
    assert P.translate(p10) == pytest.approx(p11)

    # 2D Array of points
    p10, p11 = np.array([(-1.0, 1.0), (1.0, 1.0)]), [(1, 1), (1, 3)]
    assert P.translate(p10) == pytest.approx(p11)


def test_tracker_ops_clamps_vectors_when_npoints_shrinks() -> None:
    tracker = TrackerOps(Npoints=11, front_vector=(1, 6), rear_vector=(-6, -1))

    tracker.Npoints = 5

    assert tracker.front_vector == (1, 5)
    assert tracker.rear_vector == (-5, -1)


def test_tracker_ops_normalizes_legacy_positive_rear_vector_on_init() -> None:
    tracker = TrackerOps(Npoints=5, front_vector=(1, 3), rear_vector=(3, 5))

    assert tracker.rear_vector == (-2, -1)


def test_tracker_ops_vector_softbounds_follow_npoints() -> None:
    tracker = TrackerOps(Npoints=11, front_vector=(1, 6), rear_vector=(-6, -1))

    assert tracker.param["front_vector"].bounds == (1, 11)
    assert tracker.param["front_vector"].softbounds == (1, 11)
    assert tracker.param["rear_vector"].bounds == (-11, -1)
    assert tracker.param["rear_vector"].softbounds == (-11, -1)

    tracker.Npoints = 5

    assert tracker.param["front_vector"].bounds == (1, 5)
    assert tracker.param["front_vector"].softbounds == (1, 5)
    assert tracker.param["rear_vector"].bounds == (-5, -1)
    assert tracker.param["rear_vector"].softbounds == (-5, -1)


def test_tracker_ops_vector_widgets_follow_npoints_exactly() -> None:
    tracker = TrackerOps(Npoints=11, front_vector=(1, 6), rear_vector=(-6, -1))
    pane = pn.Param(tracker, parameters=["Npoints", "front_vector", "rear_vector"])
    widgets = {widget.name: widget for widget in pane.select(pn.widgets.Widget)}

    assert widgets["Front vector"].start == 1
    assert widgets["Front vector"].end == 11
    assert widgets["Rear vector"].start == -11
    assert widgets["Rear vector"].end == -1

    tracker.Npoints = 5

    assert widgets["Front vector"].start == 1
    assert widgets["Front vector"].end == 5
    assert widgets["Rear vector"].start == -5
    assert widgets["Rear vector"].end == -1


def test_tracker_ops_recovers_from_non_positive_framerate_inputs() -> None:
    tracker = TrackerOps(fr=20)

    tracker.fr = 0
    assert tracker.fr == pytest.approx(20)
    assert tracker.dt == pytest.approx(0.05)

    tracker.dt = 0
    assert tracker.dt == pytest.approx(0.05)
    assert tracker.fr == pytest.approx(20)


def test_tracker_ops_uses_centroid_when_npoints_drops_to_zero() -> None:
    tracker = TrackerOps(Npoints=3, point_idx=2)

    tracker.Npoints = 0

    assert tracker.point_idx == -1
    assert tracker.point == "centroid"
    assert tracker.front_vector is None
    assert tracker.rear_vector is None


def ttest_param_keys():
    import inspect
    import pkgutil
    from types import ModuleType

    import larvaworld.lib

    fails = []

    print()

    def test_module(m):
        for k in m.__all__:
            p = getattr(m, k)
            if inspect.isclass(p):
                if issubclass(p, NestedConf):
                    try:
                        print(p.name)
                        assert p().param_keys.sorted == p().nestedConf.keylist
                    except:
                        print(p.name, "MOD_FAIL")
                        fails.append(p.name)
            elif isinstance(p, ModuleType):
                try:
                    get_modules(p)
                except:
                    print(p, "DIR_FAIL")

    def get_modules(m0):
        print()
        modualList = []
        for importer, modname, ispkg in pkgutil.iter_modules(m0.__path__):
            print(f"Testing module {modname}")
            print()
            m = getattr(m0, modname)
            modualList.append(m)
            if hasattr(m, "__all__"):
                test_module(m)
            elif isinstance(m, ModuleType):
                try:
                    get_modules(m)
                except:
                    print(m, "DIR_FAIL")
            print()

    get_modules(larvaworld.lib)
    print("_____FAILS_____")
    print()
    print(fails)
