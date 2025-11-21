import numpy as np
import pytest

from larvaworld.lib.param import NestedConf, OrientedPoint


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
