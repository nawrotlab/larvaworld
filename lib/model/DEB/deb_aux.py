import math

import numpy as np
from numpy.lib import scimath
from scipy.optimize import minimize


def simplex(func, x0, args=()):
    res = minimize(func, x0, args=args, method='nelder-mead', options={'xatol': 1e-8, 'disp': False}).x[0]
    return res


def beta0(x0, x1):
    x03 = x0 ** (1 / 3)
    x13 = x1 ** (1 / 3)
    a3 = math.sqrt(3)

    f1 = - 3 * x13 + a3 * np.arctan((1 + 2 * x13) / a3) - scimath.log(x13 - 1) + scimath.log(1 + x13 + x13 ** 2) / 2
    f0 = - 3 * x03 + a3 * np.arctan((1 + 2 * x03) / a3) - scimath.log(x03 - 1) + scimath.log(1 + x03 + x03 ** 2) / 2
    f = f1 - f0
    return np.real(f)
    # return f