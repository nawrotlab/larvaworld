import math

import numpy as np
from numpy.lib import scimath
from scipy.optimize import minimize


def simplex(func, x0, args=()):
    res = minimize(func, x0, args=args, method='nelder-mead', options={'xatol': 1e-8, 'disp': False}).x[0]
    return res


def beta0(x0, x1):
    '''
    Beta function used in the DEB textbook (p.58)
    Args:
        x0:float
        x1:float

    Returns:float

    '''
    x03 = x0 ** (1 / 3)
    x13 = x1 ** (1 / 3)
    a3 = math.sqrt(3)

    f1 = - 3 * x13 + a3 * np.arctan((1 + 2 * x13) / a3) - scimath.log(x13 - 1) + scimath.log(1 + x13 + x13 ** 2) / 2
    f0 = - 3 * x03 + a3 * np.arctan((1 + 2 * x03) / a3) - scimath.log(x03 - 1) + scimath.log(1 + x03 + x03 ** 2) / 2
    f = f1 - f0
    return np.real(f)
    # return f


def get_lb(kap, E_Hb, v, p_Am, E_G, k_J, p_M, eb=1.0, **kwargs):
    E_m = p_Am / v
    g = E_G / (kap * E_m)
    k_M = p_M / E_G
    k = k_J / k_M
    vHb = E_Hb * g ** 2 * k_M ** 3 / ((1 - kap) * p_Am * v ** 2)

    n = 1000 + round(1000 * max(0, k - 1))
    xb = g / (g + eb)
    xb3 = xb ** (1 / 3)
    x = np.linspace(10 ** -5, xb, n)
    dx = xb / n
    x3 = x ** (1 / 3)

    b = beta0(x, xb) / (3 * g)

    t0 = xb * g * vHb
    i = 0
    norm = 1
    ni = 100

    lb = vHb ** (1 / 3)

    while i < ni and norm > 1e-18:
        l = x3 / (xb3 / lb - b)
        s = (k - x) / (1 - x) * l / g / x
        vv = np.exp(- dx * np.cumsum(s))
        vb = vv[- 1]
        r = (g + l)
        rv = r / vv
        t = t0 / lb ** 3 / vb - dx * np.sum(rv)
        dl = xb3 / lb ** 2 * l ** 2. / x3
        dlnv = np.exp(- dx * np.cumsum(s * dl / l))
        dlnvb = dlnv[- 1]
        dt = - t0 / lb ** 3 / vb * (3 / lb + dlnvb) - dx * np.sum((dl / r - dlnv) * rv)
        lb -= t / dt  # Newton Raphson step
        norm = t ** 2
        i += 1
    return lb


def get_E0(kap, v, p_M, p_Am, E_G, eb=1.0, lb=None, **kwargs):
    k_M = p_M / E_G
    if lb is None:
        lb = get_lb(kap=kap, v=v, p_Am=p_Am, p_M=p_M, E_G=E_G, eb=eb, **kwargs)
    g = E_G * v / (kap * p_Am)
    xb = g / (g + eb)
    uE0 = np.real((3 * g / (3 * g * xb ** (1 / 3) / lb - beta0(0, xb))) ** 3)
    U0 = uE0 * v ** 2 / g ** 2 / k_M ** 3
    E0 = U0 * p_Am
    return E0

def get_E_Rm(kap, v, p_M, p_Am, E_G, lb=None, **kwargs):
    '''
        The threshold for pupation is [E_Rj]¼sj[E_Rm], which is introduced with a new parameter sj and an expression
        that gives the reference value, i.e., the maximum value for reproduction buffer density, for the onset of pupation
        Args:
            x0:float
            x1:float

        Returns:float E_Rm (J)

        '''

    if lb is None:
        lb = get_lb(kap=kap, v=v, p_Am=p_Am, p_M=p_M, E_G=E_G, **kwargs)

    k_M = p_M / E_G
    g = E_G * v / (kap * p_Am)
    E_M = p_Am / v
    k_E = g * k_M/lb
    E_Rm = (1 - kap) * g * E_M * (k_E + k_M)/(k_E - g * k_M)
    return E_Rm


def run_embryo_stage2(kap, E_Hb, v, E_G, k_J, p_M, p_Am, p_T: float = 0., dt: float = 1., E_0: float = None, **kwargs):
    E_G_per_kap = E_G / kap
    p_M_per_kap = p_M / kap
    p_T_per_kap = p_T / kap
    v_E_G_plus_P_T_per_kap = (v * E_G + p_T) / kap

    if E_0 is None:
        E_0 = get_E0(E_Hb=E_Hb, k_J=k_J, kap=kap, p_Am=p_Am, v=v, E_G=E_G, p_M=p_M, **kwargs)

    t, E, L, E_H = 0., float(E_0), 0., 0.
    done = False
    # dic=dNl.AttrDict({'t' : [], 'E':[], 'L' : [] , 'E_H' : []})
    while not done:
        L2 = L * L
        L3 = L * L2
        denom = E + E_G_per_kap * L3
        p_C = E * (v_E_G_plus_P_T_per_kap * L2 + p_M_per_kap * L3) / denom
        dL = (E * v - (p_M_per_kap * L + p_T_per_kap) * L3) / 3 / denom
        dE = -p_C
        dE_H = (1-kap) * p_C - k_J * E_H
        if E_H + dt * dE_H > E_Hb:
            dt = (E_Hb - E_H) / dE_H
            done = True
        E += dt * dE
        L += dt * dL
        E_H += dt * dE_H
        t += dt
        print(t,E_H,E_Hb)
        if E < 0 or dL < 0:
            return -1, -1, -1
    return t, E, L


def run_embryo_stage(kap, E_Hb, v, E_G, k_J, p_M, p_Am, p_T: float = 0., dt: float = 1., E_0: float = None, **kwargs):
    E_m = p_Am / v
    g = E_G / (kap * E_m)
    k_M = p_M / E_G
    L_T = p_T / p_M

    if E_0 is None:
        E_0 = get_E0(E_Hb=E_Hb, k_J=k_J, kap=kap, p_Am=p_Am, v=v, E_G=E_G, p_M=p_M, **kwargs)
    # dic = dNl.AttrDict({'t': [], 'E': [], 'L': [], 'E_H': []})
    # t, E, V, E_H = 0., float(E_0), 10**-20, 0.
    t, E, L, E_H = 0., float(E_0), 0., 0.
    while E_H < E_Hb:
        L2 = L * L
        L3 = L * L2

        # volume specific :
        e=E/E_m
        p_C = E_m * (v / L + k_M * (1 + L_T / L)) * (e * g) / (e + g)
        p_S = p_M + p_T /L
        p_G = kap * p_C - p_S
        dL=p_G / E_G

        E -= dt * p_C*L3
        L += dt * dL


        p_J = k_J * E_H
        p_R = (1 - kap) * p_C - p_J
        # dL = (p_G / E_G) ** (1 / 3)

        # dL = E * v/ (3*kap* E +3*E_G * L3) - p_S * L/ (3 *E+ 3* E_G_per_kap * L3)


        # L += dt *(p_G / E_G)**(1/3)

        E_H += dt * p_R
        t += dt

        # L=V ** (1 / 3)
        # dic.t.append(t)
        # dic.E.append(E)
        # dic.L.append(L)
        # dic.E_H.append(E_H)
    return t, E, L  # , dic


if __name__ == '__main__':
    deb_pars = {'F_m': 6.5,
                'kap_X': 0.8,
                'v': 0.12431941407809782,
                'kap': 0.9999225767377389,
                'kap_R': 0.95,
                'p_M': 242.39201174618017,
                'p_T': 0,
                'k_J': 0.002,
                'E_G': 4434.662379584172,
                'E_Hb': 1.3595083084754352e-05,
                'E_He': 0.005978262935519519,
                'z': 0.9457146211594492,
                'T_ref': 293.15,
                'T_A': 8000,
                'kap_P': 0.18,
                'kap_V': 0.99148,
                's_j': 0.999,
                'h_a': 0.0003374238984487218,
                's_G': 0.0001,
                'del_M': 0.9013633740654738,
                'p_Am': 229.251}

    E_0 = get_E0(**deb_pars)
    # print(E_0)
    # t, E, L = run_embryo_stage(E_0=E_0, **deb_pars)

    t2, E2, L2 = run_embryo_stage2(E_0=E_0, **deb_pars)

    # print(t, E, L)
    print(t2, E2, L2)
    pass
