import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
# from sympy import *

K = 5.070141129310573e-05
def qs(bX):
    K = 5.070141129310573e-05
    kE = 3.66
    X = 0.0007372827278040548
    kX = bX * K
    qE = bX * X / kE / (1 + bX * X * (kX ** -1 + kE ** -1))
    q0 = kE * qE / bX * X
    qX = 1 - q0 - qE
    return q0, qX, qE

def qsR(bX) :
    q0, qX, qE=qs(bX)
    return qE/qX-0.5

# def qsRR(bX) :
#     q0, qX, qE=qs(np.abs(bX))
#     return q0-0.5

def qsS(bX) :
    q0, qX, qE=qs(bX)
    return qE/qX-0.15

def qsRS(bXRS):
    q0R, qXR, qER = qs(bXRS[0])
    q0S, qXS, qES = qs(bXRS[1])
    wS=qES/qXS-0.15
    wR=qER/qXR-0.5
    wRS=qES-qER
    w=np.array([wS,wR,wRS])
    # print(w**2)
    r=np.sqrt(np.sum(w**2))
    return r,r

# res=fsolve(qsRS, np.array([[38575],[11572]]))
# print(res)

# raise
# print(qs(21.875))
# print(qs(17146))
# print(qs(17146))
print(qs(34106))
print(qs(17662))



bX_s=11572
bX_r=38575

for bX,lab,ss in zip([bX_r,bX_s],['Rovers', 'Sitters'], ['--', '.']) :
    K = 5.070141129310573e-05
    kE = 3.66
    X = 0.0007372827278040548
    kX = bX * K


    def qs(bX) :
        K = 5.070141129310573e-05
        kE = 3.66
        X = 0.0007372827278040548
        kX = bX * K
        qE = bX*X/kE/(1+bX*X*(kX**-1+kE**-1))
        q0 = kE*qE/bX*X
        qX = 1-q0-qE
        return q0, qX, qE



    # for bX in np.arange(10**4, 10**8, 10**2) :
    #     if q00(bX)>10**-4 :
    #         print(bX, q00(bX))

    # raise

    # function that returns dz/dt
    def model(z,t):
        dxdt = -X*bX*z[0] + kE*z[1]
        dydt = kX*(1-z[0]-z[1]) - kE*z[1]
        dzdt = [dxdt,dydt]

        return dzdt

    # initial condition
    z0 = [1,0]

    # time points
    t = np.arange(0,10, 1/(24*60))

    # solve ODE
    z = odeint(model,z0,t)

    # print(z)
    # plot results
    plt.plot(t,z[:,0],f'g{ss}',label=r'$\frac{d\theta_{0}}{dt}$'+f'{lab}')
    plt.plot(t,z[:,1],f'b{ss}',label=r'$\frac{d\theta_{E}}{dt}$'+f'{lab}')
    plt.plot(t,1-z[:,1]-z[:,0],f'r{ss}',label=r'$\frac{d\theta_{X}}{dt}$'+f'{lab}')
plt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()