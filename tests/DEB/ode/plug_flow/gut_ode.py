'''
Analysis based on p.268-270 of the DEB textbook



'''
import json
import os

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lib.conf.base.dtypes import null_dict
from lib.model.DEB.deb import DEB


def prepare_deb(q=1, dt=0.1, **kwargs):
    # Aux
    # dt = 0.1
    substrate = {'type': 'standard', 'quality': q}
    deb = DEB(species='default', substrate=substrate, constant_M_c=True, save_to='./plug_flow_ode',
              # Gut parameters
              # M_gm=10 ** -2,  # gut capacity in C-moles for unit of gut volume
              # k_dig=1,  # rate constant for digestion : k_X * y_Xg
              # f_dig=1,  # scaled functional response for digestion : M_X/(M_X+M_K_X)
              # k_abs=1,  # rate constant for absorption : k_P * y_Pc
              # f_abs=1,  # scaled functional response for absorption : M_P/(M_P+M_K_P)
              # M_c_per_cm2=10 ** -7,  # area specific amount of carriers in the gut per unit of gut surface
              # # J_g_per_cm2=8 * 10 ** -3,  # secretion rate of enzyme per unit of gut surface in sec
              # J_g_per_cm2 = 8 * 10 ** -3 / (24 * 60 * 60),  # secretion rate of enzyme per unit of gut surface in sec
              # k_c=1,  # release rate of carriers
              # k_g=0.7,  # decay rate of enzyme
              # y_P_X=0.9  # yield of product by food
              **kwargs)
    deb.grow_larva(epochs={0: null_dict('epoch', start=0.0, stop=72, substrate=substrate)})
    deb.set_steps_per_day(steps_per_day=int(24 * 60 * 60 / dt))
    # print(deb.dt*24*60*60)
    return deb


def plot_gut(deb, g, z, t, t_g, M0, N_t_gs, dt, save=False, show=True, fig=None, ax=None, title=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 5))
    t_g_ticks = [int((j + 1) * t_g * 60 / dt) for j in range(N_t_gs)]
    dig_ef = z[t_g_ticks[0], 2] / (g.y_P_X * M0)
    dig_ef = np.round(dig_ef, 2)
    print('Digestion&Uptake efficiency : ', dig_ef)
    ax.text(0.5, 0.85, f'Digestion efficiency : {dig_ef}', transform=ax.transAxes)
    for j in range(N_t_gs):
        print(f'Absorption ratio after {j + 1} residence times : ', np.round(z[t_g_ticks[j], 2] / M0, 2))
    ss = '-'
    ax.plot(t, z[:, 0], f'r{ss}', label=r'$M_{X}$')
    ax.plot(t, z[:, 1], f'b{ss}', label=r'$M_{P}$')
    ax.plot(t, z[:, 2], f'm{ss}', label=r'$M_{Pu}$')
    # ax.plot(t,z[:,3],f'g{ss}',label=r'$M_{g}$')
    # ax.plot(t,z[:,3]/g.J_g,f'g{ss}',label=r'$M_{g} ratio$')
    # ax.plot(t, z[:, 4], f'g{ss}', label=r'$M_{c}$')
    ax.set_ylabel('amount (C-moles)')
    if title is not None:
        ax.set_title(title)
    for ii in range(N_t_gs):
        ax.axvline(t_g * (ii + 1))
    ax.legend(loc='best')
    plt.subplots_adjust(hspace=0.4, left=0.1, right=0.95)
    ax.set_xlabel('time (minutes)')
    ax.set_xlim((0, None))
    if save:
        plt.savefig(f'{deb.save_to}/test_gut.png', dpi=300)
    if show:
        plt.show()


def run_gut(g, N_t_gs, dt, mode='ode'):
    M0 = g.Cmax
    t_g = M0 / (g.k_dig * g.J_g*60 / g.k_g)
    # t_g = M0 / (g.k_dig * g.J_g/ (24*60) / g.k_g)
    print('Gut residence time in minutes : ', np.round(t_g, 0))
    t0 = np.arange(0, 60 * t_g * N_t_gs + dt, dt)
    t = t0 / 60

    if mode == 'ode':

        def model(z, t, gut):
            M_X, M_P, M_Pu, M_g, M_c = z
            dM_g_dt = (gut.J_g - gut.k_g * M_g) * 1
            if M_X > 0:
                temp = gut.k_dig * gut.f_dig * M_g * 1
                dM_X_dt = - np.min([M_X, temp])
            else:
                dM_X_dt = 0
            dM_P_dt_added = -gut.y_P_X * dM_X_dt
            if M_P > 0 and M_c > 0:
                temp = gut.k_abs * gut.f_abs * M_c * 1
                dM_Pu_dt = np.min([M_P, temp])
            else:
                dM_Pu_dt = 0
            dM_P_dt = dM_P_dt_added - dM_Pu_dt
            if gut.constant_M_c:
                dM_c_dt = 0
            else:
                dM_c_dt_released = (gut.M_c_max - M_c) * gut.k_c
                dM_c_dt = dM_c_dt_released - dM_Pu_dt
            dzdt = [dM_X_dt, dM_P_dt, dM_Pu_dt, dM_g_dt, dM_c_dt]

            return dzdt

        # initial condition
        # M_X : amount of food (C-moles), M_P : amount of product (C-moles), M_Pu : amount of absorbed product (C-moles), M_g : amount of active enzyme, M_c : amount of available carriers
        z0 = [M0, 0, 0, 0, g.M_c_max]

        z = odeint(model, z0, t0, args=(g,))
    elif mode == 'gut':
        g.dict = g.init_dict()
        g.M_X = M0
        for i in range(t0.shape[0]):
            g.update()
            g.update_dict()
        z = np.vstack([g.dict['M_X'], g.dict['M_P'], np.array(g.dict['M_Pu']) / 1000, g.dict['M_g'], g.dict['R_M_g'],
                       g.dict['M_c']]).T
    return z, t, t_g, M0


def test_gut(q=1,dt=0.1, N_t_gs=2, mode='ode', title=None, fig=None, ax=None, show=True, **kwargs):
    print(mode)
    deb = prepare_deb(q=q,dt=dt, **kwargs)
    z, t, t_g, M0 = run_gut(deb.gut, N_t_gs, dt, mode=mode)
    plot_gut(deb, deb.gut, z, t, t_g, M0, N_t_gs, dt, fig=fig, ax=ax, show=show, title=title)

def compare_gut(save_to=None, save_as = 'SIMvsODE_gut_plug-flow.pdf', show=True,**kwargs) :
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(10, 6))
    axs = axs.ravel()
    test_gut(mode='gut', title='Step-wise simulation', fig=fig, ax=axs[0], show=False, **kwargs)
    test_gut(mode='ode', title='ODE calculation', fig=fig, ax=axs[1], show=False, **kwargs)
    if save_as is not None and save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(f'{save_to}/{save_as}', dpi=300)
    if show:
        plt.show()

compare_gut(save_to='./SIMvsODE', q=0.35, dt=3.34, k_g=0.7)
