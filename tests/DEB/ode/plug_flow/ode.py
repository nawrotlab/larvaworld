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

save = False
constant_M_c = True

substrate = {'type': 'standard', 'quality': 1}
deb = DEB(species='default', substrate=substrate,
          # Gut parameters
          M_gm=10 ** -2,  # gut capacity in C-moles for unit of gut volume
          k_dig=1,  # rate constant for digestion : k_X * y_Xg
          f_dig=1,  # scaled functional response for digestion : M_X/(M_X+M_K_X)
          k_abs=1,  # rate constant for absorption : k_P * y_Pc
          f_abs=1,  # scaled functional response for absorption : M_P/(M_P+M_K_P)
          M_c_per_cm2=10 ** -7,  # area specific amount of carriers in the gut per unit of gut surface
          # J_g_per_cm2=8 * 10 ** -3,  # secretion rate of enzyme per unit of gut surface in sec
          J_g_per_cm2 = 8 * 10 ** -3 / (24 * 60 * 60),  # secretion rate of enzyme per unit of gut surface in sec
          k_c=1,  # release rate of carriers
          k_g=0.7,  # decay rate of enzyme
          y_P_X=0.9  # yield of product by food
          )
deb.grow_larva(epochs={0: null_dict('epoch', start=0.0, stop=72, substrate=substrate)})
J_X_Am_per_cm3 = deb.J_X_Amm / (24 * 60 * 60)
f = deb.f
L = deb.L  # Length in cm
g=deb.gut
r_gut_V = g.r_gut_V  # gut volume per unit of body volume
r_gut_A = g.r_gut_A  # gut surface area per unit of body surface area

M_c_max = g.M_c_max  # amount of carriers in the gut surface
J_g = g.J_g  # total secretion rate of enzyme in the gut surface
M_X_full = g.Cmax
k_dig = g.k_dig
f_dig = g.f_dig
k_abs = g.k_abs
f_abs = g.f_abs
k_c = g.k_c
k_g = g.k_g
y_P_X = g.y_P_X
M_gm = g.M_gm

# Aux
N_t_gs = 2
dt = 0.1

save_to = './plug_flow_ode'
os.makedirs(save_to, exist_ok=True)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))
axs = axs.ravel()
par = 'k_abs'
for i, (k_abs, label, color) in enumerate(zip([0.4, 0.7], ['Sitters', 'Rovers'], ['Red', 'Blue'])):
    ax = axs[i]
    print()
    print(label)

    M0 = M_X_full
    t_g = M0 / (k_dig * J_g * 60 / k_g)  # gut residence time in minutes
    # t_g = M_gm * r_gut_V / J_X_Am_per_cm3 / f / 60  # gut residence time in minutes
    print('Gut residence time in minutes : ', np.round(t_g, 0))
    t0 = np.arange(0, 60 * t_g * N_t_gs + dt, dt)
    t = t0 / 60


    def model(z, t):
        M_X, M_P, M_Pu, M_g, M_c = z
        dM_g_dt = (J_g - k_g * M_g) * 1
        if M_X > 0:
            temp = k_dig * f_dig * M_g * 1
            dM_X_dt = - np.min([M_X, temp])
        else:
            dM_X_dt = 0
        dM_P_dt_added = -y_P_X * dM_X_dt
        if M_P > 0 and M_c > 0:
            temp = k_abs * f_abs * M_c * 1
            dM_Pu_dt = np.min([M_P, temp])
        else:
            dM_Pu_dt = 0
        dM_P_dt = dM_P_dt_added - dM_Pu_dt
        if constant_M_c:
            dM_c_dt = 0
        else:
            dM_c_dt_released = (M_c_max - M_c) * k_c
            dM_c_dt = dM_c_dt_released - dM_Pu_dt
        dzdt = [dM_X_dt, dM_P_dt, dM_Pu_dt, dM_g_dt, dM_c_dt]

        return dzdt


    # initial condition
    # M_X : amount of food (C-moles), M_P : amount of product (C-moles), M_Pu : amount of absorbed product (C-moles), M_g : amount of active enzyme, M_c : amount of available carriers
    z0 = [M0, 0, 0, 0, M_c_max]

    z = odeint(model, z0, t0)

    t_g_ticks = [int((j + 1) * t_g * 60 / dt) for j in range(N_t_gs)]
    dig_ef = z[t_g_ticks[0], 2] / (y_P_X * M0)
    dig_ef = np.round(dig_ef, 2)

    print('Digestion&Uptake efficiency : ', dig_ef)
    ax.text(0.5, 0.85, f'Digestion efficiency : {dig_ef}', transform=ax.transAxes)
    for j in range(N_t_gs):
        print(f'Absorption ratio after {j} residence times : ', np.round(z[t_g_ticks[j], 2] / M0, 2))
    ss = '-'
    ax.plot(t, z[:, 0], f'r{ss}', label=r'$M_{X}$')
    ax.plot(t, z[:, 1], f'b{ss}', label=r'$M_{P}$')
    ax.plot(t, z[:, 2], f'm{ss}', label=r'$M_{Pu}$')
    # ax.plot(t,z[:,3],f'g{ss}',label=r'$M_{g}$')
    # ax.plot(t,z[:,3]/J_g,f'g{ss}',label=r'$M_{g} ratio$')
    # ax.plot(t, z[:, 4], f'g{ss}', label=r'$M_{c}$')
    ax.set_ylabel('amount (C-moles)')
    ax.set_title(label)
    for ii in range(N_t_gs):
        ax.axvline(t_g * (ii + 1))
    ax.legend(loc='best')
plt.subplots_adjust(hspace=0.2, left=0.1, right=0.95)
plt.xlabel('time (minutes)')
plt.xlim((0, None))
if save:
    plt.savefig(f'{save_to}/{par}.png', dpi=300)
plt.show()
