'''
Analysis based on p.268-270 of the DEB textbook



'''

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from lib.anal.plotting import plot_debs
from lib.model.DEB.deb import DEB, deb_sim

substrates=False
plug_flow=True
idx = 89
save_to = f'./plug_flow/{idx}'

def test_substrates() :
    for s in ['standard', 'cornmeal', 'PED_tracker', 'cornmeal2', ]:
        q = 1
        deb = DEB(substrate={'quality': q, 'type': s}, assimilation_mode='sim', steps_per_day=24 * 60)
        print()
        print('substrate type : ', s)
        print('w_X : ', int(deb.substrate.get_w_X()), 'g/mole')
        print('d_X : ', int(deb.substrate.get_d_X(quality=1) * 1000), 'mg/cm**3')
        # print([[q, deb.substrate.get_X(quality=q)] for q in np.arange(0,1.01,0.3)])
        print(*[(f'quality : {q}', f'X : {np.round(deb.substrate.get_X(quality=q)*10**3, 2)} C-mmol/cm**3', f'f : {np.round(deb.substrate.get_f(K=deb.K, quality=q), 2)}') for q in
                np.arange(0, 1.01, 0.25)])

if substrates :
    test_substrates()

if plug_flow :
    show=True
    ds,dds=[],[]
    for k_abs, pref in zip([1.0,0.5], ['Rovers', 'Sitters']) :
        d,dd = deb_sim(sample='None.10_controls',id=f'{pref} SIM', model_id=f'{pref} DEB', deb_dt=60, dt=0.1, k_abs=k_abs)
        ds.append(d)
        dds.append(dd)
        break
    for m in ['assimilation','plug_flow_food', 'plug_flow_enzymes', 'food_ratio_1','food_mass']:
        save_as = f'{m}.pdf'
        plot_debs(deb_dicts=ds, save_as=save_as, mode=m, save_to=save_to, sim_only=True, show=show)
    for m in ['energy', 'growth', 'full']:
        save_as = f'{m}_vs_model.pdf'
        plot_debs(deb_dicts=ds+dds, save_as=save_as, mode=m, save_to=save_to, show=show)
