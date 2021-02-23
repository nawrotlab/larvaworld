import json

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from lib.model.larva.deb import DEB
from lib.stor.paths import Deb_path

'''
Times of hatch and puppation, lengths of young L1 and late L3 from [1] :
 - Time at hatch (days) : 0.7
 - Time from hatch to puppation (days) : 7.8
 - Length of young L1 (mm) : 0.6
 - Length of late L3 (mm) : 3.8
Wet weight of L3 from [2] :
 - Wet weight of L3 (mg) : 15
[1] I. Schumann and T. Triphan, “The PEDtracker: An Automatic Staging Approach for Drosophila melanogaster Larvae,” Front. Behav. Neurosci., vol. 14, 2020.
[2] K. G. Ormerod et al., “Drosophila development, physiology, behavior, and lifespan are influenced by altered dietary composition,” Fly (Austin)., vol. 11, no. 3, pp. 153–170, 2017.
Reserve density while feeding at libitum (f=1) :
 - e = 1 (both at hatch and at puppation)
'''

# inds=[0,1, 2, 6, 11, 14]
# inds = [0, 1, 2, 3, 4, 6, 7, 8, 11, 14, 15]


# inds = [4,6,11,12]
inds = [0, 1, 2, 3, 4, 6, 7, 8, 11, 14, 15]
R = np.array([0.7, 7.8, 0.6, 3.8, 15, 1, 1])
W = np.array([2, 5, 5, 10, 2, 5, 5])
W = W / np.sum(W)

pars0 = ['U_E__0', 'p_m', 'E_G',
         'E_H__b', 'E_R__p', 'E_H__e',
         'zoom', 'v_rate_int', 'kap_int',
         'kap_R_int', 'k_J_rate_int', 'shape_factor',
         'h_a', 'sG', 'L_0', 'd']
vals0 = [0.001, 23.57, 4400,
         0.00328786, 0.0166888, 0.05218,
         9.63228, 0.00581869, 0.631228,
         0.95, 0.002, 0.58,
         0.004218, 0.0001, 0.00001, 1]
ranges0 = [[0.000001, 0.001], [1, 100], [1000, 100000],
           [0.0001, 0.1], [0.1, 4.0], [0.1, 1.0],
           [1.0, 100.0], [0.001, 0.06], [0.1, 0.9],
           [0.9, 0.99], [0.001, 0.003], [0.1, 35],
           [0.003, 0.005], [0.0001, 0.0002], [0.00001, 0.01], [0.1, 2.0]]

try:
    with open(Deb_path) as tfp:
        species0 = json.load(tfp)
    print('Loaded existing deb')
except:
    species0 = dict(zip(pars0, vals0))

pars = [p for i, p in enumerate(pars0) if i in inds]
Npars = len(pars)
ranges = np.array([r for i, r in enumerate(ranges0) if i in inds])


def show_results(t0, t1, l0, l1, w1, e0, e1):
    print('Embryo stage : ', np.round(t0, 3), 'days')
    print('Larva stage : ', np.round(t1, 3), 'days')
    print('Larva initial length : ', np.round(l0, 3), 'mm')
    print('Larva final length : ', np.round(l1, 3), 'mm')
    print('Larva final weight : ', np.round(w1, 3), 'mg')
    print('Larva initial reserve density : ', np.round(e0, 3))
    print('Larva final reserve density : ', np.round(e1, 3))


def fit_DEB(vals, steps_per_day=24, show=False):
    species = species0.copy()
    for p, v in zip(pars, vals):
        species[p] = v
    deb = DEB(species=species, steps_per_day=steps_per_day, cv=0, aging=True, print_stage_change=show)
    c0 = False
    while not deb.puppa:
        if not deb.alive:
            return +np.inf
        f = 1
        deb.run(f)
        if deb.larva and not c0:
            c0 = True
            t0 = deb.age_day
            # w0=deb.get_W()
            l0 = deb.get_real_L() * 1000
            e0 = deb.get_reserve_density()

    t1 = deb.age_day - t0
    w1 = deb.get_W() * 1000
    l1 = deb.get_real_L() * 1000
    e1 = deb.get_reserve_density()
    r = np.array([t0, t1, l0, l1, w1, e0, e1])
    dr = np.abs(r - R) / R
    error = np.sum(dr * W)
    if show:
        show_results(t0, t1, l0, l1, w1, e0, e1)
    return error


alg_pars = {'max_num_iteration': 30,
            'population_size': 20,
            'mutation_probability': 0.1,
            'elit_ratio': 0.1,
            'crossover_probability': 0.1,
            'parents_portion': 0.2,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None}

model = ga(function=fit_DEB, dimension=Npars, variable_type='real', variable_boundaries=ranges,
           algorithm_parameters=alg_pars)

model.run()

best = species0.copy()
for p, v in zip(pars, model.best_variable):
    best[p] = v

with open(Deb_path, "w") as fp:
    json.dump(best, fp)

for k, v in best.items():
    print(f'{k} : {v}')

e = fit_DEB(model.best_variable, steps_per_day=24 * 60, show=True)

print(e)
