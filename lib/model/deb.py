'''
DEB pipeline from literature
'''
import json
import numpy as np
import pandas as pd
from numpy import real
from scipy.integrate import quad, solve_ivp

from lib.stor import paths
from lib.aux import functions as fun


class DEB:
    def __init__(self, id='DEB model',species='default', steps_per_day=1, cv=0, T=298.15, eb=1.0,base_f = 1.0,
                 aging=False, print_output=False, starvation_strategy=False, save_dict=True,
                 base_hunger=0.5, hunger_gain=10, hours_as_larva=0, simulation=True):

        # Drosophila model by default
        if species == 'default':
            with open(paths.Deb_path) as tfp:
                self.species = json.load(tfp)
        else:
            self.species = species
        self.__dict__.update(self.species)

        # DEB methods enabled
        self.starvation_strategy = starvation_strategy
        self.aging = aging

        # Hunger drive parameters
        self.hunger_gain = hunger_gain
        self.base_hunger = base_hunger

        # Aux input parameters
        self.w_X = self.w_E = self.w_V = self.w_P = 23.9  # g/mol molecular weight for water-free food, reserve, structure, product(faeces)
        self.T = T
        self.L0 = 10 ** -10
        self.hours_as_larva = hours_as_larva
        self.id = id
        self.cv = cv
        self.eb = eb
        self.base_f = base_f
        self.f=base_f
        self.print_output = print_output
        self.simulation = simulation
        self.epochs=[]
        self.dict_file = None

        # Larva stage flags
        self.stage = 'embryo'
        self.alive = True

        # Stage duration parameters
        self.age = 0
        self.birth_time_in_hours = np.nan
        self.pupation_time_in_hours = np.nan
        self.emergence_time_in_hours = np.nan
        self.death_time_in_hours = np.nan

        self.derived_pars()
        self.set_steps_per_day(steps_per_day)

        self.E = self.E0
        self.E_H = 0
        self.E_R = 0
        self.V = self.L0 ** 3
        self.update()

        self.run_embryo_stage()
        self.predict_larva_stage(f=self.base_f)

        if save_dict :
            self.dict = self.init_dict()
        else :
            self.dict=None

    def update(self):
        self.L = self.V ** (1 / 3)
        self.Lw = self.L / self.del_M
        self.Ww = self.compute_Ww()
        self.e = self.compute_e()
        self.hunger = self.compute_hunger()

    def scale_time(self):
        dt = self.dt*self.T_factor
        # print(dt)
        self.F_m_dt = self.F_m * dt
        self.v_dt = self.v * dt
        self.p_M_dt = self.p_M * dt
        self.p_T_dt = self.p_T * dt if self.p_T != 0.0 else 0.0
        self.k_J_dt = self.k_J * dt
        self.h_a_dt = self.h_a * dt ** 2

        self.p_Am_dt = self.p_Am * dt
        self.p_Amm_dt = self.p_Amm * dt
        self.k_E_dt = self.k_E * dt

    def set_steps_per_day(self, steps_per_day):
        self.steps_per_day = steps_per_day
        self.dt=1/steps_per_day
        self.scale_time()

    def derived_pars(self):
        kap = self.kap
        v = self.v
        p_Am = self.p_Am = self.z * self.p_M / kap
        self.E_M = p_Am / v
        self.E_V = self.mu_V * self.d_V / self.w_V
        k_M = self.k_M = self.p_M / self.E_G
        g = self.g = self.E_G / (kap * self.E_M)
        ii = g ** 2 * k_M ** 3 / ((1 - kap) * v ** 2)
        self.k = self.k_J / k_M
        self.U_Hb = self.E_Hb / p_Am
        self.vHb = self.U_Hb * ii
        self.U_He = self.E_He / p_Am
        self.vHe = self.U_He * ii
        self.Lm = v / (g * k_M)
        self.T_factor = np.exp(self.T_A / self.T_ref - self.T_A / self.T);  # Arrhenius factor

        lb = self.lb = self.get_length_at_birth(eb=self.eb)
        Lb = self.Lb = lb * self.Lm
        self.Lwb = Lb / self.del_M
        self.tau_b = self.get_tau_b(eb=self.eb)
        self.t_b = self.tau_b / k_M / self.T_factor

        self.k_E = v / Lb

        # For the larva the volume specific max assimilation rate p_Amm is used instead of the surface-specific p_Am
        self.p_Amm = p_Am / Lb

        self.uE0 = self.get_initial_reserve(eb=self.eb)
        self.U0 = self.uE0 * v ** 2 / g ** 2 / k_M ** 3
        self.E0 = self.U0 * p_Am
        self.Ww0 = self.E0 * self.w_E / self.mu_E  # g, initial wet weight

        self.v_Rm = (1 + lb / g) / (1 - lb)  # scaled max reprod buffer density
        self.v_Rj = self.s_j * self.v_Rm  # scaled reprod buffer density at pupation
        if self.print_output:
            print('------------------Egg------------------')
            print(f'Reserve energy  (mJ) :       {int(1000 * self.E0)}')
            print(f'Wet weight      (mg) :       {np.round(1000 * self.Ww0, 5)}')

    def get_tau_b(self, eb=1.0):
        def get_tb(x, ab, xb):
            return x ** (-2 / 3) / (1 - x) / (ab - fun.beta0(x, xb))

        g = self.g
        xb = g / (eb + g)
        ab = 3 * g * xb ** (1 / 3) / self.lb
        return 3 * quad(func=get_tb, a=1e-15, b=xb, args=(ab, xb))[0]

    def get_length_at_birth(self, eb=1.0):
        g = self.g
        k = self.k
        vHb = self.vHb

        n = 1000 + round(1000 * max(0, k - 1))
        xb = g / (g + eb)
        xb3 = xb ** (1 / 3)
        x = np.linspace(10 ** -5, xb, n)
        dx = xb / n
        x3 = x ** (1 / 3)

        b = fun.beta0(x, xb) / (3 * g)

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

    def get_initial_reserve(self, eb=1.0):
        g = self.g
        xb = g / (g + eb)
        return np.real((3 * g / (3 * g * xb ** (1 / 3) / self.lb - fun.beta0(0, xb))) ** 3)

    def predict_larva_stage(self, f=1.0):
        g = self.g
        lb = self.lb
        c1 = f / g * (g + lb) / (f - lb)
        c2 = self.k * self.vHb / lb ** 3
        rho_j = (f / lb - 1) / (f / g + 1)  # scaled specific growth rate of larva

        def get_tj(tau_j):
            ert = np.exp(- tau_j * rho_j)
            return np.abs(self.v_Rj - c1 * (1 - ert) + c2 * tau_j * ert)

        tau_j = self.tau_j = fun.simplex(get_tj, 1)
        self.lj = lb * np.exp(tau_j * rho_j / 3)
        self.t_j = tau_j / self.k_M / self.T_factor
        Lj = self.Lj = self.lj * self.Lm
        self.Lwj = Lj / self.del_M
        E_Rm = self.E_Rm = self.v_Rm * (1 - self.kap) * g * self.E_M * Lj ** 3
        self.E_Rj = E_Rm * self.s_j
        self.E_eggs = E_Rm * self.kap_R

    def predict_pupa_stage(self):
        g = self.g
        k_M = self.k_M

        def emergence(t, luEvH, terminal=True, direction=0):
            return self.vHe - luEvH[2]

        def get_te(t, luEvH):
            l = luEvH[0]
            u_E = max(1e-6, luEvH[1])
            ii = u_E + l ** 3
            dl = (g * u_E - l ** 4) / ii / 3
            du_E = - u_E * l ** 2 * (g + l) / ii
            dv_H = - du_E - self.k * luEvH[2]
            return [dl, du_E, dv_H]  # pack output

        sol = solve_ivp(fun=get_te, t_span=(0, 1000), y0=[0, self.uEj, 0], events=emergence)
        self.tau_e = sol.t_events[0][0]
        self.le, self.uEe = sol.y_events[0][0][:2]
        self.t_e = self.tau_e / k_M / self.T_factor
        self.Le = self.le * self.Lm
        self.Lwe = self.Le / self.del_M
        self.Ue = self.uEe * self.v ** 2 / g ** 2 / k_M ** 3
        self.Ee = self.Ue * self.p_Am
        self.Wwe = self.compute_Ww(V=self.Le ** 3, E=self.Ee + self.E_Rj)  # g, wet weight at emergence

        self.V=self.Le**3
        self.E=self.Ee
        self.E_H=self.E_He
        self.update()

        self.emergence_time_in_hours = self.pupation_time_in_hours + np.round(self.t_e * 24, 1)
        self.stage = 'imago'
        self.age = self.t_e
        if self.print_output:
            print('-------------Pupa stage-------------')
            print(f'Duration         (d) :      {np.round(self.t_e, 3)}')
            print('-------------Emergence--------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwe * 1000, 5)}')
            print(f'Physical length (mm) :      {np.round(self.Lwe * 10, 3)}')

    def predict_imago_stage(self, f=1.0):
        # if np.abs(self.sG) < 1e-10:
        #     self.sG = 1e-10
        # self.uh_a =self.h_a/ self.k_M ** 2 # scaled Weibull aging coefficient
        # self.lT = self.p_T/(self.p_M*self.Lm)# scaled heating length {p_T}/[p_M]Lm
        # self.li = f - self.lT;
        # self.hW3 = self.ha * f * self.g/ 6/ self.li
        # self.hW = self.hW3**(1/3) # scaled Weibull aging rate
        # self.hG = self.sG * f * self.g * self.li**2
        # self.hG3 = self.hG**3;     # scaled Gompertz aging rate
        # self.tG = self.hG/ self.hW
        # self.tG3 = self.hG3/ self.hW3 # scaled Gompertz aging rate
        # # self.tau_m = sol.t_events[0][0]
        # # self.lm, self.uEm=sol.y_events[0][0][:2]
        self.t_m = self.tau_m / self.k_M / self.T_factor
        self.Li = self.li * self.Lm
        self.Lwi = self.Li / self.del_M
        self.Ui = self.uEi * self.v ** 2 / self.g ** 2 / self.k_M ** 3
        self.Ei = self.Ui * self.p_Am
        self.Wwi = self.compute_Ww(V=self.Li ** 3, E=self.Ei + self.E_Rj)  # g, imago wet weight
        self.age = self.t_m

        self.V = self.Li ** 3
        self.E = self.Ei
        self.update()

        if self.print_output:
            print('-------------Imago stage-------------')
            print(f'Duration         (d) :      {np.round(self.t_i_cor, 3)}')
            print('---------------Emergence---------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwi * 1000, 5)}')
            print(f'Physical length (mm) :      {np.round(self.Lwi * 10, 3)}')

    def run_embryo_stage(self, dt=None):
        if dt is None:
            dt = self.dt

        kap = self.kap
        E_G = self.E_G

        t = 0

        while self.E_H < self.E_Hb:
            p_S = self.p_M_dt * self.V + self.p_T_dt * self.V ** (2 / 3)  # This is in e/t and below needs to be volume-specific
            p_C = self.E * (E_G * self.v_dt / self.V ** (1 / 3) + p_S / self.V) / (kap * self.E / self.V + E_G)
            p_G = kap * p_C - p_S
            p_J = self.k_J_dt * self.E_H
            p_R = (1 - kap) * p_C - p_J

            self.E -= p_C
            self.V += p_G / E_G
            self.E_H += p_R
            self.update()

            t += dt
        self.Eb = self.E
        L_b = self.V ** (1 / 3)
        Lw_b = L_b / self.del_M

        self.Wwb = self.compute_Ww(V=self.Lb ** 3, E=self.Eb)  # g, wet weight at birth
        self.birth_time_in_hours = np.round(self.t_b * 24, 2)
        self.stage = 'larva'
        self.age=self.t_b
        if self.print_output:
            print('-------------Embryo stage-------------')
            print(f'Duration         (d) :      predicted {np.round(self.t_b, 3)} VS computed {np.round(t, 3)}')
            print('----------------Birth----------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwb * 1000, 5)}')
            print(f'Physical length (mm) :      predicted {np.round(self.Lwb * 10, 3)} VS computed {np.round(Lw_b * 10, 3)}')

    def run_larva_stage(self, f=1.0, dt=None):
        if dt is None:
            dt = self.dt
        kap = self.kap
        E_G = self.E_G
        g = self.g
        del_M = self.del_M



        t = 0

        while self.E_R < self.E_Rj:
            p_A = self.p_Amm_dt * f * self.V
            p_S = self.p_M_dt * self.V
            p_C = self.E * (E_G * self.k_E_dt + p_S / self.V) / (kap * self.E / self.V + E_G)
            p_G = kap * p_C - p_S
            p_J = self.k_J_dt * self.E_Hb
            p_R = (1 - kap) * p_C - p_J

            self.E += (p_A - p_C)
            self.V += p_G / E_G
            self.E_R += p_R
            self.update()

            t += dt
        Lw_j = self.V ** (1 / 3) / del_M
        Ej = self.Ej = self.E
        self.Uj = Ej / self.p_Am
        self.uEj = self.lj ** 3 * (self.kap * self.kap_V + f / g)
        # self.uEj = self.Uj / self.v ** 2 * self.g ** 2 * self.k_M ** 3
        self.Wwj=self.compute_Ww(V=self.Lj ** 3, E=Ej + self.E_Rj)  # g, wet weight at pupation, including reprod buffer
        # self.Wwj = self.Lj**3 * (1 + f * self.w_V) # g, wet weight at pupation, excluding reprod buffer at pupation
        # self.Wwj += self.E_Rj * self.w_E/ self.mu_E/ self.d_E # g, wet weight including reprod buffer
        self.pupation_time_in_hours = self.birth_time_in_hours + np.round(t * 24, 1)
        self.stage = 'pupa'
        if self.print_output:
            print('-------------Larva stage-------------')
            print(f'Duration         (d) :      predicted {np.round(self.t_j, 3)} VS computed {np.round(t, 3)}')
            print('---------------Pupation---------------')
            print(f'Wet weight      (mg) :      {np.round(self.Wwj * 1000, 5)}')
            print(f'Physical length (mm) :      predicted {np.round(self.Lwj * 10, 3)} VS computed {np.round(Lw_j * 10, 3)}')

    def compute_hunger(self):
        return np.clip(self.base_hunger + self.hunger_gain * (1 - self.get_e()), a_min=0, a_max=1)

    def run(self, f=None):
        if f is None :
            f=self.base_f
        self.f=f
        self.age += self.dt

        kap = self.kap
        E_G = self.E_G

        if self.E_R < self.E_Rj:
            p_A = self.p_Amm_dt * f * self.V
            p_S = self.p_M_dt * self.V
            p_C = self.E * (E_G * self.k_E_dt + p_S / self.V) / (kap * self.E / self.V + E_G)
            p_G = kap * p_C - p_S
            p_J = self.k_J_dt * self.E_Hb
            p_R = (1 - kap) * p_C - p_J
            self.E += (p_A - p_C)
            self.V += p_G / E_G
            self.E_R += p_R
            self.update()

        elif self.stage=='larva' :
            self.pupation_time_in_hours = np.round(self.age * 24, 1)
            self.stage = 'pupa'
            # self.hours_as_larva = self.pupation_time_in_hours - self.birth_time_in_hours
        if self.dict is not None :
            self.update_dict()


    def die(self):
        self.alive = False
        self.death_time_in_hours = self.age * 24
        if self.print_output:
            print(f'Dead after {self.age} days')

    def get_f(self):
        return self.f

    def compute_Ww(self, V=None, E=None):
        if V is None :
            V=self.V
        if E is None :
            E=self.E+self.E_R
        return V * self.d_V + E * self.w_E / self.mu_E

    def compute_e(self, V=None, E=None, E_M=None):
        if V is None:
            V = self.V
        if E is None:
            E = self.E
        if E_M is None:
            E_M = self.E_M
        return E/V/E_M

    def get_Lw(self):
        # Structural L is in cm. We turn it to m
        return self.Lw * 10 / 1000

    def get_L(self):
        return self.L

    def get_Ww(self):
        return self.Ww

    def get_E_R(self):
        return self.E_R

    def get_E_H(self):
        return self.E_H

    def get_V(self):
        return self.V

    def get_E(self):
        return self.E

    def get_e(self):
        return self.e

    def grow_larva(self, hours_as_larva=None, epochs=None, fs=None):
        if epochs is None:
            if hours_as_larva is not None :
                self.hours_as_larva = hours_as_larva
                N = int(self.steps_per_day / 24 * hours_as_larva)
                for i in range(N):
                    if self.stage=='larva' :
                        self.run()
            else :
                while self.stage == 'larva':
                    self.run()
                self.hours_as_larva = self.pupation_time_in_hours-self.birth_time_in_hours
        else:
            t = 0
            Nepochs=len(epochs)
            if fs is None :
                fs=[0]*Nepochs
            elif type(fs)==float :
                fs=[fs]*Nepochs
            elif len(fs)!=Nepochs :
                raise ValueError (f'Number of functional response values : {len(fs)} does not much number of epochs : {Nepochs}')
            max_age=(self.birth_time_in_hours+hours_as_larva)/24 if hours_as_larva is not None else np.inf
            for (s0, s1), f in zip(epochs, fs):
                N0 = int(self.steps_per_day / 24 * (s0 - t))
                for i in range(N0):
                    if self.stage == 'larva' and self.age<=max_age:
                        self.run()
                N1 = int(self.steps_per_day / 24 * (s1 - s0))
                for i in range(N1):
                    if self.stage == 'larva' and self.age<=max_age:
                        self.run(f=f)
                t += s1
            if hours_as_larva is not None :
                self.hours_as_larva = hours_as_larva
                N2 = int(self.steps_per_day / 24 * (hours_as_larva - t))
                for i in range(N2):
                    if self.stage == 'larva':
                        self.run()
            else :
                while self.stage == 'larva':
                    self.run()
                self.hours_as_larva = self.pupation_time_in_hours-self.birth_time_in_hours
            self.epochs = self.store_epochs(epochs)

    def get_pupation_buffer(self):
        return self.E_R / self.E_Rj

    def get_hunger(self):
        return self.hunger

    def init_dict(self):
        dict = {
                'age': [],
                'mass': [],
                'length': [],
                'reserve': [],
                'reserve_density': [],
                'hunger': [],
                'pupation_buffer': [],
                'f': []
                }
        return dict

    def update_dict(self):
        d=self.dict
        d['age'].append(self.age*24)
        d['mass'].append(self.get_Ww()*1000)
        d['length'].append(self.get_Lw()*1000)
        d['reserve'].append(self.get_E())
        d['reserve_density'].append(self.get_e())
        d['hunger'].append(self.get_hunger())
        d['pupation_buffer'].append(self.get_pupation_buffer())
        d['f'].append(self.get_f())

    def finalize_dict(self):
        d = self.dict
        d['birth'] = self.birth_time_in_hours
        d['pupation'] = self.pupation_time_in_hours
        d['emergence'] = self.emergence_time_in_hours
        d['death'] = self.death_time_in_hours
        d['id'] = self.id
        d['simulation'] = self.simulation
        d['sim_start'] = self.hours_as_larva
        d['epochs'] = self.epochs

    def return_dict(self) :
        return self.dict

    def store_epochs(self, epochs):
        t0= self.birth_time_in_hours
        t1= self.pupation_time_in_hours
        t2= self.death_time_in_hours
        epochs = [[s0 + t0, s1 + t0] for [s0, s1] in epochs]
        for t in [t1,t2] :
            if not np.isnan(t):
             epochs = [[s0, np.clip(s1, a_min=s0, a_max=t)] for [s0, s1] in epochs if s0 <= t]
        return epochs

    def save_dict(self, path):
        if self.dict is not None :
            self.finalize_dict()
            self.dict_file=f'{path}/{self.id}.txt'
            with open(self.dict_file, "w") as fp:
                json.dump(self.dict, fp)

    def load_dict(self):
        if self.dict_file is not None :
            with open(self.dict_file) as tfp:
                d = json.load(tfp)
            return d

def deb_default(epochs=None, fs=None, base_f=1.0, steps_per_day=24 * 60, **kwargs):
    deb = DEB(steps_per_day=steps_per_day, base_f=base_f,simulation=False, **kwargs)
    deb.grow_larva(epochs=epochs, fs=fs, hours_as_larva=None)
    deb.finalize_dict()
    d=deb.return_dict()
    return d


def deb_dict(dataset, id, new_id=None, starvation_hours=None):
    if starvation_hours is None:
        starvation_hours = []
    s = dataset.step_data.xs(id, level='AgentID')
    e = dataset.endpoint_data.loc[id]
    if new_id is not None:
        id = new_id
    t0 = e['birth_time_in_hours']
    t2 = e['death_time_in_hours']
    t3 = e['hours_as_larva']
    starvation = [[s0 + t0, s1 + t0] for [s0, s1] in starvation_hours]
    if not np.isnan(t2):
        starvation = [[s0, np.clip(s1, a_min=s0, a_max=t2)] for [s0, s1] in starvation if s0 <= t2]
    dict = {'birth': t0,
            'puppation': e['pupation_time_in_hours'],
            'death': t2,
            'age': e['age'],
            'sim_start': t3,
            'mass': s['mass'].values.tolist(),
            'length': s['length'].values.tolist(),
            'reserve': s['reserve'].values.tolist(),
            'reserve_density': s['reserve_density'].values.tolist(),
            'hunger': s['hunger'].values.tolist(),
            'explore2exploit_balance': s['explore2exploit_balance'].values.tolist(),
            # 'structural_length': s['structural_length'].values.tolist(),
            # 'maturity': s['maturity'].values.tolist(),
            # 'reproduction': s['reproduction'].values.tolist(),
            # 'structure': s['structure'].values.tolist(),
            'puppation_buffer': s['puppation_buffer'].values.tolist(),
            # 'steps_per_day': e['deb_steps_per_day'],
            # 'Nticks': e['deb_Nticks'],
            'simulation': True,
            'f': s['deb_f'].values.tolist(),
            'id': id,
            'starvation': starvation}
    return dict


if __name__ == '__main__':
    eb = 1.0
    f = 1.0
    steps_per_day = 10 ** 3
    deb = DEB(eb=eb, steps_per_day=steps_per_day, print_output=True)
    deb.run_larva_stage(f=f)
    deb.predict_pupa_stage()
