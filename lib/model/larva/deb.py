'''
DEB pipeline as in DEB-IBM (see code from Netlogo below)

Changes from code from netlogo :
 - __ instead of ^ in variable names
'''
import json
import numpy as np
from lib.stor.paths import get_parent_dir, Deb_path


class DEB:
    def __init__(self, species='default', steps_per_day=1, cv=0,
                 aging=True, print_stage_change=False, starvation_strategy=False, base_hunger=0.5):
        self.base_hunger=base_hunger
        self.print_stage_change = print_stage_change
        self.starvation_strategy = starvation_strategy
        # My flags
        self.embryo = True
        self.larva = False
        self.puppa = False
        self.imago = False
        self.alive = True

        self.tick_counter = 0
        self.birth_time_in_hours = np.nan
        self.puppation_time_in_hours = np.nan
        self.death_time_in_hours = np.nan
        self.hours_as_larva = 0

        # Input params
        self.steps_per_day = steps_per_day
        self.cv = cv
        self.aging = aging
        self.species = species

        # Global parameters
        self.U_E__0 = None
        self.f = None
        self.L_0 = None

        # parameters for the environment: here only prey density
        self.X = None
        self.d_X = None

        # Individual parameters
        self.L = None
        self.dL = None
        self.U_H = None
        self.dU_H = None
        self.U_E = None
        self.dU_E = None
        self.e_scaled = None
        self.U_R = None
        self.dU_R = None
        self.U_V = None

        # Fluxes
        self.S_A = None
        self.S_C = None

        # EMBRYO
        self.e_ref = None
        self.U_E_embryo = None
        self.S_C_embryo = None
        self.U_H_embryo = None
        self.L_embryo = None
        self.dU_E_embryo = None
        self.dU_H_embryo = None
        self.dL_embryo = None

        #   parameters used to calculate the costs for an egg / initial reserves
        self.lower_bound = None
        self.upper_bound = None
        self.estimation = None
        self.lay_egg = None
        self.offspring_count = None
        self.sim = None

        # STANDARD DEB PARAMETERS
        self.g = None
        self.v_rate = None
        self.kap = None
        self.kap_R = None
        self.k_M_rate = None
        self.k_J_rate = None
        self.U_H__b = None
        self.U_H__p = 0.00001
        self.U_H__e = None
        #  parameter that is used to randomize the input parameters
        self.scatter_multiplier = None

        # PREY DYNAMICS given from netlogo interface
        self.J_XAm_rate_int = 1
        self.F_m = 1
        self.r_X = 1
        self.K_X = 1
        self.volume = 1
        self.f_scaled = 1

        # not given from netlogo interface
        self.J_XAm_rate = None
        self.K = None

        # AGING given from netlogo interface.
        self.h_a = 4.105E-4
        self.sG = -0.5
        self.background_mortality = 0.0

        # not given from netlogo interface
        self.q_acceleration = 0
        self.dq_acceleration = 0
        self.h_rate = 0
        self.dh_rate = 0
        self.age_day = 0

        # I added these because they are required for conversion
        # self.p_m = None
        # self.zoom = None
        # self.kap_int = None
        # self.E_H__b = None
        # self.E_H__p = None
        # self.E_G = None
        # self.v_rate_int = None

        if self.species == 'default':
            with open(Deb_path) as tfp:
                species = json.load(tfp)
            self.__dict__.update(species)
        else:
            self.__dict__.update(self.species)

        self.convert_parameters()
        self.f = 0
        self.lay_egg = False

        self.X = self.J_XAm_rate_int / self.F_m  # set initial value of prey to their carrying capacity

        # individual-variability  ; first their individual variability in the parameter is set
        self.individual_variability()
        #   calc-embryo-reserve-investment     ; then the initial energy is calculated for each
        self.calc_embryo_reserve_investment()
        self.hunger = self.compute_hunger()
        self.W = self.compute_wet_weight()

    def compute_hunger(self):
        try :
            h = np.clip(self.base_hunger + 1 - self.get_reserve_density(), a_min=0, a_max=1)
            return h
        except :
            return np.nan

    def run(self, f=None):
        self.age_day += 1 / self.steps_per_day
        self.tick_counter += 1
        if f is None:
            f = 1
        self.f = f
        # first all individuals calculate the change in their state variables based on the current conditions
        self.calc_dU_E(f=self.f)
        self.calc_dU_H()
        self.calc_dU_R()
        self.calc_dL()

        # if the ageing submodel is turned on, the change in damage inducing compound and damage are calculated
        if self.aging:
            self.calc_dq_acceleration()
            self.calc_dh_rate()

        # if food-dynamics = "logistic"     ; if prey dynamics are set to "logistic" the change in prey density is calculated
        #   [ask patches [calc-d_X]]
        #
        # the the state variables of the individuals and prey are updated based on the delta value
        self.update()

        # ask turtles with [U_H >= U_H^p] ;
        # mature individual check if they have enough energy in their reproduction buffer to repdroduce
        if self.U_H >= self.U_H__e:
            self.calc_lay_eggs()
            if self.lay_egg:
                # if so, they calculate how much energy to invest in an embryo
                self.calc_embryo_reserve_investment()
                # and they produce one offspring
                self.lay_eggs()

        #     do-plots                          ; then the plots are updated
        #   if count turtles = 0 [stop]


    # change in reserves: determined by the difference between assimilation (S_A) and mobilization (S_C) fluxes
    # ; when food-dynamics are constant f = the value of f_scaled set in the user interface
    # ; if food is set to  "logistic" f depends on prey density and the half-saturation coefficient (K)
    # ; for embryos f = 0 because they do not feed exogenously
    #
    # to calc-dU_E
    #
    #   if food-dynamics = "constant"
    #   [ ifelse U_H <= U_H^b
    #     [set f 0]
    #     [set f f_scaled]
    #   ]
    #   if food-dynamics = "logistic"
    #   [ ifelse U_H <= U_H^b
    #     [set f 0]
    #     [set f X / (K + X)]
    #   ]

    def calc_dU_E(self, f):
        g = self.g
        L = self.L
        k_M = self.k_M_rate
        v = self.v_rate
        if not self.U_H <= self.U_H__b:
            self.f = f
        else:
            self.f = 0
        e = v * (self.U_E / L ** 3)
        self.S_C = L ** 2 * (g * e / (g + e)) * (1 + (L / (g * (v / (g * k_M)))))
        self.S_A = self.f * L ** 2
        self.dU_E = self.S_A - self.S_C
        self.e_scaled=e

        # change in maturity is calculated (for immature individuals only)

    def calc_dU_H(self):
        k = self.kap
        k_J = self.k_J_rate
        U_H__b = self.U_H__b
        S_C = self.S_C
        U_H = self.U_H
        U_H__e = self.U_H__e
        if U_H < U_H__b:  # they only invest into maturity until they reach puberty
            self.dU_H = (1 - k) * S_C - k_J * U_H
        elif U_H__b <= U_H < U_H__e:
            if self.embryo and not self.larva:
                self.embryo = False
                self.larva = True
                self.birth_time_in_hours=self.age_day*24
                if self.print_stage_change:
                    print(f'Larval stage reached after {self.age_day} days')
            if self.puppa:
                self.dU_H = (1 - k) * S_C - k_J * U_H
            else:
                self.dU_H = 0

        elif U_H__e <= U_H:
            if self.puppa and not self.imago:
                self.puppa = False
                self.imago = True
                if self.print_stage_change:
                    print(f'Imago stage reached after {self.age_day} days')
            self.dU_H = 0

    # the following procedure calculates change in reprobuffer if mature
    def calc_dU_R(self):
        k = self.kap
        k_J = self.k_J_rate
        U_R__p=self.U_R__p
        S_C = self.S_C
        U_R = self.U_R
        if self.larva and U_R < U_R__p:
            self.dU_R = (1 - k) * S_C - k_J * U_R__p
        elif U_R >= U_R__p:
            if self.larva and not self.puppa:
                self.larva = False
                self.puppa = True
                self.puppation_time_in_hours = self.age_day * 24
                if self.print_stage_change:
                    print(f'Puppal stage reached after {self.age_day} days')
            if self.imago:
                self.dU_R = (1 - k) * S_C - k_J * U_R__p
            else:
                self.dU_R = 0

    # the following procedure calculates change in structural length, if growth in negative the individual does not have enough energy to pay somatic maintenance and the starvation submodel is run
    # where growth is set to 0 and individuals divirt enough energy from development (for juveniles) or reprodution (for adults) to pay maintenance costs
    def calc_dL(self):
        g=self.g
        L=self.L
        k_M=self.k_M_rate
        k_J=self.k_J_rate
        v=self.v_rate
        S_C=self.S_C
        S_A=self.S_A
        e=self.e_scaled
        k=self.kap
        U_H__p=self.U_H__p
        self.dL = (((v / (g * L ** 2)) * S_C) - k_M * L)/3
        # if growth is negative use starvation strategy 3 from the DEB book
        if self.dL<0 :
            if e < L / (v / (g * k_M)) and self.starvation_strategy:
                self.dL = 0
                d=(1 - k) * e * L ** 2 - k_J * U_H__p - k * L ** 2 * (L / (v / (g * k_M)) - e)
                if self.U_H < self.U_H__p:
                    self.dU_H = d
                else:
                    self.dU_R = d
                self.dU_E = S_A - e * L ** 2
                if self.U_H < U_H__p:
                    if self.dU_H < 0:
                        self.die()
                    if self.U_R < 0:
                        self.die()
            else :
                print('dd')
                self.die()


    # the following procedure calculates the change in damage enducing compounds of an individual
    def calc_dq_acceleration(self):
        g = self.g
        L = self.L
        dL = self.dL
        k_M = self.k_M_rate
        v = self.v_rate
        e = self.e_scaled
        q=self.q_acceleration
        self.dq_acceleration = (q * (L ** 3 / (v / (g * k_M)) ** 3) * self.sG + self.h_a) * e * (
                                       (v / L) - ((3 / L) * dL)) - ((3 / L) * dL) * q

    # the following procedure calculates the change in damage in the individual
    def calc_dh_rate(self):
        self.dh_rate = self.q_acceleration - ((3 / self.L) * self.dL) * self.h_rate

    def convert_parameters(self):
        self.p_am = self.p_m * self.zoom / self.kap_int
        self.U_H__b_int = self.E_H__b / self.p_am
        self.U_H__e_int = self.E_H__e / self.p_am
        self.U_R__p_int = self.E_R__p / self.p_am
        self.k_M_rate_int = self.p_m / self.E_G
        self.g_int = (self.E_G * self.v_rate_int / self.p_am) / self.kap_int

    def individual_variability(self):
        # ; individuals vary in their DEB paramters on a normal distribution with a mean on the input paramater and a coefficent of variation equal to the cv
        #   ; set cv to 0 for no variation
        #   set scatter-multiplier e ^ (random-normal 0 cv)
        scatter_multiplier = np.exp(np.random.normal(0, self.cv))
        self.J_XAm_rate = self.J_XAm_rate_int * scatter_multiplier
        self.g = self.g_int / scatter_multiplier
        self.U_H__b = self.U_H__b_int / scatter_multiplier
        self.U_R__p = self.U_R__p_int / scatter_multiplier
        self.U_H__e = self.U_H__e_int / scatter_multiplier

        self.v_rate = self.v_rate_int
        self.kap = self.kap_int
        self.kap_R = self.kap_R_int
        self.k_M_rate = self.k_M_rate_int
        self.k_J_rate = self.k_J_rate_int
        self.K = self.J_XAm_rate / self.F_m

    def calc_embryo_reserve_investment(self):

        self.L = self.L_0
        self.U_E = self.U_E__0
        self.U_H = 0
        self.U_R = 0
        self.dU_R = 0
        self.U_V = self.compute_structure()

    def calc_lay_eggs(self):
        pass

    def lay_eggs(self):
        pass

    def die(self):
        self.alive = False
        self.death_time_in_hours = self.age_day * 24
        if self.print_stage_change:
            print(f'Dead after {self.age_day} days')

    # the following procedure calculates change in prey density this procedure is only run when prey dynamics are set to "logistic" in the user interface
    #    set d_X (r_X) * X * (1 - (X / K_X))   - sum [ S_A * J_XAm_rate   ] of turtles-here / volume
    def calc_d_X(self):
        pass

    # to update
    # ; individuals update their state variables based on the calc_state variable proccesses
    #   ask turtles
    #   [
    #     set U_E U_E + dU_E / timestep
    #     set U_H U_H + dU_H / timestep
    #     set U_R U_R + dU_R    / timestep
    #     set L L + dL    / timestep
    #     if U_H > U_H^b
    #     [ set q_acceleration q_acceleration + dq_acceleration  / timestep
    #       set h_rate h_rate + dh_rate  / timestep
    #     ]
    #
    #    if aging = "on" [if ticks mod timestep = age-day [if random-float 1 < h_rate [die]] ] ;ageing related mortality
    #    if aging = "off" [if ticks mod timestep = age-day [if random-float 1 < background-mortality [die]] ]
    #  ]
    #   if food-dynamics = "logistic"[ ask patches [ set X X + d_X / timestep]]
    def update(self):
        s=self.steps_per_day
        t=self.tick_counter
        self.U_E += self.dU_E / s
        self.U_H += self.dU_H / s
        self.U_R += self.dU_R / s
        self.L += self.dL / s
        self.U_V = self.compute_structure()
        self.hunger = self.compute_hunger()
        self.W = self.compute_wet_weight()
        if self.U_H >= self.U_H__b:
            self.q_acceleration += self.dq_acceleration / s
            self.h_rate += self.dh_rate / s
        # ageing related mortality
        if self.aging:
            if t % s == 0:
                if np.random.uniform(0, 1) < self.h_rate:
                    self.die()
        # background mortality
        else:
            if t % s == 0:
                if np.random.uniform(0, 1) < self.background_mortality:
                    self.die()
        #   if food-dynamics = "logistic"[ ask patches [ set X X + d_X / timestep]]

    def get_f(self):
        return self.f

    def compute_wet_weight(self):
        physical_V = (self.L * self.shape_factor) ** 3  # in cm**3
        w = physical_V * self.d  # in g
        return w

    def get_h_rate(self):
        return self.h_rate

    def get_q_acceleration(self):
        return self.q_acceleration

    def get_real_L(self):
        # Structural L is in cm. We turn it to m
        l = self.L * self.shape_factor * 10 / 1000
        return l

    def get_L(self):
        return self.L

    def get_W(self):
        return self.W

    def get_U_E(self):
        return self.U_E

    def get_U_R(self):
        return self.U_R

    def get_U_H(self):
        return self.U_H

    def get_U_V(self):
        return self.U_V

    def get_reserve(self):
        return self.U_E*self.p_am

    def get_reserve_density(self):
        if self.embryo and not self.larva :
            return np.nan
        else :
            return self.e_scaled

    def reach_stage(self, stage='larva'):
        if stage == 'larva':
            while self.alive and not self.larva:
                f = 1
                self.run(f)

    def reach_larva_age(self, hours_as_larva, f=1):
        self.hours_as_larva=hours_as_larva
        N=int(self.steps_per_day/24*hours_as_larva)
        for i in range(N) :
            self.run(f)

    def compute_structure(self):
        return self.L ** 3 * self.E_G

    def get_puppation_buffer(self):
        if self.embryo and not self.larva :
            return np.nan
        else :
            return self.U_R/self.U_R__p


def deb_default(starvation_hours=[], base_f=1, id=None):
    # print(base_f)
    base_f=base_f
    steps_per_day = 24 * 60
    deb = DEB(species='default', steps_per_day=steps_per_day, cv=0, aging=True, print_stage_change=True)
    ww = []
    E = []
    e = []
    h = []
    # L = []
    real_L = []
    # U_H = []
    # U_R = []
    # U_V = []
    fs = []
    puppation_buffer=[]
    c0 = False
    while not deb.puppa:
        if not deb.alive:
            print(f'The organism died at {deb.death_time_in_hours} hours.')
            t1=np.nan
            break
        if deb.larva :
            if any([r1 <= (deb.age_day*24-t0) < r2 for [r1, r2] in starvation_hours]):
                f = 0
            else:
                f = base_f
        else :
            f=base_f
        ww.append(deb.get_W() * 1000)
        h.append(deb.hunger)
        real_L.append(deb.get_real_L() * 1000)
        # L.append(deb.get_L())
        E.append(deb.get_reserve())
        e.append(deb.get_reserve_density())
        # U_H.append(deb.get_U_H() * 1000)
        # U_R.append(deb.get_U_R() * 1000)
        # U_V.append(deb.get_U_V() * 1000)
        fs.append(deb.get_f())
        puppation_buffer.append(deb.get_puppation_buffer())
        deb.run(f)
        if deb.larva and not c0:
            c0 = True
            t0 = deb.birth_time_in_hours
    t1 = deb.puppation_time_in_hours
    t2=deb.death_time_in_hours
    t3=deb.hours_as_larva
    starvation=[[s0 +t0, s1 +t0] for [s0, s1] in starvation_hours]
    # print(t0,t1,starvation, deb.age_day*24)
    if not np.isnan(t2) :
        starvation = [[s0, np.clip(s1, a_min=s0, a_max=t2)] for [s0, s1] in starvation if s0<=t2]
    if id is None :
        if len(starvation)==0 :
            id = 'ad libitum'
        elif len(starvation)==1 :
            range=starvation[0]
            dur=int(range[1]- range[0])
            id = f'{dur}h starved'
        else :
            id = f'starved {len(starvation)} intervals'
    dict = {'birth': t0,
            'puppation': t1,
            'death': t2,
            'age' : deb.age_day*24+1,
            'sim_start' : t3,
            'mass': ww,
            'length': real_L,
            'reserve': E,
            'reserve_density': e,
            'hunger': h,
            # 'structural_length': L,
            # 'maturity': U_H,
            # 'reproduction': U_R,
            # 'structure': U_V,
            'puppation_buffer': puppation_buffer,
            # 'steps_per_day': deb.steps_per_day,
            # 'Nticks': deb.tick_counter,
            'simulation' : False,
            'f': fs,
            'id' : id,
            'starvation' : starvation}
    # raise
    return dict

def deb_dict(dataset, id, new_id=None, starvation_hours=[]):
    s=dataset.step_data.xs(id, level='AgentID')
    e=dataset.endpoint_data.loc[id]
    if new_id is not None :
        id=new_id
    t0=e['birth_time_in_hours']
    t2=e['death_time_in_hours']
    t3=e['hours_as_larva']
    starvation=[[s0 +t0, s1 +t0] for [s0, s1] in starvation_hours]
    # print(t0, starvation)
    if not np.isnan(t2) :
        starvation = [[s0, np.clip(s1, a_min=s0, a_max=t2)] for [s0, s1] in starvation if s0<=t2]
    dict = {'birth': t0,
            'puppation': e['puppation_time_in_hours'],
            'death': t2,
            'age' : e['age'],
            'sim_start' : t3,
            'mass': s['mass'].values.tolist(),
            'length': s['length'].values.tolist(),
            'reserve': s['reserve'].values.tolist(),
            'reserve_density': s['reserve_density'].values.tolist(),
            'hunger': s['hunger'].values.tolist(),
            'feeder_reoccurence_rate': s['feeder_reoccurence_rate'].values.tolist(),
            'explore2exploit_bias': s['explore2exploit_bias'].values.tolist(),
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
