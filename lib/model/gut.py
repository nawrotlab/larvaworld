import numpy as np


class Gut:
    def __init__(self,deb, M_gm=0.0015, V_bite=0.001, save_dict=True, **kwargs):
        self.deb=deb
        # self.mu_E=mu_E
        # self.w_V=w_V
        # self.w_E=w_E
        # self.w_X=w_X
        # self.w_P=w_P
        # self.y_P_X=y_P_X
        # self.absorption = absorption
        # Arbitrary parameters
        self.M_gm = M_gm  # Max vol specific gut capacity (mol/cm**3)
        self.V_gm = self.M_gm * self.deb.w_V / self.deb.d_V  # Max vol specific gut volume (-)
        self.V_bite = V_bite  # V_bite : Vol spec vol of food per feeding motion

        self.get_tau_gut(self.deb.base_f, self.deb.J_X_Am, self.deb.Lb)
        # print(self.V_gm)
        # raise
        # print(self.V_gm)
        self.mol_not_digested = 0
        self.mol_not_absorbed = 0
        self.mol_faeces = 0
        self.mol_absorbed = 0
        self.p_A = 0
        self.mol_ingested = 0
        # self.gut_ps = []
        self.gut_capacity = 0
        self.max_capacity = self.gut_max_capacity(self.deb.V)
        self.gut_Vmax=self.gut_Vmax(self.deb.V)
        self.gut_V=0
        self.gut_X = 0
        self.gut_P = 0
        self.gut_f = 0
        self.f = self.deb.base_f
        self.Nfeeds = 0
        if save_dict:
            self.dict = self.init_dict()
        else:
            self.dict = None

    def get_tau_gut(self, f, J_X_Am, Lb):
        # print()
        self.tau_gut = self.M_gm / (J_X_Am / Lb) / f

    def get_Nticks(self, dt):
        self.gut_Nticks = int(self.tau_gut / dt)
        # print(self.M_gm, self.V_gm, self.tau_gut * 24, self.gut_Nticks)

    def update(self, V, X_mol=0, X_V=0):
        # a = self.deb.p_Amm_dt - self.deb.J_X_Amm_dt * self.deb.y_E_X * self.deb.mu_E
        # if np.abs(a) > 10**-3:
        #     print(a, self.f, X_mol,X_V,self.deb.age)
        if X_mol>0 :
            self.Nfeeds += 1
            self.mol_ingested += X_mol
            self.gut_X += X_mol
            # self.f=X_mol/X_V
            self.f = self.compute_f(X_mol,X_V)
            # print(self.f)
        # else :
        #     print('ll')


        self.max_capacity = self.gut_max_capacity(V)
        # self.gut_Vmax = self.gut_Vmax(V)
        self.gut_capacity = self.get_capacity()
        # self.f = self.compute_f()
        # print(V-self.deb.V)
        self.digestion_capacity_per_tick = self.deb.J_X_Amm_dt*V*self.f
        # self.digestion_capacity_per_tick = self.max_capacity / self.gut_Nticks
        # self.digestion_capacity_per_tick = self.max_capacity / (self.gut_Nticks / self.f) if self.f != 0 else 0
        self.run()
        # lw=V**(1/3)/self.del_M
        # a=lw**3*0.1**2*np.pi
        # print(self.V_gm)
        # print(np.round([self.intake_as_body_volume_ratio(V), self.intake_as_gut_volume_ratio(V)], 2))
        # if X_mol>0 :
        #     print(int(X_mol/self.digestion_capacity_per_tick), int(self.get_gut_occupancy()*100))
        # if self.dict is not None:
        #     self.update_dict()

    def run(self):
        dX = np.min([self.gut_X, self.digestion_capacity_per_tick])
        # if self.gut_X<self.digestion_capacity_per_tick :
        #     print('xx',(self.digestion_capacity_per_tick-self.gut_X)/self.digestion_capacity_per_tick)
        self.gut_X -= dX
        # print(self.gut_X)
        self.gut_f += dX * self.deb.y_P_X
        self.gut_P += dX * self.deb.y_E_X
        # Absorb from product
        dPu = np.min([self.gut_P, self.digestion_capacity_per_tick*self.deb.y_E_X])
        # if self.gut_P<self.digestion_capacity_per_tick*self.deb.y_E_X :
        #     print('yy',(self.digestion_capacity_per_tick*self.deb.y_E_X-self.gut_P)/(self.digestion_capacity_per_tick*self.deb.y_E_X))
        self.gut_P -= dPu

        self.mol_absorbed += dPu
        self.p_A = dPu * self.deb.mu_E


    def get_capacity(self):
        # if len(self.gut_ps) == 0:
        #     self.gut_capacity = 0
        # else:
        # ps = np.array(self.gut_ps)
        over = self.gut_X + self.gut_P - self.max_capacity
        # over = self.gut_X + self.gut_P + self.gut_f - self.max_capacity
        # over=self.gut_X + self.gut_P - self.max_capacity
        if over > 0:
            # df = over * (1 - self.deb.y_E_X)
            dX = (self.gut_X / self.gut_X + self.gut_P) * over
            # dX = (self.gut_X / self.gut_X + self.gut_P) * (over - df)
            dP = over - dX
            # dP = over - dX - df
            self.gut_X -= dX
            self.gut_P -= dP
            # self.gut_f -= df
            self.mol_not_digested += dX
            self.mol_not_absorbed += dP
            # self.mol_faeces += df
        return self.gut_X + self.gut_P
        # return self.gut_X + self.gut_P + self.gut_f
        # while self.gut_capacity > self.max_capacity:
        #     N, P, dX = ps[-1, :]
        #     self.mol_not_digested += N * dX
        #     self.mol_faeces += P
        #     ps = np.delete(ps, (-1), axis=0)
        #     self.gut_capacity = self.gut_X + self.gut_P
        # self.gut_ps = ps.tolist()

    def get_M_ingested(self):
        return self.mol_ingested * self.deb.w_X * 1000

    def get_M_faeces(self):
        return self.mol_faeces * self.deb.w_P * 1000

    def get_M_not_digested(self):
        return self.mol_not_digested * self.deb.w_X * 1000

    def get_M_not_absorbed(self):
        return self.mol_not_absorbed * self.deb.w_P * 1000

    def get_R_absorbed(self):
        return self.mol_absorbed / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_R_faeces(self):
        return self.mol_faeces / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_R_not_digested(self):
        return self.mol_not_digested / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_gut_occupancy(self):
        return self.gut_capacity / self.max_capacity

    # def compute_occupancy(self):
    #     self.gut_V = self.

    def get_M_gut(self):
        return self.gut_capacity * self.deb.w_V * 1000

    def gut_Vmax(self, V):
        return self.V_gm * V

    def gut_max_capacity(self, V):  # in mol
        return self.M_gm * V

    def init_dict(self):
        self.dict_keys = [
            'M_gut',
            'M_ingested',
            'M_absorbed',
            'M_faeces',
            'M_not_digested',
            'M_not_absorbed',
            'R_faeces',
            'R_absorbed',
            'R_not_digested',
            'gut_occupancy',
            'gut_p_A',
            'gut_f',
        ]
        return {k: [] for k in self.dict_keys}

    def update_dict(self):
        gut_dict_values = [
            self.get_M_gut(),
            self.ingested_mass('mg'),
            self.absorbed_mass('mg'),
            self.get_M_faeces(),
            self.get_M_not_digested(),
            self.get_M_not_absorbed(),
            self.get_R_faeces(),
            self.get_R_absorbed(),
            self.get_R_not_digested(),
            self.get_gut_occupancy(),
            self.p_A / self.deb.V,
            # self.p_A * 10 ** 6,
            self.f,
        ]
        for k, v in zip(self.dict_keys, gut_dict_values):
            self.dict[k].append(v)

    # def finalize_dict(self):
    #     d = self.dict
    #     d['Nfeeds'] = self.Nfeeds
    #     d['pupation'] = self.pupation_time_in_hours
    #     d['emergence'] = self.emergence_time_in_hours
    #     d['death'] = self.death_time_in_hours
    #     d['id'] = self.id
    #     d['simulation'] = self.simulation
    #     d['sim_start'] = self.hours_as_larva
    #     d['epochs'] = self.epochs
    #     d['fr'] = 1 / (self.dt * 24 * 60 * 60)

    def compute_f(self, mol,V):
        X=mol/V
        # occ = self.get_gut_occupancy()
        return X/(self.deb.K+X)
        # return occ / (occ + self.K)

    # def intake_as_body_volume_ratio(self, V, percent=True):
    #     r = self.ingested_volume() / V
    #     if percent :
    #         r*=100
    #     return r

    # def intake_as_gut_volume_ratio(self, V, percent=True):
    #     r =  self.ingested_volume() / (self.V_gm*V)
    #     if percent :
    #         r*=100
    #     return r

    def ingested_mass(self, unit='g'):
        m = self.mol_ingested * self.deb.w_X
        if unit == 'g':
            return m
        elif unit == 'mg':
            return m * 1000

    def absorbed_mass(self, unit='g'):
        m = self.mol_absorbed * self.deb.w_E
        if unit == 'g':
            return m
        elif unit == 'mg':
            return m * 1000

    def ingested_volume(self):
        return self.mol_ingested * self.deb.w_X / self.deb.d_X
