import numpy as np



class Gut:
    def __init__(self,deb, M_gm=0.0015, V_bite=0.001, save_dict=True, **kwargs):
        self.deb=deb
        # Arbitrary parameters
        self.M_gm = M_gm  # Max vol specific gut capacity (mol/cm**3)
        self.V_gm = self.M_gm * self.deb.w_V / self.deb.d_V  # Max vol specific gut volume (-)
        self.V_bite = V_bite  # V_bite : Vol spec vol of food per feeding motion

        self.get_tau_gut(self.deb.base_f, self.deb.J_X_Am, self.deb.Lb)
        self.mol_not_digested = 0
        self.mol_not_absorbed = 0
        self.mol_faeces = 0
        self.mol_absorbed = 0
        self.p_A = 0
        self.mol_ingested = 0
        self.gut_capacity = 0
        self.max_capacity = self.gut_max_capacity(self.deb.V)
        self.Vmax=self.get_Vmax(self.deb.V)
        self.V=0
        self.gut_occupancy=self.get_gut_occupancy()
        self.gut_X = 0
        self.gut_P = 0
        self.gut_f = 0
        self.X=0
        self.f = self.deb.base_f
        self.Nfeeds = 0
        if save_dict:
            self.dict = self.init_dict()
        else:
            self.dict = None

    def get_tau_gut(self, f, J_X_Am, Lb):
        self.tau_gut = self.M_gm / (J_X_Am / Lb) / f

    def get_Nticks(self, dt):
        self.gut_Nticks = int(self.tau_gut / dt)
        # print(self.M_gm, self.V_gm, self.tau_gut * 24, self.gut_Nticks)

    def update(self, V, X_V=0):
        self.Vmax = self.get_Vmax(V)
        self.max_capacity = self.gut_max_capacity(V)
        if X_V>0 :
            self.Nfeeds += 1
            self.V += X_V
            self.mol_ingested += self.deb.substrate.C * X_V
            self.gut_X += self.deb.substrate.X *X_V
            self.X = self.compute_X()
            self.f = self.compute_f()

        dX0 = self.deb.J_X_Amm_dt * V
        dX = np.min([dX0, self.gut_X])
        # print(dX0*self.f/self.gut_X)
        # print(1/self.deb.T_factor)
        dV = dX / self.X
        self.V -= dV
        self.gut_X -= dX
        self.gut_f += dX * self.deb.y_P_X
        self.mol_absorbed += dX
        # print(np.mean(self.dict['gut_p_A_deviation'][-500:]))
        # print(np.round(self.p_A/self.deb.deb_p_A,2))
        self.p_A = dX * self.deb.mu_E*self.deb.y_E_X
        # self.p_A = self.deb.J_X_Amm_dt * V * self.deb.mu_E * self.deb.y_E_X * self.f

        self.resolve_occupancy()
        o = self.gut_occupancy = self.get_gut_occupancy()
        self.gut_capacity = self.get_capacity()








        # self.J_X_A = self.deb.J_X_Amm*self.deb.dt*V*self.f

        # dV=
        # N=self.gut_Nticks
        # Vmin=self.V
        # # k0=o/(0.00000001+o)
        # # print(self.tau_gut)
        # # k=self.V/self.gut_Nticks
        # # self.dV = self.V/(k0*N)
        # self.dV = np.min([Vmin, self.V])
        # self.dX = self.dV * self.gut_X / self.V
        # print(self.dV/self.V)
        # self.V -= self.dV
        # self.p_A = self.run(dX)


        # self.dV = self.V/self.gut_Nticks/self.gut_occupancy
        # self.J_X_A = self.max_capacity/self.gut_Nticks
        # self.J_X_A = self.deb.J_X_Amm_dt * V
        # self.p_A =self.deb.J_X_Amm_dt*V*self.f* self.deb.mu_E*self.deb.y_E_X
        # self.J_X_A = self.deb.J_X_Amm_dt*V*self.f
        # print(X_mol/self.J_X_A)
        # self.J_X_A = self.max_capacity / self.gut_Nticks
        # self.J_X_A = self.max_capacity / (self.gut_Nticks / self.f) if self.f != 0 else 0


    def run(self, dX):
        self.gut_X -= dX
        self.gut_f += dX * self.deb.y_P_X/self.deb.y_E_X
        dPu = dX
        self.mol_absorbed += dPu
        return dPu * self.deb.mu_E


    def get_capacity(self):
        return self.gut_X + self.gut_P

    def resolve_occupancy(self):
        over = self.V - self.Vmax
        # over = self.gut_X + self.gut_P - self.max_capacity
        # over = self.gut_X + self.gut_P + self.gut_f - self.max_capacity
        # over=self.gut_X + self.gut_P - self.max_capacity
        if over > 0:
            X=self.gut_X
            P=self.gut_P
            C=X+P
            rX=X/C
            dC=over*C/self.V
            # df = over * (1 - self.deb.y_E_X)
            dX = rX * dC
            # dX = (self.gut_X / self.gut_X + self.gut_P) * (over - df)
            dP = (1-rX)*dC
            # dP = over - dX - df
            self.gut_X -= dX
            self.gut_P -= dP
            # self.gut_f -= df
            self.mol_not_digested += dX
            self.mol_not_absorbed += dP
            # self.mol_faeces += df
            self.V=self.Vmax

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
        return self.V / self.Vmax
        # return self.gut_capacity / self.max_capacity

    # def compute_occupancy(self):
    #     self.gut_V = self.
    def get_dMg(self):
        J_X_A=self.deb.get_J_X_A(f=self.f)
        self.deb.J_X_A_array = np.insert(self.deb.J_X_A_array[0:-1], 0, J_X_A)
        dMg=self.deb.J_X_A_array[0]-self.deb.J_X_A_array[1]
        # dMg=(self.deb.J_X_A_array[0]-self.deb.J_X_A_array[-1])*self.deb.dt*self.deb.T_factor
        if dMg<=0:
            dMg=0
        return dMg

    def get_M_gut(self):
        return self.gut_capacity * self.deb.w_V * 1000

    def get_Vmax(self, V):
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
            'gut_p_A_deviation',
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
            self.p_A/self.deb.deb_p_A,
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

    def compute_X(self):
        X=self.gut_X/self.V if self.V>0 else 0
        return X

    def compute_f(self):
        return self.X/(self.deb.K+self.X)

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
