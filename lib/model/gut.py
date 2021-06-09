import numpy as np

class Gut :
    def __init__(self, M_gm=0.001, Lb=0.03178400207913541, f=1,absorption = 0.5, **kwargs):
        self.__dict__.update(kwargs)
        # self.mu_E=mu_E
        # self.w_V=w_V
        # self.w_E=w_E
        # self.w_X=w_X
        # self.w_P=w_P
        # self.y_P_X=y_P_X
        self.absorption=absorption
        # self.d_V=d_V
        # Arbitrary parameters
        self.M_gm = M_gm  # Max vol specific gut capacity (mol/cm**3)
        self.V_gm = self.M_gm * self.w_X / self.d_X  # Max vol specific gut volume (-)

        self.tau_gut = self.get_tau_gut(f, self.J_X_Am, Lb)
        # print(self.V_gm)
        # print(self.V_gm)
        self.mol_not_digested = 0
        self.mol_not_absorbed = 0
        self.mol_faeces = 0
        self.mol_absorbed = 0
        self.p_A = 0
        self.mol_ingested = 0
        # self.gut_ps = []
        self.gut_capacity = 0
        self.max_capacity = self.gut_max_capacity(Lb**3)
        self.gut_X = 0
        self.gut_P = 0
        self.gut_f = 0
        self.f=0

        self.init_dict()

    def get_tau_gut(self, f, J_X_Am, Lb):
        # print()
        return self.M_gm / (J_X_Am/Lb) / f

    def get_Nticks(self, dt):
        self.gut_Nticks = int(self.tau_gut / dt)
        # print(self.M_gm, self.V_gm, self.tau_gut * 24, self.gut_Nticks)

    def update(self, V, X_mol=0):

        self.mol_ingested += X_mol
        self.gut_X += X_mol
        self.max_capacity = self.gut_max_capacity(V)
        self.gut_capacity = self.get_capacity()
        self.f=self.compute_f()
        # self.digestion_capacity_per_tick = 0
        self.digestion_capacity_per_tick = self.max_capacity / (self.gut_Nticks * self.f)
        self.run()
        # print(self.get_gut_occupancy()*100)
        # if X_mol>0 :
        #     print(int(X_mol/self.digestion_capacity_per_tick), int(self.get_gut_occupancy()*100))

        self.update_dict()

    def run(self):
        dX=np.min([self.gut_X, self.digestion_capacity_per_tick])
        self.gut_X-=dX
        # print(self.gut_X)
        self.gut_f+=dX*(1-self.y_E_X)
        self.gut_P+=dX*self.y_E_X
        # Absorb from product
        dPu = np.min([self.gut_P, self.digestion_capacity_per_tick*self.y_E_X])
        # print(dPu, self.gut_P, self.digestion_capacity_per_tick)
        # dPu = np.min([self.gut_P, self.digestion_capacity_per_tick*(1-self.y_P_X)*self.absorption])
        # dPu = np.min([self.gut_P, self.digestion_capacity_per_tick*self.y_P_X])
        self.gut_P -= dPu
        # ps[:, 1] -= dPus
        # Absorb from food
        # dPu = dX*self.y_E_X
        # dPu = dX*self.y_E_X * self.absorption

        # dPu = np.sum(dPus)

        self.mol_absorbed += dPu
        self.p_A = dPu * self.mu_E
        # self.gut_ps = ps.tolist()

    def get_capacity(self):
        # if len(self.gut_ps) == 0:
        #     self.gut_capacity = 0
        # else:
            # ps = np.array(self.gut_ps)
        over=self.gut_X + self.gut_P + self.gut_f- self.max_capacity
        # over=self.gut_X + self.gut_P - self.max_capacity
        if over > 0 :
            df=over*(1-self.y_E_X)
            dX=(self.gut_X/self.gut_X + self.gut_P)*(over-df)
            dP=over-dX-df
            self.gut_X-=dX
            self.gut_P-=dP
            self.gut_f-=df
            self.mol_not_digested += dX
            self.mol_not_absorbed += dP
            self.mol_faeces += df
        return self.gut_X + self.gut_P + self.gut_f
        # while self.gut_capacity > self.max_capacity:
        #     N, P, dX = ps[-1, :]
        #     self.mol_not_digested += N * dX
        #     self.mol_faeces += P
        #     ps = np.delete(ps, (-1), axis=0)
        #     self.gut_capacity = self.gut_X + self.gut_P
            # self.gut_ps = ps.tolist()

    def get_M_absorbed(self):
        return self.mol_absorbed * self.w_E * 1000

    def get_M_ingested(self):
        return self.mol_ingested * self.w_X * 1000

    def get_M_faeces(self):
        return self.mol_faeces * self.w_P * 1000

    def get_M_not_digested(self):
        return self.mol_not_digested * self.w_X * 1000

    def get_M_not_absorbed(self):
        return self.mol_not_absorbed * self.w_P * 1000

    def get_R_absorbed(self):
        return self.mol_absorbed / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_R_faeces(self):
        return self.mol_faeces / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_R_not_digested(self):
        return self.mol_not_digested / self.mol_ingested if self.mol_ingested != 0 else 0

    def get_gut_occupancy(self):
        return self.gut_capacity / self.max_capacity

    def get_M_gut(self):
        return self.gut_capacity * self.w_V * 1000

    def get_V_gut(self):
        return self.V_gm * self.V

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
        self.dict = {k: [] for k in self.dict_keys}

    def update_dict(self):
        gut_dict_values = [
            self.get_M_gut(),
            self.get_M_ingested(),
            self.get_M_absorbed(),
            self.get_M_faeces(),
            self.get_M_not_digested(),
            self.get_M_not_absorbed(),
            self.get_R_faeces(),
            self.get_R_absorbed(),
            self.get_R_not_digested(),
            self.get_gut_occupancy(),
            self.p_A * 10**6,
            self.f
        ]
        for k, v in zip(self.dict_keys, gut_dict_values):
            self.dict[k].append(v)

    def compute_f(self):
        occ = self.get_gut_occupancy()
        return occ / (occ + 0.01)
        # return occ / (occ + self.K)