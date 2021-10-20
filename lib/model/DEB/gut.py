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
        self.V=0
        self.gut_X = 0
        self.gut_f = 0
        self.Nfeeds = 0
        self.SU = SynthesizingUnit(dt=self.deb.dt, K=self.deb.K)
        if save_dict:
            self.dict = self.init_dict()
        else:
            self.dict = None

    def get_tau_gut(self, f, J_X_Am, Lb):
        self.tau_gut = self.M_gm / (J_X_Am / Lb) / f

    def get_Nticks(self, dt):
        self.gut_Nticks = int(self.tau_gut / dt)
        # print(self.M_gm, self.V_gm, self.tau_gut * 24, self.gut_Nticks)

    def update(self, X_V=0):
        if X_V>0 :

            self.Nfeeds += 1
            self.V += X_V
            self.mol_ingested += self.deb.substrate.C * X_V
            self.gut_X += self.deb.substrate.X *X_V
        self.digest()
        self.resolve_occupancy()

    def digest2(self):
        dX, dP=self.SU.step(self.X)
        dV = dP / self.deb.substrate.C
        self.V -= dV
        if self.V<0 :
            self.V=0
        self.gut_X -= dX
        self.mol_absorbed += dP
        self.p_A = dP * self.deb.mu_E * self.deb.y_E_X

    def digest(self):
        dX0 = self.deb.J_X_Amm_dt * self.deb.V
        dX = np.min([dX0, self.gut_X])
        dV = dX / self.X if self.X!=0 else 0.0
        self.V -= dV
        self.gut_X -= dX
        self.gut_f += dX * self.deb.y_P_X
        self.mol_absorbed += dX
        self.p_A = dX * self.deb.mu_E * self.deb.y_E_X
        # self.p_A = self.deb.J_X_Amm_dt * V * self.deb.mu_E * self.deb.y_E_X * self.f





    # def resolve_occupancy(self):
    #     over = self.V - self.Vmax
    #     # over = self.gut_X + self.gut_P - self.max_capacity
    #     # over = self.gut_X + self.gut_P + self.gut_f - self.max_capacity
    #     # over=self.gut_X + self.gut_P - self.max_capacity
    #     # print(over)
    #     if over > 0:
    #
    #         X=self.gut_X
    #         P=self.gut_P
    #         C=X+P
    #         rX=X/C
    #         dC=over*C/self.V
    #         # df = over * (1 - self.deb.y_E_X)
    #         dX = rX * dC
    #         # dX = (self.gut_X / self.gut_X + self.gut_P) * (over - df)
    #         dP = (1-rX)*dC
    #         # dP = over - dX - df
    #         self.gut_X -= dX
    #         self.gut_P -= dP
    #         # self.gut_f -= df
    #         self.mol_not_digested += dX
    #         self.mol_not_absorbed += dP
    #         # self.mol_faeces += df
    #         self.V=self.Vmax

    def resolve_occupancy(self):
        dV = self.V - self.Vmax
        if dV > 0:
            # print(dV)
            dX = dV*self.X
            dX=np.min([self.gut_X, dX])
            self.gut_X -= dX
            self.mol_not_digested += dX
            self.V=self.Vmax

    @property
    def M_ingested(self):
        return self.mol_ingested * self.deb.w_X * 1000

    @property
    def M_faeces(self):
        return self.mol_faeces * self.deb.w_P * 1000

    @property
    def M_not_digested(self):
        return self.mol_not_digested * self.deb.w_X * 1000

    @property
    def M_not_absorbed(self):
        return self.mol_not_absorbed * self.deb.w_P * 1000

    @property
    def R_absorbed(self):
        return self.mol_absorbed / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_faeces(self):
        return self.mol_faeces / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_not_digested(self):
        return self.mol_not_digested / self.mol_ingested if self.mol_ingested != 0 else 0

    @ property
    def occupancy(self):
        return self.V / self.Vmax

    @ property
    def M(self):
        return self.V * self.deb.d_V * 1000

    @ property
    def Vmax(self):
        # print(self.deb.V)
        return self.V_gm * self.deb.V

    @property
    def Cmax(self):  # in mol
        return self.M_gm * self.deb.V

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
            self.M,
            self.ingested_mass('mg'),
            self.absorbed_mass('mg'),
            self.M_faeces,
            self.M_not_digested,
            self.M_not_absorbed,
            self.R_faeces,
            self.R_absorbed,
            self.R_not_digested,
            self.occupancy,
            self.p_A / self.deb.V,
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
    @property
    def X(self):
        X=self.gut_X/self.V if self.V>0 else 0
        return X

    @ property
    def f(self):
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

    def absorbed_mass(self, unit='mg'):
        m = self.mol_absorbed * self.deb.w_E
        if unit == 'g':
            return m
        elif unit == 'mg':
            return m * 1000

    @property
    def ingested_volume(self):
        return self.mol_ingested * self.deb.w_X / self.deb.d_X

# class SynthesizingUnit3:
#     def __init__(self,gut, k_E, b_X=5):
#         self.gut=gut
#         self.q0=1
#
#         self.qE=0
#         self.qX = 1-self.q0-self.qE
#         self.k_X=self.deb.K*b_X
#         self.k_E=k_E
#         self.b_X=b_X
#         self.dt=self.deb.dt
#         self.Pcum=0
#         self.Xicum=0
#         self.Xdcum=0
#
#     def step(self,X):
#         # print(X)
#         # if X in [None, np.nan]:
#         #     X=0
#         P=self.k_E*self.qE*self.dt
#         Xi=X*self.b_X*self.q0*self.dt
#         Xd=self.k_X*self.qX*self.dt
#         self.q0+=P - Xi
#         self.qE+=Xd - P
#         self.qX = 1 - self.q0 - self.qE
#         # print(self.k_E)
#
#         self.Pcum+=P
#         self.Xicum+=Xi
#         self.Xdcum+=Xd
#         # print(self.J_E_A_ratio)
#         print(self.ths)
#         # print(self.Xicum, self.Xdcum, self.Pcum)
#         return P
#
#     @ property
#     def J_E_A(self):
#         X=self.deb.substrate.X
#         o=self.k_X**-1+self.k_E**-1
#         return self.deb.y_E_X*self.b_X*X/(1+self.b_X*X*o)
#
#     @property
#     def J_E_A_ratio(self):
#         # return self.J_E_A/(self.deb.J_E_Amm)
#         return self.J_E_A/(self.deb.V*self.deb.J_E_Amm)
#
#
#
#     @ property
#     def ths(self) :
#         return [int(10**6*qq) for qq in [self.q0, self.qX,self.qE]]
#         # return self.q0, self.qX, self.qE
#
# class SynthesizingUnit2:
#     def __init__(self,deb, b_X=0.000001):
#         self.deb=deb
#         self.q0=1
#
#         # self.qE=0
#         self.qX = 1-self.q0
#         self.k_X=self.deb.K*b_X
#         # self.k_E=k_E
#         self.b_X=b_X
#         self.dt=self.deb.dt
#         self.Pcum=0
#         self.Picum=0
#         self.Xicum=0
#         self.Xcum=0
#
#     def step(self,X):
#         self.Xcum+=X
#         self.Xicum+=X
#         # print(X)
#         # if X in [None, np.nan]:
#         #     X=0
#
#         Xi=self.Xicum/self.deb.gut.Vmax*self.b_X*self.q0*self.dt
#         self.Xicum -= Xi
#         self.Picum += Xi
#         P = self.Picum/self.deb.gut.Vmax*self.k_X * self.qX * self.dt
#
#         self.Pcum += P
#         self.Picum -= P
#         # Xd=self.k_X*self.qX*self.dt
#         self.q0+=P - Xi
#         self.qX+=Xi - P
#         # self.qX = 1 - self.q0 - self.qE
#         # print(self.k_E)
#
#
#         # self.Xicum+=Xi
#
#         # print(self.J_E_A, self.J_E_A_ratio)
#         print(self.ths)
#         # try:
#         #     print(int(100*self.Xicum/self.Xcum), int(100*self.Picum/self.Xcum), int(100*self.Pcum/self.Xcum))
#         # except :
#         #     pass
#         return P
#
#     @ property
#     def J_E_A(self):
#         X=self.deb.substrate.X
#         # o=self.k_X**-1+self.k_E**-1
#         return self.deb.y_E_X*self.k_X*self.b_X*X/(self.k_X+self.b_X*X)
#     #
#     @property
#     def J_E_A_ratio(self):
#         return self.J_E_A/(self.deb.J_E_Am)
#         # return self.J_E_A/(self.deb.V*self.deb.J_E_Amm)
#
#
#
#     @ property
#     def ths(self) :
#         return [int(10**2*qq) for qq in [self.q0, self.qX]]
#         # return self.q0, self.qX, self.qE

class SynthesizingUnit:
    def __init__(self,dt, K=5*10**-5, k_X=0.5, b_X=0.2, X=None):
        self.dt=dt
        self.q0=1
        self.K=K
        self.qX = 1-self.q0
        self.k_X=K*b_X
        # self.k_X=k_X
        self.b_X=b_X
        self.X=X

    def step(self,X=None):
        if X is None :
            X=self.X
        dP=self.k_X*self.qX*self.dt
        dX=X*self.b_X*self.q0*self.dt
        self.q0+=dP - dX
        self.qX+=dX - dP
        # print(dP, dX)
        print(self.ths)
        # print(self.gut.get_R_absorbed())
        return dX, dP

    # @ property
    def J_E_A(self, deb):
        X=deb.substrate.X
        o=self.k_X**-1+deb.k_E**-1
        return deb.y_E_X*self.b_X*X/(1+self.b_X*X*o)

    # @property
    # def J_E_A_ratio(self):
    #     # return self.J_E_A/(self.deb.J_E_Amm)
    #     return self.J_E_A/(self.gut.deb.V*self.gut.deb.J_E_Amm)



    @ property
    def ths(self) :
        return [int(10**2*qq) for qq in [self.q0, self.qX]]
        # return self.q0, self.qX, self.qE

if __name__ == '__main__':
    su=SynthesizingUnit(dt=1/(24*60*60*10), X=0.93)
    Ndays=5
    Nticks=int(Ndays/su.dt)
    for i in range(Nticks) :
        su.step()
    # print(su.X/(su.K+su.X))
    # print(su.k_X/su.b_X)