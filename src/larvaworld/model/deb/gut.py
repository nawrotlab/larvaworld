import numpy as np


class Gut:
    def __init__(self, deb, M_gm=10 ** -2, y_P_X=0.9, constant_M_c=True,
                 k_abs=1, f_abs=1, k_dig=1, f_dig=1, M_c_per_cm2=5*10 ** -8, J_g_per_cm2=10 ** -2/(24*60*60), k_c=1, k_g=1,
                 save_dict=True, **kwargs):

        self.deb = deb
        # Arbitrary parameters
        self.M_gm = M_gm  # gut capacity in C-moles for unit of gut volume
        r_w2l = 0.2  # body width to length ratio
        r_gut_w = 0.7  # gut width relative to body width
        r_gut_w2L=0.5*r_w2l * r_gut_w # gut radius relative to body length
        self.r_gut_V = np.pi * r_gut_w2L  # gut volume per unit of body volume
        self.r_gut_A = 2* np.pi * r_gut_w2L  # gut surface area per unit of body surface area
        self.A_g = self.r_gut_A * self.deb.L ** 2  # Gut surface area
        self.V_gm = self.r_gut_V * self.deb.V

        # Constants
        if f_dig is None :
            f_dig=self.deb.base_f
        if k_dig is None:
            k_dig = self.deb.kap_X
        self.k_dig = k_dig  # rate constant for digestion : k_X * y_Xg
        self.f_dig = f_dig  # scaled functional response for digestion : M_X/(M_X+M_K_X)
        self.k_abs = k_abs  # rate constant for absorption : k_P * y_Pc
        self.f_abs = f_abs  # scaled functional response for absorption : M_P/(M_P+M_K_P)
        self.M_c_per_cm2 = M_c_per_cm2  # area specific amount of carriers in the gut per unit of gut surface
        self.J_g_per_cm2 = J_g_per_cm2  # secretion rate of enzyme per unit of gut surface per day
        self.k_c = k_c  # release rate of carriers
        self.k_g = k_g  # decay rate of enzyme
        self.y_P_X = y_P_X  # yield of product by food
        self.constant_M_c = constant_M_c  # yield of product by food
        self.M_c_max = self.M_c_per_cm2 * self.A_g  # amount of carriers in the gut surface
        self.J_g = self.J_g_per_cm2 * self.A_g  # total secretion rate of enzyme in the gut surface

        self.M_X = 0
        self.M_P = 0
        self.M_Pu = 0
        self.M_g = 0
        self.M_c = self.M_c_max

        self.residence_time = self.get_residence_time(self.deb.base_f, self.deb.J_X_Am, self.deb.Lb)
        self.mol_not_digested = 0
        self.mol_not_absorbed = 0
        self.mol_faeces = 0
        self.p_A = 0
        self.mol_ingested = 0
        self.V = 0
        self.gut_X = 0
        self.gut_f = 0
        self.Nfeeds = 0

        if save_dict:
            self.dict = self.init_dict()
        else:
            self.dict = None


    def update(self, V_X=0):

        self.A_g = self.r_gut_A * self.deb.L ** 2  # Gut surface area
        self.V_gm = self.r_gut_V * self.deb.V
        self.M_c_max = self.M_c_per_cm2 * self.A_g  # amount of carriers in the gut surface
        self.J_g = self.J_g_per_cm2 * self.A_g # total secretion rate of enzyme in the gut surface
        # self.M_c = self.M_c_max

        if V_X > 0:
            self.Nfeeds += 1
            self.V += V_X
            M_X_in = self.deb.substrate.X * V_X
            self.mol_ingested += M_X_in
            self.M_X += M_X_in
        self.digest()
        self.resolve_occupancy()

    def resolve_occupancy(self):
        dM=self.M_X + self.M_P-self.Cmax
        if dM>0 :
            rP=self.M_P/(self.M_P+self.M_X)
            dP=rP*dM
            self.M_P-=dP
            self.mol_not_absorbed+=dP
            dX=(1-rP)*dM
            self.M_X-=dX
            self.mol_not_digested += dX

    def digest(self):

        dt = self.deb.dt*24*60*60
        # print(dt)
        # FIXME there should be another term A_g after J_g
        self.M_g += (self.J_g*dt - self.k_g * self.M_g)
        if self.M_X > 0:
            temp = self.k_dig * self.f_dig * self.M_g
            dM_X = - np.min([self.M_X, temp])
        else:
            dM_X = 0
        self.M_X += dM_X
        dM_P_added = -self.y_P_X * dM_X
        if self.M_P > 0 and self.M_c > 0:
            temp = self.k_abs * self.f_abs * self.M_c * dt
            dM_Pu = np.min([self.M_P, temp])
        else:
            dM_Pu = 0
        self.M_P += dM_P_added - dM_Pu
        self.M_Pu += dM_Pu

        if self.constant_M_c:
            self.M_c = self.M_c_max
        else:
            dM_c_released = (self.M_c_max - self.M_c) * self.k_c * dt
            dM_c = dM_c_released - dM_Pu
            self.M_c += dM_c
        self.p_A = dM_Pu * self.deb.mu_E

    def get_residence_time(self, f, J_X_Am, Lb):
        return self.r_gut_V*self.M_gm / (J_X_Am / Lb) / f


    def get_residence_ticks(self, dt):
        self.residence_ticks = int(self.residence_time / dt)

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
        return self.M_Pu / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_faeces(self):
        return self.mol_faeces / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_not_digested(self):
        return self.mol_not_digested / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_M_c(self):
        return self.M_c / self.M_c_max

    @property
    def R_M_g(self):
        return self.M_g / self.J_g

    @property
    def R_M_X_M_P(self):
        return self.M_X / self.M_P if self.M_P != 0 else 0

    @property
    def R_M_X(self):
        return self.M_X / self.Cmax

    @property
    def R_M_P(self):
        return self.M_P / self.Cmax

    @property
    def occupancy(self):
        return self.V / self.Vmax

    @property
    def M(self):
        return self.V * self.deb.d_V * 1000

    @property
    def Vmax(self):
        return self.V_gm * self.deb.V

    @property
    def Cmax(self):  # in mol
        return self.M_gm * self.V_gm

    # def init_dict(self):
    #     self.dict_keys = [
    #         'M_gut',
    #         'M_ingested',
    #         'M_absorbed',
    #         'M_faeces',
    #         'M_not_digested',
    #         'M_not_absorbed',
    #         'R_faeces',
    #         'R_absorbed',
    #         'R_not_digested',
    #         'gut_occupancy',
    #         'gut_p_A',
    #         'gut_f',
    #         'gut_p_A_deviation',
    #         'M_X',
    #         'M_P',
    #         'M_Pu',
    #         'M_g',
    #         'M_c',
    #         'R_M_c',
    #         'R_M_g',
    #         'R_M_X_M_P',
    #     ]
    #     return {k: [] for k in self.dict_keys}
    #
    # def update_dict(self):
    #     gut_dict_values = [
    #         self.M,
    #         self.ingested_mass('mg'),
    #         self.absorbed_mass('mg'),
    #         self.M_faeces,
    #         self.M_not_digested,
    #         self.M_not_absorbed,
    #         self.R_faeces,
    #         self.R_absorbed,
    #         self.R_not_digested,
    #         self.occupancy,
    #         self.p_A / self.deb.V,
    #         self.f,
    #         self.p_A / self.deb.deb_p_A,
    #         self.M_X,
    #         self.M_P,
    #         self.M_Pu,
    #         self.M_g,
    #         self.M_c,
    #         self.R_M_c,
    #         self.R_M_g,
    #         self.R_M_X_M_P,
    #     ]
    #     for k, v in zip(self.dict_keys, gut_dict_values):
    #         self.dict[k].append(v)
    def init_dict(self):
        self.dict_keys = [
            'R_absorbed',
            'mol_ingested',
            # 'mol_absorbed',
            'gut_p_A',
            'M_X',
            'M_P',
            'M_Pu',
            'M_g',
            'M_c',
            'R_M_c',
            'R_M_g',
            'R_M_X',
            'R_M_P',
            'R_M_X_M_P'
        ]
        return {k: [] for k in self.dict_keys}

    def update_dict(self):
        gut_dict_values = [
            self.R_absorbed,
            self.mol_ingested * 1000,
            # self.mol_absorbed,
            self.p_A / self.deb.V,
            self.M_X,
            self.M_P,
            self.M_Pu*1000,
            self.M_g,
            self.M_c,
            self.R_M_c,
            self.R_M_g,
            self.R_M_X,
            self.R_M_P,
            self.R_M_X_M_P,
        ]
        for k, v in zip(self.dict_keys, gut_dict_values):
            self.dict[k].append(v)

    # @property
    # def X(self):
    #     X = self.gut_X / self.V if self.V > 0 else 0
    #     return X

    # @property
    # def f(self):
    #     return self.X / (self.deb.K + self.X)

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
        m = self.M_Pu * self.deb.w_E
        if unit == 'g':
            return m
        elif unit == 'mg':
            return m * 1000

    @property
    def ingested_volume(self):
        return self.mol_ingested * self.deb.w_X / self.deb.d_X

