
# Compound densities (g/cm**3)
substrate_dict = {
    'agar': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 16 / 1000,
        'cornmeal': 0
    },
    'standard': {  # w_X = 20.45 g/mol
        'glucose': 100 / 1000,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 50 / 1000,
        'agar': 16 / 1000,
        'cornmeal': 0
        # 'KPO4': 0.1/1000,
        # 'Na_K_tartrate': 8/1000,
        # 'NaCl': 0.5/1000,
        # 'MgCl2': 0.5/1000,
        # 'Fe2(SO4)3': 0.5/1000,
    },
    'cornmeal': {
        'glucose': 517 / 17000,
        'dextrose': 1033 / 17000,
        'saccharose': 0,
        'yeast': 0,
        'agar': 93 / 17000,
        'cornmeal': 1716 / 17000
    },
    'PED_tracker': {
        'glucose': 0,
        'dextrose': 0,
        'saccharose': 2 / 200,
        'yeast': 3 * 0.05 * 0.125 / 0.1,
        'agar': 500 * 2 / 200,
        'cornmeal': 0
    },
    #     [1] M. E. Wosniack, N. Hu, J. Gjorgjieva, and J. Berni, “Adaptation of Drosophila larva foraging in response to changes in food distribution,” bioRxiv, p. 2021.06.21.449222, 2021.
    'cornmeal2': {
        'glucose': 0,
        'dextrose': 450 / 6400,
        'saccharose': 0,
        'yeast': 90 / 6400,
        'agar': 42 / 6400,
        'cornmeal': 420 / 6400
    },
    'sucrose': {
        'glucose': 3.42 / 200,
        'dextrose': 0,
        'saccharose': 0,
        'yeast': 0,
        'agar': 0.8 / 200,
        'cornmeal': 0
    }
    # 'apple_juice': {
    #         'glucose': 0.342/200,
    #         'dextrose': 0,
    #         'saccharose': 0,
    #         'yeast': 0,
    #         'agar': 0.8 / 200,
    #         'cornmeal': 0,
    #         'apple_juice': 1.05*5/200,
    #     },

}

class Substrate:
    def __init__(self, type='standard', quality=1.0):
        self.d_water = 1
        self.d_yeast_drop = 0.125 #g/cm**3 https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi3iaeqipLxAhVPyYUKHTmpCqMQFjAAegQIAxAD&url=https%3A%2F%2Fwww.mdpi.com%2F2077-0375%2F11%2F3%2F182%2Fpdf&usg=AOvVaw1qDlMHxBPu73W8B1vZWn76
        self.V_drop = 0.05 # cm**3
        self.quality = quality # cm**3
        # Molecular weights (g/mol)
        self.w_dict={
            'glucose' : 180.18,
            'dextrose' : 198.17,
            'saccharose' : 342.30,
            'yeast' : 274.3, # Baker's yeast
            'agar' : 336.33,
            'cornmeal' : 359.33,
            'water' : 18.01528,
        }
        # Number of carbon atoms
        self.Cmol_dict = {
            'glucose': 6, # C6H12O6
            'dextrose': 6, # C6H14O7
            'saccharose': 12, # C12H22O11
            'yeast': 19,  # C19H14O2
            'agar': 0, # C24H38O19
            'cornmeal': 27,# C27H48O20
            'water': 0,
        }
        self.CHONmol_dict = {
            'glucose': {'C' : 6, 'H' : 12, 'O' : 6, 'N' : 0},  # C6H12O6
            'dextrose': {'C' : 6, 'H' : 12, 'O' : 7, 'N' : 0},  # C6H14O7
            'saccharose': {'C' : 12, 'H' : 22, 'O' : 11, 'N' : 0},  # C12H22O11
            'yeast': {'C' : 19, 'H' : 14, 'O' : 2, 'N' : 0},  # C19H14O2
            'agar': {'C' : 24, 'H' : 38, 'O' : 19, 'N' : 0}, # C24H38O19
            'cornmeal': {'C' : 27, 'H' : 48, 'O' : 20, 'N' : 0},  # C27H48O20
            'water': {'C' : 0, 'H' : 2, 'O' : 1, 'N' : 0},
        }
        self.d_dict= substrate_dict[type]
        self.d = self.d_water + sum(list(self.d_dict.values()))
        self.C=self.get_C()
        self.X=self.get_X()
        self.X_ratio=self.get_X_ratio()

    def get_d_X(self, compounds=['glucose', 'dextrose', 'yeast', 'cornmeal', 'saccharose'], quality=None):
        if quality is None:
            quality=self.quality
        d_X = 0
        for c in compounds:
            d_X += self.d_dict[c]
        return d_X * quality

    def get_w_X(self, compounds = ['glucose', 'dextrose', 'yeast', 'cornmeal', 'saccharose'], quality=None):
        ds={}
        ws={}
        d_X=self.get_d_X(compounds, quality=1)
        for c in compounds:
            ds[c]=self.d_dict[c]
            comp=self.CHONmol_dict[c]
            Cc=comp['C']
            ws[c]=12+comp['H']/Cc*1+comp['O']/Cc*16+comp['N']/Cc*14
        w_X=sum([ds[c] * ws[c] for c in compounds])/d_X
        return w_X

    def get_X(self, quality=None, compounds = ['glucose', 'dextrose', 'yeast', 'cornmeal', 'saccharose']):
        if quality is None :
            quality=self.quality
        d_X = self.get_d_X(compounds, quality)
        w_X = self.get_w_X(compounds)
        X=d_X/w_X
        return X

    def get_mol(self, V, **kwargs):
        return self.get_X(**kwargs)*V

    def get_f(self, K,**kwargs):
        X=self.get_X(**kwargs)
        return X/(K+X)

    def get_C(self, quality=None):
        C=self.d_water / self.w_dict['water'] + self.get_X(quality, compounds=list(self.d_dict.keys()))
        return C

    def get_X_ratio(self, quality=None):
        X=self.get_X(quality = quality)
        C=self.get_C(quality = quality)
        return X/C

