import pandas as pd
import param
from scipy.stats import multivariate_normal

from larvaworld.lib import aux
from larvaworld.lib.param import PositiveNumber, PositiveInteger, NestedConf, StringRobust, OptionalPositiveNumber, \
    OptionalPositiveRange, ClassAttr, ClassDict


class Compound(NestedConf):
    d=PositiveNumber(doc=f'density in g/cm**3')
    w=PositiveNumber(doc=f'molecular weight (g/mol)')
    nC=PositiveInteger(doc=f'number of carbon atoms')
    nH=PositiveInteger(doc=f'number of hydrogen atoms')
    nO=PositiveInteger(doc=f'number of oxygen atoms')
    nN=PositiveInteger(doc=f'number of nitrogen atoms')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.ww=self.nC+self.nH/12*1+self.nO/12*16+self.nN/12*14


# Compounds
compound_dict = aux.AttrDict(
{
    'glucose': Compound(w=180.18,nC=6, nH=12, nO=6),
    'dextrose': Compound(w=198.17,nC=6, nH=12, nO=7),
    'saccharose': Compound(w=342.30,nC=12, nH=22, nO=11),
    'yeast': Compound(w=274.3,nC=19, nH=14, nO=2),
    'agar': Compound(w=336.33,nC=0, nH=38, nO=19),
    'cornmeal': Compound(w=359.33,nC=27, nH=48, nO=20),
    # 'apple_juice': Compound(w=180.18,nC=6, nH=12, nO=6),
    'water': Compound(w=18.01528,nC=0, nH=2, nO=1)
}
)

all_compounds = [a for a in list(compound_dict.keys()) if a not in ['water']]
nutritious_compounds = [a for a in list(compound_dict.keys()) if a not in ['water', 'agar']]

# class Composition(param.Dict):
#     def __init__(self, default=aux.AttrDict({k : 0.0 for k in all_compounds}),**params):
#         param.Dict.__init__(self,default=default, **params)


class Substrate(NestedConf):
    composition=param.Dict({k : 0.0 for k in all_compounds},doc='The substrate composition')
    quality = param.Magnitude(1.0,doc='The substrate quality as percentage of nutrients relative to the intact substrate type')

    def __init__(self,quality =1.0,type=None,**kwargs):
        if type is not None and type in substrate_dict.keys():
            composition =substrate_dict[type].composition
        else :
            composition={k : kwargs[k] if k in kwargs.keys() else 0.0 for k in all_compounds}
        super().__init__(composition=composition,quality =quality)
        self.d_water = 1
        self.d_yeast_drop = 0.125  # g/cm**3 https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi3iaeqipLxAhVPyYUKHTmpCqMQFjAAegQIAxAD&url=https%3A%2F%2Fwww.mdpi.com%2F2077-0375%2F11%2F3%2F182%2Fpdf&usg=AOvVaw1qDlMHxBPu73W8B1vZWn76
        self.V_drop = 0.05  # cm**3
        self.d = self.d_water + sum(list(self.composition.values()))
        self.C=self.get_C()
        self.X=self.get_X()
        self.X_ratio=self.get_X_ratio()

    def get_d_X(self, compounds=None, quality=None):
        if quality is None:
            quality=self.quality
        if compounds is None:
            compounds=nutritious_compounds
        return sum([self.composition[c] for c in compounds]) * quality

    def get_w_X(self, compounds = None):
        if compounds is None:
            compounds=nutritious_compounds
        d_X=self.get_d_X(compounds, quality=1)
        if d_X>0 : return sum([self.composition[c]*compound_dict[c].ww for c in compounds])/d_X
        else :return 0.0

    def get_X(self, quality=None, compounds = None):
        if quality is None :
            quality=self.quality
        if compounds is None:
            compounds=nutritious_compounds
        d_X = self.get_d_X(compounds, quality)
        if d_X > 0: return d_X/self.get_w_X(compounds)
        else :return 0.0

    def get_mol(self, V, **kwargs):
        return self.get_X(**kwargs)*V

    def get_f(self, K,**kwargs):
        X=self.get_X(**kwargs)
        return X/(K+X)

    def get_C(self, quality=None):
        return self.d_water / compound_dict['water'].w + self.get_X(quality, compounds=all_compounds)

    def get_X_ratio(self, **kwargs):
        return self.get_X(**kwargs)/self.get_C(**kwargs)

# Compound densities (g/cm**3)
substrate_dict = aux.AttrDict(
{
    'agar': Substrate(agar=0.016),
    'standard': Substrate(glucose=0.1, yeast=0.05, agar=0.016),
    'sucrose': Substrate(glucose=0.0171, agar=0.004),
    'cornmeal': Substrate(glucose=517 / 17000,dextrose=1033 / 17000, cornmeal=1716 / 17000, agar=93 / 17000),
    'cornmeal2': Substrate(dextrose=450 / 6400, yeast=90 / 6400, cornmeal=420 / 6400, agar=42 / 6400),     #     [1] M. E. Wosniack, N. Hu, J. Gjorgjieva, and J. Berni, “Adaptation of Drosophila larva foraging in response to changes in food distribution,” bioRxiv, p. 2021.06.21.449222, 2021.
    'PED_tracker': Substrate(saccharose=0.01, yeast=0.1875,agar=5),
    # 'apple_juice': Substrate(glucose=0.00171, agar=0.004, apple_juice=0.02625)
}
)



class Odor(NestedConf):
    id = StringRobust(None, doc='The unique ID of the odorant')
    intensity = OptionalPositiveNumber(softmax=10.0, doc='The peak concentration of the odorant in micromoles')
    spread = OptionalPositiveNumber(softmax=10.0, doc='The spread of the concentration gradient around the peak')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._update_distro()

    @param.depends('intensity','spread', watch=True)
    def _update_distro(self):
        if self.intensity is not None and self.spread is not None:
            self.dist = multivariate_normal([0, 0], [[self.spread, 0], [0, self.spread]])
            self.peak_value = self.intensity / self.dist.pdf([0, 0])
        else:
            self.dist = None
            self.peak_value = 0.0

    def gaussian_value(self, pos):
        if self.dist :
            return self.dist.pdf(pos) * self.peak_value
        else :
            return None

    def draw_dist(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import multivariate_normal
        I = 2
        s = 0.0002
        r = 0.05
        x, y = np.mgrid[-r:r:.001, -r:r:.001]
        rv = multivariate_normal([0, 0], [[s, 0], [0, s]])
        p0 = rv.pdf((0, 0))
        data = np.dstack((x, y))
        z = rv.pdf(data) * I / p0
        plt.contourf(x, y, z, cmap='coolwarm')
        plt.show()




class Epoch(NestedConf):
    age_range = OptionalPositiveRange((0.0, None),softmax=100.0, hardmax=250.0, doc='The beginning and end of the epoch in hours post-hatch.')
    substrate = ClassAttr(Substrate,default=Substrate(type='standard'), doc='The substrate of the epoch')




class Life(NestedConf):
    age = OptionalPositiveNumber(0.0,softmax=100.0, hardmax=250.0, doc='The larva age in hours post-hatch at the start of the behavioral simulation. The larva will grow to that age based on the DEB model. If age is None the larva will grow to pupation.')
    epochs = ClassDict(item_type=Epoch, doc='The feeding epochs comprising life history.')


class Life3(NestedConf):
    age = OptionalPositiveNumber(0.0,softmax=100.0, hardmax=250.0, doc='The larva age in hours post-hatch at the start of the behavioral simulation. The larva will grow to that age based on the DEB model. If age is None the larva will grow to pupation.')
    age_ticks=param.List([0.0], item_type=float, doc='The larva age in hours post-hatch at the end of the rearing periods.The last-one is always equal to the final age (or None). The first is 0.0.')
    subs=param.List([], item_type=Substrate, doc='The substrates of the rearing periods.')
    reach_pupation=param.Boolean(False, doc='If True the larva will grow to pupation.')

    @param.depends('grow_to_pupation')
    def update_age_to_inf(self):
        if self.reach_pupation :
            self.age=param.Infinity
        else:
            self.age=max(self.age_ticks)

    @param.depends('age')
    def update_last_tick(self):
        if self.age not in self.age_ticks:
            self.age_ticks=sorted([t for t in self.age_ticks if t<self.age])
            self.age_ticks.append(self.age)


    @param.depends('age_ticks', 'epoch_substrates')
    def update_substrates(self):
        while self.Nmismatch!=0:
            if self.Nmismatch>0:
                self.subs+=Substrate()
            elif self.Nmismatch<0 :
                self.subs.pop()


    @property
    def Ncorrect(self):
        return len(self.age_ticks)-1

    @property
    def Ncurrent(self):
        return len(self.epoch_substrates)

    @property
    def Nmismatch(self):
        return self.Ncorrect-self.Ncurrent
