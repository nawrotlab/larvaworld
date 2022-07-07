import random
from typing import Tuple

import lib.aux.dictsNlists as dNl


class Genome:

    def __init__(self, mConf, gConf, space_dict, generation_num=None):
        # for k, v in kwargs.items():
        #     setattr(self, k, v)
        self.gConf = gConf
        self.mConf = mConf
        self.generation_num = generation_num
        self.fitness = None
        self.fitness_dict = None
        self.space_dict = space_dict

    # def crossover(self, other_parent, generation_num):
    #     gConf1=self.gConf
    #     gConf2=other_parent.gConf
    #
    #     gConf = {k: gConf1[k] if random.random() < 0.5 else gConf2[k] for k in self.space_dict.keys()}
    #     mConf_new=dNl.update_nestdict(self.mConf, gConf)
    #     # apply uniform crossover to generate a new genome
    #     return Genome(mConf=mConf_new,gConf=gConf,  generation_num=generation_num, space_dict=self.space_dict)

    # def mutation(self, **kwargs):
    #     gConf=self.gConf
    #     # mConf_f=dNl.flatten_dict(self.mConf)
    #     for k, vs in self.space_dict.items():
    #         v = self.gConf[k]
    #         if vs['dtype'] == bool:
    #             vv = self.mutate_with_probability(v, choices=[True, False], **kwargs)
    #         elif vs['dtype'] == str:
    #             vv = self.mutate_with_probability(v, choices=vs['choices'], **kwargs)
    #         else:
    #             r0, r1 = vs['min'], vs['max']
    #             range = r1 - r0
    #             if vs['dtype'] == Tuple[float]:
    #                 v0, v1 = v
    #                 vv0 = self.mutate_with_probability(v0, range=range, **kwargs)
    #                 vv1 = self.mutate_with_probability(v1, range=range, **kwargs)
    #                 vv = (vv0, vv1)
    #             else:
    #                 vv = self.mutate_with_probability(v, range=range, **kwargs)
    #         self.gConf[k]=vv
    #         # dNl.flatten_dict(self.mConf)[k]=vv
    #         # setattr(self.mConf, k, vv)
    #
    #     self.check_parameter_bounds()
    #     self.mConf = dNl.update_nestdict(self.mConf, self.gConf)
    #
    # def mutate_with_probability(self, v, Pmut, Cmut, choices=None, range=None):
    #     if random.random() < Pmut:
    #         if choices is None:
    #             if v is None:
    #                 return v
    #             else:
    #                 if range is None:
    #                     return random.gauss(v, Cmut * v)
    #                 else:
    #                     return random.gauss(v, Cmut * range)
    #         else:
    #             return random.choice(choices)
    #     else:
    #         return v
    #
    # def check_parameter_bounds(self):
    #     # mConf_f = dNl.flatten_dict(self.mConf)
    #     for k, vs in self.space_dict.items():
    #         if vs['dtype'] in [bool, str]:
    #             continue
    #
    #         else:
    #             r0, r1 = vs['min'], vs['max']
    #             v = self.gConf[k]
    #             if v is None:
    #                 self.gConf[k] = v
    #                 continue
    #             else:
    #
    #                 if vs['dtype'] == Tuple[float]:
    #                     vv0, vv1 = v
    #                     if vv0 < r0:
    #                         vv0 = r0
    #                     if vv1 > r1:
    #                         vv1 = r1
    #                     if vv0 > vv1:
    #                         vv0 = vv1
    #                     self.gConf[k] = (vv0, vv1)
    #                     # setattr(self.mConf, k, (vv0, vv1))
    #                     continue
    #                 if vs['dtype'] == int:
    #                     v = int(v)
    #                 if v < r0:
    #                     self.gConf[k] = r0
    #                     # setattr(self.mConf, k, r0)
    #                 elif v > r1:
    #                     self.gConf[k] = r1
    #                     # setattr(self.mConf, k, r1)
    #                 else:
    #                     self.gConf[k] = v
    #                     # setattr(self.mConf, k, v)
    #     self.mConf = dNl.update_nestdict(self.mConf, self.gConf)

    # def __repr__(self):
    #     fitness = None if self.fitness is None else round(self.fitness, 2)
    #     return self.__class__.__name__ + '(fitness:' + repr(fitness) + ' generation_num:' + repr(
    #         self.generation_num) + ')'

    # def get(self, rounded=False):
    #     dic = {}
    #     for k, vs in self.space_dict.items():
    #         v = self.gConf[k]
    #         if v is not None and rounded:
    #             if vs.dtype == float:
    #                 v = round(v, 2)
    #             elif vs.dtype == Tuple[float]:
    #                 v = (round(v[0], 2), round(v[1], 2))
    #         dic[k] = v
    #     return dic
    # def to_string(self):
    #     # fitness = None if self.fitness is None else round(self.fitness, 2)
    #     kwstrings = [f' {p.name}:{p.v}' for k, p in self.space_dict.items()]
    #     kwstr = ''
    #     for ii in kwstrings:
    #         kwstr = kwstr + ii
    #
    #     return kwstr

    # def to_string(self):
    #     fitness = None if self.fitness is None else round(self.fitness, 2)
    #     kwstrings = [f' {p.name}:{p.v}' for k, p in self.space_dict.items()]
    #     kwstr = ''
    #     for ii in kwstrings:
    #         kwstr = kwstr + ii
    #
    #     return '(fitness:' + repr(fitness) + kwstr + ')'
