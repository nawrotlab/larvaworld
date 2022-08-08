from lib.registry.grouptypes import GroupTypeDict
GT=GroupTypeDict()

from lib.registry.par_dict import BaseParDict

PD = BaseParDict()

from lib.registry.dist_dict import DistDict

DD = DistDict()

from lib.plot.dict import GraphDict

GD = GraphDict()

# CT.build_mDicts(PI=PI, MD=MD)


from lib.process.basic import ProcFuncDict

ProcF = ProcFuncDict()


#
#
# class Registry :
#     def __init__(self, verbose=1):
#         self.verbose = verbose
#
#     def vprint(self, text, verbose=0):
#         if verbose >= self.verbose:
#             print(text)

