import warnings



warnings.simplefilter(action='ignore')
from pint import UnitRegistry, errors

# class MyUnitRegistry(UnitRegistry):
#     def __getattr__(self, name):
#         if name[0] == '_':
#             try:
#                 value = super(MyUnitRegistry, self).__getattr__(name)
#                 return value
#             except errors.UndefinedUnitError as e:
#                 raise AttributeError()
#         else:
#             return super(MyUnitRegistry, self).__getattr__(name)


units = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)


