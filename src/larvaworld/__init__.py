#from .lib import aux, param
#from .lib import reg, plot, model, process, screen, sim, util
from . import lib, cli, gui
# # from .cli import SimModeParser
# from .lib.process.dataset import BaseLarvaDataset,LarvaDataset, LarvaDatasetCollection
# from .lib.reg import graphs
#
# # from .cli.argparser import SimModeParser
#
lib.reg.resetConfs(init=True)



__author__ = 'Panagiotis Sakagiannis'
__license__ = 'GNU GENERAL PUBLIC LICENSE'
__copyright__ = '2023, Panagiotis Sakagiannis'
__version__ = '0.0.27'
__displayname__ = 'larvaworld'
__name__ = 'larvaworld'