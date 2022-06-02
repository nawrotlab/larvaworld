import argparse
import sys
import time
import numpy as np
import warnings
import copy
import itertools
import os
# Create composite figure
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import matplotlib.pyplot as plt


import numpy as np
from matplotlib import ticker
import seaborn as sns
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
# Create composite figure
# from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
from lib.anal.evaluation import EvalRun
from lib.conf.stored.conf import kConfDict
from lib.anal.argparsers import MultiParser, update_exp_conf

from lib.anal.plot_aux import BasePlot, modelConfTable
from lib.anal.plotting import plot_dispersion

s = time.time()
MP = MultiParser(['eval_conf'])
p = MP.add()
# p.add_argument('experiment', choices=kConfDict('Ga'), help='The experiment mode')
p.add_argument('-video', '--show_screen', action="store_true", help='Whether to render the screen visualization')
# p.add_argument('-offline', '--offline', action="store_true", help='Whether to run a full LarvaworldSim environment')
#
# p.add_argument('-mID0', '--base_model', choices=kConfDict('Model'), help='The model configuration to optimize')
# p.add_argument('-mID1', '--bestConfID', type=str, help='The model configuration ID to store the best genome')

args = p.parse_args()
d = MP.get(args)
# exp = args.experiment
# base_model = args.base_model
# bestConfID = args.bestConfID
video = args.show_screen
# offline = args.offline
eval_conf=d.eval_conf
evrun = EvalRun(**eval_conf)
evrun.run(video=video)
evrun.eval()
evrun.plot_results()
evrun.plot_models()

