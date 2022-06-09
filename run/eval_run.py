import sys
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


sys.path.insert(0, '..')
from lib.eval.evaluation import EvalRun
from lib.anal.argparsers import MultiParser


MP = MultiParser(['eval_conf'])
p = MP.add()
p.add_argument('-video', '--show_screen', action="store_true", help='Whether to render the screen visualization')

args = p.parse_args()
d = MP.get(args)
video = args.show_screen
eval_conf=d.eval_conf
evrun = EvalRun(**eval_conf)
evrun.run(video=video)
evrun.eval()
evrun.plot_results()
evrun.plot_models()

