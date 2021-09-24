import sys
sys.path.insert(0, '../..')
import lib.aux.naming as nam
from lib.conf.conf import store_confs
from lib.conf.init_dtypes import store_dtypes, store_controls
import lib.aux.functions as fun
from lib.stor import paths

def store_RefPars() :
    d = {
        'length': 'body.initial_length',
        nam.freq(nam.scal(nam.vel(''))): 'brain.crawler_params.initial_freq',
        'stride_reoccurence_rate': 'brain.intermitter_params.crawler_reoccurence_rate',
        nam.mean(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_mu',
        nam.std(nam.scal(nam.chunk_track('stride', nam.dst('')))): 'brain.crawler_params.step_to_length_std',
        nam.freq('feed'): 'brain.feeder_params.initial_freq',
        # **{p: p for p in ['initial_x', 'initial_y', 'initial_front_orientation']}
    }
    fun.save_dict(d, paths.RefParsFile, use_pickle=False)

if __name__ == '__main__':
    store_dtypes()
    store_controls()
    store_confs()
    store_RefPars()