import sys
sys.path.insert(0, '../..')

from lib.conf.conf import store_confs
from lib.conf.init_dtypes import store_dtypes, store_controls

if __name__ == '__main__':
    store_dtypes()
    store_controls()
    store_confs()