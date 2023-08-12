import os

import numpy as np
# SCRIPTS_DIR =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from larvaworld.lib import reg,aux

def test_conf() :
    reg.VERBOSE=1
    conftype = 'Exp'
    id = 'dish'
    id_new = 'dish_test'
    C=reg.conf[conftype]
    ids=C.confIDs
    conf=C.getID(id)
    conf_exp=C.expand(id)


    C.setID(id_new, conf.get_copy())
    assert aux.checkEqual(ids + [id_new], C.confIDs)
    C.delete(id_new)
    assert aux.checkEqual(ids, C.confIDs)

