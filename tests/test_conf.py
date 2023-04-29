import numpy as np

from larvaworld.lib import reg,aux


def test_conf() :
    reg.VERBOSE=1
    conftype = 'Exp'
    id = 'dish'
    id_new = 'dish_test'
    ids=reg.stored.confIDs(conftype)
    conf=reg.stored.get(conftype,id)
    conf_exp=reg.stored.expand(conftype,id)


    reg.stored.set(conftype,id_new, conf.get_copy())
    assert aux.checkEqual(ids + [id_new], reg.stored.confIDs(conftype))
    reg.stored.delete(conftype,id_new)
    assert aux.checkEqual(ids, reg.stored.confIDs(conftype))

def test_reset() :
    reg.stored.resetConfs()
