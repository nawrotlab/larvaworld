import numpy as np

from larvaworld.lib import reg,aux


def test_conf() :
    reg.VERBOSE=1
    conftype = 'Exp'
    id = 'dish'
    id_new = 'dish_test'
    ids=reg.storedConf(conftype)
    conf=reg.loadConf(conftype,id)
    conf_exp=reg.expandConf(conftype,id)


    reg.saveConf(conftype,id_new, conf.get_copy())
    assert aux.checkEqual(ids + [id_new], reg.storedConf(conftype))
    reg.deleteConf(conftype,id_new)
    assert aux.checkEqual(ids, reg.storedConf(conftype))

def test_reset() :
    reg.resetConfs()
