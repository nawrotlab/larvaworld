from larvaworld.lib import reg, util


def test_conf():
    conftype = "Exp"
    id = "dish"
    id_new = "dish_test"
    C = reg.conf[conftype]
    ids = C.confIDs
    conf = C.getID(id)
    conf_exp = C.expand(id)

    C.setID(id_new, conf.get_copy())
    assert util.checkEqual(ids + [id_new], C.confIDs)
    C.delete(id_new)
    assert util.checkEqual(ids, C.confIDs)
