from pathlib import Path

from larvaworld.lib import reg, util
from larvaworld.lib.reg import config


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


def test_ref_path_to_ref_preserves_absolute_dir(tmp_path):
    dataset_dir = tmp_path / "workspace" / "datasets" / "alpha"

    path = reg.conf.Ref.path_to_Ref(dir=str(dataset_dir))

    assert path == str(dataset_dir / "data" / "conf.txt")


def test_ref_path_to_ref_roots_relative_dir_in_data_dir():
    path = reg.conf.Ref.path_to_Ref(dir="SchleyerGroup/processed/dish01")

    assert path == str(
        Path(config.DATA_DIR)
        / "SchleyerGroup"
        / "processed"
        / "dish01"
        / "data"
        / "conf.txt"
    )
