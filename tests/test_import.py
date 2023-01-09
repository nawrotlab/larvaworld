from lib import reg
from lib.process.building import import_dataset, import_datasets
from lib.process.larva_dataset import LarvaDataset


def test_import_Schleyer() :
    kws0 = {
        'datagroup_id': 'Schleyer lab',
        'group_id': 'exploration',
        'add_reference': True,
        'enrich' : True
    }

    # Merged case
    kws1 = {
        'parent_dir': 'no_odor',
        'merged': True,
        'N': 22,
        'min_duration_in_sec': 180,
        'id': f'merged_dishes',
        **kws0
    }

    # Single dish case
    kws2 = {
        'parent_dir': 'no_odor/box1-2017-06-22_09_53_27',
        'merged': False,
        'N': None,
        'min_duration_in_sec': 90,
        'id': f'dish',
        **kws0
    }


    for kws in [kws1, kws2] :
        d = import_dataset(**kws)
        assert isinstance(d, LarvaDataset)


def test_import_Jovanic() :
    kws0 = {
        'datagroup_id': 'Jovanic lab',
        # 'group_id': 'exploration',
        'add_reference': True,
        'enrich' : True,
        'merged' : False
    }

    kws1 = {
        'parent_dir': 'SS888',
        'source_ids': ['AttP240', 'SS888Imp', 'SS888'],
        # 'id': f'merged_dishes',
        **kws0
    }

    # kws2 = {
    #     'parent_dir': 'SS888_0_60',
    #     'source_ids': ['AttP240', 'SS888Imp', 'SS888'],
    #     # 'id': f'dish',
    #     **kws0
    # }



    for kws in [kws1] :
        ds = import_datasets(**kws)
        for d in ds :
            assert isinstance(d, LarvaDataset)