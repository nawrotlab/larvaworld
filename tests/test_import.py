import larvaworld
from larvaworld.lib import reg
from larvaworld.lib.process.building import import_dataset, import_datasets


def test_import_Schleyer() :
    kws0 = {
        'labID': 'Schleyer',
        'group_id': 'exploration',
        # 'enrich' : True
    }

    # Merged case
    N=60
    kws1 = {
        'parent_dir': 'exploration',
        'merged': True,
        'N': N,
        'min_duration_in_sec': 180,
        # 'id': f'40controls',
        'refID': f'exploration.{N}controls',
        **kws0
    }

    # Single dish case
    folder='dish04'
    kws2 = {
        'parent_dir': f'exploration/{folder}',
        'merged': False,
        'N': None,
        'min_duration_in_sec': 90,
        'id': folder,
        'refID': f'exploration.{folder}',
        **kws0
    }


    for kws in [kws1, kws2] :
        d = import_dataset(**kws)
        assert isinstance(d, larvaworld.lib.LarvaDataset)


def test_import_Jovanic() :
    kws0 = {
        'labID': 'Jovanic',
        # 'group_id': 'exploration',
        # 'add_reference': True,
        # 'enrich' : True,
        'merged' : False
    }

    # kws2 = {
    #     'parent_dir': 'SS888',
    #     'source_ids': ['AttP240', 'SS888Imp', 'SS888'],
    #     # 'id': f'merged_dishes',
    #     **kws0
    # }

    kws1 = {
        'parent_dir': 'Refeeding_yeast_paste_baseline',
        'source_ids': ['Fed', 'Fed_paste', 'Pd', 'Pd_paste',],
       # 'refIDs': ['AttP240.Fed', 'AttP240.Deprived', 'AttP240.Starved'],
        'colors':['green', 'lightblue', 'red', 'yellow'],
        **kws0
    }

    kws2 = {
        'parent_dir': 'Refeeding/AttP2',
        'source_ids': ['Fed', 'Refed', 'Starved'],
        'refIDs': ['AttP2.Fed', 'AttP2.Refed', 'AttP2.Starved'],
        'colors':['green', 'lightblue', 'red'],
        'time_slice':(0,60),
        **kws0
    }


    for kws in [kws1] :
        ds = import_datasets(**kws)
        for d in ds :
            assert isinstance(d, larvaworld.lib.LarvaDataset)

def xxtest_import_Berni() :
    kws0 = {
        'labID': 'Berni',
        # 'group_id': 'exploration',
        # 'add_reference': True,
        'enrich' : True,
        'merged' : False
    }

    kws1 = {
        'parent_dir': 'exploration',
        'source_ids': ['BL_22_control', 'BL_33_control', 'BL_rprhid'],
        'refIDs': ['exploration.BL_22_control', 'exploration.BL_33_control', 'exploration.BL_rprhid'],
        **kws0
    }

    for kws in [kws1]:
        ds = import_datasets(**kws)
        for d in ds:
            assert isinstance(d, larvaworld.lib.LarvaDataset)