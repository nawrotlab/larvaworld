import larvaworld
from larvaworld.lib import reg
from larvaworld.lib.process.building import import_dataset, import_datasets


def test_import_Schleyer() :
    kws0 = {
        'labID': 'Schleyer',
        'group_id': 'exploration',
    }

    # Merged case
    N=60
    kws1 = {
        'parent_dir': 'exploration',
        'merged': True,
        'N': N,
        'min_duration_in_sec': 180,
        'refID': f'exploration.{N}controls',
        **kws0
    }

    # Single dish case
    folder='dish01'
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
        'merged' : False
    }



    kws1 = {
        'parent_dir': 'ProteinDeprivation',
        'source_ids': ['Fed', 'Pd'],
        'colors':['green', 'red'],
        **kws0
    }



    for kws in [kws1] :
        ds = import_datasets(**kws)
        for d in ds :
            assert isinstance(d, larvaworld.lib.LarvaDataset)
