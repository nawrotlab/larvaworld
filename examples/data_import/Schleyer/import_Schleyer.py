from larvaworld.lib import reg


from larvaworld.lib.process.building import import_dataset

kws0 = {
        'labID': 'Schleyer',
        'group_id': 'exploration',
    }

    # Merged case
N=40
kws1 = {
    'parent_dir': 'exploration',
    'merged': True,
    'N': N,
    'min_duration_in_sec': 120,
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



d1 = import_dataset(**kws1)
d2 = import_dataset(**kws2)

