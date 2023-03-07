from larvaworld.lib import reg


from larvaworld.lib.process.building import import_dataset

kws0 = {
    'datagroup_id': 'Schleyer lab',
    'group_id': 'exploration',
    'add_reference': True,
    'enrich' : True
}

# Merged case
kws1 = {
    'parent_dir': 'exploration',
    'merged': True,
    'N': 40,
    'min_duration_in_sec': 180,
    'id': f'40controls',
    'refID': f'exploration.40controls',
    **kws0
}

# Single dish case
kws2 = {
    'parent_dir': 'exploration/dish03',
    'merged': False,
    'N': None,
    'min_duration_in_sec': 90,
    'id': f'dish03',
    'refID': f'exploration.dish03',
    **kws0
}



d1 = import_dataset(**kws1)
d2 = import_dataset(**kws2)

