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
    # 'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor',
    'merged': True,
    'N': 22,
    'min_duration_in_sec': 180,
    'id': f'merged_dishes',
    **kws0
}

# Single dish case
kws2 = {
    # 'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor/box1-2017-05-18_14_52_08',
    'merged': False,
    'N': None,
    'min_duration_in_sec': 90,
    'id': f'dish1',
    **kws0
}

kws3 = {
    # 'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor/box1-2017-06-22_09_53_27',
    'merged': False,
    'N': None,
    'min_duration_in_sec': 90,
    'id': f'dish',
    **kws0
}

# Merged case
kws4 = {
    # 'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor',
    'merged': True,
    'N': 22,
    'min_duration_in_sec': 180,
    'id': f'merged_dishes',
    **kws0
}


# d1 = import_dataset(**kws1)
# d2 = import_dataset(**kws2)
d3 = import_dataset(**kws3)

