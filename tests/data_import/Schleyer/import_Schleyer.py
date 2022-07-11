import os

from lib.registry.pars import preg
from lib.stor.managing import import_dataset

kws0 = {
    'datagroup_id': 'Schleyer lab',
    'group_id': 'exploration',
    'add_reference': False
}

# Merged case
kws1 = {
    'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor',
    'merged': True,
    'N': 22,
    'min_duration_in_sec': 180,
    'id': f'merged_dishes',
    **kws0
}

# Single dish case
kws2 = {
    'datagroup_id': 'Schleyer lab',
    'parent_dir': 'no_odor/box1-2017-05-18_14_52_08',
    'merged': False,
    'N': None,
    'min_duration_in_sec': 90,
    'id': f'dish1',
    **kws0
}

d1 = import_dataset(**kws1)
d2 = import_dataset(**kws2)

