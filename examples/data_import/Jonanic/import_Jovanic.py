from larvaworld.lib import reg, aux, plot
from larvaworld.lib.process.building import import_datasets


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

ds = import_datasets(**kws1)
