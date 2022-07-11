
from lib.stor.managing import import_Jovanic_datasets


parent_dir='SS888'
# parent_dir='SS888_0_60'
# parent_dir='18h'
# parent_dir='AttP240'


ds = import_Jovanic_datasets(parent_dir=parent_dir,source_ids=['AttP240', 'SS888Imp', 'SS888'], enrich=True)
# ds = import_Jovanic_datasets(parent_dir=parent_dir,source_ids=['AttP240','SS888Imp', 'SS888'], enrich=True, time_slice=(0,60))
print(ds)

