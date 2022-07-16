
from lib.stor.managing import import_datasets


ds = import_datasets(datagroup_id = 'Jovanic lab', parent_dir='SS888',source_ids=['AttP240', 'SS888Imp', 'SS888'], enrich=True, merged=False)
# ds = import_datasets(datagroup_id = 'Jovanic lab', parent_dir='SS888_0_60',source_ids=['AttP240', 'SS888Imp', 'SS888'], enrich=True, merged=False, time_slice=(0,60))
print(ds)

