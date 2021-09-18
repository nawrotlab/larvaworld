import sys
import argparse

sys.path.insert(0, '..')
from lib.stor.managing import build_datasets_old, analyse_datasets, visualize_datasets, enrich_datasets
import lib.aux.argparsers as prs
from lib.conf.conf import *

parser = argparse.ArgumentParser(description="Initialize processing")
parser = prs.add_data_kwargs(parser)
parser = prs.add_build_kwargs(parser)
parser = prs.add_vis_kwargs(parser)
parser = prs.add_replay_kwargs(parser)

parser.add_argument('DataGroup_id', type=str, help='The id of the DataGroup to process.')
parser.add_argument('actions', nargs='+', choices=['reg', 'init', 'build', 'enrich', 'anal', 'vis'],
                    help='The sequential processing actions to perform on the DataGroup.')


args = parser.parse_args()
data_kwargs = prs.get_data_kwargs(args)
build_kwargs = prs.get_build_kwargs(args)
vis_kwargs = prs.get_vis_kwargs(args)
replay_kwargs = prs.get_replay_kwargs(args)


DataGroup_id = args.DataGroup_id
actions = args.actions
# print(build_kwargs, data_kwargs)
# raise
if 'reg' in actions :
    setDataGroup(DataGroup_id)
if 'init' in actions :
    initializeDataGroup(DataGroup_id)
if 'build' in actions :
    build_datasets_old(DataGroup_id, **build_kwargs)
if 'enrich' in actions :
    enrich_datasets(DataGroup_id, **data_kwargs)
if 'anal' in actions :
    fig_dict =analyse_datasets(DataGroup_id, **data_kwargs)
if 'vis' in actions :
    visualize_datasets(DataGroup_id, **data_kwargs, vis_kwargs = vis_kwargs, replay_kwargs=replay_kwargs)

'''
python process.py SampleGroup reg init
python process.py SampleGroup build -raw each
python process.py SampleGroup build -raw all
python process.py SampleGroup enrich -nam dish_0 dish_1 dish_2
python process.py SampleGroup enrich -nam merged

python process.py SampleGroup anal -nam dish_0 dish_1 dish_2
python process.py SampleGroup anal -nam merged
python process.py SampleGroup vis -all -vid -va origin
python process.py SampleGroup vis -idx 1 -vid -trj 10
python process.py sample vis -all -vid -ids 0 2 4
python process.py sample vis -all -vid -tr 500 800
python process.py sample vis -all -vid -dim 0.1 0.05
python process.py sample vis -all -vid -dim 0.01 -ids 0 -fix 5
python process.py sample vis -all -vid -dim 0.01 -ids 0 -fix 5 1
python process.py sample vis -all -img
python process.py sample vis -all -img -va origin
python process.py sample vis -all -img -va origin -vc lin_color
python process.py sample vis -all -img -vcr
python process.py sample vis -all -vid -dim 0.01 -ids 0 -fix 5 -vcb -trj
python process.py sample vis -all -img overlap -dim 0.008 -ids 0 -fix 5 6 -tr 500 1200 -con
python process.py sample vis -all -vid -trj
python process.py sample vis -idx 1 -vid -blk -trj


python process.py test vis -idx 0 -vid -Nsegs
python process.py test vis -idx 0 -vid -ids 0 -fix -1 -dim 0.008 -mid
python process.py test vis -idx 0 -vid -ids 0 -fix -1 -dim 0.008 -con 
python process.py test vis -idx 0 -vid -ids 0 -fix -1 -dim 0.008 -con -Nsegs


Visualize a trajectory sample to detect feeding motions
python process.py JovanicGroup -sub FoodPatches -raw vis -img -trj -rnd -ids 0
python process.py JovanicGroup -sub FoodPatches -raw vis -img -trj -rnd -ids 0 -tkr 20000 25000 -aln center -dim 0.015 0.015
python process.py JovanicGroup -sub FoodPatches -raw vis -img -trj -rnd -ids 1 -tkr 3000 10000 -aln center -dim 0.02 0.02

smaller_dataset(larva_idx=[0], datagroup_id='JovanicGroup', subgroup='FoodPatches', dish_idx=None,dataset_name='dataset', mode='raw', tickrange=[20500, 25000])
smaller_dataset(larva_idx=[1], datagroup_id='JovanicGroup', subgroup='FoodPatches', dish_idx=None,dataset_name='dataset', mode='raw', tickrange=[3000, 10000])

python process.py JovanicGroup -sub FoodPatches -dir dataset_[0] enrich anal
python process.py JovanicGroup -sub FoodPatches -dir dataset_[1] enrich anal
python process.py JovanicGroup -sub FoodPatches -dir dataset_[0] vis -vid -trj -aln center -dim 0.015 0.015
python process.py JovanicGroup -sub FoodPatches -dir dataset_[0] vis -vid -trj -aln center -dim 0.015 0.015 -Nsegs 2
python process.py JovanicGroup -sub FoodPatches -dir dataset_[0] vis -vid -trj -aln center -dim 0.015 0.015 -Nsegs 10 -beh
python process.py JovanicGroup -sub FoodPatches -dir dataset_[1] vis -vid -trj -aln center -dim 0.02 0.02 -Nsegs 10 -beh

python process.py JovanicGroup -sub FoodPatches -dir dataset_[1] vis -vid -fix 8 -dim 0.01 0.01 -Nsegs 10 -beh

'''
