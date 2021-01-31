import sys
import argparse
import numpy as np

sys.path.insert(0, '..')
from lib.stor.datagroup import *
from lib.stor.managing import *
import lib.conf.env_modes as env
import lib.aux.argparsers as prs

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
    build_datasets(DataGroup_id, **build_kwargs)
if 'enrich' in actions :
    enrich_datasets(DataGroup_id, **data_kwargs)
if 'anal' in actions :
    analyse_datasets(DataGroup_id, **data_kwargs)
if 'vis' in actions :
    visualize_datasets(DataGroup_id, **data_kwargs, vis_kwargs = {**vis_kwargs, **replay_kwargs})
