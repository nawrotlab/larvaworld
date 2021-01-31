import os
from itertools import product

import numpy as np
import pandas as pd


def multiparse_dataset(data, parameters, flags, description_to='.', description_as=None, **kwargs):
    if not os.path.exists(description_to):
        os.makedirs(description_to)
    if description_as is None:
        description_as = 'multiparse_description.csv'
    description_file_path = os.path.join(description_to, description_as)
    filename_file_path = os.path.join(description_to, 'filename_description.csv')
    try:
        description = pd.read_csv(description_file_path, index_col=0, header=0)
        file_description = pd.read_csv(filename_file_path, index_col=0, header=0)
    except:
        description = pd.DataFrame(index=parameters, columns=flags)
        file_description = pd.DataFrame(index=parameters, columns=flags)
    for combo in product(parameters, flags):
        param = combo[0]
        flag = combo[1]
        num_segments, dataset_filename = parse_dataset(data, par=param, flag=flag, save_to=description_to,
                                                       **kwargs)
        description.loc[param, flag] = num_segments
        file_description.loc[param, flag] = dataset_filename

    description.to_csv(description_file_path, index=True, header=True)
    file_description.to_csv(filename_file_path, index=True, header=True)
    print(f'Multiparsing complete. Description in {description_file_path}')

def multiparse_dataset_by_sliding_window(data, par, flag, radius_in_ticks=10, description_to='.',
                                         description_as=None, overwrite=True, **kwargs):
    if overwrite :
        import shutil
        try :
            shutil.rmtree(description_to)
        except:
            pass
    if not os.path.exists(description_to):
        os.makedirs(description_to)
    if description_as is None:
        description_as = 'multiparse_description.csv'
    description_file_path = os.path.join(description_to, description_as)
    filename_file_path = os.path.join(description_to, 'filename_description.csv')
    offsets = []
    offset_names = []
    for i in np.arange(radius_in_ticks * 2 + 1):
        offsets.append(int(i - radius_in_ticks))
        offset_names.append(str(int(i - radius_in_ticks)))
    print(f'Offset values to be used :{offsets}')
    try:
        description = pd.read_csv(description_file_path, index_col=0, header=0)
        file_description = pd.read_csv(filename_file_path, index_col=0, header=0)
    except:
        description = pd.DataFrame(index=offset_names, columns=[flag])
        file_description = pd.DataFrame(index=offset_names, columns=[flag])
    for offset, offset_name in zip(offsets, offset_names):
        print(f'Parsing dataset at offset {offset}')
        # param = combo[0]
        # flag = combo[1]
        num_segments, dataset_filename = parse_dataset(data, par=par, flag=flag,
                                                       radius_in_ticks=radius_in_ticks, offset_in_ticks=offset,
                                                       save_to=description_to,
                                                       save_as=f'{par}_around_{flag}_offset_{offset}_dataset.csv',
                                                       **kwargs)
        description.loc[offset_name, flag] = num_segments
        file_description.loc[offset_name, flag] = dataset_filename

    description.to_csv(description_file_path, index=True, header=True)
    file_description.to_csv(filename_file_path, index=True, header=True)
    print(f'Multiparsing by parsing window complete. Description in {description_file_path}')


def parse_dataset(data, par, flag, condition='Not nan', radius_in_ticks=10, offset_in_ticks=0, save_as=None,
                  save_to='.'):
    agent_ids = data.index.unique('AgentID').values

    new_index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=('AgentID', 'Chunk'))
    segments = pd.DataFrame(index=new_index, columns=np.arange(-radius_in_ticks + offset_in_ticks,
                                                               radius_in_ticks + offset_in_ticks + 1))
    for agent_id in agent_ids:
        # print(f'Parsing chunks for {agent_id}')
        agent_data = data.xs(agent_id, level='AgentID', drop_level=True)
        df = agent_data[par]

        if condition == 'Not nan':
            flag_ticks = agent_data.index[agent_data[flag].notnull()].to_list()
        elif condition == 'True':
            flag_ticks = agent_data.index[agent_data[flag] == True].to_list()
        elif condition == 'False':
            flag_ticks = agent_data.index[agent_data[flag] == False].to_list()

        for i, tick in enumerate(flag_ticks):
            # print(tick)
            try:
                seg = df.loc[tick - radius_in_ticks + offset_in_ticks:tick + radius_in_ticks + offset_in_ticks]
                if not seg.isnull().values.any():
                    # print(tick, seg, len(seg))

                    segments.loc[(agent_id, i), :] = seg.tolist()

            except:
                pass
    if save_as is None:
        save_as = f'{par}_around_{flag}_dataset.csv'
    segment_file_path = os.path.join(save_to, save_as)
    segments.to_csv(segment_file_path, index=True, header=True)
    # print(f'Dataset saved as {segment_file_path}')
    return len(segments), save_as
