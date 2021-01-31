import sys
import argparse
import numpy as np

sys.path.insert(0, '../..')
import lib.conf.env_modes as env
from lib.conf.batch_modes import batch_types
from lib.conf import exp_types
import lib.aux.functions as fun


def add_vis_kwargs(parser):
    parser.add_argument('-vid', '--video', action="store_true", help='Select video or image')
    parser.add_argument('-img', '--image_mode', nargs='?', const='final',
                        choices=['final', 'overlap', 'snapshots'], help='Select image mode')
    parser.add_argument('-rnd', '--random_larva_colors', action="store_true", help='Color larvae with random colors')
    parser.add_argument('-beh', '--color_behavior', action="store_true", help='Color behavioral epochs')
    parser.add_argument('-trj', '--trajectories', type=float, nargs='?', const=0.0,
                        help='Show trajectories of specific time duration')
    # parser.add_argument('-trl', '--trail_decay', type=float, help='The duration of the decaying trajectory trail')
    parser.add_argument('-blk', '--black_background', action="store_true", help='Black background')

    parser.add_argument('-head', '--draw_head', action="store_true", help='Color the head and tail')

    return parser


def get_vis_kwargs(args):
    if args.video:
        mode = 'video'
    elif args.image_mode is not None:
        mode = 'image'
    else:
        mode = None
    image_mode = args.image_mode
    if args.trajectories is None:
        trajectories = False
        trail_decay_in_sec = 0.0
    else:
        trajectories = True
        trail_decay_in_sec = args.trajectories

    vis_kwargs = {'mode': mode,
                  'image_mode': image_mode,
                  'trajectories': trajectories,
                  'trail_decay_in_sec': trail_decay_in_sec,
                  'random_larva_colors': args.random_larva_colors,
                  'color_behavior': args.color_behavior,
                  'black_background': args.black_background,

                  'draw_head': args.draw_head,
                  }
    return vis_kwargs


def add_replay_kwargs(parser):
    parser.add_argument('-aln', '--align_mode', choices=['origin', 'arena', 'center'],
                        help='The alignment mode for visualization')
    parser.add_argument('-dyn', '--dynamic_color', choices=['lin', 'ang'],
                        help='Color the trajectories based on velocity')
    parser.add_argument('-ids', '--agent_ids', type=int, nargs='+', help='The indexes of larvae to visualize')
    parser.add_argument('-tkr', '--tick_range', type=int, nargs='+', help='The time range to visualize in ticks')
    parser.add_argument('-fix', '--fix_points', type=int, nargs='+',
                        help='Fixate a midline point to the center of the screen')
    parser.add_argument('-Nsegs', '--draw_Nsegs', type=int, nargs='?', const=2, help='Simplify larva body to N segments')

    parser.add_argument('-dim', '--arena_dims', type=float, nargs='+', help='The arena dimensions in m')
    parser.add_argument('-con', '--draw_contour', action="store_false", help='Hide the contour')
    parser.add_argument('-mid', '--draw_midline', action="store_false", help='Hide the midline')
    parser.add_argument('-cen', '--draw_centroid', action="store_true", help='Show the centroid')
    return parser


def get_replay_kwargs(args):
    if args.tick_range is not None:
        if len(args.tick_range) != 2 or args.tick_range[0] > args.tick_range[1]:
            raise ValueError('Inappropriate tick range')

    fix = args.fix_points
    if fix is not None:
        use_background = True
        if len(fix) == 2 and np.abs(fix[1]) == 1:
            fix_point, secondary_fix_point = fix[0], fix[0]+fix[1]
        elif len(fix) == 1:
            fix_point, secondary_fix_point = fix[0], None
        else:
            raise ValueError('Inappropriate fix points')
    else:
        fix_point, secondary_fix_point = None, None
        use_background = False
    if args.dynamic_color is None:
        dynamic_color = None
    elif args.dynamic_color == 'lin':
        dynamic_color = 'lin_color'
    elif args.dynamic_color == 'ang':
        dynamic_color = 'ang_color'

    dims = args.arena_dims
    if dims is not None:
        if len(dims) == 2:
            arena_pars = env.arena(dims[0], dims[1])
        elif len(dims) == 1:
            arena_pars = env.dish(dims[0])
        else:
            raise ValueError('Inappropriate arena dimensions')
    else:
        arena_pars = None


    replay_kwargs = {'agent_ids': args.agent_ids,
                     'align_mode': args.align_mode,
                     'fix_point': fix_point,
                     'secondary_fix_point': secondary_fix_point,
                     'draw_Nsegs': args.draw_Nsegs,
                     'use_background': use_background,
                     'time_range_in_ticks': args.tick_range,
                     'dynamic_color': dynamic_color,
                     'arena_pars': arena_pars,
                     'draw_contour': args.draw_contour,
                     'draw_midline': args.draw_midline,
                     'draw_centroid': args.draw_centroid,
                     }
    return replay_kwargs


def add_data_kwargs(parser):
    # parser.add_argument('dataset_type', type=str, help='The dataset type name')
    parser.add_argument('-fld', '--folders', nargs='+', type=str, help='Folders under the DataGroup parent dir where to search for datasets')
    parser.add_argument('-suf', '--suffixes', nargs='+', type=str, help='Suffixes of the dataset names')
    parser.add_argument('-nam', '--names', nargs='+', default=['enriched'],type=str, help='Names of the datasets')
    # parser.add_argument('-lst', '--last_common', type=str, default='processed', help='The last common folder of the datasets under the DataGroup parent dir')
    # parser.add_argument('-mode', '--mode', type=str, default='load', choices=['load', 'initialize'],help='Whether to load existing or initialize new datasets')
    parser.add_argument('-load', '--load_data', action="store_false", help='Not load the data from the datasets')
    return parser


def get_data_kwargs(args):
    data_kwargs = {
        'suffixes': args.suffixes,
        'folders': args.folders,
        'names': args.names,
        # 'last_common': args.last_common,
        # 'mode': args.mode,
        'load_data': args.load_data,
    }
    return data_kwargs

def add_build_kwargs(parser):
    parser.add_argument('-d_ids', '--dataset_ids', nargs='+', type=str, help='Ids of newly built datasets')
    parser.add_argument('-raw', '--raw_folders', nargs='+', type=str, help='Folders where to search for raw data when building a dataset')
    parser.add_argument('-t', '--min_duration_in_sec', type=float, nargs='?', default=0.0,
                        help='During dataset building, the minimum duration in sec of included larva tracks.')
    parser.add_argument('-all', '--all_folders', action="store_true", help='Create a single merged dataset from all raw folders')
    parser.add_argument('-each', '--each_folder', action="store_true", help='Create a dataset from each raw folder')
    return parser

def get_build_kwargs(args):
    if args.all_folders :
        raw_folders='all'
    elif args.each_folder :
        raw_folders='each'
    else :
        raw_folders=args.raw_folders
    build_kwargs = {
        'suffixes': args.suffixes,
        'folders': args.folders,
        'names': args.names,
        'ids': args.dataset_ids,
        'raw_folders': raw_folders,
        'min_duration_in_sec': args.min_duration_in_sec,
    }
    return build_kwargs

def add_sim_kwargs(parser):
    parser.add_argument('-id', '--sim_id', type=str, help='The id of the simulation')
    parser.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae')
    parser.add_argument('-t', '--sim_time', type=float, help='The duration of the simulation in min')
    return parser


def get_sim_kwargs(args):
    sim_kwargs = {'sim_id' :args.sim_id,
        'Nagents': args.Nagents,
                  'sim_time': args.sim_time}
    return sim_kwargs


def add_batch_kwargs(parser):
    parser.add_argument('-id_b', '--batch_id', type=str, help='The id of the batch run')
    parser.add_argument('-Nmax', '--max_Nsims', type=int, nargs='?', const=50,
                        help='The maximum number of simulations to run')
    parser.add_argument('-Nbst', '--Nbest', type=int, nargs='?', const=5,
                        help='The number of best configurations to expand')
    return parser


def get_batch_kwargs(args):
    batch_kwargs = {
        'batch_id': args.batch_id,
        'max_Nsims': args.max_Nsims,
                    'Nbest': args.Nbest}
    return batch_kwargs


def add_space_kwargs(parser):
    parser.add_argument('-par', '--pars', type=str, nargs='+', help='The parameters for space search')
    parser.add_argument('-rng', '--ranges', type=float, nargs='+', help='The range of the parameters for space search')
    parser.add_argument('-Ngrd', '--Ngrid', nargs='+', type=int, help='The number of steps for space search')
    return parser


def get_space_kwargs(args):
    Ngrid = args.Ngrid
    if Ngrid is None:
        Ngrid = [5]
    space_kwargs = {'pars': args.pars,
                    'ranges': args.ranges,
                    'Ngrid': Ngrid}
    return space_kwargs
