import sys
import numpy as np




sys.path.insert(0, '../../..')
import lib.conf.env_conf as env
from lib.conf.conf import loadConfDict
from lib.conf.init_dtypes import null_dict
import lib.aux.dictsNlists

def add_vis_kwargs(parser):
    parser.add_argument('-hide', '--show_display', action="store_false", help='Hide display')
    parser.add_argument('-vid', '--video_speed', type=float, nargs='?', const=1.0,
                        help='The fast-forward speed of the video')
    parser.add_argument('-img', '--image_mode', nargs='?', const='final',
                        choices=['final', 'overlap', 'snapshots'], help='Select image mode')
    parser.add_argument('-media', '--media_name', type=str, help='Filename for the saved video/image')
    parser.add_argument('-rnd', '--random_colors', action="store_true", help='Color larvae with random colors')
    parser.add_argument('-beh', '--color_behavior', action="store_true", help='Color behavioral epochs')
    parser.add_argument('-trj', '--trajectories', type=float, nargs='?', const=0.0,
                        help='Show trajectories of specific time duration')
    parser.add_argument('-blk', '--black_background', action="store_true", help='Black background')
    parser.add_argument('-head', '--draw_head', action="store_true", help='Color the head and tail')
    parser.add_argument('-con', '--draw_contour', action="store_false", help='Hide the contour')
    parser.add_argument('-mid', '--draw_midline', action="store_false", help='Hide the midline')
    parser.add_argument('-cen', '--draw_centroid', action="store_true", help='Show the centroid')
    parser.add_argument('-vis_clock', '--visible_clock', action="store_false", help='Visible clock')
    parser.add_argument('-vis_state', '--visible_state', action="store_false", help='Visible state')
    parser.add_argument('-vis_scale', '--visible_scale', action="store_false", help='Visible spatial scale')
    parser.add_argument('-vis_ids', '--visible_ids', action="store_true", help='Visible ids')
    return parser


def get_vis_kwargs(args):
    if args.video_speed is not None:
        mode = 'video'
        video_speed = args.video_speed
    elif args.image_mode is not None:
        mode = 'image'
        video_speed = 1
    else:
        mode = None
        video_speed = 1

    if args.trajectories is None:
        trajectories = False
        trajectory_dt = 0.0
    else:
        trajectories = True
        trajectory_dt = args.trajectories

    vis_kwargs = null_dict('visualization', mode=mode, image_mode=args.image_mode, video_speed=video_speed,
                                     show_display=args.show_display, media_name=args.media_name,
                                     draw_head=args.draw_head, draw_centroid=args.draw_centroid,
                                     draw_midline=args.draw_midline, draw_contour=args.draw_contour,
                                     trajectories=trajectories, trajectory_dt=trajectory_dt,
                                     black_background=args.black_background, random_colors=args.random_colors,
                                     color_behavior=args.color_behavior,
                                     visible_clock=args.visible_clock, visible_state=args.visible_state,
                                     visible_scale=args.visible_scale, visible_ids=args.visible_ids,
                                     )
    return vis_kwargs


def add_replay_kwargs(parser):
    parser.add_argument('-trans', '--transposition', choices=['origin', 'arena', 'center'],
                        help='The transposition mode for visualization')
    parser.add_argument('-dyn', '--dynamic_color', choices=['lin', 'ang'],
                        help='Color the trajectories based on velocity')
    parser.add_argument('-ids', '--agent_ids', type=int, nargs='+', help='The indexes of larvae to visualize')
    parser.add_argument('-tkr', '--tick_range', type=int, nargs='+', help='The time range to visualize in ticks')
    parser.add_argument('-fix', '--fix_points', type=int, nargs='+',
                        help='Fixate a midline point to the center of the screen')
    parser.add_argument('-Nsegs', '--draw_Nsegs', type=int, nargs='?', const=2,
                        help='Simplify larva body to N segments')

    parser.add_argument('-dim', '--arena_dims', type=float, nargs='+', help='The arena dimensions in m')

    return parser


def get_replay_kwargs(args):
    if args.tick_range is not None:
        if len(args.tick_range) != 2 or args.tick_range[0] > args.tick_range[1]:
            raise ValueError('Inappropriate tick range')

    fix = args.fix_points
    if fix is not None:
        use_background = True
        if len(fix) == 2 and np.abs(fix[1]) == 1:
            fix_point, secondary_fix_point = fix[0], fix[1]
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
                     'transposition': args.transposition,
                     'fix_point': fix_point,
                     'secondary_fix_point': secondary_fix_point,
                     'draw_Nsegs': args.draw_Nsegs,
                     'use_background': use_background,
                     'time_range': args.tick_range,
                     'dynamic_color': dynamic_color,
                     'arena_pars': arena_pars,

                     }
    return replay_kwargs


def add_data_kwargs(parser):
    # parser.add_argument('dataset_type', mode=str, help='The dataset mode name')
    parser.add_argument('-fld', '--folders', nargs='+', type=str,
                        help='Folders under the DataGroup parent dir where to search for datasets')
    parser.add_argument('-suf', '--suffixes', nargs='+', type=str, help='Suffixes of the dataset names')
    parser.add_argument('-nam', '--names', nargs='+', default=['enriched'], type=str, help='Names of the datasets')
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
    parser.add_argument('-raw', '--raw_folders', nargs='+', type=str,
                        help='Folders where to search for raw data when building a dataset')
    parser.add_argument('-t', '--min_duration_in_sec', type=float, nargs='?', default=0.0,
                        help='During dataset building, the minimum duration in sec of included larva tracks.')
    parser.add_argument('-all', '--all_folders', action="store_true",
                        help='Create a single merged dataset from all raw folders')
    parser.add_argument('-each', '--each_folder', action="store_true", help='Create a dataset from each raw folder')
    return parser


def get_build_kwargs(args):
    if args.all_folders:
        raw_folders = 'all'
    elif args.each_folder:
        raw_folders = 'each'
    else:
        raw_folders = args.raw_folders
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
    parser.add_argument('-id', '--sim_ID', type=str, help='The id of the simulation')
    parser.add_argument('-path', '--path', type=str, help='The path to save the simulation dataset')
    parser.add_argument('-t', '--duration', type=float, nargs='?', default=None,
                        help='The duration of the simulation in min')
    parser.add_argument('-dt', '--timestep', type=float, nargs='?', default=0.1, help='The timestep of the simulation in sec')
    parser.add_argument('-Box2D', '--Box2D', action="store_true", help='Use the Box2D physics engine')
    parser.add_argument('-sample', '--sample', type=str, nargs='?', default='reference',choices=list(loadConfDict('Ref').keys()),
                        help='The dataset to sample the parameters from')
    return parser


def get_sim_kwargs(args):
    sim_kwargs = {'sim_ID': args.sim_ID,
                  'duration': args.duration,
                  'path': args.path,
                  'timestep': args.timestep,
                  'Box2D': args.Box2D,
                  'sample': args.sample,
                  }
    return sim_kwargs


def add_life_kwargs(parser):
    parser.add_argument('-age', '--hours_as_larva', type=float, nargs='?', default=0.0,
                        help='The initial larva age since hatch in hours')
    parser.add_argument('-deb_f', '--substrate_quality', type=float, nargs='?', default=1.0,
                        help='The base deb functional response where 0 denotes no food and 1 at libitum feeding')
    parser.add_argument('-starv_h', '--epochs', type=float, nargs='+',
                        help='The starvation time intervals in hours')
    return parser


def get_life_kwargs(args):
    if args.epochs is None:
        starvation_hours = args.epochs
    else:
        if len(args.epochs) % 2 != 0:
            raise ValueError('Starvation intervals must be provided in pairs of start-stop time')
        else:
            starvation_hours = lib.aux.dictsNlists.group_list_by_n(args.epochs, 2)

    # if args.hours_as_larva is None :
    #     hours_as_larva=[0.0]
    # if args.substrate_quality is None :
    #     substrate_quality=[1.0]
    life_kwargs = {
        'hours_as_larva': args.hours_as_larva,
        'substrate_quality': args.substrate_quality,
        'epochs': starvation_hours
    }
    return life_kwargs


def add_batch_kwargs(parser):
    parser.add_argument('-id_b', '--batch_id', type=str, help='The id of the batch run')
    return parser


def get_batch_kwargs(args):
    kwargs = {
        'batch_id': args.batch_id
    }
    return kwargs


def add_optimization_kwargs(parser):
    parser.add_argument('-fit_par', '--fit_par', type=str, help='The fit parameter of the batch run')
    parser.add_argument('-minimize', '--minimize', type=bool, help='Whether to try to minimize the fit parameter')
    parser.add_argument('-threshold', '--threshold', type=float,
                        help='The fit parameter threshold for terminating the batch-run')
    parser.add_argument('-maxN', '--max_Nsims', type=int, nargs='?', default=12,
                        help='The maximum number of simulations to run')
    parser.add_argument('-Nbst', '--Nbest', type=int, nargs='?', default=4,
                        help='The number of best configurations to expand')
    return parser


def get_optimization_kwargs(args):
    kwargs = {
        'fit_par': args.fit_par,
        'minimize': args.minimize,
        'threshold': args.threshold,
        'max_Nsims': args.max_Nsims,
        'Nbest': args.Nbest}
    return kwargs


def add_space_kwargs(parser):
    parser.add_argument('-par', '--pars', type=str, nargs='+', help='The parameters for space search')
    parser.add_argument('-rng', '--ranges', type=float, nargs='+', help='The range of the parameters for space search')
    parser.add_argument('-Ngrd', '--Ngrid', nargs='+', type=int, help='The number of steps for space search')
    return parser


def get_space_kwargs(args):
    Ngrid = args.Ngrid
    if type(Ngrid) == int:
        Ngrid = [Ngrid] * len(args.pars)
    space_kwargs = {'pars': args.pars,
                    'ranges': args.ranges,
                    'Ngrid': Ngrid}
    return space_kwargs


def add_place_kwargs(parser):
    parser.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae')
    parser.add_argument('-M', '--larva_model', choices=list(loadConfDict('Model').keys()), help='The larva model to use')
    return parser


def get_place_kwargs(args):
    place_kwargs = {
        'N': args.Nagents,
        'larva_model': args.larva_model,
    }
    return place_kwargs
