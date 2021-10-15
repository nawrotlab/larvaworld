from argparse import ArgumentParser

import numpy as np

from lib.conf.stored.conf import loadConfDict
from lib.conf.base.dtypes import null_dict, arena, par_dict


class ParsArg :
    def __init__(self, short, key, **kwargs):
        self.key=key
        self.args=[f'-{short}', f'--{key}']
        self.kwargs=kwargs

    def add(self,p):
        p.add_argument(*self.args, **self.kwargs)
        return p

    def get(self, input):
        return getattr(input, self.key)

class Parser :
    def __init__(self, name):
        self.name=name
        dic=par_dict(name, argparser=True)
        try :
            self.parsargs={k : ParsArg(**v) for k,v in dic.items()}
        except :
            self.parsargs ={}
            for k, v in dic.items() :
                for kk, vv in v['content'].items() :
                    self.parsargs[kk]=ParsArg(**vv)

    def add(self, parser=None):
        if parser is None :
            parser = ArgumentParser()
        for k,v in self.parsargs.items() :
            parser=v.add(parser)
        return parser

    def get(self, input):
        dic= {k : v.get(input) for k,v in self.parsargs.items()}
        return null_dict(self.name, **dic)


class MultiParser :
    def __init__(self, names):
        self.parsers={n:Parser(n) for n in names}

    def add(self, parser=None):
        if parser is None:
            parser = ArgumentParser()
        for k,v in self.parsers.items() :
            parser=v.add(parser)
        return parser

    def get(self, input):
        return {k : v.get(input) for k,v in self.parsers.items()}

def add_exp_kwargs(parser) :
    parser.add_argument('experiment', choices=list(loadConfDict('Exp').keys()), help='The experiment mode')
    parser.add_argument('-a', '--analysis', action="store_true", help='Whether to run analysis')
    return parser

def add_vis_kwargs(p):
    p.add_argument('-hide', '--show_display', action="store_false", help='Hide display')
    p.add_argument('-vid', '--video_speed', type=float, nargs='?', const=1.0, help='The fast-forward speed of the video')
    p.add_argument('-img', '--image_mode', nargs='?', const='final',
                   choices=['final', 'overlap', 'snapshots'], help='Select image mode')
    p.add_argument('-media', '--media_name', type=str, help='Filename for the saved video/image')
    p.add_argument('-rnd', '--random_colors', action="store_true", help='Color larvae with random colors')
    p.add_argument('-beh', '--color_behavior', action="store_true", help='Color behavioral epochs')
    p.add_argument('-trj', '--trajectories', type=float, nargs='?', const=0.0,
                   help='Show trajectories of specific time duration')
    p.add_argument('-blk', '--black_background', action="store_true", help='Black background')
    p.add_argument('-head', '--draw_head', action="store_true", help='Color the head and tail')
    p.add_argument('-con', '--draw_contour', action="store_false", help='Hide the contour')
    p.add_argument('-mid', '--draw_midline', action="store_false", help='Hide the midline')
    p.add_argument('-cen', '--draw_centroid', action="store_true", help='Show the centroid')
    p.add_argument('-vis_clock', '--visible_clock', action="store_false", help='Visible clock')
    p.add_argument('-vis_state', '--visible_state', action="store_false", help='Visible state')
    p.add_argument('-vis_scale', '--visible_scale', action="store_false", help='Visible spatial scale')
    p.add_argument('-vis_ids', '--visible_ids', action="store_true", help='Visible ids')
    return p


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


def add_replay_kwargs(p):
    p.add_argument('-trans', '--transposition', choices=['origin', 'arena', 'center'],
                   help='The transposition mode for visualization')
    p.add_argument('-dyn', '--dynamic_color', choices=['lin', 'ang'],
                   help='Color the trajectories based on velocity')
    p.add_argument('-ids', '--agent_ids', type=int, nargs='+', help='The indexes of larvae to visualize')
    p.add_argument('-tkr', '--tick_range', type=int, nargs='+', help='The time range to visualize in ticks')
    p.add_argument('-fix', '--fix_points', type=int, nargs='+',
                   help='Fixate a midline point to the center of the screen')
    p.add_argument('-Nsegs', '--draw_Nsegs', type=int, nargs='?', const=2,
                   help='Simplify larva body to N segments')

    p.add_argument('-dim', '--arena_dims', type=float, nargs='+', help='The arena dimensions in m')

    return p


def get_replay_kwargs(args):
    if args.tick_range is not None:
        if len(args.tick_range) != 2 or args.tick_range[0] > args.tick_range[1]:
            raise ValueError('Inappropriate tick range')

    fix = args.fix_points
    if fix is not None:
        use_background = True
        if len(fix) == 2 and np.abs(fix[1]) == 1:
            fix_point, fix_segment = fix[0], fix[1]
        elif len(fix) == 1:
            fix_point, fix_segment = fix[0], None
        else:
            raise ValueError('Inappropriate fix points')
    else:
        fix_point, fix_segment = None, None
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
            arena_pars = arena(dims[0], dims[1])
        elif len(dims) == 1:
            arena_pars = arena(dims[0])
        else:
            raise ValueError('Inappropriate arena dimensions')
    else:
        arena_pars = None

    replay_kwargs = {'agent_ids': args.agent_ids,
                     'transposition': args.transposition,
                     'fix_point': fix_point,
                     'fix_segment': fix_segment,
                     'draw_Nsegs': args.draw_Nsegs,
                     'use_background': use_background,
                     'time_range': args.tick_range,
                     'dynamic_color': dynamic_color,
                     'arena_pars': arena_pars,

                     }
    return replay_kwargs


def add_data_kwargs(p):
    p.add_argument('-fld', '--folders', nargs='+', type=str,
                   help='Folders under the DataGroup parent dir where to search for datasets')
    p.add_argument('-suf', '--suffixes', nargs='+', type=str, help='Suffixes of the dataset names')
    p.add_argument('-nam', '--names', nargs='+', default=['enriched'], type=str, help='Names of the datasets')
    p.add_argument('-load', '--load_data', action="store_false", help='Not load the data from the datasets')
    return p


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


def add_build_kwargs(p):
    p.add_argument('-d_ids', '--dataset_ids', nargs='+', type=str, help='Ids of newly built datasets')
    p.add_argument('-raw', '--raw_folders', nargs='+', type=str,
                   help='Folders where to search for raw data when building a dataset')
    p.add_argument('-t', '--min_duration_in_sec', type=float, nargs='?', default=0.0,
                   help='During dataset building, the minimum duration in sec of included larva tracks.')
    p.add_argument('-all', '--all_folders', action="store_true",
                   help='Create a single merged dataset from all raw folders')
    p.add_argument('-each', '--each_folder', action="store_true", help='Create a dataset from each raw folder')
    return p


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


def add_sim_kwargs(p):
    p.add_argument('-id', '--sim_ID', type=str, help='The id of the simulation')
    p.add_argument('-path', '--path', type=str, help='The path to save the simulation dataset')
    p.add_argument('-t', '--duration', type=float, nargs='?', default=None,
                   help='The duration of the simulation in min')
    p.add_argument('-dt', '--timestep', type=float, nargs='?', default=0.1, help='The timestep of the simulation in sec')
    p.add_argument('-Box2D', '--Box2D', action="store_true", help='Use the Box2D physics engine')
    p.add_argument('-sample', '--sample', type=str, nargs='?', default='reference', choices=list(loadConfDict('Ref').keys()),
                   help='The dataset to sample the parameters from')
    return p


def get_sim_kwargs(args):
    sim_kwargs = {'sim_ID': args.sim_ID,
                  'duration': args.duration,
                  'path': args.path,
                  'timestep': args.timestep,
                  'Box2D': args.Box2D,
                  'sample': args.sample,
                  }
    return sim_kwargs


def add_life_kwargs(p):
    p.add_argument('-age', '--hours_as_larva', type=float, nargs='?', default=0.0,
                   help='The initial larva age since hatch in hours')
    p.add_argument('-deb_f', '--substrate_quality', type=float, nargs='?', default=1.0,
                   help='The base deb functional response where 0 denotes no food and 1 at libitum feeding')
    p.add_argument('-starv_h', '--epochs', type=float, nargs='+',
                   help='The starvation time intervals in hours')
    return p


def get_life_kwargs(args):
    if args.epochs is None:
        starvation_hours = args.epochs
    else:
        if len(args.epochs) % 2 != 0:
            raise ValueError('Starvation intervals must be provided in pairs of start-stop time')
        else:
            from lib.aux.dictsNlists import group_list_by_n
            starvation_hours = group_list_by_n(args.epochs, 2)

    # if args.hours_as_larva is None :
    #     hours_as_larva=[0.0]
    # if args.substrate_quality is None :
    #     substrate_quality=[1.0]
    life_kwargs = {
        'hours_as_larva': args.hours_as_larva,
        'epochs': starvation_hours,
        'substrate' : null_dict('substrate', quality=args.substrate_quality)
    }
    return life_kwargs


def add_batch_kwargs(p):
    p.add_argument('-id_b', '--batch_id', type=str, help='The id of the batch run')
    return p


def get_batch_kwargs(args):
    kwargs = {
        'batch_id': args.batch_id
    }
    return kwargs


def add_optimization_kwargs(p):
    p.add_argument('-fit_par', '--fit_par', type=str, help='The fit parameter of the batch run')
    p.add_argument('-minimize', '--minimize', type=bool, help='Whether to try to minimize the fit parameter')
    p.add_argument('-threshold', '--threshold', type=float,
                   help='The fit parameter threshold for terminating the batch-run')
    p.add_argument('-maxN', '--max_Nsims', type=int, nargs='?', default=12,
                   help='The maximum number of simulations to run')
    p.add_argument('-Nbst', '--Nbest', type=int, nargs='?', default=4,
                   help='The number of best configurations to expand')
    return p


def get_optimization_kwargs(args):
    kwargs = {
        'fit_par': args.fit_par,
        'minimize': args.minimize,
        'threshold': args.threshold,
        'max_Nsims': args.max_Nsims,
        'Nbest': args.Nbest}
    return kwargs


def add_space_kwargs(p):
    p.add_argument('-par', '--pars', type=str, nargs='+', help='The parameters for space search')
    p.add_argument('-rng', '--ranges', type=float, nargs='+', help='The range of the parameters for space search')
    p.add_argument('-Ngrd', '--Ngrid', nargs='+', type=int, help='The number of steps for space search')
    return p


def get_space_kwargs(args):
    Ngrid = args.Ngrid
    if type(Ngrid) == int:
        Ngrid = [Ngrid] * len(args.pars)
    space_kwargs = {'pars': args.pars,
                    'ranges': args.ranges,
                    'Ngrid': Ngrid}
    return space_kwargs


def add_place_kwargs(p):
    p.add_argument('-N', '--Nagents', type=int, help='The number of simulated larvae')
    p.add_argument('-M', '--larva_model', choices=list(loadConfDict('Model').keys()), help='The larva model to use')
    return p


def get_place_kwargs(args):
    place_kwargs = {
        'N': args.Nagents,
        'larva_model': args.larva_model,
    }
    return place_kwargs

def init_parser(description='', parsers=[]) :
    dic={
        'exp' : add_exp_kwargs,
        'vis' : add_vis_kwargs,
        'replay' : add_replay_kwargs,
        'place' : add_place_kwargs,
        'space' : add_space_kwargs,
        'opt' : add_optimization_kwargs,
        'batch' : add_batch_kwargs,
        'life' : add_life_kwargs,
        'sim' : add_sim_kwargs,
        'build' : add_build_kwargs,
        'data' : add_data_kwargs,
    }
    parser = ArgumentParser(description=description)
    for n in parsers :
        parser=dic[n](parser)
    return parser

# if __name__ == '__main__':
#     kk=Parser('sim_params')
#     print(kk)