
import numpy as np

import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x

def dst2source_evaluation(robot, source_xy):
    traj = np.array(robot.trajectory)
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    l=[]
    for label, pos in source_xy.items():
        dst2source = eudi5x(traj, np.array(pos))
        l.append(dst2source)
    m=np.mean(np.min(np.vstack(l),axis=0))
    fitness= - m/ cum_dst
    return fitness

def cum_dst(robot, **kwargs):
    return robot.cum_dst / robot.real_length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 20:
        return True
    # elif robot.negative_speed_errors >= 5:
    #     return True
    else:
        return False


fitness_funcs = dNl.NestDict({
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})



exclusion_funcs = dNl.NestDict({
    'bend_errors': bend_error_exclusion
})