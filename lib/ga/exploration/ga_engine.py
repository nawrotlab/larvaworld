import random
import multiprocessing
import math
from scipy.stats import ks_2samp
import numpy as np
from unflatten import unflatten

import lib.aux.dictsNlists as dNl
from lib.conf.base.par import ParDict
from lib.conf.stored.conf import loadRef, copyConf
from lib.ga.exploration.genome import LarvaGenome
from lib.ga.robot.larvaConfDic import LarvaConfDic
from lib.ga.robot.larva_robot import LarvaRobot
from lib.ga.util.templates import GAEngineTemplate



def target(shorts) :
    refID = 'None.100controls'
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config

    dic = ParDict(mode='load').dict
    eval = {sh: s[dic[sh]['d']].dropna().values for sh in shorts}
    return eval



class GaLarvaEngine(GAEngineTemplate):

    def __init__(self, eval_shorts=['b', 'fov', 'foa', 'tor5', 'v', 'a'],base_model='Sakagiannis2022', **kwargs):

        # print(multicore)
        # raise
        self.base_conf = copyConf(base_model, 'Model').brain
        super().__init__(genome_class=LarvaGenome, **kwargs)
        self.eval_shorts = eval_shorts
        self.eval = target(eval_shorts)



    def build_robot(self, x, y, genome, label):
    # def build_robot(self, unique_id, model, conf=None, robot_class=LarvaRobot):
        conf = self.base_conf

        # robot_class = LarvaRobot

        conf_f = dNl.flatten_dict(conf, parent_key='conf', sep='.')
        for key in LarvaConfDic.keys():
            conf_f[key] = getattr(genome, key)
        kws = dNl.AttrDict.from_nested_dicts(unflatten(conf_f))

        if self.robot_class == 'larva':
            robot = LarvaRobot(unique_id=label, model=self, **kws)
            # robot.genome=genome
        return robot

    def get_fitness(self, robot):
        robot.finalize(self.eval_shorts)
        return -np.sum([ks_2samp(self.eval[p], robot.eval[p]) for p in self.eval_shorts])
