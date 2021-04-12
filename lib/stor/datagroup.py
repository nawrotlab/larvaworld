import json
import sys
import shutil
import os
import numpy as np


from lib.stor import paths as paths

sys.path.insert(0, paths.get_parent_dir())

from lib.conf.conf import loadConf, setDataGroup, deleteConf, saveConf


class LarvaDataGroup:
    def __init__(self, id):
        try:
            temp = loadConf(id, 'Group')
        except:
            temp = setDataGroup(id)
        self.__dict__.update(temp)
        self.build_dirs()

    def delete(self):
        deleteConf(self.id, 'Group')

    def build_dirs(self):
        dir = self.get_path()
        self.raw_dir = f'{dir}/raw'
        self.proc_dir = f'{dir}/processed'
        self.plot_dir = f'{dir}/plots'
        self.vis_dir = f'{dir}/visuals'
        self.dirs = [self.raw_dir, self.proc_dir, self.plot_dir, self.vis_dir]
        for i in self.dirs:
            if not os.path.exists(i):
                os.makedirs(i)

    def save(self):
        saveConf(self, 'Group', self.id)

    def add_subgroup(self, id):
        self.subgroups += [id]
        self.save()
        # self.subgroups+=[id]
        for i in [f'{d}/{id}' for d in self.dirs]:
            if not os.path.exists(i):
                os.makedirs(i)

    def get_dirs(self, subgroup=None, raw_data=False, startswith=None, absolute=True):
        if raw_data:
            dir = self.raw_dir
        else:
            dir = self.proc_dir
        if subgroup is not None:
            dir = f'{dir}/{subgroup}'
        if startswith is None:
            dirs = os.listdir(dir)
        else:
            dirs = [f for f in os.listdir(dir) if f.startswith(startswith)]
        if absolute:
            return [os.path.join(dir, d) for d in dirs]
        else:
            return dirs

    def get_conf(self):
        return loadConf(self.conf, 'Data')

    def get_par_conf(self):
        return loadConf(loadConf(self.conf, 'Data')['par'], 'Par')

    def get_path(self):
        return f'{paths.DataFolder}/{self.path}'