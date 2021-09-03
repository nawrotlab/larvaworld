import json
import sys
import shutil
import os
import numpy as np


from lib.stor import paths as paths


sys.path.insert(0, paths.get_parent_dir())

from lib.conf.conf import loadConf, setDataGroup, deleteConf, saveConf
import lib.aux.functions as fun


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

    def get_last_common(self, dirs, raw=True):
        dirs = [fun.remove_prefix(dr, f'{self.raw_dir}/') for dr in dirs]
        pass

if __name__ == "__main__":
    from lib.stor.managing import detect_dataset
    kk=LarvaDataGroup('SchleyerGroup').raw_dir
    folder_path='/home/panos/nawrot_larvaworld/larvaworld/data/SchleyerGroup/raw/odor_conc/AM1_20/AM_Rewarded'
    dic = detect_dataset(datagroup_id='SchleyerGroup',folder_path=folder_path, raw=True)
    ids = list(dic.keys())
    dirs = list(dic.values())
    dirs = [fun.remove_prefix(dr, f'{kk}/') for dr in dirs]
    dirs = [fun.remove_suffix(dr, f'/{id}') for dr, id in zip(dirs, ids)]
    dirs=fun.unique_list(dirs)
    print(dirs)