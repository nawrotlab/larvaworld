import os

import param


from larvaworld.lib import reg, aux



class SimModeOps(aux.NestedConf):
    runtype = param.Selector(objects=reg.SIMTYPES, doc='The simulation mode')
    id=param.String(None,doc='ID of the simulation. If not specified,set according to runtype and experiment.')

    def __init__(self,runtype='Exp',**kwargs):
        self.param.add_parameter('experiment', self.exp_selector_param(runtype))
        super().__init__(runtype=runtype,**kwargs)
        if self.id is None :
            self.id=self.generate_id(self.runtype, self.experiment)
        #self.param.params('id').constant=True

    def generate_id(self, runtype,exp):
        idx = reg.next_idx(exp, conftype=runtype)
        return f'{exp}_{idx}'

    #@ staticmethod
    def exp_selector_param(self,runtype):
        # runtype = self.runtype
        defaults = {
            'Exp': 'dish',
            'Batch': 'PItest_off',
            'Ga': 'exploration',
            'Eval': 'dispersal',
            'Replay': 'replay'
        }
        kws = {
            'default': defaults[runtype],
            'doc': 'The experiment simulated'
        }
        if runtype in ['Exp', 'Batch', 'Ga']:
            ids = reg.conf[runtype].confIDs
            return param.Selector(objects=ids, **kws)
        else:
            return param.String(**kws)

class SimDataOps(SimModeOps):
    store_data = param.Boolean(True, doc='Whether to store the simulation data')
    # dir = param.Foldername(default=None, label='storage folder', doc='The directory to store data')

    def __init__(self,save_to = None,**kwargs):
        super().__init__(**kwargs)
        # Define directories
        if save_to is None:
            save_to = f'{reg.SIM_DIR}/{self.runtype.lower()}_runs'
        self.dir = f'{save_to}/{self.experiment}/{self.id}'
        self.save_to = self.dir

        self.plot_dir = f'{self.dir}/plots'
        self.data_dir = f'{self.dir}/data'

    def make_dirs(self):
        if self.store_data:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.plot_dir, exist_ok=True)

class SimTimeOps(aux.NestedConf):
    dt = aux.PositiveNumber(0.1, softmax=1.0, step=0.01, doc='The timestep of the simulation in seconds.')
    duration = aux.OptionalPositiveNumber(5.0, softmax=100.0, step=0.1,
                                          doc='The duration of the simulation in minutes.')
    Nsteps = aux.OptionalPositiveInteger(label='# simulation timesteps', doc='The number of simulation timesteps.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_Nsteps()
        # raise

    @param.depends('duration', 'dt', watch=True)
    def update_Nsteps(self):
        self.Nsteps = int(self.duration * 60 / self.dt)

    @param.depends('Nsteps', watch=True)
    def update_duration(self):
        self.duration = self.Nsteps * self.dt / 60


class SimGeneralOps(aux.NestedConf):
    Box2D = param.Boolean(False,doc='Whether to use the Box2D physics engine or not.')
    # store_data = param.Boolean(True, doc='Whether to store the simulation data')
    larva_collisions = param.Boolean(True, doc='Whether to allow overlap between larva bodies.')
    offline = param.Boolean(False,doc='Whether to launch a full Larvaworld environment')
    show_display = param.Boolean(True,doc='Whether to launch the pygame-visualization.')

    def __init__(self,offline=False, show_display=True, **kwargs):
        if offline:
            show_display=False
        super().__init__(show_display=show_display,offline=offline,**kwargs)
        # Define constant parameters
        self.scaling_factor = 1000.0 if self.Box2D else 1.0


    @param.depends('offline','show_display', watch=True)
    def disable_display(self):
        if self.offline :
            self.show_display=False

class SimOps(SimDataOps,SimTimeOps,SimGeneralOps):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

class RefDataOps(aux.NestedConf):
    refID = reg.conf.Ref.confID_selector()
    # # refID = aux.OptionalSelector(objects=[], doc='The reference dataset ID')
    dataset_dir = param.Foldername(default=None,
                                   label='directory of reference dataset',
                                   doc='The path to the stored dataset relative to Root/data. Alternative to providing refID')
    #
    conf = param.ClassSelector(default=None, class_=aux.AttrDict,
                               label='reference dataset config', doc='The stored reference dataset config')
    refDataset = param.ClassSelector(default=None, class_=BaseLarvaDataset,
                                     label='reference dataset', doc='The stored reference dataset')

    """Select a reference dataset by ID"""


    def __init__(self,**kwargs):
        super().__init__(**kwargs)
