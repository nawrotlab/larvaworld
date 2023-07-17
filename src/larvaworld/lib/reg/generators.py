import os

import param

from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import OptionalSelector


class SimTime(aux.NestedConf):
    dt = aux.PositiveNumber(0.1, softmax=1.0, step=0.01, doc='The timestep of the simulation in seconds.')
    duration = aux.OptionalPositiveNumber(5.0, softmax=100.0, step=0.1,
                                          doc='The duration of the simulation in minutes.')
    Nsteps = aux.OptionalPositiveInteger(label='# simulation timesteps', doc='The number of simulation timesteps.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_Nsteps()

    @param.depends('duration', 'dt', watch=True)
    def update_Nsteps(self):
        self.Nsteps = int(self.duration * 60 / self.dt)

    @param.depends('Nsteps', watch=True)
    def update_duration(self):
        self.duration = self.Nsteps * self.dt / 60


class SimOptions(SimTime):
    Box2D = param.Boolean(False,doc='Whether to use the Box2D physics engine or not.')
    store_data = param.Boolean(True, doc='Whether to store the simulation data')
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

from larvaworld.lib import model
from larvaworld.lib.model import Food, Border, DiffusionValueLayer, WindScape, ThermoScape, spatial, \
    FoodGrid, Life, Odor, PointAgent, OrientedAgent, Substrate


def class_generator(agent_class, mode='Unit') :
    class A(aux.NestedConf):

        def __init__(self, **kwargs):
            if hasattr(A,'distribution'):
                D=A.distribution.__class__
                ks=list(D.param.objects().keys())
                existing=[k for k in ks if k in kwargs.keys()]
                if len(existing)>0:
                    d={}
                    for k in existing :
                        d[k]=kwargs[k]
                        kwargs.pop(k)
                    kwargs['distribution']=D(**d)
            if 'c' in kwargs.keys():
                kwargs['default_color']=kwargs['c']
                kwargs.pop('c')
            if 'or' in kwargs.keys():
                kwargs['orientation']=kwargs['or']
                kwargs.pop('or')
            # if 'id' in kwargs.keys():
            #     kwargs['unique_id']=kwargs['id']
            #     kwargs.pop('id')
            if 'r' in kwargs.keys():
                kwargs['radius']=kwargs['r']
                kwargs.pop('r')
            if 'a' in kwargs.keys():
                kwargs['amount']=kwargs['a']
                kwargs.pop('a')
            if 'o' in kwargs.keys():
                assert 'odor' not in kwargs.keys()
                assert len(kwargs['o'])==3
                kwargs['odor']=dict(zip(['id', 'intensity','spread'], kwargs['o']))
                kwargs.pop('o')
            if 'sub' in kwargs.keys():
                assert 'substrate' not in kwargs.keys()
                assert len(kwargs['sub'])==2
                kwargs['substrate']=dict(zip(['quality', 'type'], kwargs['sub']))
                kwargs.pop('sub')

            super().__init__(**kwargs)

        @classmethod
        def from_entries(cls, entries):
            all_confs = []
            for gid, dic in entries.items():
                A = cls(**dic)
                gconf = aux.AttrDict(A.param.values())
                gconf.pop('name')
                if hasattr(A, 'distribution'):

                    ids = [f'{gid}_{i}' for i in range(A.distribution.N)]

                    gconf.pop('distribution')

                    try :
                        ps,ors=A.distribution()
                        confs = [{'unique_id': id, 'pos': p, 'orientation': ori, **gconf} for id, p,ori in zip(ids, ps, ors)]
                    except:
                        ps = A.distribution()
                        confs = [{'unique_id': id, 'pos': p, **gconf} for id, p in zip(ids, ps)]
                    all_confs += confs
                else:
                    gconf.unique_id=gid
                    all_confs.append(gconf)
            return all_confs


        @classmethod
        def agent_class(cls):
            return agent_class.__name__

        @classmethod
        def mode(cls):
            return mode

    A.__name__=f'{agent_class.__name__}{mode}'
    invalid = ['name', 'closed', 'visible']
    if mode=='Group':
        if issubclass(agent_class, PointAgent):
            distro=aux.Spatial_Distro
        elif issubclass(agent_class, OrientedAgent):
            distro = aux.Larva_Distro
        else :
            raise ValueError (f'No Group distribution for class {agent_class.__name__}. Change mode to Unit')
        A.param._add_parameter('distribution',aux.ClassAttr(distro, doc='The spatial distribution of the group agents'))
        invalid+=['unique_id', 'pos', 'orientation']
    elif mode=='Unit':
        pass
    for k, p in agent_class.param.params().items():
        if k not in invalid:
            A.param._add_parameter(k,p)
    return A




gen=aux.AttrDict({
    'FoodGroup':class_generator(Food, mode='Group'),
    'FoodUnit':class_generator(Food, mode='Unit'),
    'Arena':class_generator(spatial.Area,mode='Unit'),
    'Border':class_generator(Border, mode='Unit'),
    'Odor':class_generator(Odor, mode='Unit'),
    'Substrate':class_generator(Substrate, mode='Unit'),
})


class FoodConf(aux.NestedConf):
    source_groups = aux.ClassDict(item_type=gen.FoodGroup,  doc='The groups of odor or food sources available in the arena')
    source_units = aux.ClassDict(item_type=gen.FoodUnit,  doc='The individual sources  of odor or food in the arena')
    food_grid = aux.ClassAttr(FoodGrid, default=None, doc='The food grid in the arena')

class EnvConf(aux.NestedConf):
    arena = aux.ClassAttr(gen.Arena, doc='The arena configuration')
    food_params = aux.ClassAttr(FoodConf, doc='The food sources in the arena')
    border_list = aux.ClassDict(item_type=gen.Border, doc='The obstacles in the arena')
    odorscape = aux.ClassAttr(DiffusionValueLayer, default=None, doc='The obstacles in the arena')
    windscape = aux.ClassAttr(WindScape, default=None, doc='The obstacles in the arena')
    thermoscape = aux.ClassAttr(ThermoScape, default=None, doc='The obstacles in the arena')









# class DatasetSubsetConf():
#     time_range = aux.OptionalPositiveRange(softmax=1000.0, doc='Whether to only replay a defined temporal slice of the dataset.')
#     agent_ids = param.List(default=None,empty_default=True,allow_None=True, doc='Whether to only display some larvae of the dataset, defined by their indexes.')




# class ReplayConf(aux.NestedConf):
#     refID = ConfSelector('Env')
#     dir = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Env'), doc='The environment configuration ID')
#     overlap_mode = ConfSelector('Trial',default='default')
#     close_view = param.Selector(default='default', objects=stored.confIDs('Trial'), doc='The trial configuration ID')
#     fix_segment = param.ListSelector(default=['pose'],objects=reg.output_keys, doc='The data to collect as output')
#     fix_point = aux.ClassDict(item_type=LarvaGroup, doc='The larva groups')
#     draw_Nsegs = aux.ClassAttr(aux.SimConf,doc='The simulation configuration')
#     track_point = aux.ClassAttr(aux.EnrichConf, doc='The post-simulation processing')
#     time_range = ConfSelector('Exp')
#     dynamic_color = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     agent_ids = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     transposition = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     env_params = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')






# class ConfTypeSelector(OptionalSelector):
#     """Select among available configuration types"""
#     def __init__(self, **kwargs):
#         super().__init__(objects=reg.CONFTYPES, doc= 'The configuration type',**kwargs)

# class ConfType(param.Selector) :
#     """Select among available configuration types"""
#     # conftype = param.Selector(objects=reg.CONFTYPES, doc= 'The configuration type')
#
#     def __init__(self,conftype, **kwargs):
#         super().__init__(default=conftype,objects=reg.CONFTYPES, doc= 'The configuration type',**kwargs)
#
#     @ property
#     def path(self):
#         return f'{reg.CONF_DIR}/{self.default}.txt'
#
#     @property
#     def dict(self):
#         return aux.load_dict(self.path)
#
#     @property
#     def ids(self):
#         return sorted(list(self.dict.keys()))
#
#     def get(self, id):
#         if id in self.dict.keys():
#             d=self.dict[id]
#             try :
#                 return aux.AttrDict(d)
#             except:
#                 return d
#         else:
#             reg.vprint(f'{self.default} Configuration {id} does not exist', 1)
#             raise ValueError()


class ConfType(param.Parameterized) :
    """Select among available configuration types"""
    conftype = param.Selector(objects=reg.CONFTYPES, doc= 'The configuration type')
    path = param.Filename(label='path to configuration dictionary',
                           doc='The path to configuration dictionary')
    item_type =param.ClassSelector(default=None, class_=object,is_instance=False, allow_None=True)
    dict= aux.ClassDict(default=aux.AttrDict(), item_type=None, doc='The configuration dictionary')
    # ids = param.List([], item_type=str, doc='The configuration IDs.')
    id = OptionalSelector(objects=[], doc='The configuration ID')
    entry = aux.ClassAttr(default=None, class_=None, doc='The configuration')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_conftype()

    @param.depends('conftype', watch=True)
    def update_conftype(self):
        self.path = f'{reg.CONF_DIR}/{self.conftype}.txt'
        self.update_class(self.conf_class)
        self.dict=self.load()

    # @param.depends('item_type', watch=True)
    def update_class(self,c):
        self.item_type = c
        # self.param.params('item_type').class_ = c
        self.param.params('dict').item_type = c
        self.param.params('conf').class_ = c

    @param.depends('dict', watch=True)
    def update_ids(self):
        self.param.params('id').objects=self.ids

    @param.depends('id', watch=True)
    def update_entry(self):
        self.entry = self.dict[self.id]


    def get_entry(self,id):
        if id in self.dict.keys():
            return self.item_type(self.dict[id])
        else:
            reg.vprint(f'{self.conftype} Configuration {id} does not exist', 1)
            raise ValueError()

    def get_conf(self, id):
        return self.get_entry(id)

    # @property
    def load(self):
        return aux.load_dict(self.path)

    def save(self):
        return aux.save_dict(self.dict, self.path)

    @property
    def ids(self):
        return sorted(list(self.dict.keys()))

    @property
    def conf_class(self):
        c=self.conftype
        if c is None :
            return None
        elif c=='Ref':
            return str
        else :
            return aux.AttrDict






class ConfSelector(OptionalSelector):
    # conftype = ConfType(default=conftype)

    """Select among stored configurations of a given conftype by ID"""
    def __init__(self, conftype, **kwargs):

        kws={
            'objects' : reg.stored.confIDs(conftype),
            'doc' : f'The {conftype} configuration ID',
            **kwargs
        }
        super().__init__(**kws)

# class

class RefType(ConfType):
    """Select a reference dataset by ID"""
    def __init__(self, **kwargs):
        super().__init__(conftype='Ref',**kwargs)

    def get_conf(self, id):
        dir=self.get_entry(id)
        if dir is not None:
            path = f'{dir}/data/conf.txt'
            if os.path.isfile(path):
                c = aux.load_dict(path)
                if 'id' in c.keys():
                    reg.vprint(f'Loaded existing conf {c.id}', 1)
                    return c
        return None

    # def getRefDir(self, id):
    #     if id in self.dict.keys():
    #         return self.dict[id]
    #     else:
    #         reg.vprint(f'Reference dataset with ID {id} does not exist. Returning None', 1)
    #         return None
    #
    # def getRef(self, id=None, dir=None):
    #     if dir is None:
    #         dir=self.getRefDir(id)
    #     if dir is not None:
    #         path = f'{dir}/data/conf.txt'
    #         if os.path.isfile(path):
    #             c = aux.load_dict(path)
    #             if 'id' in c.keys():
    #                 reg.vprint(f'Loaded existing conf {c.id}', 1)
    #                 return c
    #     return None







class LarvaGroup(aux.NestedConf):
    model = ConfSelector('Model')
    # model = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.ModelIDs, doc='The model configuration ID')
    default_color = param.Color('black', doc='The default color of the group')
    odor = aux.ClassAttr(Odor, doc='The odor of the agent')
    distribution = aux.ClassAttr(aux.Larva_Distro,doc='The spatial distribution of the group agents')
    life_history = aux.ClassAttr(Life, doc='The life history of the group agents')
    sample = ConfSelector('Ref')
    # sample = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.RefIDs, doc='The ID of a reference dataset to sample from')
    imitation = param.Boolean(default=False, doc='Whether to imitate the reference dataset.')



    def __init__(self,id=None,**kwargs):
        super().__init__(**kwargs)
        if id is None:
            if self.model is not None :
                id = self.model
            else :
                id = 'LarvaGroup'
        self.id=id


    def entry(self, expand=False, as_entry=True):
        conf = self.nestedConf
        if expand and conf.model is not None:
            conf.model = reg.stored.getModel(conf.model)
        if as_entry :
            return aux.AttrDict({self.id: conf})
        else:
            return conf

    def __call__(self, parameter_dict={}):
        Nids=self.distribution.N
        if self.model is not None:
            m=reg.stored.getModel(self.model)
        else :
            m=None
        kws={
            'm' : m,
            'refID' : self.sample,
            'parameter_dict' : parameter_dict,
            'Nids' : Nids,
        }

        if not self.imitation:
            ps, ors = aux.generate_xyNor_distro(self.distribution)
            ids = [f'{self.id}_{i}' for i in range(Nids)]
            all_pars, refID = util.sampleRef(**kws)
        else:
            ids, ps, ors, all_pars = util.imitateRef(**kws)
        confs = []
        for id, p, o, pars in zip(ids, ps, ors, all_pars):
            conf = {
                'pos': p,
                'orientation': o,
                'default_color': self.default_color,
                'unique_id': id,
                'group': self.id,
                'odor': self.odor,
                'life_history': self.life_history,
                **pars
            }
            confs.append(conf)
        return confs







# class DatasetSubsetConf():
#     time_range = aux.OptionalPositiveRange(softmax=1000.0, doc='Whether to only replay a defined temporal slice of the dataset.')
#     agent_ids = param.List(default=None,empty_default=True,allow_None=True, doc='Whether to only display some larvae of the dataset, defined by their indexes.')




# class ReplayConf(aux.NestedConf):
#     refID = ConfSelector('Env')
#     dir = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Env'), doc='The environment configuration ID')
#     overlap_mode = ConfSelector('Trial',default='default')
#     close_view = param.Selector(default='default', objects=stored.confIDs('Trial'), doc='The trial configuration ID')
#     fix_segment = param.ListSelector(default=['pose'],objects=reg.output_keys, doc='The data to collect as output')
#     fix_point = aux.ClassDict(item_type=LarvaGroup, doc='The larva groups')
#     draw_Nsegs = aux.ClassAttr(aux.SimConf,doc='The simulation configuration')
#     track_point = aux.ClassAttr(aux.EnrichConf, doc='The post-simulation processing')
#     time_range = ConfSelector('Exp')
#     dynamic_color = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     agent_ids = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     transposition = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')
#     env_params = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')







def full_lg(id=None, expand=False,as_entry=True,**conf):
    try :
        lg=LarvaGroup(id=id,**conf)
        return lg.entry(expand=expand, as_entry=as_entry)
    except :
        raise

def GTRvsS(N=1, age=72.0, q=1.0, h_starved=0.0, sample='exploration.150controls', substrate_type='standard', pref='',
           navigator=False, expand=False, **kwargs):
    if age == 0.0:
        epochs = {}
    else:
        if h_starved == 0:
            epochs = {
                0: {'age_range': (0.0, age), 'substrate': {'type': substrate_type, 'quality': q}}
            }
        else:
            epochs = {
                0: {'age_range': (0.0, age - h_starved), 'substrate': {'type': substrate_type, 'quality': q}},
                1: {'age_range': (age - h_starved, age), 'substrate': {'type': substrate_type, 'quality': 0}},
            }
    kws0 = {
        'distribution': {'N': N, 'scale': (0.005, 0.005)},
        'life_history': {'age': age, 'epochs': epochs},
        'sample': sample,
        'expand': expand,
    }

    mcols = ['blue', 'red']
    mID0s = ['rover', 'sitter']
    lgs = {}
    for mID0, mcol in zip(mID0s, mcols):
        id = f'{pref}{mID0.capitalize()}'

        if navigator:
            mID0 = f'navigator_{mID0}'

        kws = {
            'id': id,
            'default_color': mcol,
            'model': mID0,
            **kws0
        }

        lgs.update(full_lg(**kws))
    return aux.AttrDict(lgs)









class ExpConf(aux.NestedConf):
    env_params = ConfSelector('Env')
    # env_params = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Env'), doc='The environment configuration ID')
    trials = ConfSelector('Trial',default='default')
    # trials = param.Selector(default='default', objects=stored.confIDs('Trial'), doc='The trial configuration ID')
    collections = param.ListSelector(default=['pose'],objects=reg.output_keys, doc='The data to collect as output')
    larva_groups = aux.ClassDict(item_type=LarvaGroup, doc='The larva groups')
    sim_params = aux.ClassAttr(SimOptions,doc='The simulation configuration')
    enrichment = aux.ClassAttr(aux.EnrichConf, doc='The post-simulation processing')
    experiment = ConfSelector('Exp')
    # experiment = param.Selector(default=None,empty_default=True,allow_None=True, objects=stored.confIDs('Exp'), doc='The experiment configuration ID')



    def __init__(self,id=None,**kwargs):
        super().__init__(**kwargs)

