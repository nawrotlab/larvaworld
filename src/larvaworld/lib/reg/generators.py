import os

import param


from larvaworld.lib import reg, aux, util
from larvaworld.lib.aux import OptionalSelector


class ConfType(param.Parameterized) :
    """Select among available configuration types"""
    conftype = param.Selector(objects=reg.CONFTYPES, doc= 'The configuration type')
    dict= aux.ClassDict(default=aux.AttrDict(), item_type=None, doc='The configuration dictionary')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_dict()

    @property
    def path_to_dict(self):
        return f'{reg.CONF_DIR}/{self.conftype}.txt'





    @param.depends('conftype', watch=True)
    def update_dict(self):
        self.param.params('dict').item_type = self.dict_entry_type
        self.load()



    def getID(self,id):
        if id in self.dict.keys():
            return self.dict[id]
        else:
            reg.vprint(f'{self.conftype} Configuration {id} does not exist', 1)
            raise ValueError()

    def get(self, id):
        entry=self.getID(id)
        return self.conf_class(**entry)



    def load(self):
        self.dict= aux.load_dict(self.path_to_dict)

    def save(self):
        return aux.save_dict(self.dict, self.path_to_dict)

    def reset(self,init=False):
        dd = reg.funcs.stored_confs[self.conftype]()

        if os.path.isfile(self.path_to_dict):
            if init:
                return
            else:
                # self.load()
                d = self.dict
        else:
            d = {}

        N0, N1 = len(d), len(dd)

        d.update(dd)

        Ncur = len(d)
        Nnew = Ncur - N0
        Nup = N1 - Nnew

        self.param.params('dict').item_type = self.dict_entry_type
        self.dict = d
        self.save()
        reg.vprint(f'{self.conftype}  configurations : {Nnew} added , {Nup} updated,{Ncur} now existing', 1)

    def setID(self, id, conf, mode='overwrite'):
        if id in self.dict.keys() and mode == 'update':
            self.dict[id] = self.dict[id].update_nestdict(conf.flatten())
        else:
            self.dict[id] = aux.AttrDict(conf)
        self.save()
        reg.vprint(f'{self.conftype} Configuration saved under the id : {id}', 1)

    def delete(self, id=None):
        if id is not None:
            if id in self.dict.keys():
                self.dict.pop(id, None)
                self.save()
                reg.vprint(f'Deleted {self.conftype} configuration under the id : {id}', 1)

    def expand(self, id=None, conf=None):
        if conf is None:
            if id in self.dict.keys():
                conf = self.dict[id]
            else:
                return None
        subks = reg.CONFTYPE_SUBKEYS[self.conftype]
        if len(subks) > 0:
            for subID, subk in subks.items():
                ids = reg.conf[subk].confIDs
                # ids = self.confIDs(subk)
                if subID == 'larva_groups' and subk == 'Model':
                    for k, v in conf['larva_groups'].items():
                        if v.model in ids:
                            v.model = reg.conf[subk].getID(v.model)
                            # v.model = self.get(subk, id=v.model)
                else:
                    if conf[subID] in ids:
                        conf[subID] = reg.conf[subk].getID(conf[subID])
                        # conf[subID] = self.get(subk, id=conf[subID])

        return conf

    def confID_selector(self, default=None):
        return OptionalSelector(default=default, objects=self.confIDs, doc='The configuration ID')

    @property
    def confIDs(self):
        return sorted(list(self.dict.keys()))

    @property
    def conf_class(self):
        c=self.conftype
        if c is None :
            return None
        elif c in reg.gen.keys():
            return reg.gen[c]
        else :
            return aux.AttrDict

    @property
    def dict_entry_type(self):
    #     c = self.conftype
    #     if c is None:
    #         return None
    #     elif c == 'Ref':
    #         return str
    #     else:
        return aux.AttrDict


class RefType(ConfType):

    """Select a reference dataset by ID"""


    def __init__(self, **kwargs):
        super().__init__(conftype='Ref', **kwargs)


    def getRefDir(self,id):
        assert id is not None
        return self.getID(id)

    def getRef(self, id=None, dir=None):
        if dir is None:
            dir=self.getRefDir(id)
        path = f'{dir}/data/conf.txt'
        assert os.path.isfile(path)
        c = aux.load_dict(path)
        assert 'id' in c.keys()
        reg.vprint(f'Loaded existing conf {c.id}', 1)
        return c

    def loadRef(self, id=None, dir=None, load=False, **kwargs):
        from larvaworld import LarvaDataset
        c=self.getRef(id=id, dir=dir)
        assert c is not None
        d = LarvaDataset(config=c, load_data=False)
        if load:
            d.load(**kwargs)
        reg.vprint(f'Loaded stored reference dataset : {id}', 1)
        return d

    def retrieve_dataset(self, dataset=None,load=True,**kwargs):
        if dataset is None :
            dataset=self.loadRef(load=False,**kwargs)
        if load:
            dataset.load(**kwargs)
        return dataset


    @property
    def dict_entry_type(self):
        return str


conf=aux.AttrDict({k: ConfType(conftype=k) for k in reg.CONFTYPES if k!='Ref'})


conf.Ref=RefType()


def resetConfs(conftypes=None, **kwargs):
    if conftypes is None:
        conftypes = reg.CONFTYPES

    for conftype in conftypes:
        conf[conftype].reset(**kwargs)


from larvaworld.lib.model import Food, Border, WindScape, ThermoScape, spatial, \
    FoodGrid, Life, Odor, PointAgent, OrientedAgent, Substrate, Odorscape, DiffusionValueLayer, GaussianValueLayer


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
    'Food':class_generator(Food, mode='Unit'),
    'Arena':class_generator(spatial.Area,mode='Unit'),
    'Border':class_generator(Border, mode='Unit'),
    'Odor':class_generator(Odor, mode='Unit'),
    'Substrate':class_generator(Substrate, mode='Unit'),
    'FoodGrid':class_generator(FoodGrid, mode='Unit'),
    'DiffusionValueLayer':class_generator(DiffusionValueLayer, mode='Unit'),
    'GaussianValueLayer':class_generator(GaussianValueLayer, mode='Unit'),
})

class SimDataOps(aux.NestedConf):
    runtype = param.Selector(objects=reg.SIMTYPES, doc='The simulation mode')
    id=param.String(None,doc='ID of the simulation. If not specified,set according to runtype and experiment.')
    dir = param.String(default=None, label='storage folder', doc='The directory to store data')
    # dir = param.Foldername(default=None, label='storage folder', doc='The directory to store data')

    store_data = param.Boolean(True, doc='Whether to store the simulation data')
    def __init__(self,runtype='Exp',save_to = None,**kwargs):
        self.param.add_parameter('experiment', self.exp_selector_param(runtype))
        super().__init__(runtype=runtype,**kwargs)
        if self.id is None :
            self.id=self.generate_id(self.runtype, self.experiment)
        if save_to is None:
            save_to = f'{self.path_to_runtype_data}/{self.experiment}'
        self.dir=f'{save_to}/{self.id}'

    def generate_id(self, runtype,exp):
        idx = reg.next_idx(exp, conftype=runtype)
        return f'{exp}_{idx}'

    @property
    def path_to_runtype_data(self):
        return f'{reg.SIM_DIR}/{self.runtype.lower()}_runs'

    # @property
    # def dir(self):
    #     return f'{self.save_to}/{self.id}'

    @property
    def data_dir(self):
        f = f'{self.dir}/data'
        os.makedirs(f, exist_ok=True)
        return f

    @property
    def plot_dir(self):
        f= f'{self.dir}/plots'
        os.makedirs(f, exist_ok=True)
        return f

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
            ids = conf[runtype].confIDs
            return param.Selector(objects=ids, **kws)
        else:
            return param.String(**kws)



class SimOps(SimDataOps,aux.SimTimeOps,aux.SimMetricOps,aux.SimGeneralOps):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


class FoodConf(aux.NestedConf):
    source_groups = aux.ClassDict(item_type=gen.FoodGroup,  doc='The groups of odor or food sources available in the arena')
    source_units = aux.ClassDict(item_type=gen.Food,  doc='The individual sources  of odor or food in the arena')
    food_grid = aux.ClassAttr(FoodGrid, default=None, doc='The food grid in the arena')


class EnvConf(aux.NestedConf):
    arena = aux.ClassAttr(gen.Arena, doc='The arena configuration')
    food_params = aux.ClassAttr(FoodConf, doc='The food sources in the arena')
    border_list = aux.ClassDict(item_type=gen.Border, doc='The obstacles in the arena')
    odorscape = aux.ClassAttr(Odorscape, default=None, doc='The sensory odor landscape in the arena')
    windscape = aux.ClassAttr(WindScape, default=None, doc='The wind landscape in the arena')
    thermoscape = aux.ClassAttr(ThermoScape, default=None, doc='The thermal landscape in the arena')

class LarvaGroup(aux.NestedConf):
    model = conf.Model.confID_selector()
    default_color = param.Color('black', doc='The default color of the group')
    odor = aux.ClassAttr(Odor, doc='The odor of the agent')
    distribution = aux.ClassAttr(aux.Larva_Distro,doc='The spatial distribution of the group agents')
    life_history = aux.ClassAttr(Life, doc='The life history of the group agents')
    sample = conf.Ref.confID_selector()
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
            conf.model = reg.conf.Model.getID(conf.model)
            # conf.model = reg.stored.getModel(conf.model)
        if as_entry :
            return aux.AttrDict({self.id: conf})
        else:
            return conf

    def __call__(self, parameter_dict={}):
        Nids=self.distribution.N
        if self.model is not None:
            m=reg.conf.Model.getID(self.model)
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

class ExpConf(aux.NestedConf):
    env_params = conf.Env.confID_selector()
    trials = conf.Trial.confID_selector('default')
    collections = param.ListSelector(default=['pose'],objects=reg.output_keys, doc='The data to collect as output')
    larva_groups = aux.ClassDict(item_type=LarvaGroup, doc='The larva groups')
    # sim_params = aux.ClassAttr(SimOps,doc='The simulation configuration')
    enrichment = aux.ClassAttr(aux.EnrichConf, doc='The post-simulation processing')
    experiment = conf.Exp.confID_selector()



    def __init__(self,id=None,**kwargs):
        super().__init__(**kwargs)

class DatasetConf(aux.NestedConf):
    environment = aux.ClassAttr(EnvConf, doc='The environment configuration')
    sim_options = aux.ClassAttr(SimOps, doc='The spatiotemporal resolution')
    larva_groups = aux.ClassDict(item_type=LarvaGroup, doc='The larva groups')


gen.Env=EnvConf
gen.LarvaGroup=LarvaGroup
gen.Exp=ExpConf





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












class ConfGeneratorRegistry :
    def __init__(self):
        # self.group=aux.AttrDict({k: BaseType(k=k) for k in GROUPTYPES})
        self.conf=aux.AttrDict({k: ConfType(conftype=k) for k in reg.CONFTYPES})
