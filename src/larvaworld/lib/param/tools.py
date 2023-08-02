
from larvaworld.lib import aux
from larvaworld.lib.param import Larva_Distro, Spatial_Distro, ClassAttr, NestedConf


def class_generator(A0, mode='Unit') :
    class A(NestedConf):

        def __init__(self, **kwargs):
            if hasattr(A,'distribution'):
                D=A.distribution.__class__
                ks=list(D.param.objects().keys())
                kwargs=self.shortcut(kdict={
                    'ors':'orientation_range',
                    's': 'scale',
                    # 'p': 'scale',
                    'sh': 'shape',
                },kws=kwargs)

                existing=[k for k in ks if k in kwargs.keys()]
                if len(existing)>0:
                    d={}
                    for k in existing :
                        d[k]=kwargs[k]
                        kwargs.pop(k)
                    kwargs['distribution']=D(**d)
            kwargs = self.shortcut(kdict={
                'mID': 'model',
                'c': 'default_color',
                'or': 'orientation',
                'r': 'radius',
                'a': 'amount',
            }, kws=kwargs)

            if 'life' in kwargs.keys():
                assert 'life_history' not in kwargs.keys()
                assert len(kwargs['life'])==2
                kwargs['life_history']=dict(zip(['age', 'epochs'], kwargs['life']))
                kwargs.pop('life')
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

        def shortcut(self,kdict, kws):
            for k, key in kdict.items():
                if k in kws.keys():
                    assert key not in kws.keys()
                    kws[key] = kws[k]
                    kws.pop(k)
            return kws

        @classmethod
        def from_entries(cls, entries):
            all_confs = []
            for gid, dic in entries.items():
                Ainst = cls(**dic)
                gconf = aux.AttrDict(Ainst.param.values())
                gconf.pop('name')
                if hasattr(Ainst, 'distribution'):

                    ids = [f'{gid}_{i}' for i in range(Ainst.distribution.N)]

                    gconf.pop('distribution')
                    gconf.group=gid
                    try :
                        ps,ors=Ainst.distribution()
                        confs = [{'unique_id': id, 'pos': p, 'orientation': ori, **gconf} for id, p,ori in zip(ids, ps, ors)]
                    except:
                        ps = Ainst.distribution()
                        confs = [{'unique_id': id, 'pos': p, **gconf} for id, p in zip(ids, ps)]
                    all_confs += confs
                else:
                    gconf.unique_id=gid
                    all_confs.append(gconf)
            return all_confs


        @classmethod
        def agent_class(cls):
            return A0.__name__

        @classmethod
        def mode(cls):
            return mode

    A.__name__=f'{A0.__name__}{mode}'
    invalid = ['name', 'closed', 'visible', 'selected', 'centered']
    if mode=='Group':
        if not 'pos' in A0.param.objects():
            raise ValueError (f'No Group distribution for class {A0.__name__}. Change mode to Unit')
        distro = Larva_Distro if 'orientation' in A0.param.objects() else Spatial_Distro
        A.param._add_parameter('distribution',ClassAttr(distro, doc='The spatial distribution of the group agents'))
        invalid+=['unique_id', 'pos', 'orientation']
    elif mode=='Unit':
        pass
    for k, p in A0.param.params().items():
        if k not in invalid:
            A.param._add_parameter(k,p)
    return A