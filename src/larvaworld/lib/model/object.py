import param
from agentpy import objects

from larvaworld.lib import aux
from larvaworld.lib.screen import IDBox


class Object(objects.Object):
    '''
        Basic Class for all Larvaworld model objects
        Extends the agentpy Object class by allowing recording of nested attributes

    '''

    @property
    def _log(self):
        l = self.log
        t = self.model.t
        if 't' not in l :
            l['t'] = [t]  # Initiate time dimension
        # Extend time dimension
        if t != l['t'][-1]:
            l['t'].append(t)
            for v in l.values():
                while len(v) < len(l['t']):
                    v.append(None)
        return l

    def extend_log(self,l, k,N, v):
        if k not in l:
            l[k] = [None]
        while len(l[k]) < N:
            l[k].append(None)
        l[k][-1] = v
        return l

    def connect_log(self,ls):
        t=self.type
        # Connect log to the model's dict of logs
        if t not in ls:
            ls[t] = {}
        if self.id not in ls[t]:
            ls[t][self.id] = self.log

    def nest_record(self, reporter_dic):
        self.connect_log(self.model._logs)

        l = self._log
        N = len(l['t'])
        for k, codename in reporter_dic.items():
            l=self.extend_log(l,k, N, v= aux.rgetattr(self, codename))

        self.log = l



class Named(aux.NestedConf) :
    unique_id = param.String(None, doc='The unique ID of the entity')

    def set_id(self, id):
        self.unique_id = id


class Grouped(Named) :
    group = param.String(None, doc='The unique ID of the entity"s group')


class NamedObject(Named, Object):
    def __init__(self,model=None, **kwargs):
        Named.__init__(self,**kwargs)
        Object.__init__(self, model=model)
        if self.unique_id is not None:
            self.id = self.unique_id
        self.selected = False


class GroupedObject(NamedObject):
    group = param.String(None, doc='The unique ID of the entity"s group')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.group is not None:
            self.type = self.group
