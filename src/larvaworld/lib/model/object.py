import param

from larvaworld.lib import aux
from larvaworld.lib.param import Named


class Object:
    '''
        Basic Class for all Larvaworld model objects
        Extends the agentpy Object class by allowing recording of nested attributes

    '''
    """ Base class for all objects of an agent-based models. """

    def __init__(self, model=None, id='Nada'):
        self._var_ignore = []

        self.id = id
        # self.id = model._new_id()  # Assign id to new object
        self.type = type(self).__name__
        self.log = {}

        self.model = model
        if model is not None :
            self.p = model.p

    def __repr__(self):
        return f"{self.type} (Obj {self.id})"

    def __getattr__(self, key):
        raise AttributeError(f"{self} has no attribute '{key}'.")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def _set_var_ignore(self):
        """Store current attributes to separate them from custom variables"""
        self._var_ignore = [k for k in self.__dict__.keys() if k[0] != '_']

    @property
    def vars(self):
        return [k for k in self.__dict__.keys()
                if k[0] != '_'
                and k not in self._var_ignore]

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

    def setup(self, **kwargs):
        """This empty method is called automatically at the objects' creation.
        Can be overwritten in custom sub-classes
        to define initial attributes and actions.

        Arguments:
            **kwargs: Keyword arguments that have been passed to
                :class:`Agent` or :func:`Model.add_agents`.
                If the original setup method is used,
                they will be set as attributes of the object.

        Examples:
            The following setup initializes an object with three variables::

                def setup(self, y):
                    self.x = 0  # Value defined locally
                    self.y = y  # Value defined in kwargs
                    self.z = self.p.z  # Value defined in parameters
        """

        for k, v in kwargs.items():
            setattr(self, k, v)



class NamedObject(Named, Object):
    def __init__(self,model=None, **kwargs):
        Named.__init__(self,**kwargs)
        Object.__init__(self, model=model, id = self.unique_id)
        # if self.unique_id is not None:
        #     self.id = self.unique_id



class GroupedObject(NamedObject):
    group = param.String(None, doc='The unique ID of the entity"s group')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.group is not None:
            self.type = self.group
