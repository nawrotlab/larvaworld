import param
from agentpy import objects

from larvaworld.lib import aux
from larvaworld.lib.screen import IDBox


class Object(objects.Object):
    '''
        Basic Class for all Larvaworld model objects
        Extends the agentpy Object class by allowing recording of nested attributes

    '''

    def nest_record(self, reporter_dic):
        # Connect log to the model's dict of logs
        if self.type not in self.model._logs:
            self.model._logs[self.type] = {}
        self.model._logs[self.type][self.id] = self.log
        self.log['t'] = [self.model.t]  # Initiate time dimension

        # Perform initial recording
        for var_key, codename in reporter_dic.items():
            v = aux.rgetattr(self, codename)
            self.log[var_key] = [v]

        # Set default recording function from now on
        self.record = self._record  # noqa

    def _nest_record(self, reporter_dic):

        for var_key, codename in reporter_dic.items():

            # Create empty lists
            if var_key not in self.log:
                self.log[var_key] = [None] * len(self.log['t'])

            if self.model.t != self.log['t'][-1]:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t
            self.log[var_key][-1] = aux.rgetattr(self, codename)


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
