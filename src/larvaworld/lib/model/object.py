import param
from agentpy.objects import Object

from larvaworld.lib import aux
from larvaworld.lib.screen import InputBox


class Entity(aux.NestedConf):
    default_color = param.Color('black', doc='The default color of the entity')
    unique_id = param.String(None, doc='The unique ID of the entity')
    visible = param.Boolean(True, doc='Whether the entity is visible or not')
    group = param.String(None, doc='The unique ID of the entity"s group')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.color = self.default_color
        self.selected = False

        self.id_box = InputBox(text=self.unique_id, color_inactive=self.default_color,
                               color_active=self.default_color,
                               agent=self)

    def set_color(self, color):
        self.color = color

    def set_default_color(self, color):
        self.default_color = color
        self.color=color

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text = self.unique_id

    def _draw(self,v,**kwargs):
        if self.visible :
            self.draw(v,**kwargs)
            self.id_box.draw(v)

class ModelEntity(Entity, Object):
    def __init__(self,model, **kwargs):
        Entity.__init__(self, **kwargs)
        Object.__init__(self,model=model)

    def nest_record(self, reporter_dic):
        # Connect log to the model's dict of logs
        if self.group not in self.model._logs:
            self.model._logs[self.group] = {}
        self.model._logs[self.group][self.unique_id] = self.log
        self.log['t'] = [self.model.t]  # Initiate time dimension

        # Perform initial recording
        for name, codename in reporter_dic.items():
            v = aux.rgetattr(self, codename)
            self.log[name] = [v]

        # Set default recording function from now on
        self.nest_record = self._nest_record  # noqa

    def _nest_record(self, reporter_dic):

        for name, codename in reporter_dic.items():

            # Create empty lists
            if name not in self.log:
                self.log[name] = [None] * len(self.log['t'])

            if self.model.t != self.log['t'][-1]:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t
            self.log[name][-1] = aux.rgetattr(self, codename)


class SpatialEntity(ModelEntity):
    def __init__(self, visible=False,default_color='white', **kwargs):
        super().__init__(visible=visible,default_color=default_color,**kwargs)

    def record_positions(self, label='p'):
        """ Records the positions of each agent.

        Arguments:
            label (string, optional):
                Name under which to record each position (default p).
                A number will be added for each coordinate (e.g. p1, p2, ...).
        """
        for agent, pos in self.positions.items():
            for i, p in enumerate(pos):
                agent.record(label+str(i), p)