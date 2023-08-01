import param

from larvaworld.lib.param import NestedConf





class Named(NestedConf) :
    unique_id = param.String(None, doc='The unique ID of the entity')

    def set_id(self, id):
        self.unique_id = id


class Grouped(Named) :
    group = param.String(None, doc='The unique ID of the entity"s group')

