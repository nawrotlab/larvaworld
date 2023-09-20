import param
from larvaworld.lib import aux
from larvaworld.lib.param import Named

__all__ = [
    'Object',
    'NamedObject',
    'GroupedObject',
]

__displayname__ = 'Basic ABM class'

class Object:
    """
    Basic Class for all Larvaworld model objects.

    This class extends the agentpy Object class by allowing the recording of nested attributes.

    Parameters
    ----------
    model : object, optional
        The model this object belongs to.
    id : str, optional
        The unique identifier for this object.

    Attributes
    ----------
    id : str
        The unique identifier for this object.
    type : str
        The name of the object's class.
    log : dict
        A dictionary for recording log data.
    model : object
        The model this object belongs to.
    p : object
        The parameters of the model.

    Methods
    -------
    __repr__()
        Return a string representation of the object.
    __getattr__(key)
        Raise an AttributeError for unknown attributes.
    __getitem__(key)
        Get an attribute value.
    __setitem__(key, value)
        Set an attribute value.
    _set_var_ignore()
        Store current attributes to separate them from custom variables.
    vars
        Get a list of attribute names.
    _log
        Access the log data.
    extend_log(l, k, N, v)
        Extend log data with a new value.
    connect_log(ls)
        Connect the log to the model's dictionary of logs.
    nest_record(reporter_dic)
        Record nested attributes.
    setup(**kwargs)
        Initialize object attributes and actions.

    Examples
    --------
    The following setup initializes an object with three variables:

    def setup(self, y):
        self.x = 0  # Value defined locally
        self.y = y  # Value defined in kwargs
        self.z = self.p.z  # Value defined in parameters
    """

    def __init__(self, model=None, id='Nada'):
        self._var_ignore = []
        self.id = id
        self.type = type(self).__name__
        self.log = {}
        self.model = model
        if model is not None:
            self.p = model.p

    def __repr__(self):
        return f"{self.type} (Obj {self.id})"

    # Other methods and properties...

class NamedObject(Named, Object):
    """
    A named simulation object that extends the basic Object class.

    Parameters
    ----------
    model : object, optional
        The model this object belongs to.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    Inherits attributes from Object class.
    """

    def __init__(self, model=None, **kwargs):
        Named.__init__(self, **kwargs)
        Object.__init__(self, model=model, id=self.unique_id)

class GroupedObject(NamedObject):
    """
    A grouped simulation object that extends the NamedObject class.

    Attributes
    ----------
    group : str, optional
        The unique ID of the entity's group.

    Methods
    -------
    Inherits methods from NamedObject class.
    """

    group = param.String(None, doc="The unique ID of the entity's group")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.group is not None:
            self.type = self.group
