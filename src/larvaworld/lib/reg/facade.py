from larvaworld.lib import aux


class FunctionDict:
    def __init__(self):
        self.graphs = aux.AttrDict()
        self.stored_confs = aux.AttrDict()
        self.preprocessing = aux.AttrDict()
        self.processing = aux.AttrDict()
        self.annotating = aux.AttrDict()
        self.param_computing = aux.AttrDict()

    def param(self, name):
        return self.register_func(name, "param_computing")

    def preproc(self, name):
        return self.register_func(name, "preprocessing")

    def proc(self, name):
        return self.register_func(name, "processing")

    def annotation(self, name):
        return self.register_func(name, "annotating")

    def graph(self, name):
        return self.register_func(name, "graphs")

    def stored_conf(self, name):
        return self.register_func(name, "stored_confs")


    def register_func(self, name, group):
        if not hasattr(self, group) :
            raise
        d=getattr(self,group)
        def wrapper(func):
            d[name] = func
            return func
        return wrapper


funcs=FunctionDict()