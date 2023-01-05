from lib import aux

class FunctionDict:
    def __init__(self):
        self.graphs = aux.NestDict()
        self.stored_confs = aux.NestDict()
        # from lib.registry.distro import generate_distro_database
        # self.distro_database = generate_distro_database()
        self.preprocessing = aux.NestDict()
        self.processing = aux.NestDict()
        self.annotating = aux.NestDict()
        self.param_computing = aux.NestDict()
        # self.annotation = annotation_funcs()

    def param(self, name):
        # print(name)
        return self.register_func(name, "param_computing")

    def preproc(self, name):
        return self.register_func(name, "preprocessing")

    def proc(self, name):
        return self.register_func(name, "processing")

    def annotation(self, name):
        return self.register_func(name, "annotating")

    def graph(self, name):
        # print(name)
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

    # def get_dist(self, **kwargs):
    #     from lib.registry.distro import get_dist
    #     return get_dist(**kwargs, distro_database=self.distro_database)

funcs=FunctionDict()