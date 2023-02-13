# def loader(attr):
# def add_prop(k, obj):
#    _k=f'_{k}'
import larvaworld.lib.aux.dictsNlists as dNl
from larvaworld.lib import reg


def alias(k, inverse=False):
    if not inverse:
        return f'_{k}'
    else:
        return str(k)[1:]


def datapath(self, attr):
    k = alias(attr, inverse=True)
    return reg.datapath(k, self.dir)


def dic_loader(self, attr):
    v = getattr(self, attr)
    if v is None:
        # k=alias(attr,inverse=True)
        # path=preg.datapath(k,self.dir)
        d = dNl.load_dict(self.datapath(attr))
        # print('Loaded')
        if d is not None:
            setattr(self, attr, d)
            return d
    return v


def dic_saver(self, attr, value):
    if isinstance(value, dict):
        setattr(self, attr, value)
        path = self.datapath(attr)
        # path=getattr(self,'path_dict')[attr]
        dNl.save_dict(value, path)


def setter(self, attr, value):
    setattr(self, attr, value)


def getter(self, attr):
    v = getattr(self, attr)
    return v


def custom_property(setter_func=None, getter_func=None):
    if setter_func is None:
        setter_func = setter
    if getter_func is None:
        getter_func = getter

    def attrsetter(attr):
        def set_any(self, value):
            setter_func(self, attr, value)

        return set_any

    def attrgetter(attr):
        def get_any(self):
            return getter_func(self, attr)

        return get_any

    def new_prop(_k):
        return property(fset=attrsetter(_k), fget=attrgetter(_k))

    return new_prop


def arg_dict(init_dic, prop_func, kwargs={}):
    if isinstance(init_dic, list):
        init_dic = {k: None for k in init_dic}
    dic = {}
    for k, v in init_dic.items():
        _k = alias(k)
        dic.update({_k: v, k: prop_func(_k)})
    dic.update(**kwargs)
    return dic


dic_property = custom_property(setter_func=dic_saver, getter_func=dic_loader)

ks = ['pooled_epochs', 'cycle_curves', 'chunk_dicts']
dic_manager_kwargs = arg_dict(ks, dic_property, kwargs={'datapath': datapath})








