def base(method, input, **kwargs):
    if type(input) == str:
        return method(input, **kwargs)
    elif type(input) == list:
        return [method(i, **kwargs) for i in input]


def bar(p):
    return rf'$\bar{{{p.replace("$", "")}}}$'


def wave(p):
    return rf'$\~{{{p.replace("$", "")}}}$'


def sub(p, q):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}$'


def sup(p, q):
    return rf'${{{p.replace("$", "")}}}^{{{q}}}$'


def subsup(p, q, z):
    return rf'${{{p.replace("$", "")}}}_{{{q}}}^{{{z}}}$'


def hat(p):
    return f'$\hat{{{p.replace("$", "")}}}$'


def ast(p):
    return f'${p.replace("$", "")}^{{*}}$'


def th(p):
    return fr'$\theta_{{{p.replace("$", "")}}}$'


def Delta(p):
    return fr'$\Delta{{{p.replace("$", "")}}}$'


def sum(p):
    return fr'$\sum{{{p.replace("$", "")}}}$'


def delta(p):
    return fr'$\delta{{{p.replace("$", "")}}}$'


def hat_th(p):
    return fr'$\hat{{\theta}}_{{{p}}}$'


def dot(p):
    return fr'$\dot{{{p.replace("$", "")}}}$'


def circledcirc(p):
    return f'${p.replace("$", "")}^{{\circledcirc}}$'


def circledast(p):
    return f'${p.replace("$", "")}^{{\circledast}}$'


def odot(p):
    return f'${p.replace("$", "")}^{{\odot}}$'
    # return fr'$\odot{{{p.replace("$", "")}}}$'


def paren(p):
    return fr'$({{{p.replace("$", "")}}})$'


def brack(p):
    return fr'$[{{{p.replace("$", "")}}}]$'


def ddot(p):
    return fr'$\ddot{{{p.replace("$", "")}}}$'


def dot_th(p):
    return fr'$\dot{{\theta}}_{{{p.replace("$", "")}}}$'


def ddot_th(p):
    return fr'$\ddot{{\theta}}_{{{p.replace("$", "")}}}$'


def dot_hat_th(p):
    return fr'$\dot{{\hat{{\theta}}}}_{{{p}}}$'


def ddot_hat_th(p):
    return fr'$\ddot{{\hat{{\theta}}}}_{{{p}}}$'


def lin(p):
    return fr'${{{p.replace("$", "")}}}_{{l}}$'

def dtype_name(v) :
    if v is None :
        return None
    else :
        try :
            return v.__name__
        except :
            try :
                return f'{v._name}[{v.__args__[0].__name__}]'
            except :
                return v
