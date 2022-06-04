def base(method, input, **kwargs):
    if type(input) == str:
        return method(input, **kwargs)
    elif type(input) == list:
        return [method(i, **kwargs) for i in input]


def bar(p):
    return rf'$\bar{{{p.replace("$", "")}}}$'

def tilde(p):
    return rf'$\tilde{{{p.replace("$", "")}}}$'


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

def omega(p):
    return fr'$\omega_{{{p.replace("$", "")}}}$'


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

def circle(p):
    return fr'$\mathring{{{p.replace("$", "")}}}$'


def circledcirc(p):
    return f'${p.replace("$", "")}^{{\circledcirc}}$'

def mathring(p):
    return fr'$\mathring{{{p.replace("$", "")}}}$'


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
    def typing_arg(v):
        return v.__args__[0]
    if v is None :
        n= ' '
    else :
        try :
            n= v.__name__
        except :
            try :
                n= f'{v._name}[{typing_arg(v).__name__}]'
            except :
                try:
                    v0=typing_arg(v)
                    n = f'{v._name}[{v0._name}[{typing_arg(v0).__name__}]]'
                except:
                    n = v
    return n
