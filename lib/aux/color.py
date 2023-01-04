import random
import numpy as np
from matplotlib import cm, colors



def invert_color(col, return_self=False):
    if type(col) in [list, tuple] and len(col) == 3:
        if not all([0 <= i <= 1 for i in col]):
            col = list(np.array(col) / 255)
        col = colors.rgb2hex(col)
    elif col[0] != '#':
        col = colors.cnames[col]
    table = str.maketrans('0123456789abcdef', 'fedcba9876543210')
    col2 = '#' + col[1:].lower().translate(table).upper()
    if not return_self:
        return col2
    else:
        return col, col2


def random_colors(n):
    ret = []
    r = int(random.random() * 200)
    g = int(random.random() * 200)
    b = int(random.random() * 200)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append(np.array([r, g, b]))
    return ret


def N_colors(N, as_rgb=False):
    cols=['green', 'red', 'blue', 'purple', 'orange', 'magenta', 'cyan', 'darkred', 'lightblue']
    if N<=len(cols):
        cs=cols[:N]
    elif N == 10:
        cs = ['lightgreen', 'green', 'red', 'darkred', 'lightblue', 'blue', 'darkblue', 'magenta', 'cyan', 'orange',
              'purple']
    else:
        colormap = cm.get_cmap('brg')
        cs = [colormap(i) for i in np.linspace(0, 1, N)]
    if as_rgb:
        cs = [colorname2tuple(c) for c in cs]
    return cs


def colorname2tuple(name):
    c0 = colors.to_rgb(name)
    c1 = tuple([i * 255 for i in c0])
    return c1




def col_range(q, low=(255, 0, 0), high=(255, 255, 255), mul255=False):
    rr0, gg0, bb0 = q_col1 = np.array(low) / 255
    rr1, gg1, bb1 = q_col2 = np.array(high) / 255
    qrange = np.array([rr1 - rr0, gg1 - gg0, bb1 - bb0])
    res = q_col1 + np.array([q, q, q]).T * qrange
    res = np.clip(res, a_min=0, a_max=1)
    if mul255:
        res *= 255
    return res



def get_class_by_name(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Color:

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)

    @staticmethod
    def random_color(min_r=0, min_g=0, min_b=0, max_r=255, max_g=255, max_b=255):
        r = random.randint(min_r, max_r)
        g = random.randint(min_g, max_g)
        b = random.randint(min_b, max_b)
        return r, g, b

    @staticmethod
    def random_bright(min_value=127):
        r = random.randint(min_value, 255)
        g = random.randint(min_value, 255)
        b = random.randint(min_value, 255)
        return r, g, b
