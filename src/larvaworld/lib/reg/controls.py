import json

from larvaworld.lib import reg, aux


def get_pygame_key(key):
    pygame_keys = {
        'BackSpace': 'BACKSPACE',
        'tab': 'TAB',
        'del': 'DELETE',
        'clear': 'CLEAR',
        'Return': 'RETURN',
        'Escape': 'ESCAPE',
        'space': 'SPACE',
        'exclam': 'EXCLAIM',
        'quotedbl': 'QUOTEDBL',
        '+': 'PLUS',
        'comma': 'COMMA',
        '-': 'MINUS',
        'period': 'PERIOD',
        'slash': 'SLASH',
        'numbersign': 'HASH',
        'Down:': 'DOWN',
        'Up:': 'UP',
        'Right:': 'RIGHT',
        'Left:': 'LEFT',
        'dollar': 'DOLLAR',
        'ampersand': 'AMPERSAND',
        'parenleft': 'LEFTPAREN',
        'parenright': 'RIGHTPAREN',
        'asterisk': 'ASTERISK',
    }
    return f'K_{pygame_keys[key]}' if key in list(pygame_keys.keys()) else f'K_{key}'

def init_shortcuts():
    draw = {
        'visible trail': 'p',
        '▲ trail duration': '+',
        '▼ trail duration': '-',

        'draw_head': 'h',
        'draw_centroid': 'e',
        'draw_midline': 'm',
        'draw_contour': 'c',
        'draw_sensors': 'j',
    }

    inspect = {
        'focus_mode': 'f',
        'odor gains': 'z',
        'dynamic graph': 'q',
    }

    color = {
        'black_background': 'g',
        'random_colors': 'r',
        'color_behavior': 'b',
    }

    aux = {
        'visible_clock': 't',
        'visible_scale': 'n',
        'visible_state': 's',
        'visible_ids': 'tab',
    }

    screen = {
        'move up': 'UP',
        'move down': 'DOWN',
        'move left': 'LEFT',
        'move right': 'RIGHT',
    }

    sim = {
        'larva_collisions': 'y',
        'pause': 'space',
        'snapshot': 'i',
        'delete item': 'del',

    }

    odorscape = {
        'odor_aura': 'u',
        'windscape': 'w',
        'plot odorscapes': 'o',
        **{f'odorscape {i}': i for i in range(10)},
        # 'move_right': 'RIGHT',
    }

    d = {
        'draw': draw,
        'color': color,
        'aux': aux,
        'screen': screen,
        'simulation': sim,
        'inspect': inspect,
        'landscape': odorscape,
    }

    return d


def init_controls():
    k = init_shortcuts()
    d = {'keys': {}, 'pygame_keys': {}, 'mouse': {
        'select item': 'left click',
        'add item': 'left click',
        'select item mode': 'right click',
        'inspect item': 'right click',
        'screen zoom in': 'scroll up',
        'screen zoom out': 'scroll down',
    }}
    ds = {}
    for title, dic in k.items():
        ds.update(dic)
        d['keys'][title] = dic
    d['pygame_keys'] = {k: get_pygame_key(v) for k, v in ds.items()}
    return d

class ControlRegistry :
    def __init__(self):
        self.path=f'{reg.CONF_DIR}/controls.txt'
        self.conf=init_controls()

    def save(self, conf=None):
        if conf is None:
            conf = self.conf
        with open(self.path, "w") as fp:
            json.dump(conf, fp)

    def load(self):
        with open(self.path) as tfp:
            c = json.load(tfp)
        return aux.AttrDict(c)


controls=ControlRegistry()
