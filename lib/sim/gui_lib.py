import copy
from ast import literal_eval
from typing import List, Tuple

import numpy as np
import PySimpleGUI as sg
from random import randint
import operator

from matplotlib import ticker, pyplot
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from lib.conf import agent_pars
from lib.conf.par_db import par_db

on_image = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAHGElEQVRo3u2b3W8T6RWHnzMzSbDj4KTkq1GAfFCSFrENatnQikpFC2oqRWhXq92uKm7aKy5ou9cV1/wFvQAJqTdV260qaLdSF6RsS5tN+WiRFopwTRISNuCAyRIF8jHJeObtxYyd8diYhNjBEI70KvZ4rGie9ze/c877joVAtLW19ezcuXPvpk2bIgAKxYsMQbifnDRvjcW13d1v1DY2NIm1ZM1RhmGa5tzw8PC/x8fHrymlnOzr8KKjo+NbR48e/VV3d/e+yWSC+fm5AohVnlfFD0c5/O3SJ0QjX+GdQ+8TqY4QiUTQNK3sICulsCyL+fl5RkdHr506depYLBb7LAt0T0/PD44fP3720ueDoTMDv2P6yUNEVFBay2BlndTsCD95+2e89d0+urq62LZtG4ZhUM4xOztLLBZjYmLCPHHixLtXr179K4Bs3ry54eTJk/HzQx/XfXzh97kQ04DFB3gdQIsN+3sOcfSDD+nt7WXLli0A2LaNbdtlB1jXdXRdz7y/fv068Xh87tixY7uTyeSY0d/f//OpmYd1f7nwUV7ISgAtG3IW9JIoGSSl8fZbP6K9vT0DOX17WpZVdqArKyvRNA0RF8yuXbtIJpPVhw8f/vD06dO/MHp7ew9/9p9PUQGrUGm43l//e5VP2UUELyY017fSVN/M1q1bl4+LUFVVRWVlZdmBFpEM5LTCW1pa2LNnzyEAo6mpqW3yy0SuXaShaoDu/dV8xyihlZjQWPdVAMLhcMELKueIRCK0trZ+Xdd1wwiHw5sdx862Cy0A2QClB4BLniRZpNA00ETjZY+0IJRS5KTwjP+KD7IBeLD9ys6cX+x4+RnnhJHXAjxVpxXtV7XSfRZSqjv4lQWdr4XxeXQasDIC9lGiUk/JRgDtT4bis4m0inWfmv2TUkyTlg2iaL9PK5+NpEu8nNr6FYVTMtD+W1bl6wbzjdexBuso0Iz44aswqK2gqgELtCTIg+y1J6fNVb82AaR8C0bbvbx3Z6ODfkbY3wC7N7tCsAHtPuifgiy6oO39oKpAvwH6leUJSH0PRIE2vjHujOcqpJxWsL/jAtOvQMVZMM6BJMFpBvtAnonZBapu43r66kErsHu8fv6Kq1SZBi0BFefc9tlpAVWfa0Wp/RvXo7Xn+YZqdMFptwOfpUC766m+yXfccr1bNYDT/Rr0ysLrFHE8Hw4K1/ReVGWr2Rj0vHkvqNCrAU8p9dSx9mRoe0N3k1wQdgbiUmACZkC/DvY3wd4HL3IrMh+IYp8T3G5bPWgHZMq1D6cT9Ju+zyrcRAluqRf0dv1zcDrcgcqdjGJcuIg889z1AB1cyl09aAH9GqQOgb3X8+q7QAhS33YtQ+67FUi+u0EfglTf6qoOx3HWBU4xJ2HtisatffXLYL/p1tJ2r28eHoLx9wLfTbhJ1OlYnZodxykbiCv5P/79w8KgVf7XotzuUL8B2pjX4UXcikOSoN0LqP9ybruuXwJt0vP6FSr6ZQMdPCcLtKhlpgIo5YOsfMN7L3OgxwrbjDaS26CICRJfeePyLNDlYhn+zwuCzgBULmRJg3W8kT7ueCt5an06vLWCLgd/L2wdahkwjnurp5eepZSQ1co8upySX/CcFSmaoJJtkPT6tA9yqZ7vCD4k9TRFl6NlFAbt92FZBi0e5Axgr45O77BIqdaknWcrer3soFiTZeRTU8aHxX00K0vt3paW+B8VKzFoEckCXc6WUbCOzupifLaR5cfKU7dG1g6LUHxVu5O9fAGVlZUsLCy8cDtY6Tm6rlNRUZH1uWFZFvXRRvKWec5ymZdJfnkenilFMpx+MoVSsLi4SCgUoqKiAtM0n7poUw52kX6Kqq6uDhFhYWEh85ygce/evZneN/ZH/3H13DI45dvYdjzIDrl7hSUs7SYejPNkboZEIkFnZyfRaBQR4fHjxywuLq4I1vMAXstEhEIhGhoaCIVCKKWYnJwkmUwuKKWUMTQ0dPHIkSN9+3Z/n0v/vZAN219deGBlnXa+HVJ88s8/U1e7hebmZqqrq4lGo9TU1KyoS3wRISIZbx4dHWV2dpaLFy9eVkrZ+uzs7Nz27ds/6DvQz5JpMX53FCfQG4uncFG+0kuVeACjX8TpbO0itehQU1NDOBxG07SyHrZtE4/HGR4eJh6Pc+bMmV9OT0/fMO7cufOngYGBs5ZlvfNe3xH6D7zL/8ZusrAw9xTFrt+vWhzH4Y/nf8uDqfuYpkkkEiEajZblTysAlpaWePToEaZpEovFGBwcHBgbG/soc/MbhhE5ePDgH9rb23/Y0tJCbW0thmG4PlQGm6g3R24w9eVDvta2k8b6JnS9vH5eIbhJ0LIsZmbcvHL79u3zAwMD76VSqSdZLisismPHjh93dXX9tLGx8U3DMCK8jtUm28VEIvGvW7du/XpkZOQ3ypcx/w+op8ZtEbCnywAAAABJRU5ErkJggg=='
off_image = b'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'
off_image_disabled = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsIAAA7CARUoSoAAAAWJSURBVGhD7ZtLTyxFFMd7Bhgew2OAIXgDxkQWLgYWJsaNiTFxozcuiQs/gR9Bv4crXRuJG2OIiQvj1q25CZC4IQ4wL2B4zPCGwfqVcyaHTs371Xcu/+Sf6q6qnjr1r1PVVYcmtLGx4SmEotHoB7Ozs59GIpG3y3lBxIvj4+N/h4eHH2ZmZsbLeUFAqVgsvjo9Pf3t9vY2Vc6zqAg9Pj7+3srKyvexWOzjkZERz3TC5gcR9/f33t3dnXdycuIdHh56xjG8UChULu0fsGFiYsIbHR29TaVS3yWTyW9LpdKtLUNoI/Lq2tran9PT0wuGgRZZYDzGM57jGQ/ytra2rPj9wuPjY/nqf6ChcVrv8vLyj+3t7Zem/G5ofX09lEgkfp+bm1sx9MLhsH0QmtGoXAeBAjxnaGgIB7ECMwPNUmJtp6xXFPjzbm5uvHw+7y0vL79r7D4rFAp/hc1S8bkZgffNWmcrCURk0iBQbNGCIyx24yDmnWLzdKe7QQ1Xvlwz4/b29hD7G3MbRuhPMBIPEVCZ5QPiLUGg2IO4GmY9tLabfth73flukPaFkqfblWuAVxvb45OTkx+Gx8bG3nkd1uRaQGgGA0iH+0FpX9KHhwe7tBl942ZgwtO25DWH7mC/WAtP5+EAQE/tbrGayP5UY6CE1h3vBRHd1a5AXw+cR/s73Q2KV0t7jWDghO4VtPBadH2t8bx0tEAXquULnj26DdQTV2OghUYIjumcHBcWFmzwiXsN9uCcLl2UutFo9Ek+hyO5blTsgRUaARYXFy0J8ohYkicCITQD4KI50dk6PO8vY/DgGy/0/Py8Z069NpyazWZt3IGUk5p4uQb5mUzmCYkOahCWJT+dTleoYy+1MJBCs/0Sb8zlct7V1ZU9DpNyDyjX3ohg19fXT8ggaRAoIp/onNR5o4Um0AQQyiUW3ovIUg/4lxAJUmkwOFJGKhHDRjCQQounElZ1QbxQezSzQF5wQj9knUdoqAeqHvoqNB1uly6IwHipC3J01gOBl6dSqQpZf/3gjwtSfnBw4F1cXJRL6qMloV0dbpYSxG+XLrCGUkb417+d454BoH2WEQH1udf0g8HQ5dVmjAtPhNYdqMZuCqThesZFF8g/Pz+31+yfme4ITMo9oLza891A00LXg+uZZtnMYFYDW7NCoWCXCV5c7J1JuUfks7Ozcs3eoGmhe8FOgN9hTWUtJWUPTLq/v2//xCTtsBzwyQJ51SCfNchy0oqNFaGlk+2yHbh+rx7rge0dno0HkyKsBrOHlxp77Gpgv0wd9uIajbQvaOll6IJfgF5Rw1XeDfpRLV+jI0tHr16QQYLLbn2v80FHhG4Xrt9slH646nSa4ljSXiNoe+nQBvSDGq7ybhLBXe0K9HVFaI6j/gdqkUb6vWToI7RA7Oomq/XBn2ogdCXqwh5TP1yLnYDrd5uhPmJzL2k/yAC4IM4QNhVGJMIlXyzphztJtkearjqNkg5gL3ayZePYrW3vNQVyTYp9OINhPFwsFvfYiGMsxsu3bHRG/1Ar9IvjqtMK6QBBfcAel9+Wk56rfqdYrT+6XbkG8Xjc1jN78GRoc3Pzq0Qi8SOxVv4qIa4ulYMIsZFZcXR0ZKNpu7u7lahcr+DSSPKIrayurnLcv9zZ2XkrbE5Ev+ZyuT1ORhgtx0w6E1QCsZeYRjKZtPl0spfUkDwGm8CVcV6rZTab/cl4dUG++H+5tLS0GYvF+LrULh299o5mIGs88QeO1UxRGYB+AhskDItd+Xz+n3Q6/ZGx9ajyPyzRaPRLMxI/RCKRaf5EE1Sh8Rpe3qzNdEo+1w0CsA0HwJPNjPs7k8l8Ye4PKKsIDYy481NTU18b0T8zo/LCPz2eURvGo0tm9/PKvPx+MfzZZJW3zp73H5XujC+u8bu1AAAAAElFTkSuQmCC'
on_image_disabled = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsIAAA7CARUoSoAAAAVLSURBVGhD7Zu7TytHFMbHyxvsJeb9kjCiCihRpJAobaqblFGUJmXqKMofkyZpaKhS5HaJIgUpRZQmoqAhSgEUSBZgsAE/uJhX5ht80LmH8e56fdde7PuTPu0yj/XMt7Nnd2aXxPr6uuKMjIx84LruR47jJGtJbeeVplQqOaOjo+8MDAwk7u7uyrWsWIF2FYvFf3Rbt/HnQ+oDj0Ynk8kPl5eXf9Amf6L7pW5vb9X9/b3Jaye5XE719fWpubk51dPTY/bjijba+KbN3t7d3f324uLir1rWg9HpdPrFysrKy0KhMJTNZtX19XUtu/0sLi6qyclJlUqlcLWpRCJRy4knNzc3ShusKpXKq52dnS/z+fyvSE9sbGxMrq2t/Xd8fJw+PDw0hf1oRWdxNY2Pj6tMJqMmJiZUf3//Y3ocrjQJPOG+nJ2dYWSXt7a23tMRYt+Zn5//rlqteppMB5EHi5rZ2VmEtEeTAUzGJRo3yZOv7ydo94j293v8ndjW6JDxvh7RpoBEGtsKo9FofdNTq6urampqSvX29tZynhcIIUdHR//qUb3iDA4OZnDzs0Gm0khulQCMBs/VZIC2Dw8Pv6v71OvoO7lri3nUYb5tlToRp7Z9Deos37ZanYbVaA7vON/qCU1k6kQC94oMhxFk+FuCU9doPnptkPFRqBN5YjTvKO1LE3iZtwSjMwNiDGnYaD6aEa/1czieFdXQ0JB1wQfPw5C8Cii9Wwg9omHw2NiYmSLDaCz4YNoJ8ScHpGNBCGU4SIe6hVBGY+0BBmOiUy6XzQIKpptY9cOohrESjHg+y+u2ON+w0TAXpgGYfHl5aZYGq9WqMRsLLDDbNnXGyelWQsVoisUwl4OTQGvZPF5TOsxHyOlGQsdogNEroTQZGkqlktkiLnfq7M+LpnpsM4zS5EIVXvFUKhVzAmC2zH+OoA/1JGnYaByEwoN8PONhBXFbgngOw1GvnaNamhJWjdBwb2EmDAP0/EwvTV3XNQbiRNDJ4KBxuIGGQXayGXlhKx9WnFDDCjdBGEZhIJ1Om+dnmI2RXCwWayWfgrpXV1e1v4IhG10P2dEwCoKtnpQkVOgAGNX5fN7c5LCP+IvHOzxT85sk0uUoxt+oh7ygyI7Y5IetTlSSNBUoYSheg8E4mCYf9wDy5asyqlfvFZrE1pFGhd+0pYdRPbzKPTGaF6B9WVEeJGro95uRH7Y6jcqLuiOaKvIDyP2oFBRb3bDywlbeT5LAocPvQFEif5sUBFu9RuVHkDq+RvOK/ECIeW8y7nHZsJULIj9sdRpVEKxGU2W+lftRywtb+bDywlY+qCTGaLkuAagw39pGcBSjWoJJkFe+hJdtRn7Y6kBAznwdZPCVNg5V4gegfS4KI29KgB4VMWVHo7nZtjpcvG1hZTuulK0eID/RdpQDjn7+PcfMrh5UGciDRiVA69w03UfjMdVHw9EB5EUp/IaXbHXQdrwUQTsB2q5nwZc6/T6xubn5WyaT+Wxvb08VCgVTwAtbmIkCNHpmZkYtLCyY76P5iwQ6GXGE/MHMFzPlg4ODP/f39z91Tk9Pfzw/P1dLS0tqenra10h0shUC+JQYbTs5OXltfQRtjKvQdhhMyuVyP5k244t/PXJ+0aPmCywM4dLEohAuD1S0QUa0ApiMD9LxMTrCB1SvXe0GnuHegi1M1m3/I5vNvtBZd8Zo3fCkNvvnZDL5OV41Ic7EqTM48RjReOdo+3QhLmAAwmis4ejQ8bu+Ir/SaWYpk/9XViKVSn3tuu43ujMf67t8975JDYk29UrfAP/WA2NdawNJDzlK/Q9RjPZ1HEiBtwAAAABJRU5ErkJggg=='

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'
button_kwargs = {'font': ('size', 5),
                 'size': (7, 1)
                 }
header_kwargs = {'font': ('size', 7),
                 'size': (15, 1)}
text_kwargs = {'font': ('size', 7),
               'size': (12, 1)}

def retrieve_value(v, type):
    if v in ['', 'None', None]:
        vv = None
    elif v in ['sample', 'fit']:
        vv = v
    elif type == 'bool':
        if v in ['False', False]:
            vv = False
        elif v in ['True', True]:
            vv = True
    elif type == List[Tuple[float, float]]:
        v = v.replace('{', ' ')
        v = v.replace('}', ' ')
        vv = [tuple([float(x) for x in t.split()]) for t in v.split('   ')]
    elif type == Tuple[float, float]:
        v = v.replace('{', '')
        v = v.replace('}', '')
        v = v.replace('[', '')
        v = v.replace(']', '')
        v = v.replace("'", '')
        vv = tuple([float(x) for x in v.split()])
    elif type == tuple or type == list:
        try:
            vv = literal_eval(v)
        except:
            vv = [float(x) for x in v.split()]
            if type == tuple:
                vv = tuple(vv)
    elif type(v) == type:
        vv = v
    else:
        vv = type(v)
    return vv


def get_table_data(values, pars_dict, Nagents):
    data = []
    for i in range(Nagents):
        dic = {}
        for j, (p, t) in enumerate(pars_dict.items()):
            v = values[(i, p)]
            dic[p] = retrieve_value(v, t)
        data.append(dic)
    return data


def build_table_window(data, pars_dict, title):
    text_args = {'font': 'Courier 10',
                 'size': (15, 1),
                 'justification': 'center'}
    pars = list(pars_dict.keys())
    par_types = list(pars_dict.values())
    Nagents, Npars = len(data), len(pars)
    # A HIGHLY unusual layout definition
    # Normally a layout is specified 1 ROW at a time. Here multiple rows are being contatenated together to produce the layout
    # Note the " + \ " at the ends of the lines rather than the usual " , "
    # This is done because each line is a list of lists
    layout = [[sg.Text(title, font='Default 12')]] + \
             [[sg.Text(' ', size=(2, 1))] + [sg.Text(p, key=p, enable_events=True, **text_args) for p in pars]] + \
             [[sg.T(i + 1, size=(2, 1))] + [sg.Input(data[i][p], key=(i, p), **text_args) for p in pars] for i in
              range(Nagents)] + \
             [[sg.Button('Add'), sg.Button('Remove'), sg.Button('Ok'), sg.Button('Cancel')]]

    # Create the window
    table_window = sg.Window('A Table Simulation', layout, default_element_size=(20, 1), element_padding=(1, 1),
                             return_keyboard_events=True, finalize=True)
    table_window.close_destroys_window = True
    return Nagents, Npars, pars, table_window


def gui_table(data, pars_dict, title='Agent list'):
    """
        Another simple table created from Input Text Elements.  This demo adds the ability to "navigate" around the drawing using
        the arrow keys. The tab key works automatically, but the arrow keys are done in the code below.
    """

    sg.change_look_and_feel('Dark Brown 2')  # No excuse for gray windows
    # Show a "splash" type message so the user doesn't give up waiting
    sg.popup_quick_message('Hang on for a moment, this will take a bit to create....', auto_close=True,
                           non_blocking=True)

    Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)

    current_cell = (0, 0)
    while True:  # Event Loop
        event, values = table_window.read()
        if event in (None, 'Cancel'):
            table_window.close()
            return data
            # break
        if event == 'Ok':
            data = get_table_data(values, pars_dict, Nagents)
            table_window.close()
            return data
        elem = table_window.find_element_with_focus()
        current_cell = elem.Key if elem and type(elem.Key) is tuple else (0, 0)
        r, c = current_cell

        if event.startswith('Down'):
            r = r + 1 * (r < Nagents - 1)
        elif event.startswith('Left'):
            c = c - 1 * (c > 0)
        elif event.startswith('Right'):
            c = c + 1 * (c < Npars - 1)
        elif event.startswith('Up'):
            r = r - 1 * (r > 0)
        elif event in pars:  # Perform a sort if a column heading was clicked
            col_clicked = pars.index(event)
            try:
                table = [[int(values[(row, col)]) for col in range(Npars)] for row in range(Nagents)]
                new_table = sorted(table, key=operator.itemgetter(col_clicked))
            except:
                sg.popup_error('Error in table', 'Your table must contain only ints if you wish to sort by column')
            else:
                for i in range(Nagents):
                    for j in range(Npars):
                        table_window[(i, j)].update(new_table[i][j])
                [table_window[c].update(font='Any 14') for c in pars]  # make all column headings be normal fonts
                table_window[event].update(font='Any 14 bold')  # bold the font that was clicked
        # if the current cell changed, set focus on new cell
        if current_cell != (r, c):
            current_cell = r, c
            table_window[current_cell].set_focus()  # set the focus on the element moved to
            table_window[current_cell].update(
                select=True)  # when setting focus, also highlight the data in the element so typing overwrites
        if event == 'Add':
            data = get_table_data(values, pars_dict, Nagents)
            data.append(data[r])
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
        elif event == 'Remove':
            data = get_table_data(values, pars_dict, Nagents)
            data = [d for i, d in enumerate(data) if i != r]
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
            # table_window.close()
            # gui_table(data, pars_dict, title='Agent list')

    # if clicked button to dump the table's values
    # if event.startswith('Show Table'):
    #     table = [[values[(row, col)] for col in range(Npars)] for row in range(Nagents)]
    #     sg.popup_scrolled('your_table = [ ', ',\n'.join([str(table[i]) for i in range(Nagents)]) + '  ]', title='Copy your data from here')


def update_window_from_dict(window, dict):
    if dict is not None:
        for k, v in dict.items():
            if type(v) != bool:
                window.Element(k).Update(value=v)
            else:
                window[f'TOGGLE_{k}'].metadata.state = v
                window[f'TOGGLE_{k}'].update(image_data=on_image if v else off_image)


class SectionDict:
    def __init__(self, name, dict, type_dict=None):
        self.init_dict = dict
        self.type_dict = type_dict
        self.name = name

    def init_section(self, dict=None):
        if dict is None:
            if self.init_dict is not None:
                dict = self.init_dict
            else:
                return []
        l = []
        for k, v in dict.items():
            if type(v) != bool:
                l.append([sg.Text(f'{k}:', **text_kwargs), sg.In(v, key=k, **text_kwargs)])
            else:
                l.append(bool_button(k, v))
        return l

    def get_dict(self, values, window):
        new_dict = copy.deepcopy(self.init_dict)
        if new_dict is None:
            return new_dict
        if self.type_dict is None :

            for i, (k, v) in enumerate(new_dict.items()):
                if type(v) == bool:
                    new_dict[k] = window[f'TOGGLE_{k}'].metadata.state
                else:
                    vv = values[k]
                    vv = retrieve_value(vv, type(v))
                    new_dict[k] = vv

        else :
            for i, (k, t) in enumerate(self.type_dict.items()):
                if t == bool:
                    new_dict[k] = window[f'TOGGLE_{k}'].metadata.state
                else:
                    new_dict[k] = retrieve_value(values[k], t)
        return new_dict





    # def update_section(self, window, dict):
    #     # if dict is None:
    #     #     if self.init_dict is not None:
    #     #         dict = self.init_dict
    #     #     else:
    #     #         return
    #     for k, v in dict.items():
    #         v0=self.init_dict[k]
    #         if type(v0)==bool :
    #             event=f'TOGGLE_{k}'
    #             if type(v)==bool :
    #                 window[event].metadata.state = v
    #                 window[event].update(image_data=on_image if v else off_image)
    #             else :
    #                 raise ValueError (f'Parameter {k} initial boolean value {v0} but new non boolean {v} was passed')
    #         elif type(v0)==dict :
    #             # self.update_section(window, )
    #             raise ValueError(f'Parameter {k} initial dict value {v0}')
    #         else :
    #             window.Element(k).Update(value=v)


def bool_button(name, state):
    if state:
        image = on_image
    elif not state:
        image = off_image
    elif state is None:
        image = off_image_disabled
    l = [sg.Text(f'{name} :', **text_kwargs),
         sg.Button(image_data=image, k=f'TOGGLE_{name}', border_width=0,
                   button_color=(
                       sg.theme_background_color(), sg.theme_background_color()),
                   disabled_button_color=(
                       sg.theme_background_color(), sg.theme_background_color()),
                   metadata=BtnInfo(state=state))]
    return l


class Collapsible:
    def __init__(self, name, state, content, toggle=None, disabled=False):
        self.name = name
        self.state = state
        self.content = content
        self.toggle = toggle
        if state:
            self.symbol = SYMBOL_DOWN
        else:
            self.symbol = SYMBOL_UP
        header = [sg.T(self.symbol, enable_events=True, k=f'OPEN SEC {name}', text_color='black'),
                  sg.T(name, enable_events=True, text_color='black', k=f'SEC {name} TEXT', **header_kwargs)]
        if toggle is not None:
            if disabled:
                toggle_state = None
                if toggle:
                    image = on_image_disabled
                elif not toggle:
                    image = off_image_disabled
            else:
                toggle_state = toggle
                if toggle:
                    image = on_image
                elif not toggle:
                    image = off_image
            header.append(sg.Button(image_data=image, k=f'TOGGLE_{name}', border_width=0,
                                    button_color=(
                                        sg.theme_background_color(), sg.theme_background_color()),
                                    disabled_button_color=(
                                        sg.theme_background_color(), sg.theme_background_color()),
                                    metadata=BtnInfo(state=toggle_state)))

        self.section = [header, [collapse(content, f'SEC {name}', visible=state)]]

    def get_section(self, as_col=True):
        if as_col:
            return [sg.Col(self.section)]
        else:
            return self.section

    def set_section(self, section):
        self.section = section

    def update(self, window, dict):
        if dict is None:
            if self.toggle is not None:
                window[f'TOGGLE_{self.name}'].metadata.state = None
                window[f'TOGGLE_{self.name}'].update(image_data=off_image_disabled)
            self.state = None
            window[f'OPEN SEC {self.name}'].update(SYMBOL_UP)
            window[f'SEC {self.name}'].update(visible=False)
        else:
            if self.toggle is not None:
                window[f'TOGGLE_{self.name}'].update(image_data=on_image_disabled)
            if self.state is None:
                self.state = False
            update_window_from_dict(window, dict)
        return window


def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible))


def set_kwargs(kwargs, title='Arguments', type_dict=None):
    sec_dict=SectionDict(name=title, dict=kwargs, type_dict=type_dict)
    layout=sec_dict.init_section()
    layout.append([sg.Ok(), sg.Cancel()])
    window=sg.Window(title, layout)
    while True :
        event, values = window.read()
        if event=='Ok' :
            new_kwargs=sec_dict.get_dict(values, window)
            break
        elif event=='Cancel' :
            new_kwargs=kwargs
            break
        elif 'TOGGLE' in event:
            if window[event].metadata.state is not None:
                window[event].metadata.state = not window[event].metadata.state
                window[event].update(image_data=on_image if window[event].metadata.state else off_image)
    window.close()
    del sec_dict
    return new_kwargs

    # if kwargs != {}:
    #     layout = []
    #     for i, (k, v) in enumerate(kwargs.items()):
    #         if not type(v) == dict and not type(v) == np.ndarray:
    #             layout.append([sg.Text(k, size=(20, 1)), sg.Input(default_text=str(v), k=f'KW_{i}', size=(20, 1))])
    #     layout.append([sg.Ok(), sg.Cancel()])
    #     event, values = sg.Window(title, layout).read(close=True)
    #     if event == 'Ok':
    #         for i, (k, v) in enumerate(kwargs.items()):
    #             if type(v) == np.ndarray:
    #                 continue
    #             if not type(v) == dict:
    #                 vv = values[f'KW_{i}']
    #                 if type(v) == bool:
    #                     if vv == 'False':
    #                         vv = False
    #                     elif vv == 'True':
    #                         vv = True
    #                 elif type(v) == list or type(v) == tuple:
    #                     vv = literal_eval(vv)
    #
    #                 elif v is None:
    #                     if vv == 'None':
    #                         vv = None
    #                     else:
    #                         vv = vv
    #                 else:
    #                     vv = type(v)(vv)
    #                 kwargs[k] = vv
    #             else:
    #                 kwargs[k] = set_kwargs(v, title=k)
    #
    # return kwargs


def set_agent_kwargs(agent):
    class_name = type(agent).__name__
    pars = agent_pars[class_name]
    title = f'{class_name} args'
    kwargs={}
    for p in list(pars.keys()) :
        kwargs[p] = getattr(agent,p)
    new_kwargs=set_kwargs(kwargs, title, type_dict=pars)
    for p,v in new_kwargs.items() :
        if p == 'unique_id':
            agent.set_id(v)
        else:
            setattr(agent, p, v)
    return agent


def object_menu(selected):
    object_list = ['Larva', 'Food', 'Border']
    title = 'Select object type'
    layout = [
        [sg.Text(title, **header_kwargs)],
        [sg.Listbox(default_values=[selected], values=object_list, change_submits=False, size=(20, len(object_list)),
                    key='SELECTED_OBJECT',
                    enable_events=True)],
        [sg.Ok()]]
    window = sg.Window(title, layout)
    while True:
        event, values = window.read()
        sel = values['SELECTED_OBJECT'][0]
        if event == 'Ok':
            break
    window.close()
    return sel


def delete_objects_window(selected):
    ids = [sel.unique_id for sel in selected]
    title = 'Delete objects?'
    layout = [
        [sg.Text(title, **header_kwargs)],
        [sg.Listbox(default_values=ids, values=ids, change_submits=False, size=(20, len(ids)), key='DELETE_OBJECTS',
                    enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    window = sg.Window(title, layout)
    while True:
        event, values = window.read()
        if event == 'Ok':
            res = True
            break
        elif event == 'Cancel':
            res = False
            break
    window.close()
    return res


class BtnInfo:
    def __init__(self, state=True):
        self.state = state  # Can have 3 states - True, False, None (toggle)





def draw_canvas(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


class DynamicGraph:
    def __init__(self, agent, par_shorts=[], available_pars=[]):

        self.agent = agent
        self.available_pars = available_pars
        self.par_shorts = par_shorts
        self.dt = self.agent.model.dt
        self.init_dur = 20
        self.window_size=(1550, 1200)
        self.canvas_size=(self.window_size[0]-50, self.window_size[1]-50)
        self.my_dpi=96
        self.figsize=(int(self.canvas_size[0]/self.my_dpi), int(self.canvas_size[1]/self.my_dpi))

        Ncols=4
        par_lists=[list(a) for a in np.array_split(self.available_pars, Ncols)]
        par_layout = [[sg.Text('Choose parameters')],
                      [sg.Col([*[[sg.CB(p, key=f'k_{p}')] for p in par_lists[i]]]) for i in range(Ncols)],
                      [sg.Button('Ok', **button_kwargs), sg.Button('Cancel', **button_kwargs)]
                      ]

        graph_layout = [
            # [sg.Text(f'{self.agent.unique_id} : {self.par}', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Canvas(size=(1280, 1200), key='-CANVAS-')],
            [sg.Text('Time in seconds to display on screen')],
            [sg.Slider(range=(0.1, 60), default_value=self.init_dur, size=(40, 10), orientation='h',
                       key='-SLIDER-TIME-')],
            [sg.Button('Choose', **button_kwargs)]
        ]
        layout=[[sg.Column(par_layout, key='-COL1-'), sg.Column(graph_layout, visible=False, key='-COL2-')]]
        self.window = sg.Window(f'{self.agent.unique_id} Dynamic Graph', layout, finalize=True, location=(0, 0), size=self.window_size)
        self.canvas_elem = self.window.FindElement('-CANVAS-')
        self.canvas = self.canvas_elem.TKCanvas
        self.fig_agg = None

        self.update_pars()
        self.layout=1
    def evaluate(self):
        event, values = self.window.read(timeout=0)
        if event is None:
            self.window.close()
            return False
        elif event=='Choose' :
            self.window[f'-COL2-'].update(visible=False)
            self.window[f'-COL1-'].update(visible=True)
            self.layout = 1
        elif event=='Ok' :
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            pars=[p for p in self.available_pars if values[f'k_{p}']]
            self.par_shorts=par_db.loc[par_db['par'].isin(pars)].index.tolist()
            self.update_pars()
            self.layout = 2



        elif event=='Cancel' :
            self.window[f'-COL1-'].update(visible=False)
            self.window[f'-COL2-'].update(visible=True)
            self.layout = 2

        if self.layout==2 and self.Npars>0:
            secs = values['-SLIDER-TIME-']
            Nticks = int(secs / self.dt)  # draw this many data points (on next line)
            t = self.agent.model.Nticks * self.dt
            trange = np.linspace(t - secs, t, Nticks)
            ys = self.update(Nticks)
            for ax, y in zip(self.axs, ys):
                ax.lines.pop(0)
                ax.plot(trange, y, color='black')
            self.axs[-1].set_xlim(np.min(trange), np.max(trange))
            self.fig_agg.draw()
        return True

    def update(self, Nticks):
        y_nan = np.ones(Nticks) * np.nan
        ys = []
        for p, v in self.yranges.items():
            self.yranges[p] = np.append(v, getattr(self.agent, p))
            dif = self.yranges[p].shape[0] - Nticks
            if dif >= 0:
                y = self.yranges[p][-Nticks:]
            else:
                y = y_nan
                y[-dif:] = self.yranges[p]
                # y = np.pad(self.yranges[p], (-dif, 0), constant_values=np.nan)
            ys.append(y)
        return ys

    def update_pars(self):
        self.pars, self.sim_symbols, self.exp_symbols, self.units, self.ylims, self.par_collects = [
            par_db[['par', 'symbol', 'exp_symbol', 'unit', 'lim', 'collect']].loc[self.par_shorts].values[:, k].tolist() for k in
            range(6)]

        self.Npars = len(self.pars)
        self.yranges = {}

        self.fig, axs = plt.subplots(self.Npars, 1, figsize=self.figsize, dpi=self.my_dpi, sharex=True)
        if self.Npars > 1:
            self.axs = axs.ravel()
        else:
            self.axs = [axs]
        Nticks = int(self.init_dur / self.dt)
        for i, (ax, p, l, u, lim, p_col) in enumerate(zip(self.axs, self.pars, self.sim_symbols, self.units, self.ylims, self.par_collects)):
            if hasattr(self.agent, p_col) :
                p0=p_col
            else :
                p0=p
            self.yranges[p0] = np.ones(Nticks) * np.nan
            ax.grid()
            ax.plot(range(Nticks), self.yranges[p0], color='black', label=l)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.legend(loc='upper right')
            ax.set_ylabel(u, fontsize=10)
            if lim is not None :
                ax.set_ylim(lim)
            ax.tick_params(axis='y', which='major', labelsize=10)
            if i == self.Npars - 1:
                ax.set_xlabel('time, $sec$')
                ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
        self.fig.subplots_adjust(bottom=0.15, top=0.95, left=0.1, right=0.99, wspace=0.01, hspace=0.05)
        if self.fig_agg:
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_canvas(self.canvas, self.fig)


