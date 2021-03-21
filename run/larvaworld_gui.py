#!/usr/bin/env python
# !/usr/bin/env python
import ast
import copy
from ast import literal_eval

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import matplotlib
import inspect
import sys

from tkinter import *
import pyscreenshot as ImageGrab

from lib.aux.collecting import effector_collection
from lib.conf import exp_types, default_sim, growing_rover, mock_larva, box2d_space, larva_place_modes, \
    food_place_modes, arena_shapes, pref_exp_np
from lib.model.envs._food import Food
from lib.model.larva._larva import Larva
from lib.sim.gui_lib import gui_table, retrieve_value
from lib.sim.single_run import generate_config, run_sim, next_idx, init_sim

sys.path.insert(0, '..')
from lib.anal.plotting import *
from lib.stor.larva_dataset import LarvaDataset
from lib.stor.paths import SingleRunFolder, RefFolder, get_parent_dir

matplotlib.use('TkAgg')


# Class holding the button graphic info. At this time only the state is kept
class BtnInfo:
    def __init__(self, state=True):
        self.state = state  # Can have 3 states - True, False, None (toggle)


on_image = b'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'

off_image = b'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'

off_image_disabled = b'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'
on_image_disabled = b'iVBORw0KGgoAAAANSUhEUgAAAFoAAAAnCAYAAACPFF8dAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsIAAA7CARUoSoAAAAVLSURBVGhD7Zu7TytHFMbHyxvsJeb9kjCiCihRpJAobaqblFGUJmXqKMofkyZpaKhS5HaJIgUpRZQmoqAhSgEUSBZgsAE/uJhX5ht80LmH8e56fdde7PuTPu0yj/XMt7Nnd2aXxPr6uuKMjIx84LruR47jJGtJbeeVplQqOaOjo+8MDAwk7u7uyrWsWIF2FYvFf3Rbt/HnQ+oDj0Ynk8kPl5eXf9Amf6L7pW5vb9X9/b3Jaye5XE719fWpubk51dPTY/bjijba+KbN3t7d3f324uLir1rWg9HpdPrFysrKy0KhMJTNZtX19XUtu/0sLi6qyclJlUqlcLWpRCJRy4knNzc3ShusKpXKq52dnS/z+fyvSE9sbGxMrq2t/Xd8fJw+PDw0hf1oRWdxNY2Pj6tMJqMmJiZUf3//Y3ocrjQJPOG+nJ2dYWSXt7a23tMRYt+Zn5//rlqteppMB5EHi5rZ2VmEtEeTAUzGJRo3yZOv7ydo94j293v8ndjW6JDxvh7RpoBEGtsKo9FofdNTq6urampqSvX29tZynhcIIUdHR//qUb3iDA4OZnDzs0Gm0khulQCMBs/VZIC2Dw8Pv6v71OvoO7lri3nUYb5tlToRp7Z9Deos37ZanYbVaA7vON/qCU1k6kQC94oMhxFk+FuCU9doPnptkPFRqBN5YjTvKO1LE3iZtwSjMwNiDGnYaD6aEa/1czieFdXQ0JB1wQfPw5C8Cii9Wwg9omHw2NiYmSLDaCz4YNoJ8ScHpGNBCGU4SIe6hVBGY+0BBmOiUy6XzQIKpptY9cOohrESjHg+y+u2ON+w0TAXpgGYfHl5aZYGq9WqMRsLLDDbNnXGyelWQsVoisUwl4OTQGvZPF5TOsxHyOlGQsdogNEroTQZGkqlktkiLnfq7M+LpnpsM4zS5EIVXvFUKhVzAmC2zH+OoA/1JGnYaByEwoN8PONhBXFbgngOw1GvnaNamhJWjdBwb2EmDAP0/EwvTV3XNQbiRNDJ4KBxuIGGQXayGXlhKx9WnFDDCjdBGEZhIJ1Om+dnmI2RXCwWayWfgrpXV1e1v4IhG10P2dEwCoKtnpQkVOgAGNX5fN7c5LCP+IvHOzxT85sk0uUoxt+oh7ygyI7Y5IetTlSSNBUoYSheg8E4mCYf9wDy5asyqlfvFZrE1pFGhd+0pYdRPbzKPTGaF6B9WVEeJGro95uRH7Y6jcqLuiOaKvIDyP2oFBRb3bDywlbeT5LAocPvQFEif5sUBFu9RuVHkDq+RvOK/ECIeW8y7nHZsJULIj9sdRpVEKxGU2W+lftRywtb+bDywlY+qCTGaLkuAagw39pGcBSjWoJJkFe+hJdtRn7Y6kBAznwdZPCVNg5V4gegfS4KI29KgB4VMWVHo7nZtjpcvG1hZTuulK0eID/RdpQDjn7+PcfMrh5UGciDRiVA69w03UfjMdVHw9EB5EUp/IaXbHXQdrwUQTsB2q5nwZc6/T6xubn5WyaT+Wxvb08VCgVTwAtbmIkCNHpmZkYtLCyY76P5iwQ6GXGE/MHMFzPlg4ODP/f39z91Tk9Pfzw/P1dLS0tqenra10h0shUC+JQYbTs5OXltfQRtjKvQdhhMyuVyP5k244t/PXJ+0aPmCywM4dLEohAuD1S0QUa0ApiMD9LxMTrCB1SvXe0GnuHegi1M1m3/I5vNvtBZd8Zo3fCkNvvnZDL5OV41Ic7EqTM48RjReOdo+3QhLmAAwmis4ejQ8bu+Ir/SaWYpk/9XViKVSn3tuu43ujMf67t8975JDYk29UrfAP/WA2NdawNJDzlK/Q9RjPZ1HEiBtwAAAABJRU5ErkJggg=='

food_pars = {
    'unique_id': str,
    'pos': tuple,
    'radius': float,
    'amount': float,
    'odor_id': str,
    'odor_intensity': float,
    'odor_spread': float
}

larva_pars = {
    'unique_id': str,
}


def draw_canvas(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


# def save_canvas(canvas, filepath):
#     r = Tk()
#     x2=r.winfo_rootx()+canvas.winfo_x()
#     y2=r.winfo_rooty()+canvas.winfo_y()
#     x1=x2+canvas.winfo_width()
#     y1=y2+canvas.winfo_height()
#     ImageGrab.grab().crop((x2,y2,x1,y1)).save(filepath)

def save_plot(fig, save_to, save_as):
    if fig is not None:
        layout = [
            [sg.Text('Filename', size=(10, 1)), sg.In(default_text=save_as, k='SAVE_AS', size=(80, 1))],
            [sg.Text('Directory', size=(10, 1)), sg.In(save_to, k='SAVE_TO', size=(80, 1)),
             sg.FolderBrowse(initial_folder=get_parent_dir(), key='SAVE_TO', change_submits=True)],
            [sg.Ok(), sg.Cancel()]]

        event, values = sg.Window('Save figure', layout).read(close=True)
        if event == 'Ok':
            save_as = values['SAVE_AS']
            save_to = values['SAVE_TO']
            filepath = os.path.join(save_to, save_as)
            fig.savefig(filepath, dpi=300)
            # save_canvas(window['GRAPH_CANVAS'].TKCanvas, filepath)
            # figure_agg.print_figure(filepath)
            print(f'Plot saved as {save_as}')


def set_agent_kwargs(agent, pars, title='Agent args'):
    layout = []
    for i, (p, t) in enumerate(pars.items()):
        layout.append([sg.Text(p, size=(20, 1)), sg.Input(default_text=getattr(agent, p), key=f'kw_{p}', size=(20, 1))])
    layout.append([sg.Ok(), sg.Cancel()])
    event, values = sg.Window(title, layout).read(close=True)
    if event == 'Ok':
        for i, (p, t) in enumerate(pars.items()):
            v = values[f'kw_{p}']
            # print(v, type(v))
            if p == 'unique_id':
                agent.set_id(str(v))
            else:
                setattr(agent, p, retrieve_value(v, t))
    return agent


def set_kwargs(kwargs, title='Arguments'):
    if kwargs != {}:
        layout = []
        for i, (k, v) in enumerate(kwargs.items()):
            if not type(v) == dict and not type(v) == np.ndarray:
                layout.append([sg.Text(k, size=(20, 1)), sg.Input(default_text=str(v), k=f'KW_{i}', size=(20, 1))])
        layout.append([sg.Ok(), sg.Cancel()])
        event, values = sg.Window(title, layout).read(close=True)
        if event == 'Ok':
            for i, (k, v) in enumerate(kwargs.items()):
                if type(v) == np.ndarray:
                    continue
                if not type(v) == dict:
                    vv = values[f'KW_{i}']
                    if type(v) == bool:
                        if vv == 'False':
                            vv = False
                        elif vv == 'True':
                            vv = True
                    elif type(v) == list or type(v) == tuple:
                        vv = literal_eval(vv)

                    elif v is None:
                        if vv == 'None':
                            vv = None
                        else:
                            vv = vv
                    else:
                        vv = type(v)(vv)
                    kwargs[k] = vv
                else:
                    kwargs[k] = set_kwargs(v, title=k)

    return kwargs

def get_agent_list(agents, pars) :
    data = []
    for f in agents:
        dic = {}
        for p in pars:
            dic[p] = getattr(f, p)
        data.append(dic)
    return data


def get_graph_kwargs(func):
    signature = inspect.getfullargspec(func)
    kwargs = dict(zip(signature.args[-len(signature.defaults):], signature.defaults))
    for k in ['datasets', 'labels', 'save_to', 'save_as', 'return_fig', 'deb_dicts']:
        if k in kwargs.keys():
            del kwargs[k]
    return kwargs


def update_data_list(window, data):
    window.Element('DATASET_IDS').Update(values=list(data.keys()))


def change_dataset_id(window, values, data):
    if len(values['DATASET_IDS']) > 0:
        old_id = values['DATASET_IDS'][0]
        l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(k='NEW_ID', size=(10, 1))],
             [sg.Ok(), sg.Cancel()]]
        e, v = sg.Window('Change dataset ID', l).read(close=True)
        if e == 'Ok':
            data[v['NEW_ID']] = data.pop(old_id)
            update_data_list(window, data)
    return data


def draw_figure(window, func, func_kwargs, data, figure_agg):
    if func is not None and len(list(data.keys())) > 0:
        if figure_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(figure_agg)
        try:
            fig, save_to, save_as = func(datasets=list(data.values()), labels=list(data.keys()),
                                         return_fig=True, **func_kwargs)  # call function to get the figure
            figure_agg = draw_canvas(window['GRAPH_CANVAS'].TKCanvas, fig)  # draw the figure
        except:
            print('Plot not available for these datasets')
    return figure_agg, fig, save_to, save_as


def update_func(window, values, func, func_kwargs, graph_dict):
    if len(values['GRAPH_LIST']) > 0:
        choice = values['GRAPH_LIST'][0]
        if graph_dict[choice] != func:
            func = graph_dict[choice]
            func_kwargs = get_graph_kwargs(func)
        window['GRAPH_CODE'].update(inspect.getsource(func))
    return func, func_kwargs


def update_model(larva_model, window, collapsibles, sectiondicts):
    for name, dict in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                          [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                           larva_model['body_params']]):
        collapsibles[name].update(window, dict)
    module_dict = larva_model['neural_params']['modules']
    for k, v in module_dict.items():
        collapsibles[k.upper()].update(window, larva_model['neural_params'][f'{k}_params'])
    module_dict_upper = copy.deepcopy(module_dict)
    for k in list(module_dict_upper.keys()):
        module_dict_upper[k.upper()] = module_dict_upper.pop(k)
    collapsibles['BRAIN'].update(window, module_dict_upper)


def update_environment(env_params, window, collapsibles, sectiondicts, food_list):
    arena_params = env_params['arena_params']
    for k, v in arena_params.items():
        window.Element(k).Update(value=v)

    food_list = env_params['food_params']['food_list']
    place_params = env_params['place_params']
    update_placement(place_params, window, collapsibles, sectiondicts, food_list)
    return food_list




def update_placement(place_params, window, collapsibles, sectiondicts, food_list):
    window.Element('Nagents').Update(value=place_params['initial_num_flies'])
    window.Element('larva_place_mode').Update(value=place_params['initial_fly_positions']['mode'])
    window.Element('larva_positions').Update(value=place_params['initial_fly_positions']['loc'])
    update_food_placement(window, food_list, place_params=place_params)

def update_food_placement(window, food_list, place_params=None) :
    if len(food_list) > 0 :
        Nfood=len(food_list)
        food_place_mode=None
        food_positions=None
    else :
        if place_params is None :
            return
        Nfood = place_params['initial_num_food']
        food_place_mode = place_params['initial_food_positions']['mode']
        food_positions = place_params['initial_food_positions']['loc']
    window.Element('Nfood').Update(value=Nfood)
    window.Element('food_place_mode').Update(value=food_place_mode)
    window.Element('food_positions').Update(value=food_positions)


def init_model(larva_model, collapsibles={}, sectiondicts={}):
    # update_window_dict(model['sensorimotor_params'], window)
    # window = collapsibles['ENERGETICS'].update(model['energetics_params'], window)
    # window = collapsibles['BODY'].update(model['body_params'], window)

    for name, dict, kwargs in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                                  [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                                   larva_model['body_params']],
                                  [{}, {'toggle': True, 'disabled': True}, {}]):
        sectiondicts[name] = SectionDict(name, dict)
        collapsibles[name] = Collapsible(name, False, sectiondicts[name].init_section(), **kwargs)

    module_conf = []
    for k, v in larva_model['neural_params']['modules'].items():
        d = SectionDict(k, larva_model['neural_params'][f'{k}_params'])
        s = Collapsible(k.upper(), False, d.init_section(), toggle=v)
        collapsibles[s.name] = s
        sectiondicts[d.name] = d
        module_conf.append(s.get_section())
    collapsibles['BRAIN'] = Collapsible('BRAIN', False, module_conf)

    model_layout = [
        collapsibles['PHYSICS'].get_section(),
        collapsibles['BODY'].get_section(),
        collapsibles['ENERGETICS'].get_section(),
        collapsibles['BRAIN'].get_section()
    ]

    collapsibles['MODEL'] = Collapsible('MODEL', True, model_layout)

    return [collapsibles['MODEL'].get_section()]


def init_environment(env_params, collapsibles={}, sectiondicts={}):
    sectiondicts['ARENA'] = SectionDict('ARENA', env_params['arena_params'])
    collapsibles['ARENA'] = Collapsible('ARENA', False, sectiondicts['ARENA'].init_section())

    larva_place_conf = [
        [sg.Text('# larvae:', size=(12, 1)), sg.In(1, key='Nagents', **text_kwargs)],
        [sg.Text('placement:', size=(12, 1)),
         sg.Combo(larva_place_modes, key='larva_place_mode', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Text('positions:', size=(12, 1)), sg.In(None, key='larva_positions', **text_kwargs)],
        # [sg.Text('orientations:', size=(12, 1)), sg.In(None, key='larva_orientations', **text_kwargs)],
    ]

    # food_list_conf = []
    # for f in env_params['food_params']['food_list']:
    #     d = SectionDict(k, larva_model['neural_params'][f'{k}_params'])
    #     s = Collapsible(k.upper(), False, d.init_section(), toggle=v)
    #     collapsibles[s.name] = s
    #     sectiondicts[d.name] = d
    #     food_list_conf.append(s.get_section())
    # collapsibles['FOOD_LIST'] = Collapsible('FOOD_LIST', False, food_list_conf)

    food_place_conf = [
        [sg.Text('# food:', size=(12, 1)), sg.In(1, key='Nfood', **text_kwargs)],
        [sg.Text('placement:', size=(12, 1)),
         sg.Combo(food_place_modes, key='food_place_mode', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Text('positions:', size=(12, 1)), sg.In(None, key='food_positions', **text_kwargs)],
        [sg.Button('Food list', **button_kwargs)]

    ]

    odor_conf = [

    ]

    collapsibles['LARVA_PLACEMENT'] = Collapsible('LARVA_PLACEMENT', False, larva_place_conf)
    collapsibles['FOOD_PLACEMENT'] = Collapsible('FOOD_PLACEMENT', False, food_place_conf)
    collapsibles['ODORS'] = Collapsible('ODORS', False, odor_conf)

    env_layout = [
        collapsibles['ARENA'].get_section(),
        collapsibles['LARVA_PLACEMENT'].get_section(),
        collapsibles['FOOD_PLACEMENT'].get_section(),
        collapsibles['ODORS'].get_section()
    ]

    collapsibles['ENVIRONMENT'] = Collapsible('ENVIRONMENT', True, env_layout)

    return [collapsibles['ENVIRONMENT'].get_section()]


def update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list):
    if values['EXP'] != '':
        exp = values['EXP']
        exp_conf = copy.deepcopy(exp_types[exp])
        update_model(exp_conf['fly_params'], window, collapsibles, sectiondicts)

        food_list=update_environment(exp_conf['env_params'], window, collapsibles, sectiondicts, food_list)

        if 'sim_params' not in exp_conf.keys():
            exp_conf['sim_params'] = default_sim.copy()
        sim_params = exp_conf['sim_params']
        window.Element('sim_time_in_min').Update(value=sim_params['sim_time_in_min'])
        output_dict = {}
        for k in output_keys:
            # if k in sim_params['collect_effectors'] :
            #     output_dict[f'collect_{k}']=True
            # else :
            #     output_dict[f'collect_{k}'] = False
            if k in exp_conf['collect_effectors']:
                output_dict[k] = True
            else:
                output_dict[k] = False
        collapsibles['OUTPUT'].update(window, output_dict)
        sim_id = f'{exp}_{next_idx(exp)}'
        window.Element('sim_id').Update(value=sim_id)
        common_folder = f'single_runs/{exp}'
        window.Element('common_folder').Update(value=common_folder)
        return food_list


def collapse(layout, key, visible=True):
    """
    Helper function that creates a Column that can be later made hidden, thus appearing "collapsed"
    :param layout: The layout for the section
    :param key: Key used to make this seciton visible / invisible
    :return: A pinned column that can be placed directly into your layout
    :rtype: sg.pin
    """
    return sg.pin(sg.Column(layout, key=key, visible=visible))


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
            update_window_dict(window, dict)
        return window


# def collapsible_section(sec_name, state, section, toggle=None, disabled=False):
#     if state:
#         symbol = SYMBOL_DOWN
#     else:
#         symbol = SYMBOL_UP
#     header = [sg.T(symbol, enable_events=True, k=f'OPEN SEC {sec_name}', text_color='black'),
#               sg.T(sec_name, enable_events=True, text_color='black', k=f'SEC {sec_name} TEXT', **header_kwargs)]
#     if toggle is not None:
#         if disabled:
#             toggle_state = None
#             if toggle:
#                 image = on_image_disabled
#             elif not toggle:
#                 image = off_image_disabled
#         else:
#             toggle_state = toggle
#             if toggle:
#                 image = on_image
#             elif not toggle:
#                 image = off_image
#         header.append(sg.Button(image_data=image, k=f'TOGGLE_{sec_name}', border_width=0,
#                                 button_color=(
#                                     sg.theme_background_color(), sg.theme_background_color()),
#                                 disabled_button_color=(
#                                     sg.theme_background_color(), sg.theme_background_color()),
#                                 metadata=BtnInfo(state=toggle_state)))
#
#     l = [header,
#          [collapse(section, f'SEC {sec_name}', visible=state)]]
#     return l


def bool_button(name, state):
    if state:
        image = on_image
    elif not state:
        image = off_image
    elif state is None:
        image = off_image_disabled
    l = [sg.Text(f'{name} :', size=(12, 1)),
         sg.Button(image_data=image, k=f'TOGGLE_{name}', border_width=0,
                   button_color=(
                       sg.theme_background_color(), sg.theme_background_color()),
                   disabled_button_color=(
                       sg.theme_background_color(), sg.theme_background_color()),
                   metadata=BtnInfo(state=state))]
    return l


class SectionDict:
    def __init__(self, name, dict):
        self.init_dict = dict
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
                l.append([sg.Text(f'{k}:', size=(12, 1)), sg.In(v, key=k, **text_kwargs)])
            else:
                l.append(bool_button(k, v))
        return l

    def get_dict(self, values, window):
        new_dict = copy.deepcopy(self.init_dict)
        if new_dict is None:
            return new_dict
        for i, (k, v) in enumerate(new_dict.items()):
            if type(v) == bool:
                new_dict[k] = window[f'TOGGLE_{k}'].metadata.state
            else:
                vv = values[k]
                vv=retrieve_value(vv, type(v))
                new_dict[k] = vv
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


def update_window_dict(window, dict):
    if dict is not None:
        for k, v in dict.items():
            if type(v) != bool:
                window.Element(k).Update(value=v)
            else:
                window[f'TOGGLE_{k}'].metadata.state = v
                window[f'TOGGLE_{k}'].update(image_data=on_image if v else off_image)


def build_analysis_tab():
    fig, save_to, save_as, figure_agg = None, '', '', None
    func, func_kwargs = None, {}
    data = {}
    data_list = [
        [sg.Text('DATASETS', **header_kwargs)],
        [sg.Listbox(values=[], change_submits=False, size=(20, len(data.keys())), key='DATASET_IDS',
                    enable_events=True)],
        [sg.FolderBrowse(button_text='Add', initial_folder=SingleRunFolder, key='DATASET_DIR', change_submits=True,
                         enable_events=True, **button_kwargs)],
        [sg.Button('Remove', **button_kwargs), sg.Button('Add ref', **button_kwargs),
         sg.Button('Change ID', **button_kwargs)],
        # [sg.Text(' ' * 12)]
    ]

    dim = 2000
    figure_w, figure_h = dim, dim
    graph_dict = {
        'crawl_pars': plot_crawl_pars,
        'angular_pars': plot_ang_pars,
        'endpoint_params': plot_endpoint_params,
        'stride_Dbend': plot_stride_Dbend,
        'stride_Dorient': plot_stride_Dorient,
        'interference': plot_interference,
        'dispersion': plot_dispersion,
        'stridesNpauses': plot_stridesNpauses,
        'turn_duration': plot_turn_duration,
        'turns': plot_turns,
        'odor_concentration': plot_odor_concentration,
        'pathlength': plot_pathlength,
        'food_amount': plot_food_amount,
        'gut': plot_gut,
        'barplot': barplot,
        'deb': plot_debs,
    }
    graph_list = [
        [sg.Text('GRAPHS', **header_kwargs)],
        [sg.Listbox(values=list(graph_dict), change_submits=True, size=(20, len(list(graph_dict))), key='GRAPH_LIST')],
        [sg.Button('Graph args', **button_kwargs), sg.Button('Draw', **button_kwargs),
         sg.Button('Save', **button_kwargs)]]

    graph_code = sg.Col([[sg.MLine(size=(70, 30), key='GRAPH_CODE')]])
    graph_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='GRAPH_CANVAS')]])
    graph_instructions = sg.Col([[sg.Pane([graph_canvas, graph_code], size=(figure_w, figure_h))],
                                 [sg.Text('Grab square above and slide upwards to view source code for graph')]])

    analysis_layout = [
        [sg.Col(data_list)],
        [sg.Col(graph_list), graph_instructions]
    ]
    return analysis_layout, graph_dict, data, func, func_kwargs, fig, save_to, save_as, figure_agg


def build_simulation_tab():
    sim_datasets = []

    larva_model = copy.deepcopy(mock_larva)
    env_params = pref_exp_np
    food_list = env_params['food_params']['food_list']

    module_dict = larva_model['neural_params']['modules']
    module_keys = list(module_dict.keys())

    collapsibles = {}
    sectiondicts = {}

    exp_layout = [
        [sg.Text('Experiment:', size=(10, 1)),
         sg.Combo(list(exp_types.keys()), key='EXP', enable_events=True, readonly=True, font=('size', 10),
                  size=(15, 1))],
        [sg.Button('Load', **button_kwargs), sg.Button('Configure', **button_kwargs),
         sg.Button('Run', **button_kwargs)]
         ]

    output_keys = list(effector_collection.keys())
    output_dict = dict(zip(output_keys, [False] * len(output_keys)))
    # output_dict=dict(zip([f'collect_{k}' for k in output_keys], [False]*len(output_keys)))
    sectiondicts['OUTPUT'] = SectionDict('OUTPUT', output_dict)
    collapsibles['OUTPUT'] = Collapsible('OUTPUT', False, sectiondicts['OUTPUT'].init_section())

    sim_conf = [[sg.Text('Sim id:', size=(12, 1)), sg.In('unnamed_sim', key='sim_id', **text_kwargs)],
                [sg.Text('Path:', size=(12, 1)), sg.In('single_runs', key='common_folder', **text_kwargs)],
                [sg.Text('Duration (min):', size=(12, 1)), sg.In(3, key='sim_time_in_min', **text_kwargs)],
                [sg.Text('Timestep (sec):', size=(12, 1)), sg.In(0.1, key='dt', **text_kwargs)],
                bool_button('Box2D', False),
                collapsibles['OUTPUT'].get_section()
                ]

    collapsibles['CONFIGURATION'] = Collapsible('CONFIGURATION', False, sim_conf)

    conf_layout = [
        collapsibles['CONFIGURATION'].get_section()
    ]

    model_layout = init_model(larva_model, collapsibles, sectiondicts)

    env_layout = init_environment(env_params, collapsibles, sectiondicts)

    simulation_layout = [
        [sg.Col(exp_layout)],
        [sg.Col(conf_layout)],
        [sg.Col(model_layout)],
        [sg.Col(env_layout)]
    ]
    return simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list


def build_model_tab():
    model_layout = []
    return model_layout


def eval_analysis(event, values, window, func, func_kwargs, data, figure_agg, fig, save_to, save_as, graph_dict):
    if event == 'DATASET_DIR':
        if values['DATASET_DIR'] != '':
            d = LarvaDataset(dir=values['DATASET_DIR'])
            data[d.id] = d
            update_data_list(window, data)

            # window['DATASET_DIR'] = ''
    elif event == 'Add ref':
        d = LarvaDataset(dir=RefFolder)
        data[d.id] = d
        window.Element('DATASET_IDS').Update(values=list(data.keys()))
    elif event == 'Remove':
        if len(values['DATASET_IDS']) > 0:
            id = values['DATASET_IDS'][0]
            data.pop(id, None)
            update_data_list(window, data)
    elif event == 'Change ID':
        data = change_dataset_id(window, values, data)
    elif event == 'Save':
        save_plot(fig, save_to, save_as)
    elif event == 'Graph args':
        func_kwargs = set_kwargs(func_kwargs, title='Graph arguments')
    elif event == 'Draw':
        figure_agg, fig, save_to, save_as = draw_figure(window, func, func_kwargs, data, figure_agg)
    func, func_kwargs = update_func(window, values, func, func_kwargs, graph_dict)
    # print(values['DATASET_DIR'], type(values['DATASET_DIR']))
    # print(window.FindElement('DATASET_DIR').values)
    return window, func, func_kwargs, data, figure_agg, fig, save_to, save_as


def eval_model(event, values, window):
    return window


def get_model(window, values, module_keys, sectiondicts, collapsibles, base_model):
    module_dict = dict(zip(module_keys, [window[f'TOGGLE_{k.upper()}'].metadata.state for k in module_keys]))
    base_model['neural_params']['modules'] = module_dict

    for name, pars in zip(['PHYSICS', 'ENERGETICS', 'BODY'],
                          ['sensorimotor_params', 'energetics_params', 'body_params']):
        if collapsibles[name].state is None:
            base_model[pars] = None
        else:
            base_model[pars] = sectiondicts[name].get_dict(values, window)
        # collapsibles[name].update(window,dict)

    # module_conf = []
    for k, v in module_dict.items():
        base_model['neural_params'][f'{k}_params'] = sectiondicts[k].get_dict(values, window)
        # collapsibles[k.upper()].update(window,larva_model['neural_params'][f'{k}_params'])
    return base_model


def get_environment(window, values, module_keys, sectiondicts, collapsibles, base_environment, food_list):
    base_environment['place_params']['initial_num_flies'] = int(values['Nagents'])
    base_environment['place_params']['initial_fly_positions']['mode'] = str(values['larva_place_mode'])
    # larva_loc=values['larva_positions']
    #
    # base_environment['place_params']['initial_fly_positions']['loc'] = np.array(literal_eval(larva_loc))

    base_environment['place_params']['initial_num_food'] = int(values['Nfood'])
    base_environment['place_params']['initial_food_positions']['mode'] = str(values['food_place_mode'])

    base_environment['food_params']['food_list'] = food_list
    # food_loc = values['food_positions']
    # if food_loc is None or food_loc=='' :
    #     food_loc=None
    # else:
    #     food_loc= np.array(literal_eval(food_loc))
    # base_environment['place_params']['initial_food_positions']['loc'] =food_loc

    if window['TOGGLE_Box2D'].metadata.state:
        base_environment['space_params'] = box2d_space
    base_environment['arena_params'] = sectiondicts['ARENA'].get_dict(values, window)
    return base_environment


def get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list):
    exp = values['EXP']
    exp_conf = copy.deepcopy(exp_types[exp])

    sim_params = copy.deepcopy(default_sim)
    sim_params['sim_time_in_min'] = float(values['sim_time_in_min'])
    sim_params['dt'] = float(values['dt'])

    sim_params['collect_effectors'] = [k for k in output_keys if window[f'TOGGLE_{k}'].metadata.state == True]
    # sim_params['collect_effectors'] = dict(zip(output_keys, [window[f'TOGGLE_{f"collect_{k}"}'].metadata.state for k in output_keys]))

    env_params = get_environment(window, values, module_keys, sectiondicts, collapsibles, exp_conf['env_params'],
                                 food_list)
    # print(env_params['place_params'])

    fly_params = get_model(window, values, module_keys, sectiondicts, collapsibles, exp_conf['fly_params'])

    sim_config = {'sim_id': str(values['sim_id']),
                  'common_folder': str(values['common_folder']),
                  'enrich': True,
                  'experiment': exp,
                  'sim_params': sim_params,
                  'env_params': env_params,
                  'fly_params': fly_params,
                  }
    return sim_config


def eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys,
                    food_list):
    if event.startswith('OPEN SEC'):
        sec_name = event.split()[-1]
        if collapsibles[sec_name].state is not None:
            collapsibles[sec_name].state = not collapsibles[sec_name].state
            window[event].update(SYMBOL_DOWN if collapsibles[sec_name].state else SYMBOL_UP)
            window[f'SEC {sec_name}'].update(visible=collapsibles[sec_name].state)
    # elif event == 'EXP':
    elif event == 'Load':
        food_list=update_sim(window, values, collapsibles, sectiondicts, output_keys, food_list)
    elif 'TOGGLE' in event:
        if window[event].metadata.state is not None:
            window[event].metadata.state = not window[event].metadata.state
            window[event].update(image_data=on_image if window[event].metadata.state else off_image)
    elif event == 'Food list':
        food_list = gui_table(food_list, food_pars)
        update_food_placement(window, food_list, place_params=None)

    elif event == 'Configure':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list)
            env = init_sim(
                fly_params=sim_config['fly_params'],
                env_params=sim_config['env_params'])
            while env.is_running:
                if len(env.selected_agents) == 0:
                    env.step()
                    env.render()
                else:
                    env.render()
                    sel = env.selected_agents[0]
                    if isinstance(sel, Food):
                        sel = set_agent_kwargs(sel, food_pars)
                    elif isinstance(sel, Larva):
                        sel = set_agent_kwargs(sel, larva_pars)
                    sel.selected = False
                    env.selected_agents.remove(sel)
            food_list = get_agent_list(env.get_food(), food_pars)
            update_food_placement(window, food_list, place_params=None)


            # place_params = setup_sim(
            #     fly_params=sim_config['fly_params'],
            #     env_params=sim_config['env_params'])
            # update_placement(place_params, window, collapsibles, sectiondicts)



    elif event == 'Run':
        if values['EXP'] != '':
            sim_config = get_sim_config(window, values, module_keys, sectiondicts, collapsibles, output_keys, food_list)
            vis_kwargs = {'mode': 'video'}
            d = run_sim(**sim_config, **vis_kwargs)
            if d is not None:
                sim_datasets.append(d)
    return food_list

# -------------------------------- GUI Starts Here -------------------------------#
# fig = your figure you want to display.  Assumption is that 'fig' holds the      #
#       information to display.                                                   #
# --------------------------------------------------------------------------------#
sg.theme('LightGreen')

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'

button_kwargs = {'font': ('size', 6),
                 'size': (7, 1)
                 }
header_kwargs = {'font': ('size', 10),
                 'size': (15, 1)}

text_kwargs = {'font': ('size', 10),
               'size': (15, 1)}


def run_gui():
    analysis_layout, graph_dict, data, func, func_kwargs, fig, save_to, save_as, figure_agg = build_analysis_tab()
    # fig, save_to, save_as, figure_agg = None, '', '', None
    # func, func_kwargs = None, {}
    # data = {}
    # data_list = [
    #     [sg.Text('DATASETS', **header_kwargs)],
    #     [sg.Listbox(values=[], change_submits=True, size=(20, len(data.keys())), key='DATASET_IDS')],
    #     [sg.FolderBrowse(button_text='Add', initial_folder=SingleRunFolder, key='DATASET_DIR', change_submits=True,
    #                      **button_kwargs)],
    #     [sg.Button('Remove', **button_kwargs), sg.Button('Add ref', **button_kwargs),
    #      sg.Button('Change ID', **button_kwargs)],
    #     # [sg.Text(' ' * 12)]
    # ]
    #
    # dim = 1000
    # figure_w, figure_h = dim, dim
    # graph_dict = {
    #     'crawl_pars': plot_crawl_pars,
    #     'angular_pars': plot_ang_pars,
    #     'endpoint_params': plot_endpoint_params,
    #     'stride_Dbend': plot_stride_Dbend,
    #     'stride_Dorient': plot_stride_Dorient,
    #     'interference': plot_interference,
    #     'dispersion': plot_dispersion,
    #     'stridesNpauses': plot_stridesNpauses,
    #     'turn_duration': plot_turn_duration,
    #     'turns': plot_turns,
    #     'pathlength': plot_pathlength,
    #     'food_amount': plot_food_amount,
    #     'gut': plot_gut,
    #     'barplot': barplot,
    #     'deb': plot_debs,
    # }
    # graph_list = [
    #     [sg.Text('GRAPHS', **header_kwargs)],
    #     [sg.Listbox(values=list(graph_dict), change_submits=True, size=(20, len(list(graph_dict))), key='GRAPH_LIST')],
    #     [sg.Button('Set args', **button_kwargs), sg.Button('Draw', **button_kwargs), sg.Button('Save', **button_kwargs)]]
    #
    # graph_code = sg.Col([[sg.MLine(size=(70, 30), key='GRAPH_CODE')]])
    # graph_canvas = sg.Col([[sg.Canvas(size=(figure_w, figure_h), key='GRAPH_CANVAS')]])
    # graph_instructions = sg.Col([[sg.Pane([graph_canvas, graph_code], size=(figure_w, figure_h))],
    #                              [sg.Text('Grab square above and slide upwards to view source code for graph')]])
    #
    # analysis_layout = [
    #     [sg.Col(data_list)],
    #     [sg.Col(graph_list), graph_instructions]
    # ]

    simulation_layout, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys, food_list = build_simulation_tab()
    model_layout = build_model_tab()

    layout = [
        [sg.TabGroup([[
            sg.Tab('Model', model_layout, background_color='darkseagreen', key='MODEL_TAB'),
            sg.Tab('Simulation', simulation_layout, background_color='darkseagreen', key='SIMULATION_TAB'),
            sg.Tab('Analysis', analysis_layout, background_color='darkseagreen', key='ANALYSIS_TAB')]],
            key='ACTIVE_TAB', tab_location='top', selected_title_color='purple')]
    ]

    window = sg.Window('Larvaworld gui', layout, resizable=True, finalize=True, size=(2000, 1200))

    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break
        tab = values['ACTIVE_TAB']
        if tab == 'ANALYSIS_TAB':
            window, func, func_kwargs, data, figure_agg, fig, save_to, save_as = eval_analysis(event, values, window,
                                                                                               func,
                                                                                               func_kwargs, data,
                                                                                               figure_agg, fig, save_to,
                                                                                               save_as, graph_dict)
        elif tab == 'MODEL_TAB':
            window = eval_model(event, values, window)
        elif tab == 'SIMULATION_TAB':

            food_list = eval_simulation(event, values, window, sim_datasets, collapsibles, module_keys, sectiondicts, output_keys,
                            food_list)
    window.close()

if __name__ == "__main__":
    run_gui()
