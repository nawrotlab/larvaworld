import copy
import PySimpleGUI as sg

from lib.gui.gui_lib import CollapsibleDict, Collapsible, save_gui_conf, delete_gui_conf, b12_kws, \
    b6_kws, CollapsibleTable, graphic_button, t10_kws, t12_kws, t18_kws, w_kws, default_run_window, col_kws, col_size
from lib.conf.conf import loadConfDict, loadConf
import lib.conf.dtype_dicts as dtypes


def init_model(collapsibles={}):
    l1=[]
    for name, kwargs in zip(['Physics', 'Energetics', 'Body', 'Odor'],
                                    [{}, {'toggle': True, 'disabled': True}, {}, {}]):
        n=name.lower()
        collapsibles[name] = CollapsibleDict(name, False, dict=dtypes.get_dict(n),type_dict=dtypes.get_dict_dtypes(n), **kwargs)
        l1.append(collapsibles[name].get_section())
    l1.append(collapsibles['odor_gains'].get_section())
    non_brain_layout = sg.Col(l1, **col_kws, size=col_size(0.25))

    l2 = []
    for k in dtypes.module_keys:
        s = CollapsibleDict(k.upper(), False, dict=dtypes.get_dict(k), type_dict=dtypes.get_dict_dtypes(k),
                            dict_name=k.upper(), toggle=True, auto_open=False, disp_name=k)
        collapsibles.update(s.get_subdicts())
        l2.append(s.get_section())
    collapsibles['Brain'] = Collapsible('Brain', True, l2)
    brain_layout = sg.Col([collapsibles['Brain'].get_section()],**col_kws, size=col_size(0.25))

    model_layout = [[brain_layout, non_brain_layout]]

    collapsibles['Model'] = Collapsible('Model', True, model_layout)
    return collapsibles['Model'].get_section()


def update_model(larva_model, window, collapsibles):
    for name in ['Physics', 'Energetics', 'Body', 'Odor']:
        collapsibles[name].update(window, larva_model[name.lower()])
    module_dict = larva_model['brain']['modules']
    for k, v in module_dict.items():
        dic = larva_model['brain'][f'{k}_params']
        if k == 'olfactor':
            if dic is not None:
                odor_gains = dic['odor_dict']
                dic.pop('odor_dict')
            else :
                odor_gains = {}
            collapsibles['odor_gains'].update_table(window, odor_gains)
        collapsibles[k.upper()].update(window, dic)
    module_dict_upper = copy.deepcopy(module_dict)
    for k in list(module_dict_upper.keys()):
        module_dict_upper[k.upper()] = module_dict_upper.pop(k)
    collapsibles['Brain'].update(window, module_dict_upper, use_prefix=False)


def get_model(window, values, collapsibles):
    module_dict = dict(zip(dtypes.module_keys, [window[f'TOGGLE_{k.upper()}'].get_state() for k in dtypes.module_keys]))
    model = {}
    model['brain'] = {}
    model['brain']['modules'] = module_dict

    for name in ['Physics', 'Energetics', 'Body', 'Odor']:
        model[name.lower()] = None  if collapsibles[name].state is None else collapsibles[name].get_dict(values, window)

    for k, v in module_dict.items():
        model['brain'][f'{k}_params'] = collapsibles[k.upper()].get_dict(values, window)
    if model['brain']['olfactor_params'] is not None:
        model['brain']['olfactor_params']['odor_dict'] = collapsibles['odor_gains'].dict
    model['brain']['nengo'] = False
    return copy.deepcopy(model)


def build_model_tab():
    dicts={}
    graph_lists={}
    collapsibles={}
    s1 = CollapsibleTable('odor_gains', True, headings=['id', 'mean', 'std'], dict={},
                          disp_name='Odor gains',type_dict=dtypes.get_dict_dtypes('odor_gain'))

    collapsibles.update(**s1.get_subdicts())

    l_mod0 = [sg.Col([
        [sg.Text('Larva model',**t10_kws),
         graphic_button('load', 'LOAD_MODEL', tooltip='Load a larva model to inspect its parameters.'),
         graphic_button('data_add', 'SAVE_MODEL', tooltip='Save a new larva model to use in simulations.'),
         graphic_button('data_remove', 'DELETE_MODEL', tooltip='Delete an existing larva model.')],
         [sg.Combo(list(loadConfDict('Model').keys()), key='MODEL_CONF', enable_events=True, readonly=True,
                   **t18_kws, tooltip='The currently loaded larva model.')],
    ])]

    l_mod1 = init_model(collapsibles)

    l_mod = [[sg.Col([l_mod0, l_mod1],vertical_alignment='t')]]
    return l_mod, collapsibles, graph_lists, dicts


def eval_model(event, values, window, collapsibles, dicts, graph_lists):
    if event == 'LOAD_MODEL':
        if values['MODEL_CONF'] != '':
            conf = loadConf(values['MODEL_CONF'], 'Model')
            update_model(conf, window, collapsibles)

    elif event == 'SAVE_MODEL':
        model = get_model(window, values, collapsibles)
        save_gui_conf(window, model, 'Model')

    elif event == 'DELETE_MODEL':
        delete_gui_conf(window, values, 'Model')

    return dicts, graph_lists

if __name__ == "__main__":
    sg.theme('LightGreen')
    n='model'
    l, c, g, d = build_model_tab()
    w = sg.Window(f'{n} gui', l, size=(1800, 1200), **w_kws, location=(300, 100))

    while True:
        e, v = w.read()
        if e in (None, 'Exit'):
            break
        default_run_window(w, e, v, c, g)
        d, g = eval_model(e,v,w, collapsibles=c, dicts=d, graph_lists=g)
    w.close()

