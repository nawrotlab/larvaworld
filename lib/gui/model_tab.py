import copy

import PySimpleGUI as sg

from lib.conf.dtype_dicts import odor_gain_pars
from lib.conf import test_larva, module_keys
from lib.gui.gui_lib import CollapsibleDict, button_kwargs, Collapsible, text_kwargs, header_kwargs, set_agent_dict, \
    buttonM_kwargs, save_gui_conf, delete_gui_conf
from lib.conf.conf import loadConfDict, loadConf


def init_model(larva_model, collapsibles={}):
    for name, dict, kwargs in zip(['PHYSICS', 'ENERGETICS', 'BODY', 'ODOR'],
                                  [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                                   larva_model['body_params'], larva_model['odor_params']],
                                  [{}, {'toggle': True, 'disabled': True}, {}, {}]):
        collapsibles[name] = CollapsibleDict(name, True, dict=dict, type_dict=None, **kwargs)

    module_conf = []
    for k, v in larva_model['neural_params']['modules'].items():
        dic = larva_model['neural_params'][f'{k}_params']
        if k == 'olfactor':
            # odor_gains=dic['odor_dict']
            dic.pop('odor_dict')
        s = CollapsibleDict(k.upper(), False, dict=dic, dict_name=k.upper(), toggle=v)
        collapsibles.update(s.get_subdicts())
        module_conf.append(s.get_section())
    odor_gain_conf = [sg.Button('ODOR GAINS', **buttonM_kwargs)]
    # odor_gain_conf = [sg.Text('odor gains:', **text_kwargs), sg.Button('Odor gains', **button_kwargs)]
    module_conf.append(odor_gain_conf)
    collapsibles['BRAIN'] = Collapsible('BRAIN', True, module_conf)
    brain_layout = sg.Col([collapsibles['BRAIN'].get_section()])
    non_brain_layout = sg.Col([collapsibles['PHYSICS'].get_section(),
                               collapsibles['BODY'].get_section(),
                               collapsibles['ENERGETICS'].get_section(),
                               collapsibles['ODOR'].get_section()
                               ])

    model_layout = [[brain_layout, non_brain_layout]]

    collapsibles['MODEL'] = Collapsible('MODEL', False, model_layout)
    return collapsibles['MODEL'].get_section()


def update_model(larva_model, window, collapsibles):
    for name, dict in zip(['PHYSICS', 'ENERGETICS', 'BODY', 'ODOR'],
                          [larva_model['sensorimotor_params'], larva_model['energetics_params'],
                           larva_model['body_params'], larva_model['odor_params']]):
        collapsibles[name].update(window, dict)
    module_dict = larva_model['neural_params']['modules']
    odor_gains = {}
    for k, v in module_dict.items():
        dic = larva_model['neural_params'][f'{k}_params']
        if k == 'olfactor':
            if dic is not None:
                odor_gains = dic['odor_dict']
                dic.pop('odor_dict')
        collapsibles[k.upper()].update(window, dic)
    module_dict_upper = copy.deepcopy(module_dict)
    for k in list(module_dict_upper.keys()):
        module_dict_upper[k.upper()] = module_dict_upper.pop(k)
    collapsibles['BRAIN'].update(window, module_dict_upper, use_prefix=False)
    return odor_gains


def get_model(window, values, collapsibles, odor_gains):
    module_dict = dict(zip(module_keys, [window[f'TOGGLE_{k.upper()}'].metadata.state for k in module_keys]))
    model = {}
    model['neural_params'] = {}
    model['neural_params']['modules'] = module_dict

    for name, pars in zip(['PHYSICS', 'ENERGETICS', 'BODY', 'ODOR'],
                          ['sensorimotor_params', 'energetics_params', 'body_params', 'odor_params']):
        if collapsibles[name].state is None:
            model[pars] = None
        else:
            model[pars] = collapsibles[name].get_dict(values, window)
        # collapsibles[name].update(window,dict)

    for k, v in module_dict.items():
        model['neural_params'][f'{k}_params'] = collapsibles[k.upper()].get_dict(values, window)
        # collapsibles[k.upper()].update(window,larva_model['neural_params'][f'{k}_params'])
    if model['neural_params']['olfactor_params'] is not None:
        model['neural_params']['olfactor_params']['odor_dict'] = odor_gains
    model['neural_params']['nengo'] = False
    return copy.deepcopy(model)


def build_model_tab(collapsibles, dicts):
    larva_model = copy.deepcopy(test_larva)
    dicts['odor_gains'] = larva_model['neural_params']['olfactor_params']['odor_dict']
    # module_dict = larva_model['neural_params']['modules']

    l_mod0 = [sg.Col([
        [sg.Text('Larva model:', **header_kwargs),
         sg.Combo(list(loadConfDict('Model').keys()), key='MODEL_CONF', enable_events=True, readonly=True,
                  **text_kwargs)],
        [sg.Button('Load', key='LOAD_MODEL', **button_kwargs),
         sg.Button('Save', key='SAVE_MODEL', **button_kwargs),
         sg.Button('Delete', key='DELETE_MODEL', **button_kwargs)]
    ])]

    l_mod1 = init_model(larva_model, collapsibles)

    l_mod = [[sg.Col([l_mod0, l_mod1])]]
    return l_mod, collapsibles, dicts


def eval_model(event, values, window, collapsibles, dicts):
    if event == 'LOAD_MODEL':
        if values['MODEL_CONF'] != '':
            conf = loadConf(values['MODEL_CONF'], 'Model')
            dicts['odor_gains'] = update_model(conf, window, collapsibles)

    elif event == 'SAVE_MODEL':
        model = get_model(window, values, collapsibles, dicts['odor_gains'])
        save_gui_conf(window, model, 'Model')

    elif event == 'DELETE_MODEL':
        delete_gui_conf(window, values, 'Model')

    elif event == 'ODOR GAINS':
        dicts['odor_gains'] = set_agent_dict(dicts['odor_gains'], odor_gain_pars, title='Odor gains')

    return dicts
