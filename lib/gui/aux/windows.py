import copy
import operator
import os

import PySimpleGUI as sg

import lib.aux.dictsNlists
from lib.aux import colsNstr as fun
from lib.conf.conf import loadConfDict, saveConf, loadConf
from lib.conf.dtypes import null_dict
from lib.gui.aux.functions import retrieve_value, t_kws, b6_kws, w_kws, col_size
from lib.gui.aux.buttons import named_bool_button


from lib.stor import paths


def get_table(v, pars_dict, Nagents):
    data = []
    for i in range(Nagents):
        dic = {}
        for j, (p, t) in enumerate(pars_dict.items()):
            dic[p] = retrieve_value(v[(i, p)], t)
        data.append(dic)
    return data


def table_window(data, pars_dict, title, return_layout=False):
    d = pars_dict
    t12_kws_c = {**t_kws(12),
                 'justification': 'center'}

    ps = list(d.keys())
    Nagents, Npars = len(data), len(ps)

    l0 = [sg.T(' ', **t_kws(2))] + [sg.T(p, k=p, enable_events=True, **t12_kws_c) for p in ps]
    l1 = [[sg.T(i + 1, **t_kws(2))] + [sg.In(data[i][p], k=(i, p), **t12_kws_c) if type(d[p]) != list else sg.Combo(
        d[p], default_value=data[i][p], k=(i, p), enable_events=True, readonly=True,
        **t_kws(12)) for p in ps] for i in range(Nagents)]
    l2 = [sg.B(ii, **b6_kws) for ii in ['Add', 'Remove', 'Ok', 'Cancel']]

    l = [l0, *l1, l2]

    if return_layout:
        return l

    w = sg.Window(title, l, default_element_size=(20, 1), element_padding=(1, 1),
                  return_keyboard_events=True, finalize=True, force_toplevel=True)
    w.close_destroys_window = True
    return Nagents, Npars, ps, w


def gui_table(data, pars_dict, title='Agent list', sortable=False):
    data0 = copy.deepcopy(data)
    """
        Another simple table created from Input Text Elements.  This demo adds the ability to "navigate" around the drawing using
        the arrow keys. The tab key works automatically, but the arrow keys are done in the code below.
    """

    sg.change_look_and_feel('Dark Brown 2')  # No excuse for gray windows
    # Show a "splash" mode message so the user doesn't give up waiting
    sg.popup_quick_message('Hang on for a moment, this will take a bit to create....', auto_close=True,
                           non_blocking=True)

    Nrows, Npars, ps, w = table_window(data, pars_dict, title)
    while True:
        e, v = w.read()
        if e in (None, 'Cancel'):
            w.close()
            return data0
        if e == 'Ok':
            data = get_table(v, pars_dict, Nrows)
            w.close()
            return data
        elem = w.find_element_with_focus()
        cell = elem.Key if elem and type(elem.Key) is tuple else (0, 0)
        r, c = cell
        try:
            if e.startswith('Down'):
                r = r + 1 * (r < Nrows - 1)
            elif e.startswith('Left'):
                c = c - 1 * (c > 0)
            elif e.startswith('Right'):
                c = c + 1 * (c < Npars - 1)
            elif e.startswith('Up'):
                r = r - 1 * (r > 0)
        except:
            pass
        if sortable and e in ps:  # Perform a sort if a column heading was clicked
            try:
                table = [[int(v[(r, c)]) for c in range(Npars)] for r in range(Nrows)]
                new_table = sorted(table, key=operator.itemgetter(ps.index(e)))
            except:
                sg.popup_error('Error in table', 'Your table must contain only ints if you wish to sort by column')
            else:
                for i in range(Nrows):
                    for j in range(Npars):
                        w[(i, j)].update(new_table[i][j])
                [w[c].update(font='Any 14') for c in ps]  # make all column headings be normal fonts
                w[e].update(font='Any 14 bold')  # bold the font that was clicked
        # if the current cell changed, set focus on new cell
        if cell != (r, c):
            cell = r, c
            w[cell].set_focus()  # set the focus on the element moved to
            w[cell].update(
                select=True)  # when setting focus, also highlight the data in the element so typing overwrites
        if e == 'Add':
            data = get_table(v, pars_dict, Nrows)
            try:
                new_row = data[r]
            except:
                new_row = {k: None for k in pars_dict.keys()}
            data.append(new_row)
            w.close()
            Nrows, Npars, ps, w = table_window(data, pars_dict, title)
        elif e == 'Remove':
            data = get_table(v, pars_dict, Nrows)
            data = [d for i, d in enumerate(data) if i != r]
            w.close()
            Nrows, Npars, ps, w = table_window(data, pars_dict, title)


def delete_objects_window(selected):
    ids = [sel.unique_id for sel in selected]
    title = 'Delete objects?'
    l = [
        [sg.T(title)],
        [sg.Listbox(default_values=ids, values=ids, size=(20, len(ids)), k='DELETE_OBJECTS', enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    w = sg.Window(title, l)
    while True:
        e, v = w.read()
        if e == 'Ok':
            res = True
            break
        elif e == 'Cancel':
            res = False
            break
    w.close()
    return res


def object_menu(selected, **kwargs):
    objects = ['', 'Larva', 'Food', 'Border']
    title = 'Select object_class mode'
    layout = [
        [sg.T(title)],
        [sg.Listbox(default_values=[selected], values=objects, change_submits=False, size=(20, len(objects)),
                    key='SELECTED_OBJECT',
                    enable_events=True)],
        [sg.Ok(), sg.Cancel()]]
    w = sg.Window(title, layout, **kwargs)
    while True:
        e, v = w.read()
        if e == 'Ok':
            sel = v['SELECTED_OBJECT'][0]
            break
        elif e in (None, 'Cancel'):
            sel = selected
            break
    w.close()
    return sel


def set_kwargs(dic, title='Arguments', type_dict=None, **kwargs):
    from lib.gui.tabs.gui import check_toggles
    from lib.gui.aux.elements import SectionDict
    sec_dict = SectionDict(name=title, dict=dic, type_dict=type_dict)
    l = sec_dict.layout
    l.append([sg.Ok(), sg.Cancel()])
    w = sg.Window(title, l, **w_kws, **kwargs)
    while True:
        e, v = w.read()
        if e == 'Ok':
            new_dic = sec_dict.get_dict(v, w)
            break
        elif e in ['Cancel', None]:
            new_dic = dic
            break
        check_toggles(w, e)
    w.close()
    del sec_dict
    return new_dic


def set_agent_kwargs(agent, **kwargs):
    class_name = type(agent).__name__
    type_dict = null_dict('agent', 'dtype')
    title = f'{class_name} args'
    dic = {}
    for p in list(type_dict.keys()):
        dic[p] = getattr(agent, p)
    new_dic = set_kwargs(dic, title, type_dict=type_dict, **kwargs)
    for p, v in new_dic.items():
        if p == 'unique_id':
            agent.set_id(v)
        else:
            setattr(agent, p, v)
    return agent


def save_conf_window(conf, conftype, disp=None):
    from lib.gui.aux.elements import NamedList
    if disp is None:
        disp = conftype
    temp = NamedList('save_conf', key=f'{disp}_ID',
                     choices=list(loadConfDict(conftype).keys()),
                     readonly=False, enable_events=False, header_kws={'text' : f'Store new {disp}'})
    l = [
        temp.get_layout(),
        [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window(f'{disp} configuration', l).read(close=True)
    if e == 'Ok':
        id = v[f'{disp}_ID']
        saveConf(conf, conftype, id)
        return id
    elif e == 'Cancel':
        return None


def import_window(datagroup_id,raw_dic):
    from lib.gui.tabs.gui import check_togglesNcollapsibles
    from lib.gui.aux.elements import CollapsibleDict
    g = loadConf(datagroup_id, 'Group')
    group_dir = f'{paths.path("DATA")}/{g["path"]}'
    raw_folder = f'{group_dir}/raw'
    proc_folder = f'{group_dir}/processed'

    M, E = 'Merge', 'Enumerate'
    E0 = f'{E}_id'
    proc_dir = {}
    N = len(raw_dic)
    raw_ids = list(raw_dic.keys())
    raw_dirs = list(raw_dic.values())
    temp=fun.remove_prefix(raw_dirs[0], f'{raw_folder}/')
    groupID0=fun.remove_suffix(temp, f'/{raw_ids[0]}')
    # raw_dirs0 = [fun.remove_prefix(dr, f'{raw_folder}/') for dr in raw_dirs]
    # raw_dirs0 = [fun.remove_suffix(dr, f'/{id}') for dr, id in zip(raw_dirs0, raw_ids)]
    # raw_dirs0 = fun.unique_list(raw_dirs0)
    # groupID0 = raw_dirs0[0] if len(raw_dirs0) == 1 else ''
    if N == 0:
        return proc_dir
    w_size = (1200, 800)
    h_kws = {'font': ('Helvetica', 8, 'bold'), 'justification': 'center', **t_kws(30)}
    l00 = [sg.T('Group ID :', **t_kws(8)), sg.In(default_text=groupID0, k='import_group_id', **t_kws(20)),
                   *named_bool_button(name=M, state=False, toggle_name=None),
                   *named_bool_button(name=E, state=False, toggle_name=None, disabled=False),
                  sg.Ok(), sg.Cancel()]

    l01 = [sg.Col([[sg.T('RAW DATASETS', **h_kws), sg.T(**t_kws(8)), sg.T('NEW DATASETS', **h_kws)],
        *[[sg.T(id, **t_kws(30)), sg.T('  -->  ', **t_kws(8)), sg.In(id, k=f'new_{id}', **t_kws(30))] for id in  raw_ids]],
        vertical_scroll_only=True, scrollable=True, expand_y=True, vertical_alignment='top',
        size=col_size(y_frac=0.4, win_size=w_size))]

    s1 = CollapsibleDict('build_conf', disp_name='Configuration',text_kws=t_kws(20), state=True)
    c = {}
    for s in [s1]:
        c.update(**s.get_subdicts())
    l = [l01, l00,s1.get_layout()]
    w = sg.Window('Build new datasets from raw files', l, size=w_size)
    while True:
        e, v = w.read()

        if e in (None, 'Exit', 'Cancel'):
            w.close()
            break
        else:
            gID = v['import_group_id']
            toggled = check_togglesNcollapsibles(w, e, v, c)
            merge = w[f'TOGGLE_{M}'].get_state()
            for i, (id, dir) in enumerate(raw_dic.items()):
                if i != 0:
                    w.Element(f'new_{id}').Update(visible=not merge)
            enum = w[f'TOGGLE_{E}'].get_state()
            if E in toggled:
                if not enum:
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f'new_{id}').Update(value=id)
                else:
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f'new_{id}').Update(value=f'{gID}_{i}')
            if e == 'Ok':
                conf = s1.get_dict(values=v, window=w)
                kws = {
                    'datagroup_id': datagroup_id,
                    'group_id': gID,
                    **conf}
                w.close()
                from lib.stor.managing import build_dataset
                targets = [f.replace(raw_folder, proc_folder) for f in raw_dirs]
                if not merge:
                    print(f'------ Building {N} discrete datasets ------')
                    for target, source_id, source in zip(targets, raw_ids, raw_dirs):
                        target_id = v[f'new_{source_id}']
                        if datagroup_id in ['Berni lab']:
                            target = f'{target}/{target_id}'
                            source_files=[os.path.join(source, n) for n in os.listdir(source) if
                                  n.startswith(source_id)]
                            dd = build_dataset(id=target_id, target_dir=target, source_files=source_files, **kws)
                        elif datagroup_id in ['Jovanic lab']:
                            target = f'{target}/{target_id}'
                            dd = build_dataset(id=target_id, target_dir=target, source_dir=source, **kws)
                        elif datagroup_id in ['Schleyer lab']:
                            target = target.replace(source_id, target_id)
                            dd = build_dataset(id=target_id, target_dir=target, source_dir=[source], **kws)
                        if dd is not None:
                            proc_dir[target_id] = dd
                        else:
                            del proc_dir[target_id]

                else:
                    print(f'------ Building a single merged dataset ------')
                    target_id0 = v[f'new_{raw_ids[0]}']

                    if datagroup_id in ['Berni lab']:
                        target0 = f'{targets[0]}/{target_id0}'
                        source_files = lib.aux.dictsNlists.flatten_list([[os.path.join(source, n) for n in os.listdir(source) if
                                                                          n.startswith(source_id)] for source_id, source in raw_dic.items()])
                        dd = build_dataset(id=target_id0, target_dir=target0, source_files=source_files, **kws)
                    elif datagroup_id in ['Schleyer lab']:
                        target0 = targets[0].replace(raw_ids[0], target_id0)
                        dd = build_dataset(id=target_id0, target_dir=target0, source_dir=raw_dirs, **kws)
                    elif datagroup_id in ['Jovanic lab']:
                        raise NotImplemented
                    proc_dir[dd.id] = dd
                break
    return proc_dir



def change_dataset_id(dic, old_ids):
    k = 'NEW_ID'
    for old_id in old_ids:
        l = [[sg.Text('Enter new dataset ID', size=(20, 1)), sg.In(default_text=old_id, k=k, size=(10, 1))],
             [sg.Button('Store'), sg.Ok(), sg.Cancel()]]
        e1, v1 = sg.Window('Change dataset ID', l).read(close=True)
        new_id = v1[k]
        if e1 == 'Ok':
            dic[new_id] = dic.pop(old_id)
        elif e1 == 'Store':
            d = dic[old_id]
            d.set_id(new_id)
            dic[new_id] = dic.pop(old_id)
    return dic

def larvagroup_window(**kwargs) :
    from lib.gui.aux.elements import CollapsibleDict
    from lib.gui.tabs.gui import check_togglesNcollapsibles
    k='LarvaGroup'
    # dic = null_dict(k)
    c0=CollapsibleDict(k, use_header=False, as_entry='Group ID', subdict_state=True, col_idx=[[0,1,2,3,4,7],[5],[6]])
    c=c0.get_subdicts()
    l=c0.get_layout(as_col=False)+[[sg.Ok(), sg.Cancel()]]
    kws=w_kws
    kws['default_element_size']=(16,1)
    w = sg.Window(k, l,size=col_size(0.8,0.5), **kws, **kwargs)
    while True:
        e, v = w.read()
        if e == 'Ok':
            dic = c0.get_dict(v, w)
            break
        elif e in ['Cancel', None]:
            dic = {}
            break
        check_togglesNcollapsibles(w, e, v, c)
    w.close()
    return dic

if __name__ == "__main__":
    dic=larvagroup_window()