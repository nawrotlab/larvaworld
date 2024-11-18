import os

import PySimpleGUI as sg

from ...gui import gui_aux
from ...lib import reg, util

__all__ = [
    "table_window",
    "delete_objects_window",
    "object_menu",
    "set_kwargs",
    "set_agent_kwargs",
    "save_conf_window",
    "delete_conf_window",
    "add_ref_window",
    "save_ref_window",
    "import_window",
    "change_dataset_id",
    "entry_window",
]

__displayname__ = "GUI pop-up windows"


def table_window(data, pars_dict, title, return_layout=False):
    d = pars_dict
    t12_kws_c = {**gui_aux.t_kws(12), "justification": "center"}

    ps = list(d.keys())
    Nagents, Npars = len(data), len(ps)

    l0 = [sg.T(" ", **gui_aux.t_kws(2))] + [
        sg.T(p, k=p, enable_events=True, **t12_kws_c) for p in ps
    ]
    l1 = [
        [sg.T(i + 1, **gui_aux.t_kws(2))]
        + [
            sg.In(data[i][p], k=(i, p), **t12_kws_c)
            if type(d[p]) != list
            else sg.Combo(
                d[p],
                default_value=data[i][p],
                k=(i, p),
                enable_events=True,
                readonly=True,
                **gui_aux.t_kws(12),
            )
            for p in ps
        ]
        for i in range(Nagents)
    ]
    l2 = [sg.B(ii, **gui_aux.b6_kws) for ii in ["Add", "Remove", "Ok", "Cancel"]]

    l = [l0, *l1, l2]

    if return_layout:
        return l

    w = sg.Window(
        title,
        l,
        default_element_size=(20, 1),
        element_padding=(1, 1),
        return_keyboard_events=True,
        finalize=True,
        force_toplevel=True,
    )
    w.close_destroys_window = True
    return Nagents, Npars, ps, w


def delete_objects_window(selected):
    ids = [sel.unique_id for sel in selected]
    title = "Delete objects?"
    l = [
        [sg.T(title)],
        [
            sg.Listbox(
                default_values=ids,
                values=ids,
                size=(20, len(ids)),
                k="DELETE_OBJECTS",
                enable_events=True,
            )
        ],
        [sg.Ok(), sg.Cancel()],
    ]
    w = sg.Window(title, l)
    while True:
        e, v = w.read()
        if e == "Ok":
            res = True
            break
        elif e == "Cancel":
            res = False
            break
    w.close()
    return res


def object_menu(selected, **kwargs):
    objects = ["", "Larva", "Food", "Border"]
    title = "Select object_class mode"
    layout = [
        [sg.T(title)],
        [
            sg.Listbox(
                default_values=[selected],
                values=objects,
                change_submits=False,
                size=(20, len(objects)),
                key="SELECTED_OBJECT",
                enable_events=True,
            )
        ],
        [sg.Ok(), sg.Cancel()],
    ]
    w = sg.Window(title, layout, **kwargs)
    while True:
        e, v = w.read()
        if e == "Ok":
            sel = v["SELECTED_OBJECT"][0]
            break
        elif e in (None, "Cancel"):
            sel = selected
            break
    w.close()
    return sel


def set_kwargs(dic, title="Arguments", type_dict=None, **kwargs):
    from ...gui.tabs.larvaworld_gui import check_toggles
    from .elements import SectionDict

    sec_dict = SectionDict(name=title, dict=dic, type_dict=type_dict)
    l = sec_dict.layout
    l.append([sg.Ok(), sg.Cancel()])
    w = sg.Window(title, l, **gui_aux.w_kws, **kwargs)
    while True:
        e, v = w.read()
        if e == "Ok":
            new_dic = sec_dict.get_dict(v, w)
            break
        elif e in ["Cancel", None]:
            new_dic = dic
            break
        check_toggles(w, e)
    w.close()
    del sec_dict
    return new_dic


def set_agent_kwargs(agent, **kwargs):
    return agent
    # if isinstance(agent,)

    class_name = type(agent).__name__
    if class_name == "Food":
        k = "food"
    elif class_name == "Larva":
        k = "Model"
    else:
        return agent
    # type_dict = reg.par.get_null(k)
    title = f"{class_name} args"
    dic = {}
    for p in list(type_dict.keys()):
        dic[p] = getattr(agent, p)
    new_dic = set_kwargs(dic, title, type_dict=type_dict, **kwargs)
    for p, v in new_dic.items():
        if p == "unique_id":
            agent.set_id(v)
        else:
            setattr(agent, p, v)
    return agent


def save_conf_window(conf, conftype, disp=None):
    from .elements import NamedList

    if disp is None:
        disp = conftype
    temp = NamedList(
        "save_conf",
        key=f"{disp}_ID",
        choices=reg.conf[conftype].confIDs,
        readonly=False,
        enable_events=False,
        header_kws={"text": f"Store new {disp}"},
    )
    l = [temp.get_layout(), [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window(f"{disp} configuration", l).read(close=True)
    if e == "Ok":
        id = v[f"{disp}_ID"]
        reg.conf[conftype].setID(conf=conf, id=id)
        return id
    elif e == "Cancel":
        return None


def delete_conf_window(id, conftype, disp=None):
    if disp is None:
        disp = conftype
    l = [
        [
            sg.Col(
                [
                    [
                        sg.Text(
                            f"Are you sure you want to delete \n "
                            f"the {disp} configuration with ID : {id} ?",
                            size=(40, 3),
                        )
                    ],
                    [sg.Ok(), sg.Cancel()],
                ],
                justification="center",
                vertical_alignment="center",
                element_justification="center",
                pad=(20, 20),
            )
        ]
    ]
    e, v = sg.Window("Delete configuration", l, size=(500, 250)).read(close=True)
    if e == "Ok":
        # reg.stored.delete(id=id, conftype=conftype)
        reg.conf[conftype].delete(id)
        return True
    elif e == "Cancel":
        return False


def add_ref_window():
    from .elements import NamedList

    k = "ID"
    temp = NamedList(
        "Reference ID : ",
        key=k,
        choices=reg.conf.Ref.confIDs,
        size=(30, 10),
        select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
        drop_down=False,
        readonly=True,
        enable_events=False,
        header_kws={"text": "Select reference datasets"},
    )
    l = [temp.get_layout(), [sg.Ok(), sg.Cancel()]]
    e, v = sg.Window("Load reference datasets", l).read(close=True)
    if e == "Ok":
        return {id: reg.conf.Ref.loadRef(id=id) for id in v[k]}
    elif e == "Cancel":
        return None


def save_ref_window(d):
    k = "ID"
    l = [
        [
            sg.Text("Reference ID : ", size=(12, 1)),
            sg.In(default_text=f"{d.group_id}.{d.id}", k=k, size=(30, 1)),
        ],
        [sg.Ok(), sg.Cancel()],
    ]
    e, v = sg.Window("Store reference dataset", l).read(close=True)
    if e == "Ok":
        d.save_config(refID=v[k])


def import_window(labID, raw_dic):
    from ...gui.tabs.larvaworld_gui import check_togglesNcollapsibles
    from .elements import PadDict

    g = reg.conf.LabFormat.get(labID)
    # group_dir = g.path
    raw_folder = g.raw_folder
    proc_folder = g.processed_folder

    M, E = "Merge", "Enumerate"
    E0 = f"{E}_id"
    proc_dir = {}
    N = len(raw_dic)
    raw_ids = list(raw_dic.keys())
    raw_dirs = list(raw_dic.values())
    temp = util.remove_prefix(raw_dirs[0], f"{raw_folder}/")
    groupID0 = util.remove_suffix(temp, f"/{raw_ids[0]}")
    if N == 0:
        return proc_dir
    w_size = (1200, 800)
    h_kws = {
        "font": ("Helvetica", 8, "bold"),
        "justification": "center",
        **gui_aux.t_kws(30),
    }
    l00 = sg.Col(
        [
            [
                sg.T("Group ID :", **gui_aux.t_kws(8)),
                sg.In(default_text=groupID0, k="import_group_id", **gui_aux.t_kws(20)),
            ],
            [
                *gui_aux.named_bool_button(name=M, state=False, toggle_name=None),
                *gui_aux.named_bool_button(
                    name=E, state=False, toggle_name=None, disabled=False
                ),
            ],
            [sg.Ok(), sg.Cancel()],
        ]
    )

    l01 = [
        sg.Col(
            [
                [
                    sg.T("RAW DATASETS", **h_kws),
                    sg.T(**gui_aux.t_kws(8)),
                    sg.T("NEW DATASETS", **h_kws),
                ],
                *[
                    [
                        sg.T(id, **gui_aux.t_kws(30)),
                        sg.T("  -->  ", **gui_aux.t_kws(8)),
                        sg.In(id, k=f"new_{id}", **gui_aux.t_kws(30)),
                    ]
                    for id in raw_ids
                ],
            ],
            vertical_scroll_only=True,
            scrollable=True,
            expand_y=True,
            vertical_alignment="top",
            size=gui_aux.col_size(y_frac=0.4, win_size=w_size),
        )
    ]

    s1 = PadDict(
        "build_conf",
        disp_name="Configuration",
        text_kws=gui_aux.t_kws(20),
        header_width=30,
        background_color="purple",
    )

    l2 = [
        [
            sg.Frame(
                title="Options",
                layout=[[sg.Col(s1.layout), l00]],
                title_color="green",
                background_color="blue",
                border_width=8,
                pad=(40, 40),
                element_justification="center",
                title_location=sg.TITLE_LOCATION_TOP,
            )
        ]
    ]

    c = {}
    for s in [s1]:
        c.update(**s.get_subdicts())
    l = [l01, l2]
    w = sg.Window("Build new datasets from raw files", l, size=w_size)
    while True:
        e, v = w.read()

        if e in (None, "Exit", "Cancel"):
            w.close()
            break
        else:
            gID = v["import_group_id"]
            toggled = check_togglesNcollapsibles(w, e, v, c)
            merge = w[f"TOGGLE_{M}"].get_state()
            for i, (id, dir) in enumerate(raw_dic.items()):
                if i != 0:
                    w.Element(f"new_{id}").Update(visible=not merge)
            enum = w[f"TOGGLE_{E}"].get_state()
            if E in toggled:
                if not enum:
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f"new_{id}").Update(value=id)
                else:
                    for i, (id, dir) in enumerate(raw_dic.items()):
                        w.Element(f"new_{id}").Update(value=f"{gID}_{i}")
            if e == "Ok":
                conf = s1.get_dict(v, w)
                kws = {**conf}
                w.close()
                targets = [f.replace(raw_folder, proc_folder) for f in raw_dirs]
                if not merge:
                    print(f"------ Building {N} discrete datasets ------")
                    for target, source_id, source in zip(targets, raw_ids, raw_dirs):
                        target_id = v[f"new_{source_id}"]
                        if labID in ["Berni"]:
                            kws0 = {
                                "id": target_id,
                                "target_dir": f"{target}/{target_id}",
                                "source_files": [
                                    os.path.join(source, n)
                                    for n in os.listdir(source)
                                    if n.startswith(source_id)
                                ],
                                **kws,
                            }
                        elif labID in ["Jovanic"]:
                            kws0 = {
                                "id": target_id,
                                "target_dir": f"{target}/{target_id}",
                                "source_dir": source,
                                "source_id": source_id,
                                **kws,
                            }
                        elif labID in ["Schleyer"]:
                            kws0 = {
                                "id": target_id,
                                "target_dir": target.replace(source_id, target_id),
                                "source_dir": [source],
                                **kws,
                            }
                        elif labID in ["Arguello"]:
                            kws0 = {
                                "id": target_id,
                                "target_dir": f"{target}/{target_id}",
                                "source_files": [
                                    os.path.join(source, n)
                                    for n in os.listdir(source)
                                    if n.startswith(source_id)
                                ],
                                **kws,
                            }
                        dd = g.import_dataset(**kws0)

                        if dd is not None:
                            proc_dir[target_id] = dd
                        else:
                            del proc_dir[target_id]

                else:
                    print("------ Building a single merged dataset ------")
                    target_id0 = v[f"new_{raw_ids[0]}"]

                    if labID in ["Berni", "Arguello"]:
                        kws0 = {
                            "id": target_id0,
                            "target_dir": f"{targets[0]}/{target_id0}",
                            "source_files": util.flatten_list(
                                [
                                    [
                                        os.path.join(source, n)
                                        for n in os.listdir(source)
                                        if n.startswith(source_id)
                                    ]
                                    for source_id, source in raw_dic.items()
                                ]
                            ),
                            **kws,
                        }
                    elif labID in ["Schleyer"]:
                        kws0 = {
                            "id": target_id0,
                            "target_dir": targets[0].replace(raw_ids[0], target_id0),
                            "source_dir": raw_dirs,
                            **kws,
                        }
                    elif labID in ["Jovanic"]:
                        raise NotImplementedError
                    dd = g.import_dataset(**kws0)
                    proc_dir[dd.id] = dd
                break
    return proc_dir


def change_dataset_id(dic, old_ids):
    k = "NEW_ID"
    for old_id in old_ids:
        l = [
            [
                sg.Text("Enter new dataset ID", size=(20, 1)),
                sg.In(default_text=old_id, k=k, size=(10, 1)),
            ],
            [sg.Button("Store"), sg.Ok(), sg.Cancel()],
        ]
        e1, v1 = sg.Window("Change dataset ID", l).read(close=True)
        new_id = v1[k]
        if e1 == "Ok":
            dic[new_id] = dic.pop(old_id)
        elif e1 == "Store":
            d = dic[old_id]
            d.set_id(new_id)
            dic[new_id] = dic.pop(old_id)
    return dic


def entry_window(index, dict_name, base_dict={}, id=None, **kwargs):
    from .elements import PadDict

    c0 = PadDict(dict_name)
    l0 = c0.get_layout(pad=(20, 20))
    lID = [
        sg.T(f"{index}:", **gui_aux.t_kws(10), tooltip="The ID of the new entry"),
        sg.In(key="kID", **gui_aux.t_kws(30), pad=((0, 100), (0, 0))),
    ]
    l1 = sg.Pane([sg.Col([[*lID, sg.Ok(), sg.Cancel()]])], border_width=8, pad=(20, 20))
    l = [[l0], [l1]]
    kws = gui_aux.w_kws
    kws["default_element_size"] = (16, 1)
    w = sg.Window(
        gui_aux.get_disp_name(dict_name),
        l,
        size=gui_aux.col_size(0.8, 0.5),
        **kws,
        **kwargs,
    )
    if id is not None:
        c0.update(w, base_dict[id])
        w.Element("kID").Update(value=id)
    while True:
        e, v = w.read()
        if e == "Ok":
            dic = c0.get_dict(v, w)
            new_id = w.Element("kID").get()
            entry = {new_id: dic}
            if new_id in [None, ""]:
                sg.popup_no_buttons(
                    f"{index} not provided",
                    title="No ID!",
                    auto_close_duration=2,
                    auto_close=True,
                )
                continue
            elif new_id in base_dict.keys():
                choice, _ = sg.Window(
                    "Overwrite?",
                    [
                        [sg.T(f"{index} {new_id} already exists.", pad=(20, 20))],
                        [sg.T("Overwrite it? "), sg.Yes(), sg.No()],
                    ],
                    disable_close=True,
                ).read(close=True)
                if choice == "No":
                    continue
                else:
                    break
            else:
                break
        elif e in ["Cancel", None]:
            entry = {}
            break
    w.close()
    return entry
