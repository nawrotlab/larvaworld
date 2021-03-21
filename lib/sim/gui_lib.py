from ast import literal_eval

import PySimpleGUI as sg
from random import randint
import operator

def retrieve_value(v, type) :
    if v in ['', 'None', None]:
        vv = None
    elif v in ['sample', 'fit']:
        vv = v
    elif type=='bool' :
        if v in ['False', False]:
            vv = False
        elif v in ['True', True]:
            vv = True
    elif type == tuple or type == list:
        try:
            vv = literal_eval(v)
        except:
            vv= [float(x) for x in v.split()]
            if type == tuple :
                vv=tuple(vv)
    elif type(v) == type:
        vv=v
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


def build_table_window(data, pars_dict, title) :
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

def gui_table(data, pars_dict, title='Agent list') :
    """
        Another simple table created from Input Text Elements.  This demo adds the ability to "navigate" around the drawing using
        the arrow keys. The tab key works automatically, but the arrow keys are done in the code below.
    """

    sg.change_look_and_feel('Dark Brown 2')  # No excuse for gray windows
    # Show a "splash" type message so the user doesn't give up waiting
    sg.popup_quick_message('Hang on for a moment, this will take a bit to create....', auto_close=True, non_blocking=True)

    Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)

    current_cell = (0, 0)
    while True:  # Event Loop
        event, values = table_window.read()
        if event in (None, 'Cancel'):
            table_window.close()
            return data
            # break
        if event == 'Ok' :
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
        elif event in pars:         # Perform a sort if a column heading was clicked
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
                [table_window[c].update(font='Any 14') for c in pars]     # make all column headings be normal fonts
                table_window[event].update(font='Any 14 bold')                    # bold the font that was clicked
        # if the current cell changed, set focus on new cell
        if current_cell != (r, c):
            current_cell = r, c
            table_window[current_cell].set_focus()          # set the focus on the element moved to
            table_window[current_cell].update(select=True)  # when setting focus, also highlight the data in the element so typing overwrites
        if event == 'Add':
            data = get_table_data(values, pars_dict, Nagents)
            data.append(data[r])
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
        elif event == 'Remove':
            data = get_table_data(values, pars_dict, Nagents)
            data=[d for i,d in enumerate(data) if i!=r]
            table_window.close()
            Nagents, Npars, pars, table_window = build_table_window(data, pars_dict, title)
            # table_window.close()
            # gui_table(data, pars_dict, title='Agent list')


    # if clicked button to dump the table's values
        # if event.startswith('Show Table'):
        #     table = [[values[(row, col)] for col in range(Npars)] for row in range(Nagents)]
        #     sg.popup_scrolled('your_table = [ ', ',\n'.join([str(table[i]) for i in range(Nagents)]) + '  ]', title='Copy your data from here')


