import threading
import PySimpleGUI as sg
import matplotlib

from lib.gui.intro_tab import build_intro_tab, eval_intro_tab
from lib.gui.video_tab import build_video_tab, eval_video_tab
from lib.gui.batch_tab import build_batch_tab, eval_batch, get_batch
import lib.gui.gui_lib as gui
from lib.gui.model_tab import build_model_tab, eval_model
from lib.gui.simulation_tab import build_sim_tab, eval_sim
from lib.gui.analysis_tab import build_analysis_tab, eval_analysis

matplotlib.use('TkAgg')


class LarvaworldGui:

    def __init__(self, tabs=None):
        self.tab_dict = {'intro': 'introduction',
                         'model': 'larva models',
                         'exp': 'simulations',
                         'batch': 'batch runs',
                         'anal': 'analysis',
                         'video': 'videos'}

        if tabs is None:
            tabs = list(self.tab_dict.keys())
        # sg.change_look_and_feel('Dark Blue 3')
        sg.theme('LightGreen')
        self.background_color = None
        self.collapsibles = {}
        self.graph_lists = {}
        self.dicts = {
            'sim_results': {'datasets': []},
            'batch_kwargs': None,
            'batch_results': {},
            'analysis_data': {},
        }
        self.tabs = tabs
        c = {'layout': self.get_layout(), 'size': gui.window_size, 'location': (300, 100), **gui.w_kws}
        self.window = sg.Window('Larvaworld gui', **c)

    def run(self):
        while True:
            e, v = self.window.read()
            if e in (None, 'Exit'):
                break
            gui.default_run_window(self.window, e, v, self.collapsibles, self.graph_lists)

            tab = v['ACTIVE_TAB'].split()[0]
            k, kk = self.eval_tab(tab, event=e, values=v, window=self.window,
                                  collapsibles=self.collapsibles, dicts=self.dicts,
                                  graph_lists=self.graph_lists)

            # if dicts['batch_kwargs'] :
            #     thread = threading.Thread(target=batch_thread, args=(dicts['batch_kwargs'], W, dicts),daemon=True)
            #     thread.start()
            #     dicts['batch_kwargs'] = None
            #
            #
            # elif e == '-THREAD-':  # Thread has completed
            #     thread.join(timeout=0)
            #     # print('Thread finished')
            #     # sg.popup_animated(None)  # stop animination in case one is running
            #     thread = None  # reset variables for next run
            #     # thread, message, progress, timeout = None, '', 0, None  # reset variables for next run
            #     graph_lists['BATCH'].update(W, dicts['batch_results']['fig_dict'])
            # print(v)
        self.window.close()

    def build_tab(self, name):
        if name == 'model':
            return build_model_tab()
        elif name == 'exp':
            return build_sim_tab()
        elif name == 'batch':
            return build_batch_tab()
        elif name == 'anal':
            return build_analysis_tab()
        elif name == 'video':
            return build_video_tab()
        elif name == 'intro':
            return build_intro_tab()

    def eval_tab(self, name, **kwargs):
        if name == 'model':
            return eval_model(**kwargs)
        elif name == 'exp':
            return eval_sim(**kwargs)
        elif name == 'batch':
            return eval_batch(**kwargs)
        elif name == 'anal':
            return eval_analysis(**kwargs)
        elif name == 'video':
            return eval_video_tab(**kwargs)
        elif name == 'intro':
            return eval_intro_tab(**kwargs)

    def get_layout(self):
        ts = []
        for n in self.tabs:
            l, c, g, d = self.build_tab(n)
            self.collapsibles.update(c)
            self.graph_lists.update(g)
            self.dicts.update(d)
            ts.append(sg.Tab(self.tab_dict[n], l, background_color=self.background_color, key=f'{n} TAB', ))
        return [[sg.TabGroup([ts], key='ACTIVE_TAB', tab_location='topleft', selected_title_color='darkblue',
                             font=("Helvetica", 13, "normal"),
                             title_color='grey', selected_background_color=None,
                             tab_background_color='lightgrey', background_color=None)]]
    # def batch_thread(kwargs, window, dicts):
    #     """
    #     A worker thread that communicates with the GUI through a global message variable
    #     This thread can block for as long as it wants and the GUI will not be affected
    #     :param seconds: (int) How long to sleep, the ultimate blocking call
    #     """
    #     # progress = 0
    #     # print('Thread started - will sleep for {} seconds'.format(seconds))
    #     # for i in range(int(seconds * 10)):
    #     #     time.sleep(.1)  # sleep for a while
    #     #     progress += 100 / (seconds * 10)
    #     #     window.write_event_value('-PROGRESS-', progress)
    #
    #     df, fig_dict = batch_run(**kwargs)
    #     df_ax, df_fig = render_mpl_table(df)
    #     fig_dict['dataframe'] = df_fig
    #     dicts['batch_results']['df'] = df
    #     dicts['batch_results']['fig_dict'] = fig_dict
    #
    #     window.write_event_value('-THREAD-', '*** The thread says.... "I am finished" ***')


