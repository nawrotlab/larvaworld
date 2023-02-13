import PySimpleGUI as sg

from larvaworld.lib import reg, sim
from larvaworld.gui import gui_aux



class ImportTab(gui_aux.GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'
        self.Ctrac, self.Cenr = 'purple', 'cyan'

    def update(self, w, c, conf, id=None):
        path = conf['path']
        w[f'BROWSE {self.raw_key}'].InitialFolder = f'{reg.DATA_DIR}/{path}/raw'
        w[f'BROWSE {self.proc_key}'].InitialFolder = f'{reg.DATA_DIR}/{path}/processed'
        for n in ['Tracker', 'enrichment']:
            c[n].update(w, conf[n])

    def build(self):
        x, y = 0.17, 0.3
        x1=0.35
        kR, kP = self.raw_key, self.proc_key
        d = {kR: {}, kP: {}}
        sl1 = gui_aux.SelectionList(tab=self, disp='Data format/lab', buttons=['load'])
        dl1 = gui_aux.DataList(kR, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'change_ID', 'browse'],
                               raw=True, size=(50, 10))
        dl2 = gui_aux.DataList(kP, tab=self, dict=d[kP],
                               buttons=['replay', 'imitate', 'enrich', 'select_all', 'remove', 'change_ID', 'save_ref','browse'],
                               aux_cols=['N', 'duration', 'quality'], size=(55, 9))

        pd2 = gui_aux.PadDict('enrichment', background_color=self.Cenr, header_width=125,
                              subconfs={'preprocessing': {'text_kws': gui_aux.t_kws(14)},
                               'processing': {'Ncols': 2, 'text_kws': gui_aux.t_kws(9)},
                               'annotation': {'Ncols': 2, 'text_kws': gui_aux.t_kws(9)},
                               'metric_definition': {'header_width': 60,'text_kws': gui_aux.t_kws(9)}}
                              )
        pd1 = gui_aux.PadDict('Tracker', background_color=self.Ctrac, text_kws=gui_aux.t_kws(8), header_width=22,
                              subconfs={'filesystem': {'header_width': 20, 'value_kws': gui_aux.t_kws(5)},
                                       'resolution': {'header_width': 20, 'text_kws': gui_aux.t_kws(13)},
                                       'arena': {'header_width': 20, 'text_kws': gui_aux.t_kws(7)}}
                              )
        dd1 = gui_aux.gui_col([sl1, pd1], x_frac=x, as_pane=True, pad=(0, 0))
        dd2 = gui_aux.gui_row([dl1, dl2], x_frac=1 - x - x1, y_frac=y, x_fracs=[x1, 1 - x1 - x], as_pane=True, pad = None)
        dd3 = pd2.get_layout()
        dd5 = sg.Col([dd2, dd3[0]])
        g1 = gui_aux.ButtonGraphList(self.name, tab=self, fig_dict={})
        l = [[dd1, dd5]]
        c = {}
        for s in [pd1, pd2]:
            c.update(**s.get_subdicts())
        return l, c, {g1.name: g1}, d

    def eval(self, e, v, w, c, d, g):
        return d, g

    def imitate(self, conf):
        run = sim.ExpRun(parameters=conf)
        run.simulate()

        # for d in run.datasets:
        #     f = sim.ExpFitter(refID=d.config['sample'])
        #     fit = f.compare(d, save_to_config=True)
        #     print(d.id, fit)


if __name__ == "__main__":
    from larvaworld.gui.tabs.larvaworld_gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui.run()
