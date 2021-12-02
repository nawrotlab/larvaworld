import PySimpleGUI as sg

from lib.gui.aux.elements import ButtonGraphList, CollapsibleDict, DataList, SelectionList, PadDict
from lib.gui.aux.functions import gui_cols, t_kws, gui_row, gui_col, col_size, gui_rowNcol, window_size
from lib.gui.tabs.tab import GuiTab
from lib.sim.single.single_run import SingleRun
from lib.conf.base import paths


class ImportTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'
        self.Ctrac, self.Cenr = 'purple', 'cyan'
        self.fields = ['tracker', 'enrichment']

    def update(self, w, c, conf, id=None):
        path = conf['path']
        w[f'BROWSE {self.raw_key}'].InitialFolder = f'{paths.path("DATA")}/{path}/raw'
        w[f'BROWSE {self.proc_key}'].InitialFolder = f'{paths.path("DATA")}/{path}/processed'
        for n in self.fields:
            c[n].update(w, conf[n])

    def build(self):
        x, y = 0.17, 0.3
        x1=0.35
        kR, kP = self.raw_key, self.proc_key
        d = {kR: {}, kP: {}}
        sl1 = SelectionList(tab=self, disp='Data format/lab', buttons=['load'])
        dl1 = DataList(kR, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'change_ID', 'browse'],
                       raw=True, size=(50, 10))
        dl2 = DataList(kP, tab=self, dict=d[kP],
                       buttons=['replay', 'imitate', 'enrich', 'select_all', 'remove', 'change_ID', 'save_ref','browse'],
                       aux_cols=['N', 'duration', 'quality'], size=(55, 9))
        pd1 = PadDict('tracker', background_color=self.Ctrac, text_kws=t_kws(8),header_width=22,
                     subconfs={'filesystem': {'header_width': 20, 'value_kws': t_kws(5)},
                               'resolution': {'header_width': 20, 'text_kws': t_kws(13)},
                               'arena': {'header_width': 20, 'text_kws': t_kws(7)}}
                     )
        pd2 = PadDict('enrichment', background_color=self.Cenr, header_width=125,
                     subconfs={'preprocessing': {'text_kws': t_kws(14)},
                               'to_drop': {'Ncols': 2, 'text_kws': t_kws(9)},
                               'processing': {'Ncols': 2, 'text_kws': t_kws(9)},
                               'metric_definition': {'header_width': 60,'text_kws': t_kws(9)}}
                     )
        dd1 = gui_col([sl1, pd1], x_frac=x, as_pane=True, pad=(0,0))
        dd2 = gui_row([dl1, dl2], x_frac=1 - x, y_frac=y, x_fracs=[x1, 1 - x1 - x], as_pane=True)
        dd3 = pd2.get_layout()
        dd5 = sg.Col([dd2, dd3[0]])
        g1 = ButtonGraphList(self.name, tab=self, fig_dict={})
        l = [[dd1, dd5]]
        c = {}
        for s in [pd1, pd2]:
            c.update(**s.get_subdicts())
        return l, c, {g1.name: g1}, d

    def eval(self, e, v, w, c, d, g):
        return d, g

    def imitate(self, conf):
        from lib.anal.comparing import ExpFitter
        dd = SingleRun(**conf).run()
        for d in dd:
            f = ExpFitter(d.config['sample'])
            fit = f.compare(d, save_to_config=True)
            print(d.id, fit)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['import','analysis', 'settings'])
    larvaworld_gui.run()
