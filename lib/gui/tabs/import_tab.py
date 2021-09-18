from lib.gui.aux.elements import ButtonGraphList, CollapsibleDict, DataList, SelectionList
from lib.gui.aux.functions import gui_col
from lib.gui.tabs.tab import GuiTab
from lib.stor import paths



class ImportTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'

        self.fields = ['tracker', 'parameterization', 'enrichment']

    def update(self, w, c, conf, id=None):
        path=conf['path']
        w[f'BROWSE {self.raw_key}'].InitialFolder = f'{paths.DataFolder}/{path}/raw'
        w[f'BROWSE {self.proc_key}'].InitialFolder = f'{paths.DataFolder}/{path}/processed'
        for n in self.fields:
            c[n].update(w, conf[n])

    def build(self):
        kR, kP = self.raw_key, self.proc_key
        d = {kR: {}, kP: {}}

        sl1 = SelectionList(tab=self, disp='Data format/lab', buttons=['load'])
        dl1 = DataList(name=self.raw_key, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'change_ID', 'browse'],
                       raw=True, size=(25,5))
        dl2 = DataList(name=self.proc_key, tab=self, dict=d[kP],
                       buttons=['replay', 'enrich', 'select_all', 'remove', 'change_ID', 'browse'],
                       aux_cols=['N', 'duration', 'quality'], size=(40,5))
        c1,c2,c3=[CollapsibleDict(n, default=True, toggled_subsections=None) for n in self.fields]
        g1 = ButtonGraphList(name=self.name, tab=self, fig_dict={})

        l = [[
            gui_col([sl1, c1], 0.25),
            gui_col([dl1,c2], 0.25),
            gui_col([dl2, c3], 0.5)
        ]]

        c = {}
        for s in [c1, c2, c3]:
            c.update(**s.get_subdicts())

        g = {g1.name: g1}
        return l, c, g, d

    def eval(self, e, v, w, c, d, g):
        # print(e)
        return d, g


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'analysis', 'settings'])

    larvaworld_gui.run()
