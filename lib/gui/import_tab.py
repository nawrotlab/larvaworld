from lib.gui.gui_lib import ButtonGraphList, CollapsibleDict, DataList, SelectionList
from lib.gui.aux import gui_col
from lib.gui.tab import GuiTab
from lib.stor.datagroup import LarvaDataGroup
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

        sl1 = SelectionList(tab=self, disp='Data format/lab', actions=['load'])
        dl1 = DataList(name=self.raw_key, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'changeID', 'browse'],raw=True)
        dl2 = DataList(name=self.proc_key, tab=self, dict=d[kP],
                       buttons=['replay', 'enrich', 'select_all', 'remove', 'changeID', 'browse'])
        c1,c2,c3=[CollapsibleDict(n, False, default=True, toggled_subsections=None) for n in self.fields]
        g1 = ButtonGraphList(name=self.name, fig_dict={})

        l = [[
            gui_col([sl1, c1], 0.25),
            gui_col([dl1,c2], 0.25),
            gui_col([dl2, c3], 0.25)
        ]]

        c = {}
        for s in [c1, c2, c3]:
            c.update(**s.get_subdicts())

        g = {g1.name: g1}
        return l, c, g, d

    def eval(self, e, v, w, c, d, g):
        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'analysis', 'settings'])

    larvaworld_gui.run()
