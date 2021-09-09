from lib.gui.gui_lib import ButtonGraphList, CollapsibleDict, DataList, SelectionList
from lib.gui.aux import gui_col
from lib.gui.tab import GuiTab
from lib.stor.datagroup import LarvaDataGroup



class ImportTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'
        self.raw_ids_key = f'{self.raw_key}_IDS'
        self.proc_ids_key = f'{self.proc_key}_IDS'
        self.raw_folder = None
        self.proc_folder = None

    def update(self, w, c, conf, id=None):
        datagroup = LarvaDataGroup(id)
        w[f'BROWSE {self.raw_key}'].InitialFolder = datagroup.raw_dir
        # w[self.raw_key].InitialFolder = datagroup.raw_dir
        self.raw_folder = datagroup.raw_dir
        w[f'BROWSE {self.proc_key}'].InitialFolder = datagroup.proc_dir
        # w[self.proc_key].InitialFolder = datagroup.proc_dir
        self.proc_folder = datagroup.proc_dir
        c['enrichment'].update(w, datagroup.get_conf()['enrich'])
        # c['arena'].update(w, datagroup.get_conf()['arena_pars'])

    def build(self):
        kR, kP = self.raw_key, self.proc_key
        d = {kR: {}, kP: {}}

        lR = DataList(name=self.raw_key, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'changeID', 'browse'],raw=True)
        lP = DataList(name=self.proc_key, tab=self, dict=d[kP],
                       buttons=['replay', 'enrich', 'select_all', 'remove', 'changeID', 'browse'])
        lG = SelectionList(tab=self, disp='Data group', actions=['load'])

        # lGA = CollapsibleDict('arena', False, default=True)



        g1 = ButtonGraphList(name=self.name, fig_dict={})
        s1 = CollapsibleDict('enrichment', False, default=True, toggled_subsections=None)

        l = [[
            gui_col([lG, lR, lP], 0.25),
            # gui_col([lG,lGA, lR, lP], 0.25),
            gui_col([s1], 0.25)
        ]]

        c = {}
        for s in [s1]:
            c.update(**s.get_subdicts())

        g = {g1.name: g1}
        return l, c, g, d

    def eval(self, e, v, w, c, d, g):

        return d, g


if __name__ == "__main__":
    from lib.gui.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'analysis'])
    larvaworld_gui.run()
