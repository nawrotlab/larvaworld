from lib.gui.aux.elements import ButtonGraphList, CollapsibleDict, DataList, SelectionList
from lib.gui.aux.functions import gui_cols
from lib.gui.tabs.tab import GuiTab
from lib.sim.single.single_run import SingleRun
from lib.conf.base import paths



class ImportTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw_key = 'raw_data'
        self.proc_key = 'imported_data'

        self.fields = ['tracker', 'parameterization', 'enrichment']

    def update(self, w, c, conf, id=None):
        path=conf['path']
        w[f'BROWSE {self.raw_key}'].InitialFolder = f'{paths.path("DATA")}/{path}/raw'
        w[f'BROWSE {self.proc_key}'].InitialFolder = f'{paths.path("DATA")}/{path}/processed'
        for n in self.fields:
            c[n].update(w, conf[n])

    def build(self):
        kR, kP = self.raw_key, self.proc_key
        d = {kR: {}, kP: {}}

        sl1 = SelectionList(tab=self, disp='Data format/lab', buttons=['load'])
        dl1 = DataList(kR, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'change_ID', 'browse'],
                       raw=True, size=(25,5))
        dl2 = DataList(kP, tab=self, dict=d[kP],buttons=['replay', 'imitate', 'enrich', 'select_all', 'remove', 'change_ID', 'browse'],
                       aux_cols=['N', 'duration', 'quality'], size=(40,5))
        c1,c2,c3=[CollapsibleDict(n) for n in self.fields]
        g1 = ButtonGraphList(self.name, tab=self, fig_dict={})
        l = gui_cols(cols=[[sl1, c1], [dl1,c2], [dl2, c3]], x_fracs=[0.25, 0.25, 0.5])

        c = {}
        for s in [c1, c2, c3]:
            c.update(**s.get_subdicts())

        return l, c, {g1.name: g1}, d

    def eval(self, e, v, w, c, d, g):
        # print(e)
        return d, g

    def imitate(self, conf):
        from lib.anal.comparing import ExpFitter

        dd = SingleRun(**conf).run()
        f = ExpFitter(dd.config['env_params']['larva_groups']['ImitationGroup']['sample'])
        fit = f.compare(dd, save_to_config=True)
        print(fit)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'settings'])

    larvaworld_gui.run()
