from lib.gui.aux.elements import ButtonGraphList, CollapsibleDict, DataList, SelectionList, PadDict
from lib.gui.aux.functions import gui_cols, t_kws
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
        kws = {'background_color': 'purple'}
        sl1 = SelectionList(tab=self, disp='Data format/lab', buttons=['load'])
        dl1 = DataList(kR, tab=self, dict=d[kR], buttons=['import', 'select_all', 'remove', 'change_ID', 'browse'],
                       raw=True, size=(30,5))
        dl2 = DataList(kP, tab=self, dict=d[kP],buttons=['replay', 'imitate', 'enrich', 'select_all', 'remove', 'change_ID','save_ref', 'browse'],
                       aux_cols=['N', 'duration', 'quality'], size=(40,5))
        # c1=PadDict('tracker', Ncols=1, text_kws=t_kws(10), **kws)
        # c2=PadDict('parameterization', Ncols=1, text_kws=t_kws(16), **kws)
        # c3=PadDict('enrichment', col_idx=[[4,5,0],[1],[2],[3]], text_kws=t_kws(8), **kws)
        # c3=PadDict('enrichment', col_idx=[[4,5,0,3],[1],[2]], text_kws=t_kws(8), **kws,
        #            subconfs={'preprocessing' : {'text_kws' : t_kws(14)},
        #                      'to_drop' : {'Ncols' : 2},
                             # 'types' : {'Ncols' : 2},
                             # })
        # c1,c2,c3=[PadDict(n, Ncols=2, text_kws=t_kws(10)) for n in self.fields]
        c1,c2,c3=[CollapsibleDict(n, state=True) for n in self.fields]
        g1 = ButtonGraphList(self.name, tab=self, fig_dict={})
        l = gui_cols(cols=[[sl1, c1], [dl1,dl2, c2], [c3]], x_fracs=[0.22, 0.25, 0.55])
        # l = gui_cols(cols=[[sl1, c1], [dl1,c2], [dl2, c3]], x_fracs=[0.25, 0.25, 0.5])
        # l = gui_cols(cols=[[sl1, c1], [dl1,c2], [dl2, c3]], x_fracs=[0.25, 0.25, 0.5], as_pane=True)

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
        for d in dd :
            f = ExpFitter(d.config['sample'])
            fit = f.compare(d, save_to_config=True)
            print(d.id, fit)


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    # larvaworld_gui = LarvaworldGui(tabs=['import'])
    larvaworld_gui = LarvaworldGui(tabs=['import', 'settings'])

    larvaworld_gui.run()
