from gui.tabs.tab import GuiTab

from gui.aux import elements as gui_el, functions as gui_fun
from lib import reg


class AnalysisTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        d = {self.name: {}}
        dl1 = gui_el.DataList('Datasets', tab=self, dict=d[self.name],
                      buttons=['replay', 'add_ref', 'select_all', 'remove', 'change_ID', 'browse'],
                       aux_cols=['N', 'duration', 'quality'], size=(28,20)
                       )
        g1 = gui_el.ButtonGraphList(self.name, tab=self, fig_dict=reg.graphs.dict, canvas_size=gui_fun.col_size(x_frac=0.5, y_frac=0.8))
        l = gui_fun.gui_cols(cols=[[dl1], [g1.canvas], [g1]], x_fracs=[0.25, 0.52, 0.2], as_pane=True, pad=(10,20))
        return l, {}, {g1.name: g1}, d

    def eval(self, e, v, w, c, d, g):
        return d, g


if __name__ == "__main__":
    from gui.tabs.gui import LarvaworldGui
    larvaworld_gui = LarvaworldGui(tabs=['analysis'])
    larvaworld_gui.run()
