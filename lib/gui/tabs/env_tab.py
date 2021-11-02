from lib.gui.aux.elements import CollapsibleTable, SelectionList, PadDict
from lib.gui.aux.functions import gui_cols
from lib.gui.aux.buttons import GraphButton
from lib.gui.tabs.tab import GuiTab


class EnvTab(GuiTab):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.canvas_size=(800,800)
        self.S, self.L, self.B = 'Source', 'Larva', 'Border'
        self.Su, self.Sg = f'{self.S.lower()}_units', f'{self.S.lower()}_groups'
        # self.Lu, self.Lg=f'{self.L.lower()}_units', f'{self.L.lower()}_groups'
        self.Bg = f'{self.B.lower()}_list'

    def update(self, w, c, conf, id=None):
        for n in [self.Bg, 'arena', 'odorscape', 'windscape']:
            c[n].update(w, conf[n] if n in conf.keys() else {})
        for n in [self.Sg, self.Su, 'food_grid']:
            c[n].update(w, conf['food_params'][n])
        # self.base_dict['env_db'] = self.set_env_db(env=conf)
        # w.write_event_value('RESET_ARENA', 'Draw the initial arena')

    def get(self, w, v, c, as_entry=False):
        return {
            'food_params': {n: c[n].get_dict(v, w) for n in [self.Sg, self.Su, 'food_grid']},
            **{n: c[n].get_dict(v, w) for n in [self.Bg, 'arena', 'odorscape', 'windscape']},
            # 'windscape' : None
        }


    def build(self):
        s2 = CollapsibleTable(self.Sg, dict_name='SourceGroup', state=True, index='Group ID',
                              col_widths=[10, 3, 8, 7, 6], num_rows=5,
                              heading_dict={'N': 'distribution.N', 'color': 'default_color', 'odor': 'odor.odor_id',
                                            'amount': 'amount'}, )
        s3 = CollapsibleTable(self.Su, dict_name='source', state=True, index='ID', col_widths=[10, 8, 8, 8], num_rows=5,
                              heading_dict={'color': 'default_color', 'odor': 'odor.odor_id', 'amount': 'amount'}, )
        s4 = CollapsibleTable(self.Bg, dict_name='border_list', index='ID', col_widths=[10, 8, 16], num_rows=5,
                              heading_dict={'color': 'default_color', 'points': 'points'}, state=True)


        s5 = PadDict('arena', header_width=23, after_header=[GraphButton('Button_Burn', 'RESET_ARENA',
                                                                         tooltip='Reset to the initial arena. All drawn items will be erased.'),
                                                             GraphButton('Globe_Active', 'NEW_ARENA',
                                                                         tooltip='Create a new arena.All drawn items will be erased.')])

        s6 = PadDict('food_grid', header_width=26,toggle=True)
        s7 = PadDict('odorscape', header_width=31)
        s8 = PadDict('windscape', header_width=31)

        c = {}
        for s in [s2, s3, s4, s5, s6, s7, s8]:
            c.update(s.get_subdicts())
        l1 = [c[n].get_layout(as_pane=True)[0] for n in [self.Sg, self.Su, 'food_grid']]
        c2 = PadDict(self.S, content=l1, header_width=34)
        c.update(c2.get_subdicts())
        sl1 = SelectionList(tab=self, buttons=['save', 'delete'], disp=self.name, width=30)
        l = gui_cols([[sl1, s7,s8, s4], [c2]], x_fracs=[0.25,0.25], as_pane=True, pad=(10,10))
        self.layout = l
        return l, c, {}, {}

    # def run(self, v, w,c,d,g, conf,id):
    #     sim=null_dict('sim_params', sim_ID='env_test', duration=0.5)
    #     exp_conf=null_dict('exp_conf', env_params=conf, sim_params=sim)
    #     exp_conf['life_params']=loadConf(exp_conf['life_params'], 'Life')
    #     # p = self.base_list.progressbar
    #     # p.run(w, max=N)
    #     exp_conf['experiment'] = 'test'
    #     exp_conf['save_data_flag'] = False
    #     exp_conf['vis_kwargs'] = null_dict('visualization', mode='video', video_speed=60)
    #     res = run_sim(**exp_conf)
    #     return d, g


if __name__ == "__main__":
    from lib.gui.tabs.gui import LarvaworldGui

    larvaworld_gui = LarvaworldGui(tabs=['environment'])
    larvaworld_gui.run()
