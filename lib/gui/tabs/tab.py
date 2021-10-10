from lib.conf.stored.conf import loadConf
from lib.gui.aux.elements import GuiElement


class GuiTab(GuiElement):
    def __init__(self, name, gui, conftype=None, dtype=None):
        super().__init__(name)
        # self.name = name
        self.gui = gui
        self.conftype = conftype
        self.dtype = dtype
        self.selectionlists = {}
        self.datalists = {}
        self.graphlists = {}
        # super().__init__(name)
        # self.graph_list=None

    @property
    def graph_list(self):
        gs = self.gui.graph_lists
        n = self.name
        if n in list(gs.keys()):
            return gs[n]
        else:
            return None

    @property
    def canvas_k(self):
        g=self.graph_list
        return g.canvas_key if g is not None else None

    @property
    def graphlist_k(self):
        g = self.graph_list
        return g.list_key if g is not None else None

    @property
    def base_list(self):
        return self.selectionlists[self.conftype] if self.conftype is not None else None

    @property
    def datalist(self):
        return self.datalists[self.name] if self.name in list(self.datalists.keys()) else None

    @property
    def base_dict(self):
        ds=self.gui.dicts
        n=self.name
        if n in list(ds.keys()) :
            return ds[n]
        else :
            return None

    def current_ID(self, v):
        l=self.base_list
        return v[l.k] if l is not None else None

    def current_conf(self, v):
        id=self.current_ID(v)
        return loadConf(id, self.conftype) if id is not None else None

    def build(self):
        return None, {}, {}, {}

    def eval0(self, e, v):

        w = self.gui.window
        c = self.gui.collapsibles
        g = self.gui.graph_lists
        d = self.gui.dicts
        for sl_name,sl in self.selectionlists.items():
            sl.eval(e, v, w, c, d, g)
        for dl_name,dl in self.datalists.items():
            dl.eval(e, v, w, c, d, g)
        for dl_name,dl in self.graphlists.items():
            dl.eval(e, v, w, c, d, g)
        for dl_name,dl in c.items():
            try :
                dl.eval(e, v, w, c, d, g)
            except :
                pass
        self.eval(e, v, w, c, d, g)

    def run(self, v, w,c, d, g, conf, id):
        pass
        # return d, g

    def eval(self, e, v, w, c, d, g):
        pass

    def get(self, w, v, c):
        return None

    def update(self, w, c, conf, id):
        pass

    def fork(self, func, kwargs):
        import os
        import signal
        import sys
        def handle_signal(signum, frame):
            print('Caught signal "%s"' % signum)
            if signum == signal.SIGTERM:
                print('SIGTERM. Exiting!')
                sys.exit(1)
            elif signum == signal.SIGHUP:
                print('SIGHUP')
            elif signum == signal.SIGUSR1:
                print('SIGUSR1 Calling wait()')
                pid, status = os.wait()
                print('PID was: %s.' % pid)

        print('Starting..')
        signal.signal(signal.SIGCHLD, handle_signal)
        signal.signal(signal.SIGHUP, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGUSR1, handle_signal)

        try:
            ff_pid = os.fork()
        except OSError as err:
            print('Unable to fork: %s' % err)
        if ff_pid > 0:
            # Parent.
            print('First fork.')
            print('Child PID: %d' % ff_pid)
        elif ff_pid == 0:
            res=func(**kwargs)
            # return res
            # sys.exit(0)



if __name__ == "__main__":
    pass
    # sg.theme('LightGreen')
    # n = 'intro'
    # l, c, g, d = build_intro_tab()
    # w = sg.Window(f'{n} gui', l, size=(1800, 1200), **w_kws, location=(300, 100))
    #
    # while True:
    #     e, v = w.read()
    #     if e in (None, 'Exit'):
    #         break
    #     run0(w, e, v, c, g)
    #     d, g = eval_intro_tab(e, v, w, collapsibles=c, dicts=d, graph_lists=g)
    # w.close()
