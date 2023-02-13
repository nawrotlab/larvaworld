
from larvaworld.lib import reg
from larvaworld.gui.gui_aux.elements import GuiElement

class GuiTab(GuiElement):
    def __init__(self, name, gui, conftype=None, dtype=None):
        super().__init__(name)
        self.gui = gui
        self.conftype = conftype
        self.dtype = dtype
        self.selectionlists = {}
        self.datalists = {}
        self.graphlists = {}

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
        return reg.loadConf(id=id, conftype=self.conftype) if id is not None else None

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

class DrawTab(GuiTab):
    def __init__(self,canvas_size = (800, 800), **kwargs):
        super().__init__(**kwargs)
        self.canvas_size = canvas_size

    @property
    def s(self):
        return self.base_dict['s']

    def get_drag_ps(self, scaled=False):
        d = self.base_dict
        p1, p2 = d['start_point'], d['end_point']
        return [self.scale_xy(p1), self.scale_xy(p2)] if scaled else [p1, p2]

    def set_drag_ps(self, p1=None, p2=None):
        d = self.base_dict
        if p1 is not None:
            d['start_point'] = p1
        if p2 is not None:
            d['end_point'] = p2

    def scale_xy(self, xy, reverse=False):
        if xy is None:
            return None
        W, H = self.graph_list.canvas_size
        s = self.s
        x, y = xy
        if reverse:
            return x * s + W / 2, y * s + H / 2
        else:
            return (x - W / 2) / s, (y - H / 2) / s

    def aux_reset(self):
        dic = self.base_dict
        dic['dragging'], dic['current'] = False, {}
        dic['start_point'], dic['end_point'], dic['prior_rect'] = None, None, None


