import os
import numpy as np



os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from lib import reg, aux, screen

class ScreenManager:
    def __init__(self, model,  vis_kwargs=None,**kwargs):

        self.model = model
        if vis_kwargs is None:
            vis_kwargs = reg.get_null('visualization', mode=None)
        self.vis_kwargs = aux.AttrDict(vis_kwargs)
        self.mode = self.vis_kwargs.render.mode
        show_display = self.vis_kwargs.render.show_display

        if self.mode is None and not show_display:
            reg.vprint('Storage of media or visualization not requested.')
            self.active=False
        else :
            self.active=True
            self.build(**kwargs)



    def build(self,background_motion=None, traj_color=None,allow_clicks=True, **kwargs):
        self.s = self.model.scaling_factor

        self.image_mode = self.vis_kwargs.render.image_mode
        self.screen_kws = self.define_screen_kws()

        self.__dict__.update(self.vis_kwargs.draw)
        self.__dict__.update(self.vis_kwargs.color)
        self.__dict__.update(self.vis_kwargs.aux)
        self.dynamic_graphs = []
        self.focus_mode = False
        self.intro_text = False
        self.selected_type = ''


        self.mousebuttondown_pos = None
        self.mousebuttonup_pos = None

        self.selected_agents = []

        self.allow_clicks = allow_clicks

        self.snapshot_interval = int(60 / self.model.dt)

        self.traj_color = traj_color
        self.background_motion = background_motion
        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.selection_color = np.array([255, 0, 0])

        self.snapshot_counter = 0
        self.odorscape_counter = 0



        self.bg = self.background_motion
        self.v = None
        self.pygame_keys = None

        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.sim_clock = screen.SimulationClock(self.model.dt, color=self.scale_clock_color)
        self.sim_scale = screen.SimulationScale(self.model.arena_dims[0], color=self.scale_clock_color)
        self.sim_state = screen.SimulationState(model=self.model, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)
        self.add_screen_texts(list(self.model.odor_layers.keys()), color=self.scale_clock_color)
        self.input_box = screen.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                                  center=True, w=120 * 4, h=32 * 4, font=pygame.font.SysFont("comicsansms", 32 * 2))


    def define_screen_kws(self):
        m = self.model
        mode = self.vis_kwargs.render.mode
        show_display = self.vis_kwargs.render.show_display
        media_name = self.vis_kwargs.render.media_name
        video_speed = self.vis_kwargs.render.video_speed
        if media_name is None:
            media_name = str(m.id)

        self.space_bounds = aux.get_arena_bounds(m.space.dims, self.s)
        self.window_dims = aux.get_window_dims(m.space.dims)
        screen_kws = {
            'window_dims': self.window_dims,
            'space_bounds': self.space_bounds,
            'caption': media_name,
            'dt': m.dt,
            'fps': int(video_speed / m.dt),
            'show_display': show_display,
            # 'record_video_to':show_display,
        }
        os.makedirs(m.save_to, exist_ok=True)
        if mode == 'video':
            screen_kws['record_video_to'] = f'{m.save_to}/{media_name}.mp4'
        if mode == 'image':
            screen_kws['record_image_to'] = f'{m.save_to}/{media_name}_{self.image_mode}.png'
        return screen_kws

    def add_agent(self, **kwargs):
        return self.model.add_agent(**kwargs)

    def get_all_objects(self):
        return self.model.get_all_objects()

    @ property
    def odor_layers(self):
        return self.model.odor_layers

    def add_screen_texts(self, names, color):
        for name in names:
            text = screen.InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def step(self, tick=None):
        if self.active :
            self.sim_clock.tick_clock()
            if self.mode == 'video':
                if self.image_mode != 'snapshots' or self.snapshot_tick:
                    self.render(tick)
            elif self.mode == 'image':
                if self.image_mode == 'overlap':
                    self.render(tick)
                elif self.image_mode == 'snapshots' and self.snapshot_tick:
                    self.capture_snapshot(tick)

    def finalize(self, tick=None):
        if self.active:
            if self.image_mode == 'overlap':
                self.v.render()
            elif self.image_mode == 'final':
                self.capture_snapshot(tick)
            if self.v:
                self.v.close()

    def capture_snapshot(self, tick):
        self.render(tick)
        self.model.toggle('snapshot #')
        self.v.render()

    def create_screen_texts(self, color):
        names = [
            'trajectory_dt',
            'trails',
            'focus_mode',
            'draw_centroid',
            'draw_head',
            'draw_midline',
            'draw_contour',
            'draw_sensors',
            'visible_clock',
            'visible_ids',
            'visible_state',
            'visible_scale',
            'odor_aura',
            'color_behavior',
            'random_colors',
            'black_background',
            'larva_collisions',
            'zoom',
            'snapshot #',
            'odorscape #',
            'windscape',
            'is_paused',
        ]
        return {name: screen.InputBox(text=name, color_active=color, color_inactive=color) for name in names}

    def set_default_colors(self, black_background):
        if black_background:
            tank_color = (0, 0, 0)
            screen_color = (50, 50, 50)
            scale_clock_color = (255, 255, 255)
            default_larva_color = np.array([255, 255, 255])
        else:
            tank_color = (255, 255, 255)
            screen_color = (200, 200, 200)
            scale_clock_color = (0, 0, 0)
            default_larva_color = np.array([0, 0, 0])
        return tank_color, screen_color, scale_clock_color, default_larva_color

    def set_background(self, width, height):
        if self.bg is not None:
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(ROOT_DIR, 'background.png')
            print('Loading background image from', path)
            self.bgimage = pygame.image.load(path)
            self.bgimagerect = self.bgimage.get_rect()
            self.tw = self.bgimage.get_width()
            self.th = self.bgimage.get_height()
            self.th_max = int(height / self.th) + 2
            self.tw_max = int(width / self.tw) + 2
        else:
            self.bgimage = None
            self.bgimagerect = None

    def draw_agents(self, screen):
        m=self.model
        for o in m.get_food():
            if o.visible:
                o.draw(screen, filled=True if o.amount > 0 else False)
                o.id_box.draw(screen, screen_pos=self.space2screen_pos(o.get_position()))

        for g in m.get_flies():
            if g.visible:
                if self.color_behavior:
                    g.update_behavior_dict()
                g.draw(screen, self)
                g.id_box.draw(screen, screen_pos=self.space2screen_pos(g.get_position()))

        if self.trails:
            if self.trajectory_dt is None:
                self.trajectory_dt = 0.0
            draw_trajectories(space_dims=m.arena_dims * self.s, agents=m.get_flies(), screen=screen,
                              decay_in_ticks=int(self.trajectory_dt / m.dt),
                              traj_color=self.traj_color)

    def render(self, tick=None):
        m = self.model
        if self.bg is not None and tick is not None:
            bg = self.bg[:, tick - 1]
        else:
            bg = [0, 0, 0]
        if self.v is None:
            self.v = self.initialize(bg=bg)
        elif self.v.close_requested():
            self.v.close()
            self.v = None
            m.running = False
            return
        if self.image_mode != 'overlap':
            self.draw_arena(self.v, bg)

        self.draw_agents(self.v)

        if self.v.show_display:
            evaluate_input(self, self.v)
            evaluate_graphs(self)
        if self.image_mode != 'overlap':

            self.draw_aux(self.v)
            self.v.render()

    def initialize(self, bg):

        v = screen.Viewer(**self.screen_kws)

        self.display_configuration(v)
        self.render_aux()
        self.set_background(*v.display_dims)

        self.draw_arena(v, bg)

        print('Screen opened')
        return v

    def display_configuration(self, screen):
        if self.intro_text:
            box = screen.InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                           text=self.model.configuration_text,
                           color_active=pygame.Color('white'),
                           visible=True,
                           center=True, w=220 * 4, h=200 * 4,
                           font=pygame.font.SysFont("comicsansms", 25))

            for i in range(10):
                box.draw(screen)
                screen.render()
            box.visible = False

    def space2screen_pos(self, pos):
        if pos is None or any(np.isnan(pos)):
            return None
        try:
            return self.v._transform(pos)
        except:
            X, Y = self.model.arena_dims * self.s
            X0, Y0 = self.window_dims

            p = pos[0] * 2 / X, pos[1] * 2 / Y
            pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
            return pp

    def draw_aux(self, screen):
        screen.draw_arena(self.model.space.vertices, self.tank_color, self.screen_color)
        if self.visible_clock:
            self.sim_clock.draw_clock(screen)
        if self.visible_scale:
            self.sim_scale.draw_scale(screen)
        if self.visible_state:
            self.sim_state.draw_state(screen)
        self.draw_screen_texts(screen)

    def draw_screen_texts(self, screen):
        for text in list(self.screen_texts.values()) + [self.input_box]:
            if text and text.start_time < pygame.time.get_ticks() < text.end_time:
                text.visible = True
                text.draw(screen)
            else:
                text.visible = False

    def draw_arena(self, v, bg):
        m = self.model

        arena_drawn = False
        for id, layer in m.odor_layers.items():
            if layer.visible:
                layer.draw(v)
                arena_drawn = True
                break
        if not arena_drawn and m.food_grid is not None:
            m.food_grid.draw(v)
            arena_drawn = True

        if not arena_drawn:
            v.draw_polygon(m.space.vertices, color=self.tank_color)
            self.draw_background(v, bg)

        if m.windscape is not None and m.windscape.visible:
            m.windscape.draw(v)

        for i, b in enumerate(m.borders):
            b.draw(v)

    def render_aux(self):
        self.sim_clock.render_clock(*self.window_dims)
        self.sim_scale.render_scale(*self.window_dims)
        self.sim_state.render_state(*self.window_dims)
        for t in self.screen_texts.values():
            t.render(*self.window_dims)

    def draw_background(self, v, bg):
        if self.bgimage is not None and self.bgimagerect is not None:
            x, y, a = bg
            try:
                min_x = int(np.floor(x))
                min_y = -int(np.floor(y))

                surface = v._window
                for py in np.arange(min_y - 1, self.th_max + min_y, 1):
                    for px in np.arange(min_x - 1, self.tw_max + min_x, 1):
                        if a != 0.0:
                            # px,py=aux.rotate_point_around_point((px,py),-a)
                            pass
                        p = ((px - x) * (self.tw - 1), (py + y) * (self.th - 1))
                        surface.blit(self.bgimage, p)
            except:
                pass

    def toggle(self, name, value=None, show=False, minus=False, plus=False, disp=None):
        m = self.model
        if disp is None:
            disp = name

        if name == 'snapshot #':
            self.v.snapshot_requested = int(m.Nticks * m.dt)
            value = self.snapshot_counter
            self.snapshot_counter += 1
        elif name == 'odorscape #':
            reg.graphs.dict['odorscape'](odor_layers = m.odor_layers,save_to=m.plot_dir, show=show, scale=self.s, idx=self.odorscape_counter)
            value = self.odorscape_counter
            self.odorscape_counter += 1
        elif name == 'trajectory_dt':
            if minus:
                dt = -1
            elif plus:
                dt = +1
            self.trajectory_dt = np.clip(self.trajectory_dt + 5 * dt, a_min=0, a_max=np.inf)
            value = self.trajectory_dt

        if value is None:
            setattr(self, name, not getattr(self, name))
            value = 'ON' if getattr(self, name) else 'OFF'
        self.screen_texts[name].flash_text(f'{disp} {value}')
        # self.screen_texts[name].text = f'{disp} {value}'
        # self.screen_texts[name].end_time = pygame.time.get_ticks() + 2000
        # self.screen_texts[name].start_time = pygame.time.get_ticks() + int(self.dt * 1000)

        if name == 'visible_ids':
            for a in m.get_flies() + m.get_food():
                a.id_box.visible = not a.id_box.visible
        elif name == 'color_behavior':
            if not self.color_behavior:
                for f in m.get_flies():
                    f.set_color(f.default_color)
        elif name == 'random_colors':
            for f in m.get_flies():
                color = aux.random_colors(1)[0] if self.random_colors else f.default_color
                f.set_color(color)
        elif name == 'black_background':
            self.update_default_colors()
        elif name == 'larva_collisions':

            self.eliminate_overlap()

    def update_default_colors(self):
        if self.black_background:
            self.tank_color = (0, 0, 0)
            self.screen_color = (50, 50, 50)
            self.scale_clock_color = (255, 255, 255)
        else:
            self.tank_color = (255, 255, 255)
            self.screen_color = (200, 200, 200)
            self.scale_clock_color = (0, 0, 0)
        for i in [self.sim_clock, self.sim_scale, self.sim_state] + list(self.screen_texts.values()):
            i.set_color(self.scale_clock_color)


    def apply_screen_zoom(self, screen, d_zoom):
        screen.zoom_screen(d_zoom)
        self.sim_scale = screen.SimulationScale(self.model.arena_dims[0] * screen.zoom, color=self.sim_scale.color)
        self.sim_scale.render_scale(*self.window_dims)



    @property
    def snapshot_tick(self):
        return (self.model.Nticks - 1) % self.snapshot_interval == 0


    def add_screen_texts(self, names, color):
        for name in names:
            text = screen.InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def generate_larva_color(self):
        return aux.random_colors(1)[0] if self.random_colors else self.default_larva_color



    def eliminate_overlap(self):
        pass


def evaluate_input(m, screen):

    if m.pygame_keys is None :
        m.pygame_keys = reg.controls.load()['pygame_keys']

    d_zoom = 0.01
    ev = pygame.event.get()
    for e in ev:
        if e.type == pygame.QUIT:
            screen.close()
        if e.type == pygame.KEYDOWN:
            for k, v in m.pygame_keys.items():
                if e.key == getattr(pygame, v):
                    eval_keypress(k, screen, m)

        if m.allow_clicks:
            if e.type == pygame.MOUSEBUTTONDOWN:
                m.mousebuttondown_pos = screen.mouse_position
            elif e.type == pygame.MOUSEBUTTONUP:
                p = screen.mouse_position
                if e.button == 1:
                    if not eval_selection(m, p, ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL):
                        m.add_agent(agent_class=m.selected_type, p0=tuple(p),
                                        p1=tuple(m.mousebuttondown_pos))

                elif e.button == 3:
                    from gui.gui_aux.windows import set_agent_kwargs, object_menu
                    loc = tuple(np.array(screen.w_loc) + np.array(pygame.mouse.get_pos()))
                    if len(m.selected_agents) > 0:
                        for sel in m.selected_agents:
                            sel = set_agent_kwargs(sel, location=loc)
                    else:
                        m.selected_type = object_menu(m.selected_type, location=loc)
                elif e.button in [4, 5]:
                    m.apply_screen_zoom(screen, d_zoom=-d_zoom if e.button == 4 else d_zoom)
                    m.toggle(name='zoom', value=screen.zoom)
    if m.focus_mode and len(m.selected_agents) > 0:
        try:
            sel = m.selected_agents[0]
            screen.move_center(pos=sel.get_position())
        except:
            pass


def eval_keypress(k, screen, model):
    from lib.model.agents._larva_sim import LarvaSim
    from lib.model.agents._larva import Larva
    # print(k)
    if k == '▲ trail duration':
        model.toggle('trajectory_dt', plus=True, disp='trail duration')
    elif k == '▼ trail duration':
        model.toggle('trajectory_dt', minus=True, disp='trail duration')
    elif k == 'visible trail':
        model.toggle('trails')
    elif k == 'pause':
        model.toggle('is_paused')
    elif k == 'move left':
        screen.move_center(-0.05, 0)
    elif k == 'move right':
        screen.move_center(+0.05, 0)
    elif k == 'move up':
        screen.move_center(0, +0.05)
    elif k == 'move down':
        screen.move_center(0, -0.05)
    elif k == 'plot odorscapes':
        model.toggle('odorscape #', show=pygame.key.get_mods() & pygame.KMOD_CTRL)
    elif 'odorscape' in k:
        idx = int(k.split(' ')[-1])
        try :
            layer_id = list(model.odor_layers.keys())[idx]
            layer = model.odor_layers[layer_id]
            layer.visible = not layer.visible
            model.toggle(layer_id, 'ON' if layer.visible else 'OFF')
        except :
            pass
    elif k == 'snapshot':
        model.toggle('snapshot #')
    elif k == 'windscape' :
        try :
            model.windscape.visible = not model.windscape.visible
            model.toggle('windscape', 'ON' if model.windscape.visible else 'OFF')
        except :
            pass
    elif k == 'delete item':
        from gui.gui_aux.windows import delete_objects_window
        if delete_objects_window(model.selected_agents):
            for f in model.selected_agents:
                model.selected_agents.remove(f)
                model.delete_agent(f)
    elif k == 'dynamic graph':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, Larva):
                from gui.gui_aux.elements import DynamicGraph
                model.dynamic_graphs.append(DynamicGraph(agent=sel))
    elif k == 'odor gains':
        if len(model.selected_agents) > 0:
            sel = model.selected_agents[0]
            if isinstance(sel, LarvaSim) and sel.brain.olfactor is not None:
                from gui.gui_aux.windows import set_kwargs
                sel.brain.olfactor.gain = set_kwargs(sel.brain.olfactor.gain, title='Odor gains')
    else:
        model.toggle(k)


def evaluate_graphs(m):
    for g in m.dynamic_graphs:
        running = g.evaluate()
        if not running:
            m.dynamic_graphs.remove(g)
            del g


def eval_selection(m, p, ctrl):
    res = False if len(m.selected_agents) == 0 else True
    for f in m.get_all_objects():
        if f.contained(p):
            if not f.selected:
                f.selected = True
                m.selected_agents.append(f)
            elif ctrl:
                f.selected = False
                m.selected_agents.remove(f)
            res = True
        elif f.selected and not ctrl:
            f.selected = False
            m.selected_agents.remove(f)
    return res


def draw_trajectories(space_dims, agents, screen, decay_in_ticks=None, traj_color=None):
    trajs = [fly.trajectory for fly in agents]
    if traj_color is not None:
        traj_cols = [traj_color.xs(fly.unique_id, level='AgentID') for fly in agents]
    else:
        traj_cols = [np.array([(0, 0, 0) for t in traj]) for traj, fly in zip(trajs, agents)]

    if decay_in_ticks is not None:
        trajs = [t[-decay_in_ticks:] for t in trajs]
        traj_cols = [t[-decay_in_ticks:] for t in traj_cols]

    for fly, traj, traj_col in zip(agents, trajs, traj_cols):
        # This is the case for simulated larvae where no values are np.nan
        if not np.isnan(traj).any():
            parsed_traj = [traj]
            parsed_traj_col = [traj_col]
        elif np.isnan(traj).all():
            continue
        # This is the case for larva trajectories derived from experiments where some values are np.nan
        else:
            traj_x = np.array([x for x, y in traj])
            ds, de = aux.parse_array_at_nans(traj_x)
            parsed_traj = [traj[s:e] for s, e in zip(ds, de)]
            parsed_traj_col = [traj_col[s:e] for s, e in zip(ds, de)]

        for t, c in zip(parsed_traj, parsed_traj_col):
            # If trajectory has one point, skip

            if len(t) < 2:
                pass
            else:
                if traj_color is None:
                    screen.draw_polyline(t, color=fly.default_color, closed=False, width=0.003 * space_dims[0])
                else:
                    c = [tuple(float(x) for x in s.strip('()').split(',')) for s in c]
                    c = [s if not np.isnan(s).any() else (255, 0, 0) for s in c]
                    screen.draw_polyline(t, color=c, closed=False, width=0.01 * space_dims[0], dynamic_color=True)


