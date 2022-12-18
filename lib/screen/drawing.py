import os

import numpy as np
import pygame

from lib.registry import reg
from lib.screen.rendering import Viewer, InputBox, SimulationClock, SimulationScale, SimulationState, draw_trajectories
from lib.screen.screen_aux import get_window_dims, get_arena_bounds
from lib.aux import dictsNlists as dNl, colsNstr as cNs
from lib.screen.input_lib import evaluate_input, evaluate_graphs


class ScreenManager:
    def __init__(self, model,  vis_kwargs=None,**kwargs):

        self.model = model
        if vis_kwargs is None:
            vis_kwargs = reg.get_null('visualization', mode=None)
        self.vis_kwargs = dNl.NestDict(vis_kwargs)
        self.mode = self.vis_kwargs.render.mode
        show_display = self.vis_kwargs.render.show_display

        if self.mode is None and not show_display:
            reg.vprint('Storage of media or visualization not requested.')
            self.active=False
        else :
            self.active=True
            self.build(**kwargs)



    def build(self,show_conf_text=False,background_motion=None, traj_color=None,odor_aura = False, allow_clicks=True, **kwargs):
        self.s = self.model.scaling_factor

        self.image_mode = self.vis_kwargs.render.image_mode
        self.screen_kws = self.define_screen_kws()

        self.__dict__.update(self.vis_kwargs.draw)
        self.__dict__.update(self.vis_kwargs.color)
        self.__dict__.update(self.vis_kwargs.aux)
        self.dynamic_graphs = []
        self.focus_mode = False
        self.selected_type = ''


        self.odor_aura = odor_aura
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


        self.show_conf_text = show_conf_text

        self.bg = self.background_motion
        self.v = None
        self.pygame_keys = None

        self.tank_color, self.screen_color, self.scale_clock_color, self.default_larva_color = self.set_default_colors(
            self.black_background)

        self.sim_clock = SimulationClock(self.model.dt, color=self.scale_clock_color)
        self.sim_scale = SimulationScale(self.model.arena_dims[0], color=self.scale_clock_color)
        self.sim_state = SimulationState(model=self.model, color=self.scale_clock_color)

        self.screen_texts = self.create_screen_texts(color=self.scale_clock_color)
        self.add_screen_texts(list(self.model.odor_layers.keys()), color=self.scale_clock_color)
        self.input_box = InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
                                  center=True, w=120 * 4, h=32 * 4, font=pygame.font.SysFont("comicsansms", 32 * 2))


    def define_screen_kws(self):
        m = self.model
        mode = self.vis_kwargs.render.mode
        show_display = self.vis_kwargs.render.show_display
        media_name = self.vis_kwargs.render.media_name
        video_speed = self.vis_kwargs.render.video_speed
        if media_name is None:
            media_name = m.id

        self.space_bounds = get_arena_bounds(m.arena_dims, self.s)
        self.window_dims = get_window_dims(m.arena_dims)
        screen_kws = {
            'window_dims': self.window_dims,
            'space_bounds': self.space_bounds,
            'caption': media_name,
            'dt': m.dt,
            'fps': int(video_speed / m.dt),
            'show_display': show_display,
            # 'record_video_to':show_display,
        }
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
            text = InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def step(self, tick=None):
        if self.active :
            self.sim_clock.tick_clock()
            m = self.model
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
        return {name: InputBox(text=name, color_active=color, color_inactive=color) for name in names}

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
            m.is_running = False
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

        v = Viewer(**self.screen_kws)

        self.display_configuration(v)
        self.render_aux()
        self.set_background(*v.display_dims)

        self.draw_arena(v, bg)

        print('Screen opened')
        return v

    def display_configuration(self, screen):
        if self.show_conf_text:
            box = InputBox(screen_pos=self.space2screen_pos((0.0, 0.0)),
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
            X, Y = self.model.space_dims
            X0, Y0 = self.window_dims

            p = pos[0] * 2 / X, pos[1] * 2 / Y
            pp = ((p[0] + 1) * X0 / 2, (-p[1] + 1) * Y0)
            return pp

    def draw_aux(self, screen):
        screen.draw_arena(self.model.tank_shape, self.tank_color, self.screen_color)
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
            v.draw_polygon(m.tank_shape, color=self.tank_color)
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
                if a == 0.0:
                    surface = v._window
                    for py in np.arange(min_y - 1, self.th_max + min_y, 1):
                        for px in np.arange(min_x - 1, self.tw_max + min_x, 1):
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
            self.plot_odorscape(save_to=m.save_to, show=show, scale=self.s, idx=self.odorscape_counter)
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
                color = cNs.random_colors(1)[0] if self.random_colors else f.default_color
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
        self.sim_scale = SimulationScale(self.model.arena_dims[0] * screen.zoom, color=self.sim_scale.color)
        self.sim_scale.render_scale(*self.window_dims)



    @property
    def snapshot_tick(self):
        return (self.model.Nticks - 1) % self.snapshot_interval == 0


    def add_screen_texts(self, names, color):
        for name in names:
            text = InputBox(text=name, color_active=color, color_inactive=color)
            self.screen_texts[name] = text

    def generate_larva_color(self):
        return cNs.random_colors(1)[0] if self.random_colors else self.default_larva_color

    # @property
    # def configuration_text(self):
    #     text = f"Simulation configuration : \n" \
    #            "\n" \
    #            f"Experiment : {self.experiment}\n" \
    #            f"Simulation ID : {self.id}\n" \
    #            f"Duration (min) : {self.duration}\n" \
    #            f"Timestep (sec) : {self.dt}\n" \
    #            f"Parent path : {self.save_to}"
    #     return text



    def plot_odorscape(self, scale=1.0, idx=0, **kwargs):
        from lib.plot.scape import plot_surface
        for id, layer in self.model.odor_layers.items():
            X, Y = layer.meshgrid
            x = X * 1000 / scale
            y = Y * 1000 / scale
            plot_surface(x=x, y=y, z=layer.get_grid(), vars=[r'x $(mm)$', r'y $(mm)$'], target=r'concentration $(μM)$',
                         title=f'{id} odorscape', save_as=f'{id}_odorscape_{idx}', **kwargs)

    def eliminate_overlap(self):
        pass


