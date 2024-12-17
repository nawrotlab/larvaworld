"""
Screen management for pygame-based simulation visualization
"""

import math
import os
import sys
import agentpy
import imageio
import numpy as np
import pandas as pd
import param
from param import Boolean, String
from shapely import geometry

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from ... import ROOT_DIR, vprint
from .. import reg, util

from ..param import (
    Area2DPixel,
    NestedConf,
    OptionalSelector,
    PositiveInteger,
    PositiveNumber,
)

from ..screen import (
    ScreenMsgText,
    ScreenTextBoxRect,
    SidePanel,
    SimulationClock,
    SimulationScale,
    SimulationState,
)

__all__ = [
    "MediaDrawOps",
    "AgentDrawOps",
    "ColorDrawOps",
    "ScreenOps",
    "GA_ScreenManager",
    "ScreenManager",
]


class MediaDrawOps(NestedConf):
    """
    Options regarding the media (videos,images) to be stored during simulation.
    """

    image_mode = OptionalSelector(
        objects=["final", "snapshots", "overlap"], doc="When to save images."
    )
    image_file = String(
        None,
        doc="Filename for the saved image. File extension png sutomatically added.",
    )
    snapshot_interval_in_sec = PositiveInteger(
        60, softmax=100, doc="Sec between snapshots"
    )
    video_file = String(
        None,
        doc="Filename for the saved video. File extension mp4 sutomatically added.",
    )
    media_dir = String(
        None,
        doc="Directory where to save media. Defaults tp model.dir if not provided.",
    )
    fps = PositiveInteger(60, softmax=100, doc="Video speed")
    save_video = Boolean(False, doc="Whether to save a video.")
    vis_mode = OptionalSelector(objects=["video", "image"], doc="Screen mode.")
    show_display = Boolean(False, doc="Whether to launch the pygame-visualization.")

    @property
    def active(self):
        return (
            self.save_video
            or self.image_mode
            or self.show_display
            or (self.vis_mode is not None)
        )

    @property
    def video_filepath(self):
        if self.media_dir is not None and self.video_file is not None:
            return f"{self.media_dir}/{self.video_file}.mp4"
        else:
            return None

    @property
    def image_filepath(self):
        if self.media_dir is not None and self.image_file is not None:
            return f"{self.media_dir}/{self.image_file}.png"
        else:
            return None

    @property
    def overlap_mode(self):
        return self.image_mode == "overlap"

    def new_video_writer(self, fps, video_filepath=None):
        if self.save_video:
            if video_filepath is None:
                video_filepath = self.video_filepath
            os.makedirs(self.media_dir, exist_ok=True)
            vid_writer = imageio.get_writer(video_filepath, mode="I", fps=fps)
            vprint(f"Video will be saved as {video_filepath}", 1)
        else:
            vid_writer = None
        return vid_writer

    def new_image_writer(self, image_filepath=None):
        if self.image_mode:
            if image_filepath is None:
                image_filepath = self.image_filepath
            os.makedirs(self.media_dir, exist_ok=True)
            img_writer = imageio.get_writer(image_filepath, mode="i")
            vprint(f"Image will be saved as {image_filepath}", 1)
        else:
            img_writer = None
        return img_writer


class AgentDrawOps(NestedConf):
    """
    Options for drawing the agents on the screen.
    """

    visible_trails = Boolean(False, doc="Draw the larva trajectories")
    trail_dt = PositiveNumber(20, step=0.2, doc="Duration of the drawn trajectories")
    trail_color = param.Selector(
        objects=["normal", "linear", "angular"],
        doc="Whether to display larva tracks according to the instantaneous forward or angular velocity.",
    )
    draw_sensors = Boolean(False, doc="Draw the larva sensors")
    draw_contour = Boolean(True, doc="Draw the larva contour")
    draw_segs = Boolean(True, doc="Draw the larva body segments")
    draw_midline = Boolean(True, doc="Draw the larva midline")
    draw_centroid = Boolean(False, doc="Draw the larva centroid")
    draw_head = Boolean(False, doc="Draw the larva head")
    draw_orientations = Boolean(False, doc="Draw the larva body vector orientations")


class ColorDrawOps(NestedConf):
    """
    Options regarding coloring.
    """

    intro_text = Boolean(True, doc="Show the introductory configuration screen")
    odor_aura = Boolean(False, doc="Draw the aura around odor sources")
    allow_clicks = Boolean(True, doc="Whether to allow input from display")
    black_background = Boolean(False, doc="Set the background color to black")
    random_colors = Boolean(False, doc="Color each larva with a random color")
    color_behavior = Boolean(
        False, doc="Color the larvae according to their instantaneous behavior"
    )
    panel_width = PositiveInteger(0, doc="The width of the side panel in pixels")


class ScreenOps(ColorDrawOps, AgentDrawOps, MediaDrawOps):
    pass


class ScreenArea(Area2DPixel):
    def __init__(self, model, **kwargs):
        self.model = model
        self.space_dims = self.model.p.env_params.arena.dims
        super().__init__(dims=util.get_window_dims(self.space_dims), **kwargs)

    def space2screen_pos(self, pos):
        return self.adjust_pos_to_area(
            pos=pos, area=self.model.space, scaling_factor=self.model.scaling_factor
        )

    def get_rect_at_screen_pos(self, pos=(0, 0)):
        return self.get_rect_at_pos(self.space2screen_pos(pos))


class ScreenAreaZoomable(ScreenArea):
    zoom = PositiveNumber(1.0, doc="Zoom factor")
    center = param.Parameter(np.array([0.0, 0.0]), doc="Center xy")
    center_lim = param.Parameter(np.array([0.0, 0.0]), doc="Center xy lim")
    _scale = param.Parameter(np.array([[1.0, 0.0], [0.0, -1.0]]), doc="Scale of xy")
    _translation = param.Parameter(np.zeros(2), doc="Translation of xy")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_bounds()

    @property
    def display_size(self):
        return (np.array(self.dims) / self.zoom).astype(int)

    @param.depends("zoom", "center", watch=True)
    def set_bounds(self):
        s, z = self.model.scaling_factor, self.zoom
        rw, rh = self.w / self.space_dims[0], self.h / self.space_dims[1]
        self._scale = np.array([[rw, 0.0], [0.0, -rh]]) / z / s
        self._translation = np.array(self.dims) / 2 + self.center / z / s * [-rw, rh]
        self.center_lim = (z - 1) * s * np.array(self.space_dims) / 2

    def _transform(self, position):
        return np.round(self._scale.dot(position) + self._translation).astype(int)

    def move_center(self, dx=0, dy=0, pos=None):
        if pos is None:
            pos = self.center - self.center_lim * [dx, dy]
        self.center = np.clip(pos, self.center_lim, -self.center_lim)

    def zoom_screen(self, sign, pos=None):
        d_zoom = -0.01 * sign
        if pos is None:
            pos = self.mouse_position
        if 0.001 <= self.zoom + d_zoom <= 1:
            self.zoom = np.round(self.zoom + d_zoom, 2)
            self.center = np.clip(
                self.center - np.array(pos) * d_zoom, self.center_lim, -self.center_lim
            )
        if self.zoom == 1.0:
            self.center = np.array([0.0, 0.0])

    @param.depends("zoom", watch=True)
    def update_scale(self):
        def closest(lst, k):
            return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]

        def compute_lines(x, y, scale):
            return [
                [(x - scale / 2, y), (x + scale / 2, y)],
                [(x + scale / 2, y * 0.75), (x + scale / 2, y * 1.25)],
                [(x - scale / 2, y * 0.75), (x - scale / 2, y * 1.25)],
            ]

        w_in_mm = self.space_dims[0] * self.zoom * 1000
        # Get 1/10 of max real dimension, transform it to mm and find the closest reasonable scale
        scale_in_mm = closest(
            lst=[
                0.1,
                0.25,
                0.5,
                0.75,
                1,
                2.5,
                5,
                7.5,
                10,
                25,
                50,
                75,
                100,
                250,
                500,
                750,
                1000,
            ],
            k=w_in_mm / 10,
        )
        try:
            S = self.screen_scale
            S.text_font.set_text(f"{scale_in_mm} mm")
            S.lines = compute_lines(S.x, S.y, scale_in_mm / w_in_mm * self.w)
        except:
            pass


class ScreenAreaPygame(ScreenAreaZoomable, ScreenOps):
    caption = param.String("", doc="The caption of the screen window")
    scene = param.String(None, doc="The scene ID to be loaded from file")

    def __init__(self, background_motion=None, **kwargs):
        super().__init__(**kwargs)
        self.bg = background_motion

        m = self.model
        if m.offline:
            self.show_display = False
        if self.video_file is None:
            self.video_file = str(m.id)
        if self.image_file is None:
            self.image_file = str(m.id)
        if self.media_dir is None:
            self.media_dir = m.dir
        self._fps = int(self.fps / m.dt)
        if self.vis_mode == "video" and not self.save_video:
            self.show_display = True

        if self.caption is None:
            self.caption = str(m.id)

        pygame.init()
        os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (1550, 400)
        self.v = self.init_screen()
        # self._t = pygame.time.Clock()

        if self.bg is not None:
            self.set_background()
        else:
            self.bgimage = None
            self.bgimagerect = None

    @property
    def mouse_position(self):
        p = np.array(pygame.mouse.get_pos()) - self._translation
        return np.linalg.inv(self._scale).dot(p)

    @property
    def new_display_surface(self):
        return pygame.Surface(self.display_size, pygame.SRCALPHA)

    def _draw_arena(self, tank_color, screen_color):
        surf1 = self.new_display_surface
        surf2 = self.new_display_surface
        vs = [self._transform(v) for v in self.model.space.vertices]
        pygame.draw.polygon(surf1, tank_color, vs, 0)
        pygame.draw.rect(surf2, screen_color, surf2.get_rect())
        surf2.blit(surf1, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        self.v.blit(surf2, (0, 0))

    def init_screen(self):
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if self.show_display:
            v = pygame.display.set_mode((self.w + self.panel_width, self.h), flags)
            pygame.display.set_caption(self.caption)
            pygame.event.set_allowed(pygame.QUIT)
        else:
            v = pygame.Surface(self.display_size, flags)
        return v

    def draw_circle(
        self, position=(0, 0), radius=0.1, color=(0, 0, 0), filled=True, width=0.01
    ):
        p = self._transform(position)
        r = int(self._scale[0, 0] * radius)
        w = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(self.v, color, p, r, w)

    def draw_polygon(self, vertices, color=(0, 0, 0), filled=True, width=0.01):
        if vertices is not None and len(vertices) > 1:
            vs = [self._transform(v) for v in vertices]
            w = 0 if filled else int(self._scale[0, 0] * width)
            pygame.draw.polygon(self.v, color, vs, w)

    def draw_convex(self, points, **kwargs):
        from scipy.spatial import ConvexHull

        ps = np.array(points)
        vs = ps[ConvexHull(ps).vertices].tolist()
        self.draw_polygon(vs, **kwargs)

    def draw_grid(self, all_vertices, colors, filled=True, width=0.01):
        all_vertices = [
            [self._transform(v) for v in vertices] for vertices in all_vertices
        ]
        w = 0 if filled else int(self._scale[0, 0] * width)
        for vs, c in zip(all_vertices, colors):
            pygame.draw.polygon(self.v, c, vs, w)

    def draw_polyline(self, vertices, color=(0, 0, 0), closed=False, width=0.01):
        vs = [self._transform(v) for v in vertices]
        w = int(self._scale[0, 0] * width)
        if isinstance(color, list):
            for v1, v2, c in zip(vs[:-1], vs[1:], color):
                pygame.draw.lines(self.v, c, closed=closed, points=[v1, v2], width=w)
        else:
            pygame.draw.lines(self.v, color, closed=closed, points=vs, width=w)

    def draw_line(self, start, end, color=(0, 0, 0), width=0.01):
        start = self._transform(start)
        end = self._transform(end)
        w = int(self._scale[0, 0] * width)
        pygame.draw.line(self.v, color, start, end, w)

    def draw_transparent_circle(
        self, position=(0, 0), radius=0.1, color=(0, 0, 0, 125), filled=True, width=0.01
    ):
        r = int(self._scale[0, 0] * radius)
        s = pygame.Surface((2 * r, 2 * r), pygame.HWSURFACE | pygame.SRCALPHA)
        w = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(s, color, (r, r), radius, w)
        self.v.blit(s, self._transform(position) - r)

    def draw_text_box(self, font, rect):
        self.v.blit(font, rect)

    def draw_envelope(self, points, **kwargs):
        vs = list(geometry.MultiPoint(points).envelope.exterior.coords)
        self.draw_polygon(vs, **kwargs)

    def draw_arrow_line(
        self, start, end, color=(0, 0, 0), width=0.01, dl=0.02, phi=0, s=10
    ):
        a0 = math.atan2(end[1] - start[1], end[0] - start[0])
        l0 = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        w = int(self._scale[0, 0] * width)
        pygame.draw.line(self.v, color, self._transform(start), self._transform(end), w)

        a = a0 + np.pi / 2
        sin0, cos0 = math.sin(a) * s, math.cos(a) * s
        sin1, cos1 = math.sin(a - np.pi * 2 / 3) * s, math.cos(a - np.pi * 2 / 3) * s
        sin2, cos2 = math.sin(a + np.pi * 2 / 3) * s, math.cos(a + np.pi * 2 / 3) * s

        l = 0 + phi * dl
        while l < l0:
            pos = self._transform(
                (start[0] + math.cos(a0) * l, start[1] + math.sin(a0) * l)
            )
            p0 = (pos[0] + sin0, pos[1] + cos0)
            p1 = (pos[0] + sin1, pos[1] + cos1)
            p2 = (pos[0] + sin2, pos[1] + cos2)
            pygame.draw.polygon(self.v, color, (p0, p1, p2))
            l += dl

    def set_background(self):
        path = f"{ROOT_DIR}/lib/screen/background.png"
        print("Loading background image from", path)
        self.bgimage = pygame.image.load(path)
        self.bgimagerect = self.bgimage.get_rect()
        self.tw = self.bgimage.get_width()
        self.th = self.bgimage.get_height()
        self.th_max = int(self.v.get_height() / self.th) + 2
        self.tw_max = int(self.v.get_width() / self.tw) + 2

    def draw_background(self, bg=[0, 0, 0]):
        if self.bgimage is not None and self.bgimagerect is not None:
            x, y, a = bg
            try:
                min_x = int(np.floor(x))
                min_y = -int(np.floor(y))

                for py in np.arange(min_y - 1, self.th_max + min_y, 1):
                    for px in np.arange(min_x - 1, self.tw_max + min_x, 1):
                        if a != 0.0:
                            pass
                        p = ((px - x) * (self.tw - 1), (py + y) * (self.th - 1))
                        self.v.blit(self.bgimage, p)
            except:
                pass


class ScreenManager(ScreenAreaPygame):
    """
    Base class managing the pygame screen.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_type = ""
        self.selected_agents = []
        self.selection_color = "red"

        self.dynamic_graphs = []
        self.focus_mode = False

        self.snapshot_interval = int(self.snapshot_interval_in_sec / self.model.dt)

        self.snapshot_counter = 0
        self.odorscape_counter = 0

        self.pygame_keys = None

        self.vid_writer = None
        self.img_writer = None
        self.initialized = False

    def increase_fps(self):
        if self._fps < 60:
            self._fps += 1
        vprint(f"viewer.fps: {self._fps}", 1)

    def decrease_fps(self):
        if self._fps > 1:
            self._fps -= 1
        vprint(f"viewer.fps: {self._fps}", 1)

    def draw_agents(self):
        """
        Draw the agents on the screen
        """
        for o in self.model.sources:
            o._draw(v=self)
        for g in self.model.agents:
            g._draw(v=self)

    def check(self, **kwargs):
        """
        Check whether to initialize or close the display
        """
        # if self.v is None:
        if self.active and not self.initialized:
            self.initialize(**kwargs)
        elif self.close_requested():
            self.close()

    def close(self):
        """
        Close the pygame display
        """
        pygame.display.quit()
        if self.vid_writer:
            self.vid_writer.close()
        if self.img_writer:
            self.img_writer.close()
        vprint("Screen closed", 1)
        self.model.running = False
        vprint("Terminated by the user", 3)
        return

    @staticmethod
    def close_requested():
        if pygame.display.get_init():
            return pygame.event.peek(pygame.QUIT)
        return False

    def render(self, **kwargs):
        """
        Draw the display and evaluate user-input
        """
        if self.active:
            self.check(**kwargs)
            if not self.overlap_mode:
                self.draw_arena()

            self.draw_agents()
            if self.show_display:
                self.evaluate_input()
                self.evaluate_graphs()
            if not self.overlap_mode:
                self._draw_arena(self.tank_color, self.screen_color)
                self.draw_aux()
                self._render()

    def _render(self):
        if self.show_display:
            pygame.display.flip()
            image = pygame.surfarray.pixels3d(self.v)
            # self._t.tick(self.manager._fps)
        else:
            image = pygame.surfarray.array3d(self.v)
        if self.vid_writer:
            self.vid_writer.append_data(np.flipud(np.rot90(image)))
        if self.img_writer:
            self.img_writer.append_data(np.flipud(np.rot90(image)))
            self.img_writer = None
        return image

    def initialize(self, **kwargs):
        """
        Initialize the pygame display
        """
        self.vid_writer = self.new_video_writer(fps=self._fps)
        self.img_writer = self.new_image_writer()
        if self.scene is not None:
            path = f"{ROOT_DIR}/lib/sim/ga_scenes/{self.scene}.txt"
            self.model.objects = agentpy.AgentList(
                model=self.model, objs=self.load_scene_from_file(path, m=self.model)
            )

        self.build_aux()
        self.draw_arena()
        self.initialized = True
        vprint("Screen opened", 1)

    def evaluate_graphs(self):
        """
        Evaluation of dynamic graphs on the screen.
        """
        for g in self.dynamic_graphs:
            running = g.evaluate()
            if not running:
                self.dynamic_graphs.remove(g)
                del g

    @property
    def screen_color(self):
        return (200, 200, 200) if not self.black_background else (50, 50, 50)

    @property
    def tank_color(self):
        return util.Color.WHITE if not self.black_background else util.Color.BLACK

    @property
    def sidepanel_color(self):
        return util.Color.BLACK if not self.black_background else util.Color.WHITE

    @property
    def snapshot_tick(self):
        return (self.model.Nticks - 1) % self.snapshot_interval == 0

    @property
    def snapshot_valid(self):
        return (
            self.vis_mode == "image"
            and self.image_mode == "snapshots"
            and self.snapshot_tick
        )

    @property
    def render_valid(self):
        m = self.vis_mode
        return (m == "image" and self.overlap_mode) or (
            m == "video" and (self.image_mode != "snapshots" or self.snapshot_tick)
        )

    def step(self):
        self.check()
        if self.active:
            self.screen_clock.tick_clock()
            if self.render_valid:
                self.render()
            if self.snapshot_valid:
                self.capture_snapshot()

    def draw_arena_tank(self):
        """
        Draw the tank of the arena with optional background
        """
        self.draw_polygon(self.model.space.vertices, color=self.tank_color)
        self.draw_background(
            self.bg[:, self.model.t - 1] if self.bg is not None else [0, 0, 0]
        )

    def toggle(self, name, value=None, show=False, minus=False, plus=False, disp=None):
        """
        Presentation of user-input-induced changes on screen
        """
        m = self.model
        if disp is None:
            disp = name
        if name == "snapshot #":
            self.img_writer = self.new_image_writer(
                f"{self.caption}_at_{int(m.Nticks * m.dt)}_sec.png"
            )
            value = self.snapshot_counter
            self.snapshot_counter += 1
        elif name == "odorscape #":
            reg.graphs.dict["odorscape"](
                odor_layers=m.odor_layers,
                save_to=m.plot_dir,
                show=show,
                scale=m.scaling_factor,
                idx=self.odorscape_counter,
            )
            value = self.odorscape_counter
            self.odorscape_counter += 1
        elif name == "trail_dt":
            if minus:
                dt = -1
            elif plus:
                dt = +1
            self.trail_dt = np.clip(self.trail_dt + 5 * dt, a_min=0, a_max=np.inf)
            value = self.trail_dt
        elif name == "trail_color":
            obs = self.param.trail_color.objects
            self.trail_color = obs[(obs.index(self.trail_color) + 1) % len(obs)]
            value = self.trail_color

        if value is None:
            setattr(self, name, not getattr(self, name))
            value = "ON" if getattr(self, name) else "OFF"

        self.screen_texts[name].flash_text(f"{disp} {value}")

        if name == "random_colors":
            for f in m.agents:
                color = (
                    util.random_colors(1)[0] if self.random_colors else f.default_color
                )
                f.set_default_color(color)
        elif name == "black_background":
            for a in (
                m.get_all_objects()
                + [self.screen_clock, self.screen_scale, self.screen_state]
                + list(self.screen_texts.values())
            ):
                a.invert_default_color()
        # elif name == 'larva_collisions':
        #
        #     m.eliminate_overlap()

    def evaluate_input(self):
        """
        Evaluation of user input through keyboard and mouse.
        """
        if self.pygame_keys is None:
            self.pygame_keys = reg.controls.load()["pygame_keys"]

        ev = pygame.event.get()
        for e in ev:
            if e.type == pygame.QUIT:
                self.close()
                sys.exit()

            elif e.type == pygame.KEYDOWN and (e.key == 93 or e.key == 270):
                self.increase_fps()
            elif e.type == pygame.KEYDOWN and (e.key == 47 or e.key == 269):
                self.decrease_fps()

            if e.type == pygame.KEYDOWN:
                for k, v in self.pygame_keys.items():
                    if e.key == getattr(pygame, v):
                        self.eval_keypress(k)

            if self.allow_clicks:
                if e.type == pygame.MOUSEWHEEL:
                    self.zoom_screen(e.y, pos=self.mouse_position)
                    self.toggle(name="zoom", value=self.zoom)
                elif e.type == pygame.MOUSEBUTTONUP:
                    if e.button == 1:
                        if not self.eval_selection(
                            p=self.mouse_position,
                            ctrl=pygame.key.get_mods() & pygame.KMOD_CTRL,
                        ):
                            #     self.model.add_agent(agent_class=self.selected_type, p0=tuple(p),
                            #                 p1=tuple(self.mousebuttondown_pos))
                            pass

                    elif e.button == 3:
                        from ...gui.gui_aux.windows import (
                            object_menu,
                            set_agent_kwargs,
                        )

                        loc = tuple(
                            np.array(
                                [
                                    int(x)
                                    for x in os.environ["SDL_VIDEO_WINDOW_POS"].split(
                                        ","
                                    )
                                ]
                            )
                            + np.array(pygame.mouse.get_pos())
                        )
                        if len(self.selected_agents) > 0:
                            for sel in self.selected_agents:
                                sel = set_agent_kwargs(sel, location=loc)
                        else:
                            self.selected_type = object_menu(
                                self.selected_type, location=loc
                            )

        if self.focus_mode and len(self.selected_agents) > 0:
            try:
                sel = self.selected_agents[0]
                self.move_center(pos=sel.get_position())
            except:
                pass
        # print(self.selected_agents)

    def eval_keypress(self, k):
        """
        Evaluation of keyboard input.
        """
        m = self.model
        if k == "visible_ids":
            for a in m.agents + m.sources:
                temp = a.id_box.toggle_vis()
            self.toggle(k, "ON" if temp else "OFF", disp="IDs")
        elif k == "visible_clock":
            vis = self.screen_clock.toggle_vis()
            self.toggle(k, "ON" if vis else "OFF", disp="clock")
        elif k == "visible_scale":
            vis = self.screen_scale.toggle_vis()
            self.toggle(k, "ON" if vis else "OFF", disp="scale")
        elif k == "visible_state":
            vis = self.screen_state.toggle_vis()
            self.toggle(k, "ON" if vis else "OFF", disp="state")
        elif k == "▲ trail duration":
            self.toggle("trail_dt", plus=True, disp="trail duration")
        elif k == "▼ trail duration":
            self.toggle("trail_dt", minus=True, disp="trail duration")
        elif k == "visible_trails":
            self.toggle(k, disp="trails")
        elif k == "pause":
            m.is_paused = not m.is_paused
            self.toggle("is_paused", "ON" if m.is_paused else "OFF")
        elif k == "move left":
            self.move_center(-0.05, 0)
        elif k == "move right":
            self.move_center(+0.05, 0)
        elif k == "move up":
            self.move_center(0, +0.05)
        elif k == "move down":
            self.move_center(0, -0.05)
        elif k == "plot odorscapes":
            self.toggle("odorscape #", show=pygame.key.get_mods() & pygame.KMOD_CTRL)
        elif "odorscape" in k:
            idx = int(k.split(" ")[-1])
            try:
                layer_id = list(m.odor_layers.keys())[idx]
                layer = m.odor_layers[layer_id]
                vis = layer.toggle_vis()
                self.toggle(layer_id, "ON" if vis else "OFF")
            except:
                pass
        elif k == "snapshot":
            self.toggle("snapshot #")
        elif k == "windscape":
            try:
                vis = m.windscape.toggle_vis()
                self.toggle("windscape", "ON" if vis else "OFF")
            except:
                pass
        elif k == "delete item":
            from ...gui.gui_aux.windows import delete_objects_window

            if delete_objects_window(self.selected_agents):
                for f in self.selected_agents:
                    self.selected_agents.remove(f)
                    m.delete_agent(f)
        elif k == "dynamic graph":
            from ..model.agents._larva import Larva

            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
                if isinstance(sel, Larva):
                    from ...gui.gui_aux import DynamicGraph

                    self.dynamic_graphs.append(DynamicGraph(agent=sel))
        elif k == "odor gains":
            if len(self.selected_agents) > 0:
                sel = self.selected_agents[0]
                from ..model.agents._larva_sim import LarvaSim

                if isinstance(sel, LarvaSim) and sel.brain.olfactor is not None:
                    from ...gui.gui_aux.windows import set_kwargs

                    sel.brain.olfactor.gain = set_kwargs(
                        sel.brain.olfactor.gain, title="Odor gains"
                    )
        elif k == "larva_collisions":
            m.larva_collisions = not m.larva_collisions
            # m.eliminate_overlap()
        else:
            self.toggle(k)

    def eval_selection(self, p, ctrl):
        """
        Selection of items on the screen by mouse-clicks.
        """
        res = False if len(self.selected_agents) == 0 else True
        for f in self.model.get_all_objects():
            if f.contained(p):
                if not f.selected:
                    f.selected = True
                    self.selected_agents.append(f)
                elif ctrl:
                    f.selected = False
                    self.selected_agents.remove(f)
                res = True
            elif f.selected and not ctrl:
                f.selected = False
                self.selected_agents.remove(f)
        return res

    def build_aux(self):
        """
        Generate additional items on screen
        """
        m = self.model
        self.input_box = ScreenTextBoxRect(
            text_color="lightgreen",
            color="white",
            frame_rect=self.get_rect_at_screen_pos(),
            font_type="comicsansms",
            font_size=40,
        )
        if self.intro_text:
            box = ScreenTextBoxRect(
                text=m.configuration_text,
                text_color="lightgreen",
                color="white",
                visible=True,
                frame_rect=self.get_rect_at_screen_pos(),
                font_type="comicsansms",
                font_size=30,
            )
            box.draw(self)
            self._render()
            pygame.time.wait(2000)
            box.visible = False
            self.draw_arena_tank()

        kws = {
            "reference_area": self,
            "color": self.sidepanel_color,
        }

        self.screen_clock = SimulationClock(
            sim_step_in_sec=m.dt, pos=self.item_pos("clock"), **kws
        )
        self.screen_scale = SimulationScale(pos=self.item_pos("scale"), **kws)
        self.screen_state = SimulationState(model=m, pos=self.item_pos("state"), **kws)
        self.screen_texts = util.AttrDict(
            {
                name: ScreenMsgText(text=name, **kws)
                for name in [
                    "trail_dt",
                    "trail_color",
                    "visible_trails",
                    "focus_mode",
                    "draw_centroid",
                    "draw_head",
                    "draw_midline",
                    "draw_contour",
                    "draw_sensors",
                    "draw_orientations",
                    "draw_segs",
                    "visible_clock",
                    "visible_ids",
                    "visible_state",
                    "visible_scale",
                    "odor_aura",
                    "color_behavior",
                    "random_colors",
                    "black_background",
                    "larva_collisions",
                    "zoom",
                    "snapshot #",
                    "odorscape #",
                    "windscape",
                    "is_paused",
                ]
                + list(m.odor_layers.keys())
            }
        )

        self.side_panel = SidePanel(self) if self.panel_width > 0 else None

    def capture_snapshot(self):
        """
        Capture an image snapshot of the current display
        """
        self.render()
        self.toggle("snapshot #")
        self._render()

    def draw_arena(self):
        """
        Draw the arena and sensory landscapes
        """
        self.draw_arena_tank()
        m = self.model
        arena_drawn = False
        for id, layer in m.odor_layers.items():
            if layer.visible:
                layer.draw(self)
                arena_drawn = True
                break

        if not arena_drawn and m.food_grid is not None:
            m.food_grid._draw(v=self)

        if m.windscape is not None:
            m.windscape._draw(v=self)

        for b in m.borders:
            b._draw(v=self)

    def item_pos(self, item):
        item_pos_scale = util.AttrDict(
            {
                "clock": (0.85, 0.94),
                "scale": (0.1, 0.04),
                "state": (0.85, 0.94),
            }
        )
        assert item in item_pos_scale
        return self.get_relative_pos(item_pos_scale[item])

    def item_textfonts(self):
        rel_pos = {
            "clock": [
                (0.85, 0.94),
                [(0.91, 1.0), (0.95, 1.0), (1.0, 1.0), (1.04, 1.1)],
                [(1 / 40), (1 / 40), (1 / 50), (1 / 50)],
            ],
            "scale": [(0.1, 0.04), [(1, 1.5)], [(1 / 40)]],
            "state": [(0.85, 0.94), [(1, 1)], [(1 / 40)]],
        }
        rel = pd.DataFrame.from_dict(
            rel_pos,
            columns=["pos2screen", "text2pos", "fontsize2screen"],
            orient="index",
        )
        rel["pos"] = rel["pos2screen"].apply(self.get_relative_pos)

        def temp(alist):
            return [self.get_relative_font_size(aa) for aa in alist]

        rel["font_size"] = rel["fontsize2screen"].apply(temp)

        def temp2(alist, p):
            return [self.get_relative_pos(aa, reference=p) for aa in alist]

        rel["text_center"] = rel[["text2pos", "pos"]].apply(temp2)

    def draw_aux(self):
        """
        Draw additional items on screen
        """
        try:
            for t in [self.screen_clock, self.screen_scale, self.screen_state]:
                t._draw(self)
            for t in list(self.screen_texts.values()) + [self.input_box]:
                t.visible = t.start_time < pygame.time.get_ticks() < t.end_time
                t._draw(self)
        except:
            pass

        if self.side_panel is not None:
            self.side_panel.draw(self)

    def load_scene_from_file(self, file_path, m):
        from ..model.envs.obstacle import Box, Wall

        obs = []
        with open(file_path) as f:
            n = 1
            for line in f:
                ws = line.split()

                # skip empty lines
                if len(ws) == 0:
                    n += 1
                    continue

                # skip comments in file
                if ws[0][0] == "#":
                    n += 1
                    continue

                if ws[0] == "Box":
                    obs.append(
                        Box(
                            x=int(ws[1]),
                            y=int(ws[2]),
                            size=int(ws[3]),
                            model=m,
                            color="lightgreen",
                            unique_id=f"Box_{n}",
                        )
                    )
                elif ws[0] == "Wall":
                    obs.append(
                        Wall(
                            point1=geometry.Point(int(ws[1]), int(ws[2])),
                            point2=geometry.Point(int(ws[3]), int(ws[4])),
                            model=m,
                            color="lightgreen",
                            unique_id=f"Wall_{n}",
                        )
                    )
                elif ws[0] == "Light":
                    from ..model.modules.rot_surface import LightSource

                    obs.append(
                        LightSource(
                            x=int(ws[1]),
                            y=int(ws[2]),
                            emitting_power=int(ws[3]),
                            model=m,
                            unique_id=f"LightSource_{n}",
                        )
                    )

                n += 1

        return obs

    def finalize(self):
        """
        Apply final actions before closing the screen manager
        """
        if self.active:
            if self.overlap_mode:
                self._render()
                pygame.time.wait(5000)
            elif self.image_mode == "final":
                self.capture_snapshot()
        self.close()


class GA_ScreenManager(ScreenManager):
    """
    Screen manager for the Genetic Algorithm simulations.
    """

    def __init__(
        self, model, black_background=True, panel_width=600, scene="no_boxes", **kwargs
    ):
        super().__init__(
            model=model,
            black_background=black_background,
            panel_width=panel_width,
            caption=f"GA {model.experiment} : {model.id}",
            scene=scene,
            **kwargs,
        )
