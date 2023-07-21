import math
import os
import numpy as np
import imageio
import param
from shapely import geometry

from larvaworld.lib.param import Viewable, PointPositioned, ViewableGroupedObject, PositiveRange, PositiveNumber, \
    PositiveIntegerRange, PointRelPositioned

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from larvaworld.lib import aux




class Viewer(object):
    def __init__(self, model, window_dims, caption="", fps=10, dt=0.1, record_video_to=None,
                 record_image_to=None, panel_width=0):
        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (1550, 400)
        self.w_loc = [int(x) for x in os.environ['SDL_VIDEO_WINDOW_POS'].split(',')]
        # self.speed = speed

        self.model = model
        self.space_bounds = self.model.space.scaled_range
        self.zoom = 1
        self.caption = caption
        self.width,self.height = window_dims
        self.window_dims = window_dims
        self.panel_width = panel_width
        self.panel_rect = pygame.Rect(self.width, 0, self.panel_width, self.height)
        self._t = pygame.time.Clock()
        self._fps = fps
        self.dt = dt
        self.center = np.array([0.0, 0.0])
        self.center_lim = np.array([0.0, 0.0])
        self.snapshot_requested=None
        self.display_size = self.zoomed_window_dims
        self._window = self.init_screen()
        self.objects = []


        if record_video_to:
            self._video_writer = imageio.get_writer(record_video_to, mode='I', fps=self._fps)
        else:
            self._video_writer = None

        if record_image_to:

            self._image_writer = imageio.get_writer(record_image_to, mode='i')
        else:
            self._image_writer = None

        self._scale = np.array([[1., .0], [.0, -1.]])
        self._translation = np.zeros(2)
        self.set_bounds(*self.space_bounds)

    def increase_fps(self):
        if self._fps < 60:
            self._fps += 1
        print('viewer.fps:', self._fps)

    def decrease_fps(self):
        if self._fps > 1:
            self._fps -= 1
        print('viewer.fps:', self._fps)

    def put(self, obj):
        if isinstance(obj, list):
            self.objects.extend(obj)
        else:
            self.objects.append(obj)

    def remove(self, obj):
        self.objects.remove(obj)

    def save(self, filename_pattern='scene', file_path='saved_scenes/'):
        date_time = aux.TimeUtil.format_date_time()
        file_name = filename_pattern + '_' + date_time + ".txt"
        file_path = file_path + file_name

        with open(file_path, 'w') as f:
            line1 = '# First uncommented line must starts with "Scene"'
            line2 = '# This is the syntax for each kind of object:'
            line3 = '# Scene WIDTH HEIGHT'
            line4 = '# Wall X1 Y1 X2 Y2'
            line5 = '# Box X Y SIZE'
            line6 = '# Light X Y EMITTING_POWER'

            f.write(line1 + '\n')
            f.write(line2 + '\n')
            f.write(line3 + '\n')
            f.write(line4 + '\n')
            f.write(line5 + '\n')
            f.write(line6 + '\n')
            f.write('\n')

            f.write(self.get_saved_scene_repr() + '\n')  # scene size


        f.closed
        print('Scene saved:', file_path)

    def get_saved_scene_repr(self):
        return self.__class__.__name__ + ' ' + str(self.width) + ' ' + str(self.height)

    def draw_panel_rect(self):
        pygame.draw.rect(self._window, aux.Color.BLACK, self.panel_rect)

    @ property
    def display_dims(self):
        return self._window.get_width(), self._window.get_height()

    def draw_arena(self, tank_color, screen_color):
        surf1 = pygame.Surface(self.display_size, pygame.SRCALPHA)
        surf2 = pygame.Surface(self.display_size, pygame.SRCALPHA)
        vs = [self._transform(v) for v in self.model.space.vertices]
        pygame.draw.polygon(surf1, tank_color, vs, 0)
        pygame.draw.rect(surf2, screen_color, surf2.get_rect())
        surf2.blit(surf1, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        self._window.blit(surf2, (0, 0))

    def init_screen(self):
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF
        if self.model.show_display:
            window = pygame.display.set_mode((self.width + self.panel_width, self.height), flags)
            pygame.display.set_caption(self.caption)
            pygame.event.set_allowed(pygame.QUIT)
        else:
            window = pygame.Surface(self.display_size, flags)
        return window


    @ property
    def zoomed_window_dims(self):
        return (np.array(self.window_dims) / self.zoom).astype(int)


    def zoom_screen(self, d_zoom, pos=None):
        if pos is None:
            pos = self.mouse_position
        if 0.001 <= self.zoom + d_zoom <= 1:
            self.zoom = np.round(self.zoom + d_zoom, 2)
            self.display_size = self.zoomed_window_dims
            self.center = np.clip(self.center - pos * d_zoom, self.center_lim, -self.center_lim)
        if self.zoom == 1.0:
            self.center = np.array([0.0, 0.0])
        self.set_bounds(*self.space_bounds)


    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        x = self.display_size[0] / (right - left)
        y = self.display_size[1] / (top - bottom)
        self._scale = np.array([[x, .0], [.0, -y]])
        self._translation = np.array([(-left * self.zoom) * x, (-bottom * self.zoom) * y]) + self.center * [-x, y]
        self.center_lim = (1 - self.zoom) * np.array([left, bottom])


    def _transform(self, position):
        return np.round(self._scale.dot(position) + self._translation).astype(int)

    def draw_circle(self, position=(0, 0), radius=.1, color=(0, 0, 0), filled=True, width=.01):
        p = self._transform(position)
        r = int(self._scale[0, 0] * radius)
        w = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(self._window, color, p, r, w)

    def draw_polygon(self, vertices, color=(0, 0, 0), filled=True, width=.01):
        vs = [self._transform(v) for v in vertices]
        w = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.polygon(self._window, color, vs, w)

    def draw_convex(self, points, **kwargs):
        from scipy.spatial import ConvexHull

        ps=np.array(points)
        vs = ps[ConvexHull(ps).vertices].tolist()
        self.draw_polygon(vs, **kwargs)

    def draw_grid(self, all_vertices, colors, filled=True, width=.01):
        all_vertices = [[self._transform(v) for v in vertices] for vertices in all_vertices]
        w = 0 if filled else int(self._scale[0, 0] * width)
        for vs, c in zip(all_vertices, colors):
            pygame.draw.polygon(self._window, c, vs, w)

    def draw_polyline(self, vertices, color=(0, 0, 0), closed=False, width=.01, dynamic_color=False):
        vs = [self._transform(v) for v in vertices]
        w = int(self._scale[0, 0] * width)
        if not dynamic_color:
            pygame.draw.lines(self._window, color, closed, vs, w)
        else:
            for v1, v2, c in zip(vs[:-1], vs[1:], color):
                pygame.draw.lines(self._window, c, closed, [v1, v2], w)

    def draw_line(self, start, end, color=(0, 0, 0), width=.01):
        start = self._transform(start)
        end = self._transform(end)
        w = int(self._scale[0, 0] * width)
        pygame.draw.line(self._window, color, start, end, w)

    def draw_transparent_circle(self, position=(0, 0), radius=.1, color=(0, 0, 0, 125), filled=True, width=.01):
        r = int(self._scale[0, 0] * radius)
        s = pygame.Surface((2 * r, 2 * r), pygame.HWSURFACE | pygame.SRCALPHA)
        w = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(s, color, (r, r), radius, w)
        self._window.blit(s, self._transform(position) - r)

    def draw_text_box(self, text_font, text_position):
        self._window.blit(text_font, text_position)

    def draw_envelope(self, points, **kwargs):

        vs = list(geometry.MultiPoint(points).envelope.exterior.coords)
        self.draw_polygon(vs, **kwargs)

    def draw_arrow_line(self, start, end, color=(0, 0, 0), width=.01, dl=0.02, phi=0, s=10):
        a0 = math.atan2(end[1] - start[1], end[0] - start[0])
        l0 = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        w = int(self._scale[0, 0] * width)
        pygame.draw.line(self._window, color, self._transform(start), self._transform(end), w)

        a = a0 + np.pi / 2
        sin0, cos0 = math.sin(a) * s, math.cos(a) * s
        sin1, cos1 = math.sin(a - np.pi * 2 / 3) * s, math.cos(a - np.pi * 2 / 3) * s
        sin2, cos2 = math.sin(a + np.pi * 2 / 3) * s, math.cos(a + np.pi * 2 / 3) * s

        l = 0+phi*dl
        while l < l0:
            pos = self._transform((start[0] + math.cos(a0) * l, start[1] + math.sin(a0) * l))
            p0 = (pos[0] + sin0, pos[1] + cos0)
            p1 = (pos[0] + sin1, pos[1] + cos1)
            p2 = (pos[0] + sin2, pos[1] + cos2)
            pygame.draw.polygon(self._window, color, (p0, p1, p2))
            l += dl


    @property
    def mouse_position(self):
        p = np.array(pygame.mouse.get_pos()) - self._translation
        return np.linalg.inv(self._scale).dot(p)

    def render(self):
        if self.model.show_display:
            pygame.display.flip()
            image = pygame.surfarray.pixels3d(self._window)
            self._t.tick(self._fps)
        else:
            image = pygame.surfarray.array3d(self._window)
        if self._video_writer:
            self._video_writer.append_data(np.flipud(np.rot90(image)))
        if self.snapshot_requested :
            self._image_writer = imageio.get_writer(f'{self.caption}_at_{self.snapshot_requested}_sec.png', mode='i')
            self.snapshot_requested=None
        if self._image_writer:
            self._image_writer.append_data(np.flipud(np.rot90(image)))
            self._image_writer = None
        return image

    @staticmethod
    def close_requested():
        if pygame.display.get_init():
            return pygame.event.peek(pygame.QUIT)
        return False

    def close(self):
        pygame.display.quit()
        if self._video_writer:
            self._video_writer.close()
        if self._image_writer:
            self._image_writer.close()
        del self

        print('Screen closed')

    def move_center(self, dx=0, dy=0, pos=None):
        if pos is None:
            pos = self.center - self.center_lim * [dx, dy]
        self.center = np.clip(pos, self.center_lim, -self.center_lim)
        self.set_bounds(*self.space_bounds)

    @staticmethod
    def load_from_file(file_path, model, **kwargs):
        from larvaworld.lib.model.envs.obstacle import Wall, Box
        with open(file_path) as f:
            line_number = 1

            for line in f:
                words = line.split()

                # skip empty lines
                if len(words) == 0:
                    line_number += 1
                    continue

                # skip comments in file
                if words[0][0] == '#':
                    line_number += 1
                    continue

                if words[0] == 'Scene':
                    # pass
                    # width = int(words[1])
                    # height = int(words[2])
                    viewer = Viewer(model=model,**kwargs)
                # elif words[0] == 'SensorDrivenRobot':
                #     x = float(words[1])
                #     y = float(words[2])
                #     robot = SensorDrivenRobot(x, y, ROBOT_SIZE, ROBOT_WHEEL_RADIUS)
                #     robot.label = line_number
                #     viewer.put(robot)
                elif words[0] == 'Box':
                    x = int(words[1])
                    y = int(words[2])
                    size = int(words[3])
                    box = Box(x, y, size,model=model, default_color='lightgreen')
                    box.label = line_number
                    viewer.put(box)
                elif words[0] == 'Wall':
                    x1 = int(words[1])
                    y1 = int(words[2])
                    x2 = int(words[3])
                    y2 = int(words[4])

                    point1 = geometry.Point(x1, y1)
                    point2 = geometry.Point(x2, y2)
                    wall = Wall(point1, point2,model=model, default_color='lightgreen')
                    wall.label = line_number
                    viewer.put(wall)
                elif words[0] == 'Light':
                    from larvaworld.lib.model.modules.rot_surface import LightSource
                    x = int(words[1])
                    y = int(words[2])
                    emitting_power = int(words[3])
                    light = LightSource(x, y, emitting_power, aux.Color.YELLOW, aux.Color.BLACK,model=model)
                    light.label = line_number
                    viewer.put(light)

                line_number += 1

        return viewer



class InputBox(Viewable):
    default_color = param.Color('lightblue')
    visible = param.Boolean(False)

    def __init__(self, text='', color_inactive=None, center=False, w=140, h=32,
                 screen_pos=None, linewidth=0.01, show_frame=False,
                 agent=None, end_time=0, start_time=0, font=None, **kwargs):
        super().__init__(**kwargs)
        # self.screen_pos = screen_pos
        self.linewidth = linewidth
        self.show_frame = show_frame

        self.color_active = self.default_color
        if color_inactive is None:
            color_inactive = pygame.Color('lightgreen')
        self.color_inactive = color_inactive
        self.active = False
        if font is None:
            pygame.init()
            font = pygame.font.Font(None, 32)
        self.font = font
        self.text = text
        self.text_font = None
        self.agent = agent
        self.end_time = end_time
        self.start_time = start_time
        self.center = center
        self.w = w
        self.h = h
        if screen_pos is not None:
            self.shape = self.set_shape(screen_pos)
        else:
            self.shape = None

    def draw(self, v,**kwargs):
        # if self.visible:
        if self.agent is not None:
            pos=self.agent.get_position()
            screen_pos = self.agent.model.screen_manager.space2screen_pos(pos)
            self.shape = self.set_shape(screen_pos)
            self.color = self.agent.default_color
        if self.shape is not None:
            # Render the current text.
            lines = self.text.splitlines()
            txt_surfaces = [self.font.render(l, True, self.color) for l in lines]
            # Blit the text.
            for i, s in enumerate(txt_surfaces):
                v.draw_text_box(s, (self.shape.x + 5, self.shape.y + 5 + i * 100))
            if self.show_frame:
                # Blit the input_box rect.
                v.draw_polygon(self.shape, color=self.color, filled=False, width=self.linewidth)
        elif self.text_font is not None:
            self.text_font = self.font_large.render(self.text, 1, self.color)
            v.draw_text_box(self.text_font, self.text_font_r)

    def switch(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # self.set_shape(event.pos)
            # If the user clicked on the input_box rect.
            if self.shape.collidepoint(event.pos):
                # Toggle the active variable.
                self.active = not self.active
            else:
                self.active = False
                # Change the current color of the input box.
            self.color = self.color_active if self.active else self.color_inactive

    def get_input(self, event):
        if self.visible:
            self.switch(event)
            if event.type == pygame.KEYDOWN:
                if self.active:
                    if event.key == pygame.K_RETURN:
                        self.submit()
                    elif event.key == pygame.K_BACKSPACE:
                        self.text = self.text[:-1]
                    else:
                        self.text += event.unicode

    def submit(self):
        print(self.text)
        self.visible = False

    def set_shape(self, pos):
        if pos is not None and not any(np.isnan(pos)):
            if self.center:
                return pygame.Rect(pos[0] - self.w / 2, pos[1] - self.h / 2, self.w, self.h)
            else:
                return pygame.Rect(pos[0], pos[1], self.w, self.h)
        else:
            return None

    def render(self, width, height):
        # Scale to screen
        x_pos = int(width * 0.85)
        y_pos = int(height * 0.1)
        large_font_size = int(1 / 20 * width)

        # Fonts
        self.font_large = pygame.font.SysFont("SansitaOne.tff", large_font_size)

        # Hour
        self.text_font = self.font_large.render(self.text, 1, self.color)  # zero-pad hours to 2 digits
        self.text_font_r = self.text_font.get_rect()
        self.text_font_r.center = (x_pos * 0.91, y_pos)

    def flash_text(self, text, t=2):
        self.text = text
        self.end_time = pygame.time.get_ticks() + t * 1000
        self.start_time = pygame.time.get_ticks() + int(0.1 * 1000)


class IDBox(InputBox):
    def __init__(self, agent,**kwargs):
        super().__init__(agent=agent, text=agent.unique_id, color_inactive=agent.default_color,default_color=agent.default_color,**kwargs)

class LabelledGroupedObject(ViewableGroupedObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id_box = IDBox(agent=self)

    def _draw(self,v,**kwargs):
        if self.visible :
            self.draw(v,**kwargs)
            if self.id_box.visible:
                self.id_box.draw(v)




class PointRelPositionedViewable(PointRelPositioned, Viewable) :

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TextFont(PointRelPositionedViewable) :
    text = param.String('', doc='The text to draw')
    font_size_scale = PositiveNumber(1/40, doc='The font size relative to the window size')
    font_type = param.Parameter("Trebuchet MS", doc='The font type to use')
    text_centre_scale = PositiveRange((0.9, 0.9),softmax=10.0, step=0.01, doc='The text center position relative to the position')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font = None
        self.text_centre = None
        self.text_font = None
        self.text_font_r = None
        self.update_font()
        self.render_text()

    @param.depends('text','default_color','text_centre_scale', watch=True)
    def render_text(self):
        self.text_font = self.font.render(self.text, 1, self.default_color)  # zero-pad hours to 2 digits
        self.text_font_r = self.text_font.get_rect()
        self.text_font_r.center = self.text_centre




    @param.depends('pos', 'window_dims', 'text_centre_scale', watch=True)
    def update_font(self):
        width, height = self.window_dims
        font_size=int(width *self.font_size_scale)
        self.font = pygame.font.SysFont(self.font_type, font_size)
        dx, dy =self.text_centre_scale
        self.text_centre = (self.x * dx, self.y * dy)

    def draw(self, v,**kwargs):
        v.draw_text_box(self.text_font, self.text_font_r)

    def set_text(self, text):
        self.text = text


class SimulationClock(PointRelPositionedViewable):
    pos_scale = PositiveRange((0.94, 0.04))


    def __init__(self, sim_step_in_sec,**kwargs):
        super().__init__(**kwargs)
        # Time Info
        self.sim_step_in_dms = int(sim_step_in_sec * 100)
        self.time_in_min = 0
        self.dmsecond = 0
        self.second = 0
        self.minute = 0
        self.hour = 0

        width, height=self.window_dims
        large_font_size = int(1 / 40 * width)
        small_font_size = int(1 / 50 * width)

        # Fonts
        self.font_large = pygame.font.SysFont("Trebuchet MS", large_font_size)
        self.font_small = pygame.font.SysFont("Trebuchet MS", small_font_size)

        self.hour_centre=(self.x * 0.91, self.y)
        self.minute_centre=(self.x * 0.95, self.y)
        self.second_centre=(self.x * 1, self.y)
        self.msecond_centre=(self.x * 1.04, self.y * 1.1)


    def tick_clock(self):
        # self.counter += 1
        self.dmsecond += self.sim_step_in_dms
        if self.dmsecond >= 100:
            self.second += 1
            self.dmsecond -= 100
            if self.second >= 60:
                self.minute += 1
                self.second -= 60
                if self.minute >= 60:
                    self.hour += 1
                    self.minute -= 60

    # def render(self, width, height):
    #     # Scale to screen
    #     x_pos = int(width * 0.94)
    #     y_pos = int(height * 0.04)
    #     large_font_size = int(1 / 40 * width)
    #     small_font_size = int(1 / 50 * width)
    #
    #     # Fonts
    #     self.font_large = pygame.font.SysFont("Trebuchet MS", large_font_size)
    #     self.font_small = pygame.font.SysFont("Trebuchet MS", small_font_size)
    #
    #     # Hour
    #     self.hour_font = self.font_large.render("{0:02}".format(self.hour), 1, self.color)  # zero-pad hours to 2 digits
    #     self.hour_font_r = self.hour_font.get_rect()
    #     self.hour_font_r.center = (x_pos * 0.91, y_pos)
    #     # Minute
    #     self.minute_font = self.font_large.render(":{0:02}".format(self.minute), 1,
    #                                               self.color)  # zero-pad minutes to 2 digits
    #     self.minute_font_r = self.minute_font.get_rect()
    #     self.minute_font_r.center = (x_pos * 0.95, y_pos)
    #     # Second
    #     self.second_font = self.font_large.render(":{0:02}:".format(self.second), 1,
    #                                               self.color)  # zero-pad seconds to 2 digits
    #     self.second_font_r = self.second_font.get_rect()
    #     self.second_font_r.center = (x_pos, y_pos)
    #     # Milisecond
    #     self.dmsecond_font = self.font_small.render("{0:02}".format(self.dmsecond), 1,
    #                                                 self.color)  # zero-pad miliseconds to 2 digits
    #     self.msecond_font_r = self.dmsecond_font.get_rect()
    #     self.msecond_font_r.center = (x_pos * 1.04, y_pos * 1.1)

    def draw(self, v,**kwargs):
        c=self.default_color
        self.hour_font = self.font_large.render("{0:02}".format(self.hour), 1, c)  # zero-pad hours to 2 digits
        self.hour_font_r = self.hour_font.get_rect()
        self.hour_font_r.center = self.hour_centre

        self.minute_font = self.font_large.render(":{0:02}".format(self.minute), 1,c)  # zero-pad minutes to 2 digits
        self.minute_font_r = self.minute_font.get_rect()
        self.minute_font_r.center = self.minute_centre

        self.second_font = self.font_large.render(":{0:02}:".format(self.second), 1,c)  # zero-pad seconds to 2 digits
        self.second_font_r = self.second_font.get_rect()
        self.second_font_r.center = self.second_centre

        self.dmsecond_font = self.font_small.render("{0:02}".format(self.dmsecond), 1,c)  # zero-pad miliseconds to 2 digits
        self.msecond_font_r = self.dmsecond_font.get_rect()
        self.msecond_font_r.center = self.msecond_centre

        v.draw_text_box(self.hour_font, self.hour_font_r)
        v.draw_text_box(self.minute_font, self.minute_font_r)
        v.draw_text_box(self.second_font, self.second_font_r)
        v.draw_text_box(self.dmsecond_font, self.msecond_font_r)


class SimulationScale(TextFont):
    real_width = PositiveNumber(0.1, softmax=10.0, doc='The width of the Arena possibly zoomed')
    font_size_scale = PositiveNumber(1 / 40)
    pos_scale = PositiveRange((0.1, 0.04))
    text_centre_scale = PositiveRange((1, 1.5))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = None
        self.update_scale()
        # Get 1/10 of max real dimension, transform it to mm and find the closest reasonable scale
        # w_in_mm = self.real_width * 1000
        # self.scale_in_mm = self.compute_scale_in_mm(w_in_mm)
        # self.scale_in_mm = self.closest(
        #     lst=[0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000], k=w_in_mm / 10)
        # I don't exactly understand why this works...
        # self.scale_to_draw = self.scale_in_mm / w_in_mm


        # self.window_width, self.window_height = window_dims
        # self.x_pos = int(self.window_width * 0.1)
        # self.y_pos = int(self.window_height * 0.04)
        #
        # font_size = int(1 / 40 * self.window_width)
        # self.font = pygame.font.SysFont("Trebuchet MS", font_size)
        # self.font_center=(self.x_pos, self.y_pos * 1.5)


    @param.depends('real_width', watch=True)
    def update_scale(self):
        w_in_mm = self.real_width * 1000
        self.scale_in_mm = self.closest(
            lst=[0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000], k=w_in_mm / 10)
        self.scale_to_draw = self.scale_in_mm / w_in_mm
        self.text=f'{self.scale_in_mm} mm'
        self.lines = self.compute_lines(self.x, self.y, self.scale_to_draw * self.window_dims[0])

        # self.scale_font = self.font.render(self.text, 1, self.color)
        # self.scale_font_r = self.scale_font.get_rect()
        # self.scale_font_r.center = self.font_center

    def compute_lines(self, x, y, scale):
        return [[(x - scale / 2, y), (x + scale / 2, y)],
                [(x + scale / 2, y * 0.75), (x + scale / 2, y * 1.25)],
                [(x - scale / 2, y * 0.75), (x - scale / 2, y * 1.25)]]

    def closest(self, lst, k):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]

    # def render(self, width, height):
    #     # Scale to screen
    #     scale_to_screen = self.scale_to_draw * width
    #     x_pos = int(width * 0.1)
    #     y_pos = int(height * 0.04)
    #
    #     self.lines = self.compute_lines(x_pos, y_pos, scale_to_screen)
    #
    #     font_size = int(1 / 40 * width)
    #     self.font = pygame.font.SysFont("Trebuchet MS", font_size)
    #     self.scale_font = self.font.render(self.text, 1, self.color)
    #     self.scale_font_r = self.scale_font.get_rect()
    #     self.scale_font_r.center = (x_pos, y_pos * 1.5)

    def draw(self, v,**kwargs):
        for line in self.lines:
            pygame.draw.line(v._window, self.default_color, line[0], line[1], 1)
        super().draw(v,**kwargs)
        # self.scale_font = self.font.render(self.text, 1, self.color)
        # self.scale_font_r = self.scale_font.get_rect()
        # self.scale_font_r.center = self.font_center
        # v.draw_text_box(self.scale_font, self.scale_font_r)


class SimulationState(TextFont):
    font_size_scale = PositiveNumber(1 / 40)
    pos_scale = PositiveRange((0.85, 0.94))
    text_centre_scale = PositiveRange((1, 1))

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        # self.Nagents = 0
        # self.text = ''
        # self.text = f'# larvae : {self.Nagents}'
        # self.window_width, self.window_height = window_dims
        # self.x_pos = int(self.window_width * 0.85)
        # self.y_pos = int(self.window_height * 0.94)
        # font_size = int(1 / 40 * self.window_width)
        # self.font = pygame.font.SysFont("Trebuchet MS", font_size)
        # self.font_center = (self.x_pos, self.y_pos)

    # def render(self, width, height):
    #     x_pos = int(width * 0.85)
    #     y_pos = int(height * 0.94)
    #     font_size = int(1 / 40 * width)
    #
    #     self.font = pygame.font.SysFont("Trebuchet MS", font_size)
    #     self.state_font = self.font.render(self.text, 1, self.color)
    #     self.state_font_r = self.state_font.get_rect()
    #     self.state_font_r.center = (x_pos, y_pos)

    # def draw(self, v,**kwargs):
    #     self.state_font = self.font.render(self.text, 1, self.color)
    #     self.state_font_r = self.state_font.get_rect()
    #     self.state_font_r.center = self.font_center
    #     # self.state_font = self.font.render(self.text, 1, self.color)
    #     v.draw_text_box(self.state_font, self.state_font_r)

    # def set_text(self, text):
    #     self.text = text


# class SimulationState(aux.Viewable):
#
#     def __init__(self, model,window_dims, **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         # self.Nagents = 0
#         self.text = ''
#         # self.text = f'# larvae : {self.Nagents}'
#         self.window_width, self.window_height = window_dims
#         self.x_pos = int(self.window_width * 0.85)
#         self.y_pos = int(self.window_height * 0.94)
#         font_size = int(1 / 40 * self.window_width)
#         self.font = pygame.font.SysFont("Trebuchet MS", font_size)
#         self.font_center = (self.x_pos, self.y_pos)
#
#     # def render(self, width, height):
#     #     x_pos = int(width * 0.85)
#     #     y_pos = int(height * 0.94)
#     #     font_size = int(1 / 40 * width)
#     #
#     #     self.font = pygame.font.SysFont("Trebuchet MS", font_size)
#     #     self.state_font = self.font.render(self.text, 1, self.color)
#     #     self.state_font_r = self.state_font.get_rect()
#     #     self.state_font_r.center = (x_pos, y_pos)
#
#     def draw(self, v,**kwargs):
#         self.state_font = self.font.render(self.text, 1, self.color)
#         self.state_font_r = self.state_font.get_rect()
#         self.state_font_r.center = self.font_center
#         # self.state_font = self.font.render(self.text, 1, self.color)
#         v.draw_text_box(self.state_font, self.state_font_r)
#
#     def set_text(self, text):
#         self.text = text

#
# def blit_text(surface, text, pos, font=None, color=pygame.Color('white')):
#     if font is None:
#         font = pygame.font.SysFont('Arial', 20)
#     words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
#     space = font.size(' ')[0]  # The width of a space.
#     max_width, max_height = surface.get_size()
#     x, y = pos
#     for line in words:
#         for word in line:
#             word_surface = font.render(word, 0, color)
#             word_width, word_height = word_surface.get_size()
#             if x + word_width >= max_width:
#                 x = pos[0]  # Reset the x.
#                 y += word_height  # Start on new row.
#             surface.blit(word_surface, (x, y))
#             x += word_width + space
#         x = pos[0]  # Reset the x.
#         y += word_height  # Start on new row.
