import math
import os
import time

import numpy as np
import pygame

from lib.aux import functions as fun


class GuppiesViewer(object):
    def __init__(self, width, height, caption="", fps=10, dt=0.1, show_display=True, record_video_to=None,
                 record_image_to=None, zoom=1):
        x = 1550
        y = 400
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)
        self.w_loc = [int(x) for x in os.environ['SDL_VIDEO_WINDOW_POS'].split(',')]
        self.zoom = zoom
        self.caption = caption
        self.window_size = width, height
        self.show_display = show_display
        self._t = pygame.time.Clock()
        self._fps = fps
        self.dt = dt
        self.center = np.array([0.0, 0.0])
        self.center_lim = np.array([0.0, 0.0])

        self.display_size = self.scale_dims()
        self._window = self.init_screen()

        # winInfo = pygame.PygameWindowInfo()

        if record_video_to:
            import imageio
            self._video_writer = imageio.get_writer(record_video_to, mode='I', fps=self._fps)
        else:
            self._video_writer = None

        if record_image_to:
            import imageio
            self._image_writer = imageio.get_writer(record_image_to, mode='i')
        else:
            self._image_writer = None

        self._scale = np.array([[1., .0], [.0, -1.]])
        self._translation = np.zeros(2)

    def draw_arena(self, tank_shape, tank_color, screen_color):
        # t0=time.time()
        surf1 = pygame.Surface(self.display_size, pygame.SRCALPHA)
        surf2 = pygame.Surface(self.display_size, pygame.SRCALPHA)
        tank_shape = [self._transform(v) for v in tank_shape]
        pygame.draw.polygon(surf1, tank_color, tank_shape, 0)

        # screen_shape = [self._transform(v) for v in screen_shape]
        # pygame.draw.polygon(surf2, screen_color, screen_shape, 0)
        pygame.draw.rect(surf2, screen_color, surf2.get_rect())
        # surf1.blit(surf2, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        surf2.blit(surf1, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        self._window.blit(surf2, (0, 0))
        # t1 = time.time()
        # print()
        # print(np.round((t1-t0)*1000))

    def init_screen(self):
        if self.show_display:
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
            window = pygame.display.set_mode(self.window_size, flags)
            pygame.display.set_caption(self.caption)
            pygame.event.set_allowed(pygame.QUIT)
        else:
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF
            window = pygame.Surface(self.display_size, flags)
        return window

    def scale_dims(self):
        _width = int(self.window_size[0] / self.zoom)
        _height = int(self.window_size[1] / self.zoom)
        return _width, _height

    def zoom_screen(self, d_zoom, pos=None):
        if pos is None:
            pos = self.get_mouse_position()
        if 0.001 <= self.zoom + d_zoom <= 1:
            self.zoom = np.round(self.zoom + d_zoom, 2)
            self.display_size = self.scale_dims()
            self.center = np.clip(self.center - pos * d_zoom, self.center_lim, -self.center_lim)
        if self.zoom == 1.0:
            self.center = np.array([0.0, 0.0])

    def __del__(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scale_x = self.display_size[0] / (right - left)
        scale_y = self.display_size[1] / (top - bottom)
        self._scale = np.array([[scale_x, .0], [.0, -scale_y]])
        self._translation = np.array([(-left * self.zoom) * scale_x, (-bottom * self.zoom) * scale_y])
        self._translation += self.center * [-scale_x, scale_y]
        self.center_lim = (1 - self.zoom) * np.array([left, bottom])

    def _transform(self, position):
        return np.round(self._scale.dot(position) + self._translation).astype(int)

    def draw_circle(self, position=(0, 0), radius=.1, color=(0, 0, 0), filled=True, width=.01):
        position = self._transform(position)
        radius = int(self._scale[0, 0] * radius)
        # width = 0 if filled else int(np.ceil(radius * width))
        width = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(self._window, color, position, radius, width)

    def draw_polygon(self, vertices, color=(0, 0, 0), filled=True, width=.01):
        vertices = [self._transform(v) for v in vertices]
        width = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.polygon(self._window, color, vertices, 0 if filled else width)

    def draw_grid(self, all_vertices, colors, filled=True, width=.01):
        all_vertices = [[self._transform(v) for v in vertices] for vertices in all_vertices]
        width = 0 if filled else int(self._scale[0, 0] * width)
        for vertices, color in zip(all_vertices, colors):
            pygame.draw.polygon(self._window, color, vertices, 0 if filled else width)

    def draw_polyline(self, vertices, color=(0, 0, 0), closed=False, width=.01, dynamic_color=False):
        vertices = [self._transform(v) for v in vertices]
        width = int(self._scale[0, 0] * width)
        if not dynamic_color:
            pygame.draw.lines(self._window, color, closed, vertices, width)
        else:
            for v1, v2, c in zip(vertices[:-1], vertices[1:], color):
                pygame.draw.lines(self._window, c, closed, [v1, v2], width)

    def draw_line(self, start, end, color=(0, 0, 0), width=.01):
        start = self._transform(start)
        end = self._transform(end)
        width = int(self._scale[0, 0] * width)
        pygame.draw.line(self._window, color, start, end, width)

    def draw_transparent_circle(self, position=(0, 0), radius=.1, color=(0, 0, 0, 125), filled=True, width=.01):
        radius = int(self._scale[0, 0] * radius)
        s = pygame.Surface((2 * radius, 2 * radius), pygame.HWSURFACE | pygame.SRCALPHA)
        width = 0 if filled else int(self._scale[0, 0] * width)
        pygame.draw.circle(s, color, (radius, radius), radius, width)
        self._window.blit(s, self._transform(position) - radius)

    def draw_text_box(self, text_font, text_position):
        self._window.blit(text_font, text_position)

    def draw_arrow(self, start, end, color=(0, 0, 0), width=.01):
        size = 4
        start = self._transform(start)
        end = self._transform(end)
        pygame.draw.line(self._window, color, start, end, 2)
        rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
        pygame.draw.polygon(self._window, color, (
            (end[0] + size * math.sin(math.radians(rotation)), end[1] + size * math.cos(math.radians(rotation))), (
                end[0] + size * math.sin(math.radians(rotation - 120)),
                end[1] + size * math.cos(math.radians(rotation - 120))),
            (end[0] + size * math.sin(math.radians(rotation + 120)),
             end[1] + size * math.cos(math.radians(rotation + 120)))))

    def get_array(self):
        image_data = pygame.surfarray.array3d(self._window)
        return image_data

    def get_mouse_position(self):
        mouse_pos = np.array(pygame.mouse.get_pos()) - self._translation
        return np.linalg.inv(self._scale).dot(mouse_pos)

    def render(self):
        if self.show_display:
            pygame.display.flip()
            image = pygame.surfarray.pixels3d(self._window)
            self._t.tick(self._fps)
        else:
            image = pygame.surfarray.array3d(self._window)
        if self._video_writer:
            self._video_writer.append_data(np.flipud(np.rot90(image)))
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
        # pygame.quit()
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


class ScreenItem:
    def __init__(self, color=None):
        if color is None:
            self.color = (0, 0, 0)
        else:
            self.color = color

    def set_color(self, color):
        self.color = color


class InputBox(ScreenItem):
    def __init__(self, visible=False, text='', color_inactive=None, color_active=None,
                 screen_pos=None, linewidth=0.01, show_frame=False, agent=None, end_time=0, start_time=0):
        super().__init__(color=color_active)
        self.screen_pos = screen_pos
        self.linewidth = linewidth
        self.show_frame = show_frame
        # self.set_shape(self.screen_pos)
        if color_active is None:
            color_active = pygame.Color('dodgerblue2')
        self.color_active = color_active
        if color_inactive is None:
            color_inactive = pygame.Color('lightskyblue3')
        self.color_inactive = color_inactive
        self.visible = visible
        self.active = False
        self.font = pygame.font.Font(None, 32)
        self.text = text
        self.text_font = None
        self.agent = agent
        self.end_time = end_time
        self.start_time = start_time
        self.shape = None

    def draw(self, viewer):
        if self.visible:
            if self.agent is not None:
                self.set_shape(self.agent.model.space2screen_pos(self.agent.get_position()))
                self.color = self.agent.default_color
            if self.shape is not None:
                # Render the current text.
                txt_surface = self.font.render(self.text, True, self.color)
                # Blit the text.
                viewer.draw_text_box(txt_surface, (self.shape.x + 5, self.shape.y + 5))
                # viewer._window.blit(txt_surface, (self.shape.x + 5, self.shape.y + 5))
                if self.show_frame:
                    # Blit the input_box rect.
                    viewer.draw_polygon(self.shape, color=self.color, filled=False, width=self.linewidth)
                    # pygame.draw.rect(viewer._window, self.color, self.shape, self.linewidth)
            elif self.text_font is not None:
                self.text_font = self.font_large.render(self.text, 1, self.color)
                viewer.draw_text_box(self.text_font, self.text_font_r)

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
            self.shape = pygame.Rect(pos[0], pos[1], 140, 32)
        else:
            self.shape = None

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


class SimulationClock(ScreenItem):

    def __init__(self, sim_step_in_sec, color=None):
        super().__init__(color=color)
        # Time Info
        self.sim_step_in_dms = int(sim_step_in_sec * 100)
        self.time_in_min = 0
        self.dmsecond = 0
        self.second = 0
        self.minute = 0
        self.hour = 0
        self.counter = 0

        self.timer_on = False
        self.next_on = None
        self.next_off = None
        self.timer_opened = False
        self.timer_closed = False

    def tick_clock(self):
        self.counter += 1
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
        self.check_timer()

    def render_clock(self, width, height):
        # Scale to screen
        x_pos = int(width * 0.94)
        y_pos = int(height * 0.04)
        large_font_size = int(1 / 40 * width)
        small_font_size = int(1 / 50 * width)

        # Fonts
        self.font_large = pygame.font.SysFont("Trebuchet MS", large_font_size)
        self.font_small = pygame.font.SysFont("Trebuchet MS", small_font_size)

        # Hour
        self.hour_font = self.font_large.render("{0:02}".format(self.hour), 1, self.color)  # zero-pad hours to 2 digits
        self.hour_font_r = self.hour_font.get_rect()
        self.hour_font_r.center = (x_pos * 0.91, y_pos)
        # Minute
        self.minute_font = self.font_large.render(":{0:02}".format(self.minute), 1,
                                                  self.color)  # zero-pad minutes to 2 digits
        self.minute_font_r = self.minute_font.get_rect()
        self.minute_font_r.center = (x_pos * 0.95, y_pos)
        # Second
        self.second_font = self.font_large.render(":{0:02}:".format(self.second), 1,
                                                  self.color)  # zero-pad seconds to 2 digits
        self.second_font_r = self.second_font.get_rect()
        self.second_font_r.center = (x_pos, y_pos)
        # Milisecond
        self.dmsecond_font = self.font_small.render("{0:02}".format(self.dmsecond), 1,
                                                    self.color)  # zero-pad miliseconds to 2 digits
        self.msecond_font_r = self.dmsecond_font.get_rect()
        self.msecond_font_r.center = (x_pos * 1.04, y_pos * 1.1)

    def draw_clock(self, viewer):
        self.hour_font = self.font_large.render("{0:02}".format(self.hour), 1, self.color)  # zero-pad hours to 2 digits
        self.minute_font = self.font_large.render(":{0:02}".format(self.minute), 1,
                                                  self.color)  # zero-pad minutes to 2 digits
        self.second_font = self.font_large.render(":{0:02}:".format(self.second), 1,
                                                  self.color)  # zero-pad seconds to 2 digits
        self.dmsecond_font = self.font_small.render("{0:02}".format(self.dmsecond), 1,
                                                    self.color)  # zero-pad miliseconds to 2 digits

        viewer.draw_text_box(self.hour_font, self.hour_font_r)
        viewer.draw_text_box(self.minute_font, self.minute_font_r)
        viewer.draw_text_box(self.second_font, self.second_font_r)
        viewer.draw_text_box(self.dmsecond_font, self.msecond_font_r)

    def set_timer(self, on_ticks, off_ticks):
        self.Ndurs = len(on_ticks)
        self.timer_on_ticks, self.timer_off_ticks = on_ticks, off_ticks
        self.dur_idx = 0
        self.next_on, self.next_off = self.timer_on_ticks[self.dur_idx], self.timer_off_ticks[self.dur_idx]
        self.timer_on = False

    def check_timer(self):
        self.timer_opened = False
        self.timer_closed = False
        if not self.timer_on and self.next_on is not None:
            if self.counter >= self.next_on:
                self.timer_on = True
                self.timer_opened = True
                self.dur_idx += 1
                if self.dur_idx < self.Ndurs:
                    self.next_on = self.timer_on_ticks[self.dur_idx]
                else:
                    self.next_on = None
        elif self.timer_on and self.next_off is not None:
            if self.counter >= self.next_off:
                self.timer_on = False
                self.timer_closed = True
                if self.dur_idx < self.Ndurs:
                    self.next_off = self.timer_off_ticks[self.dur_idx]
                else:
                    self.next_on = None


class SimulationScale(ScreenItem):

    def __init__(self, real_width, scaling_factor, color=None, space_in_mm = False):
        super().__init__(color=color)

        # Get 1/10 of max real dimension, transform it to mm and find the closest reasonable scale
        real_width_in_mm = real_width if space_in_mm else real_width*1000
        self.scale_in_mm = self.closest(lst=[1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000],
                                        k=real_width_in_mm / 10)
        # I don't exactly understand why this works...
        self.scale_to_draw = self.scale_in_mm / real_width_in_mm
        self.lines = None

    def compute_lines(self, x, y, scale):
        return [[(x - scale / 2, y), (x + scale / 2, y)],
                [(x + scale / 2, y * 0.75), (x + scale / 2, y * 1.25)],
                [(x - scale / 2, y * 0.75), (x - scale / 2, y * 1.25)]]

    def closest(self, lst, k):
        return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]

    def render_scale(self, width, height):
        # Scale to screen
        scale_to_screen = self.scale_to_draw * width
        x_pos = int(width * 0.1)
        y_pos = int(height * 0.04)
        font_size = int(1 / 40 * width)
        self.lines = self.compute_lines(x_pos, y_pos, scale_to_screen)

        self.font = pygame.font.SysFont("Trebuchet MS", font_size)
        self.scale_font = self.font.render(f'{self.scale_in_mm} mm', 1, self.color)
        self.scale_font_r = self.scale_font.get_rect()
        self.scale_font_r.center = (x_pos, y_pos * 1.5)

    def draw_scale(self, viewer):
        for line in self.lines:
            pygame.draw.line(viewer._window, self.color, line[0], line[1], 1)
        self.scale_font = self.font.render(f'{self.scale_in_mm} mm', 1, self.color)
        viewer.draw_text_box(self.scale_font, self.scale_font_r)


class SimulationState(ScreenItem):

    def __init__(self, model, color=None):
        super().__init__(color=color)
        self.model = model
        # self.Nagents = 0
        self.text = ''
        # self.text = f'# larvae : {self.Nagents}'

    def render_state(self, width, height):
        x_pos = int(width * 0.85)
        y_pos = int(height * 0.94)
        font_size = int(1 / 40 * width)

        self.font = pygame.font.SysFont("Trebuchet MS", font_size)
        self.state_font = self.font.render(self.text, 1, self.color)
        self.state_font_r = self.state_font.get_rect()
        self.state_font_r.center = (x_pos, y_pos)

    def draw_state(self, viewer):
        self.state_font = self.font.render(self.text, 1, self.color)
        viewer.draw_text_box(self.state_font, self.state_font_r)

    def set_text(self, text):
        self.text = text


def draw_velocity_arrow(_screen, agent):
    start = agent.get_centroid_position()
    lin_vel = np.array(agent.get_head().get_linearvelocity_amp())
    if lin_vel < 0.001:
        # FIXME This produces bug
        # _screen.draw_circle(start, agent.get_sim_length() / 5, color=(255, 0, 0), width=.01)
        pass
    else:
        _screen.draw_arrow(start, start + lin_vel / 100, color=(0, 0, 255), width=.01)


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
            ds, de = fun.parse_array_at_nans(traj_x)
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
