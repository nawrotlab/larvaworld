"""
Screen renderable items for pygame-based simulation visualization
"""

import os

import param

from ..param import (
    Area2DPixel,
    IntegerTuple,
    NestedConf,
    NumericTuple2DRobust,
    Pos2D,
    PositiveInteger,
    PositiveNumber,
    PosPixelRel2Area,
    Viewable,
    ViewableToggleable,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

__all__ = [
    "IDBox",
    "ScreenTextBoxRect",
    "ScreenMsgText",
    "ScreenTextBox",
    "SimulationClock",
    "SimulationScale",
    "SimulationState",
]


class ScreenTextFont(NestedConf):
    text_color = param.Color("black", doc="The color of the text")
    text = param.String("", doc="The text to draw")
    font_size = PositiveInteger(20, doc="The font size")
    font_type = param.Parameter("Trebuchet MS", doc="The font type to use")
    text_centre = NumericTuple2DRobust(doc="The text center position")

    def __init__(self, end_time=0, start_time=0, **kwargs):
        self.font = None
        self.text_font = None
        self.text_font_r = None
        super().__init__(**kwargs)
        self.end_time = end_time
        self.start_time = start_time
        if not self.font:
            self.update_font()

    @param.depends("text", "text_color", "text_centre", watch=True)
    def render_text(self):
        if not self.font:
            self.update_font()
        if self.N_text_lines == 1:
            self.text_font = self.font.render(
                self.text, 1, self.text_color
            )  # zero-pad hours to 2 digits
            self.text_font_r = self.text_font.get_rect()
            self.text_font_r.center = self.text_centre
        else:
            N = self.N_text_lines
            ls = self.text_lines
            self.text_font = []
            self.text_font_r = []
            x0, y0 = self.text_centre
            for i in range(N):
                f = self.font.render(ls[i], True, self.text_color)
                r = f.get_rect()
                r.center = x0, y0 + (i - int(N / 2)) * 50
                self.text_font.append(f)
                self.text_font_r.append(r)

    @property
    def text_lines(self):
        return self.text.splitlines()

    @property
    def N_text_lines(self):
        return len(self.text_lines)

    @param.depends("font_size", watch=True)
    def update_font(self):
        pygame.init()
        self.font = pygame.font.SysFont(self.font_type, self.font_size)

    def draw(self, v, **kwargs):
        if self.text_font is None or self.text_font_r is None:
            self.render_text()
        if self.N_text_lines == 1:
            v.draw_text_box(self.text_font, self.text_font_r)
        else:
            for i in range(self.N_text_lines):
                v.draw_text_box(self.text_font[i], self.text_font_r[i])

    def set_text(self, text):
        self.text = text

    def flash_text(self, text, t=2):
        self.set_text(text)
        self.end_time = pygame.time.get_ticks() + t * 1000
        self.start_time = pygame.time.get_ticks() + int(0.1 * 1000)


class ScreenTextFontRel(ScreenTextFont):
    text_centre_scale = param.NumericTuple(
        (0.9, 0.9),
        # text_centre_scale = PositiveRange((0.9, 0.9), softmax=10.0, step=0.01,
        doc="The text center position relative to the position",
    )
    font_size_scale = PositiveNumber(
        1 / 40, doc="The font size relative to the window size"
    )
    reference_object = param.ClassSelector(
        class_=PosPixelRel2Area, doc="The object hosting the text"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_font_size(self.reference_object)
        self.update_font_centre_pos(self.reference_object)

    @param.depends("reference_object.pos", "text_centre_scale", watch=True)
    def update_font_centre_pos(self, obj):
        dx, dy = self.text_centre_scale
        self.text_centre = (obj.x * dx, obj.y * dy)

    # @param.depends('reference_object', watch=True)
    def update_font_size(self, obj):
        self.font_size = int(obj.reference_area.w * self.font_size_scale)


class ScreenTextBoxRect(ScreenTextFont, Viewable):
    visible = param.Boolean(False)
    frame_rect = param.ClassSelector(class_=pygame.Rect, doc="The frame rectangle")
    linewidth = PositiveNumber(10.0, doc="The linewidth to draw the box")
    show_frame = param.Boolean(True, doc="Draw the rectangular frame around the text")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text_centre = self.frame_rect.center

    def draw(self, v, **kwargs):
        if self.show_frame and self.frame_rect is not None:
            pygame.draw.rect(
                v.v,
                color=self.text_color,
                rect=self.frame_rect,
                width=int(self.linewidth),
            )

        super().draw(v=v, **kwargs)


class ScreenTextBox(ScreenTextFont, ViewableToggleable, Area2DPixel):
    dims = IntegerTuple(default=(140, 32))
    visible = param.Boolean(False)
    linewidth = PositiveNumber(0.001, doc="The linewidth to draw the box")
    show_frame = param.Boolean(True, doc="Draw the rectangular frame around the text")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_rect = None

    def set_frame_rect(self, pos=None, **kwargs):
        return self.get_rect_at_pos(pos, **kwargs)

    def draw(self, v, **kwargs):
        if self.show_frame:
            if self.frame_rect is not None:
                # v.draw_polygon(self.shape, color=self.color, filled=False, width=self.linewidth)
                # pygame.draw.rect(v._window, color=self.color, rect=self.shape)
                pygame.draw.rect(
                    v.v,
                    color=self.color,
                    rect=self.frame_rect,
                    width=int(v._scale[0, 0] * self.linewidth),
                )


class IDBox(ScreenTextFont, ViewableToggleable):
    visible = param.Boolean(False)
    agent = param.ClassSelector(class_=Pos2D, doc="The agent owning the ID")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_font()
        self.update_agent()

    def update_agent(self):
        self.text_color = self.agent.color
        self.set_text(self.agent.unique_id)

    # @param.depends('agent.pos', watch=True)
    def update_font_centre_pos(self, v):
        pos = self.agent.get_position()
        x, y = v.space2screen_pos(pos)
        self.text_centre = x + 50, y + 12

    def draw(self, v, **kwargs):
        self.update_font_centre_pos(v)
        self.update_agent()
        ScreenTextFont.draw(self, v=v, **kwargs)


class PosPixelRel2AreaViewable(PosPixelRel2Area, Viewable):
    pass


class ScreenMsgText(ScreenTextFontRel, Viewable):
    text_centre_scale = param.NumericTuple(
        (0.91, 1), doc="The text center position relative to the position"
    )
    font_size_scale = PositiveNumber(
        1 / 25, doc="The font size relative to the window size"
    )
    font_type = param.Parameter(default="SansitaOne.tff")

    def __init__(self, reference_area, **kwargs):
        reference_object = PosPixelRel2Area(
            reference_area=reference_area, pos_scale=(0.95, 0.1)
        )
        super().__init__(reference_object=reference_object, **kwargs)

    def draw(self, v, **kwargs):
        ScreenTextFont.draw(self, v=v, **kwargs)
        # self.text_font.draw(v, **kwargs)

    def set_default_color(self, color):
        super().set_default_color(color)
        self.text_color = self.color


class SimulationClock(PosPixelRel2AreaViewable):
    pos_scale = param.NumericTuple((0.94, 0.04))

    def __init__(self, sim_step_in_sec, **kwargs):
        super().__init__(**kwargs)
        # Time Info
        self.sim_step_in_dms = int(sim_step_in_sec * 100)
        self.time_in_min = 0
        self.dmsecond = 0
        self.second = 0
        self.minute = 0
        self.hour = 0

        kws = {
            "reference_object": self,
            "text_color": self.color,
        }

        self.text_fonts = {
            "hour": ScreenTextFontRel(
                font_size_scale=(1 / 40), text_centre_scale=(0.91, 1.0), **kws
            ),
            "minute": ScreenTextFontRel(
                font_size_scale=(1 / 40), text_centre_scale=(0.95, 1.0), **kws
            ),
            "second": ScreenTextFontRel(
                font_size_scale=(1 / 50), text_centre_scale=(1.0, 1.0), **kws
            ),
            "dmsecond": ScreenTextFontRel(
                font_size_scale=(1 / 50), text_centre_scale=(1.04, 1.1), **kws
            ),
        }

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

    def draw(self, v, **kwargs):
        for k, f in self.text_fonts.items():
            t = getattr(self, k)
            if k != "hour":
                f.set_text(f":{t:02}")
            else:
                f.set_text(f"{t:02}")
            f.draw(v, **kwargs)

    def set_default_color(self, color):
        super().set_default_color(color)
        for k, v in self.text_fonts.items():
            v.text_color = self.color


class SimulationScale(PosPixelRel2AreaViewable):
    pos_scale = param.NumericTuple((0.1, 0.04))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kws = {
            "reference_object": self,
            "text_color": self.color,
        }
        self.text_font = ScreenTextFontRel(
            font_size_scale=(1 / 40), text_centre_scale=(1, 1.5), **kws
        )

        self.lines = None
        # self.update_scale()

    # @param.depends('reference_area.zoom', watch=True)
    # def update_scale(self):
    #     def closest(lst, k):
    #         return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]
    #
    #     w_in_mm = self.reference_area.model.space.w * self.reference_area.zoom * 1000
    #     # Get 1/10 of max real dimension, transform it to mm and find the closest reasonable scale
    #     self.scale_in_mm = closest(
    #         lst=[0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000], k=w_in_mm / 10)
    #     self.text_font.set_text(f'{self.scale_in_mm} mm')
    #     self.lines = self.compute_lines(self.x, self.y, self.scale_in_mm / w_in_mm * self.reference_area.w)
    #
    # def compute_lines(self, x, y, scale):
    #     return [[(x - scale / 2, y), (x + scale / 2, y)],
    #             [(x + scale / 2, y * 0.75), (x + scale / 2, y * 1.25)],
    #             [(x - scale / 2, y * 0.75), (x - scale / 2, y * 1.25)]]

    def draw(self, v, **kwargs):
        for line in self.lines:
            pygame.draw.line(v.v, self.color, line[0], line[1], 1)
        # v.draw_text_box(self.text_font, self.text_font_r)
        self.text_font.draw(v, **kwargs)

    def set_default_color(self, color):
        super().set_default_color(color)
        self.text_font.text_color = self.color


class SimulationState(PosPixelRel2AreaViewable):
    pos_scale = param.NumericTuple((0.85, 0.94))

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        kws = {
            "reference_object": self,
            "text_color": self.color,
        }
        self.text_font = ScreenTextFontRel(
            font_size_scale=(1 / 40), text_centre_scale=(1, 1), **kws
        )

    def set_text(self, text):
        self.text_font.set_text(text)

    def draw(self, v, **kwargs):
        self.text_font.draw(v, **kwargs)

    def set_default_color(self, color):
        super().set_default_color(color)
        self.text_font.text_color = self.color
