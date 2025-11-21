"""
Screen renderable items for pygame-based simulation visualization
"""

from __future__ import annotations
from typing import Any

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

__all__: list[str] = [
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

    def __init__(self, end_time: int = 0, start_time: int = 0, **kwargs: Any):
        self.font = None
        self.text_font = None
        self.text_font_r = None
        super().__init__(**kwargs)
        self.end_time = end_time
        self.start_time = start_time
        if not self.font:
            self.update_font()

    @param.depends("text", "text_color", "text_centre", watch=True)
    def render_text(self) -> None:
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
    def text_lines(self) -> list[str]:
        return self.text.splitlines()

    @property
    def N_text_lines(self) -> int:
        return len(self.text_lines)

    @param.depends("font_size", watch=True)
    def update_font(self) -> None:
        import pygame

        pygame.init()
        self.font = pygame.font.SysFont(self.font_type, self.font_size)

    def draw(self, v: Any, **kwargs: Any) -> None:
        if self.text_font is None or self.text_font_r is None:
            self.render_text()
        if self.N_text_lines == 1:
            v.draw_text_box(self.text_font, self.text_font_r)
        else:
            for i in range(self.N_text_lines):
                v.draw_text_box(self.text_font[i], self.text_font_r[i])

    def set_text(self, text: str) -> None:
        self.text = text

    def flash_text(self, text: str, t: int = 2) -> None:
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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.update_font_size(self.reference_object)
        self.update_font_centre_pos(self.reference_object)

    @param.depends("reference_object.pos", "text_centre_scale", watch=True)
    def update_font_centre_pos(self, obj: Any) -> None:
        dx, dy = self.text_centre_scale
        self.text_centre = (obj.x * dx, obj.y * dy)

    # @param.depends('reference_object', watch=True)
    def update_font_size(self, obj: Any) -> None:
        self.font_size = int(obj.reference_area.w * self.font_size_scale)


class ScreenTextBoxRect(ScreenTextFont, Viewable):
    """
    Text box with rectangular frame at fixed screen position.

    Displays text within a rectangular frame, commonly used for
    labels and status indicators in pygame visualizations.

    Attributes:
        visible: Whether the text box is visible
        frame_rect: The rectangular frame object
        linewidth: Width of the frame border
        show_frame: Whether to draw the rectangular frame

    Example:
        >>> text_box = ScreenTextBoxRect(text="Status", frame_rect=rect)
        >>> text_box.draw(viewer)
    """

    visible = param.Boolean(False)
    frame_rect = param.ClassSelector(class_=object, doc="The frame rectangle")
    linewidth = PositiveNumber(10.0, doc="The linewidth to draw the box")
    show_frame = param.Boolean(True, doc="Draw the rectangular frame around the text")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.text_centre = self.frame_rect.center

    def draw(self, v: Any, **kwargs: Any) -> None:
        if self.show_frame and self.frame_rect is not None:
            import pygame

            pygame.draw.rect(
                v.v,
                color=self.text_color,
                rect=self.frame_rect,
                width=int(self.linewidth),
            )

        super().draw(v=v, **kwargs)


class ScreenTextBox(ScreenTextFont, ViewableToggleable, Area2DPixel):
    """
    Toggle-able text box with frame at pixel coordinates.

    Displays text within a rectangular area that can be toggled on/off,
    with optional frame rendering for UI elements.

    Attributes:
        dims: Dimensions as (width, height) pixel tuple
        visible: Whether the text box is visible
        linewidth: Width of the frame border
        show_frame: Whether to draw the rectangular frame

    Example:
        >>> text_box = ScreenTextBox(text="Info", dims=(200, 40))
        >>> text_box.toggle()  # Show/hide
    """

    dims = IntegerTuple(default=(140, 32))
    visible = param.Boolean(False)
    linewidth = PositiveNumber(0.001, doc="The linewidth to draw the box")
    show_frame = param.Boolean(True, doc="Draw the rectangular frame around the text")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.frame_rect = None

    def set_frame_rect(self, pos: Any | None = None, **kwargs: Any):
        return self.get_rect_at_pos(pos, **kwargs)

    def draw(self, v: Any, **kwargs: Any) -> None:
        if self.show_frame:
            if self.frame_rect is not None:
                # v.draw_polygon(self.shape, color=self.color, filled=False, width=self.linewidth)
                # pygame.draw.rect(v._window, color=self.color, rect=self.shape)
                import pygame

                pygame.draw.rect(
                    v.v,
                    color=self.color,
                    rect=self.frame_rect,
                    width=int(v._scale[0, 0] * self.linewidth),
                )


class IDBox(ScreenTextFont, ViewableToggleable):
    """
    Text box displaying agent ID that follows the agent.

    Renders agent unique ID as text near the agent position,
    using the agent's color for visibility during visualization.

    Attributes:
        visible: Whether the ID box is visible
        agent: The agent whose ID is displayed

    Example:
        >>> id_box = IDBox(agent=larva_agent)
        >>> id_box.draw(viewer)  # Draws ID text near agent
    """

    visible = param.Boolean(False)
    agent = param.ClassSelector(class_=Pos2D, doc="The agent owning the ID")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.update_font()
        self.update_agent()

    def update_agent(self) -> None:
        self.text_color = self.agent.color
        self.set_text(self.agent.unique_id)

    # @param.depends('agent.pos', watch=True)
    def update_font_centre_pos(self, v: Any) -> None:
        pos = self.agent.get_position()
        x, y = v.space2screen_pos(pos)
        self.text_centre = x + 50, y + 12

    def draw(self, v: Any, **kwargs: Any) -> None:
        self.update_font_centre_pos(v)
        self.update_agent()
        ScreenTextFont.draw(self, v=v, **kwargs)


class PosPixelRel2AreaViewable(PosPixelRel2Area, Viewable):
    pass


class ScreenMsgText(ScreenTextFontRel, Viewable):
    """
    Message text display with relative positioning.

    Displays temporary or persistent messages at screen positions
    relative to a reference area, used for notifications and alerts.

    Attributes:
        text_centre_scale: Text center position relative to reference
        font_size_scale: Font size relative to window size
        font_type: Font type for rendering

    Example:
        >>> msg = ScreenMsgText(reference_area=screen_area, text="Paused")
        >>> msg.draw(viewer)
    """

    text_centre_scale = param.NumericTuple(
        (0.91, 1), doc="The text center position relative to the position"
    )
    font_size_scale = PositiveNumber(
        1 / 25, doc="The font size relative to the window size"
    )
    font_type = param.Parameter(default="SansitaOne.tff")

    def __init__(self, reference_area: Any, **kwargs: Any) -> None:
        reference_object = PosPixelRel2Area(
            reference_area=reference_area, pos_scale=(0.95, 0.1)
        )
        super().__init__(reference_object=reference_object, **kwargs)

    def draw(self, v: Any, **kwargs: Any) -> None:
        ScreenTextFont.draw(self, v=v, **kwargs)
        # self.text_font.draw(v, **kwargs)

    def set_default_color(self, color: Any) -> None:
        super().set_default_color(color)
        self.text_color = self.color


class SimulationClock(PosPixelRel2AreaViewable):
    """
    Clock display for simulation time tracking.

    Renders current simulation time in HH:MM:SS:ds format (hours:minutes:seconds:deciseconds)
    at a fixed screen position during visualization.

    Attributes:
        pos_scale: Position scale relative to screen area
        sim_step_in_dms: Simulation step size in deciseconds
        hour, minute, second, dmsecond: Current time components
        text_fonts: Dictionary of font objects for each time component

    Example:
        >>> clock = SimulationClock(sim_step_in_sec=0.1, reference_area=screen)
        >>> clock.tick_clock()  # Advance by one time step
        >>> clock.draw(viewer)  # Render current time
    """

    pos_scale = param.NumericTuple((0.94, 0.04))

    def __init__(self, sim_step_in_sec: float, **kwargs: Any) -> None:
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

    def tick_clock(self) -> None:
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

    def draw(self, v: Any, **kwargs: Any) -> None:
        for k, f in self.text_fonts.items():
            t = getattr(self, k)
            if k != "hour":
                f.set_text(f":{t:02}")
            else:
                f.set_text(f"{t:02}")
            f.draw(v, **kwargs)

    def set_default_color(self, color: Any) -> None:
        super().set_default_color(color)
        for k, v in self.text_fonts.items():
            v.text_color = self.color


class SimulationScale(PosPixelRel2AreaViewable):
    """
    Scale bar display for spatial reference.

    Renders a scale bar indicating spatial dimensions in millimeters,
    helping interpret distances in the visualization.

    Attributes:
        pos_scale: Position scale relative to screen area
        text_font: Font object for scale label
        lines: Line segments forming the scale bar

    Example:
        >>> scale = SimulationScale(reference_area=screen)
        >>> scale.draw(viewer)  # Draws scale bar with mm label
    """

    pos_scale = param.NumericTuple((0.1, 0.04))

    def __init__(self, **kwargs: Any) -> None:
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

    def draw(self, v: Any, **kwargs: Any) -> None:
        for line in self.lines:
            import pygame

            pygame.draw.line(v.v, self.color, line[0], line[1], 1)
        # v.draw_text_box(self.text_font, self.text_font_r)
        self.text_font.draw(v, **kwargs)

    def set_default_color(self, color: Any) -> None:
        super().set_default_color(color)
        self.text_font.text_color = self.color


class SimulationState(PosPixelRel2AreaViewable):
    """
    Simulation state display for runtime information.

    Renders current simulation state (running, paused, etc.) and
    other status information during visualization.

    Attributes:
        pos_scale: Position scale relative to screen area
        model: Reference to the simulation model
        text_font: Font object for state text

    Example:
        >>> state = SimulationState(model=sim_model, reference_area=screen)
        >>> state.set_text("PAUSED")
        >>> state.draw(viewer)
    """

    pos_scale = param.NumericTuple((0.85, 0.94))

    def __init__(self, model: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        kws = {
            "reference_object": self,
            "text_color": self.color,
        }
        self.text_font = ScreenTextFontRel(
            font_size_scale=(1 / 40), text_centre_scale=(1, 1), **kws
        )

    def set_text(self, text: str) -> None:
        self.text_font.set_text(text)

    def draw(self, v: Any, **kwargs: Any) -> None:
        self.text_font.draw(v, **kwargs)

    def set_default_color(self, color: Any) -> None:
        super().set_default_color(color)
        self.text_font.text_color = self.color
