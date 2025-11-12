"""
Screen side-panel for pygame-based simulation visualization
"""

from __future__ import annotations

import math
import os

import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from .. import util

__all__: list[str] = [
    "SidePanel",
]


class SidePanel:
    """
    A class representing a side panel for displaying information in a Pygame viewer.
    """

    FONT_SIZE = 30
    LINE_SPACING_MIN = 25
    LINE_SPACING_MAX = 35
    SCENE_HEIGHT_THRESHOLD = 700
    DEFAULT_MARGIN = 35
    LEFT_MARGIN = 30

    def __init__(self, viewer):
        """
        Initializes a SidePanel instance.

        :param viewer: The Pygame viewer associated with this side panel.
        """
        self.viewer = viewer
        self.line_num = None
        self.line_spacing = (
            self.LINE_SPACING_MAX
            if self.viewer.h > self.SCENE_HEIGHT_THRESHOLD
            else self.LINE_SPACING_MIN
        )
        try:
            import pygame

            if pygame.font:
                self.font = pygame.font.Font(None, self.FONT_SIZE)
            else:
                self.font = None
        except Exception:
            self.font = None
        try:
            import pygame

            self.panel_rect = pygame.Rect(
                self.viewer.w, 0, self.viewer.panel_width, self.viewer.h
            )
        except Exception:
            self.panel_rect = None

    def display_ga_info(self):
        """
        Displays information about the Genetic Algorithm (GA) on the side panel.
        """
        v = self.viewer
        m = v.model
        self.line_num = 1
        """
        Renders introductory information on the side panel.
        """
        cur_t = util.TimeUtil.current_time_millis()
        cum_t = math.floor((cur_t - m.start_total_time) / 1000)
        lines = [
            "Total time: " + util.TimeUtil.format_time_seconds(cum_t),
            "Generation: " + str(m.generation_num),
            "Population: " + str(len(m.agents)) + "/" + str(m.selector.Nagents),
            "Generation real-time: " + util.TimeUtil.format_time_seconds(m.t * m.dt),
            "",
        ]

        for line in lines:
            self.render_line(line)
        """
        Renders results and information about the current state.
        """
        g0 = m.best_genome
        if g0 is not None:
            self.render_line("Max fitness: " + str(round(g0.fitness, 2)))
            if g0.fitness_dict is not None:
                for name, dic in g0.fitness_dict.items():
                    self.render_line(f"{name} error: ")
                    for short, ks in dic.items():
                        self.render_line(
                            f"{short}: " + str(np.round(ks, 2)), self.LEFT_MARGIN
                        )
            self.render_line("Best genome: ")
            for k, p in m.selector.space_objs.items():
                self.render_line(f"{p.name}: {g0.gConf[k]}", self.LEFT_MARGIN)
        else:
            self.render_line("No best genome yet!")
        """
        Renders control instructions on the side panel.
        """
        lines = [
            "+ : increase scene speed",
            "- : decrease scene speed",
            "R : restart",
            "ESC : quit",
        ]
        self.render_line("")
        self.render_line("Controls:")
        for line in lines:
            self.render_line(line, self.LEFT_MARGIN)

    def render_line(self, text, extra_margin=0):
        """
        Renders a text line on the side panel.

        :param text: The text to render.
        :param extra_margin: Additional margin for the text.
        """
        line = self.font.render(text, 1, self.viewer.tank_color)
        x = self.viewer.w + self.DEFAULT_MARGIN + extra_margin
        y = self.line_num * self.line_spacing
        import pygame

        lint_pos = pygame.Rect(x, y, 20, 20)
        self.viewer.draw_text_box(line, lint_pos)
        self.line_num += 1

    def draw(self, v, **kwargs):
        """
        Draws the side panel on the Pygame viewer.

        :param v: The Pygame viewer.
        :param kwargs: Additional drawing arguments.
        """
        # Draw a black background for the side panel
        import pygame

        if self.panel_rect is not None:
            pygame.draw.rect(v.v, v.sidepanel_color, self.panel_rect)
        v.draw_line((v.w, 0), (v.w, v.h), color=util.Color.RED)
        self.display_ga_info()
