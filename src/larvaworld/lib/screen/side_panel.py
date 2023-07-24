import math
import os
import numpy as np
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from larvaworld.lib import aux



class SidePanel:
    FONT_SIZE = 30
    LINE_SPACING_MIN = 25
    LINE_SPACING_MAX = 35
    SCENE_HEIGHT_THRESHOLD = 700
    DEFAULT_MARGIN = 35
    LEFT_MARGIN = 30

    def __init__(self, viewer):
        self.viewer = viewer
        self.line_num = None
        self.line_spacing = self.LINE_SPACING_MAX if self.viewer.h > self.SCENE_HEIGHT_THRESHOLD else self.LINE_SPACING_MIN
        if pygame.font:
            self.font = pygame.font.Font(None, self.FONT_SIZE)
        else :
            self.font = None
        self.panel_rect = pygame.Rect(self.viewer.w, 0, self.viewer.manager.panel_width, self.viewer.h)



    #@ property
    def render_intro(self):
        m = self.viewer.manager.model
        cur_t = aux.TimeUtil.current_time_millis()
        cum_t = math.floor((cur_t - m.start_total_time) / 1000)
        lines = [
            'Total time: ' + aux.TimeUtil.format_time_seconds(cum_t),
            'Generation: ' + str(m.generation_num),
            'Population: ' + str(len(m.agents)) + '/' + str(m.Nagents),
            'Generation real-time: ' + aux.TimeUtil.format_time_seconds(m.generation_step_num*m.dt),
            '',
        ]

        for line in lines :
            self.render_line(line)

    def render_controls(self):
        lines = [
            '+ : increase scene speed',
            '- : decrase scene speed',
            'R : restart',
            'ESC : quit',
        ]
        self.render_line('')
        self.render_line('Controls:')
        for line in lines:
            self.render_line(line, self.LEFT_MARGIN)

    def render_results(self):
        m = self.viewer.manager.model
        best_gen=m.best_genome
        if best_gen is not None :
            self.render_line('Max fitness: ' + str(round(best_gen.fitness, 2)))
            if best_gen.fitness_dict is not None :
                for name, dic in best_gen.fitness_dict.items():
                    self.render_line(f'{name} error: ')
                    for short, ks in dic.items():
                        self.render_line(f'{short}: ' + str(np.round(ks, 2)), self.LEFT_MARGIN)
            self.render_line('Best genome: ')
            for k, p in m.space_dict.items():
                self.render_line(f'{p.name}: {best_gen.gConf[k]}', self.LEFT_MARGIN)
        else :
            self.render_line('No best genome yet!')


    def display_ga_info(self):
        self.line_num = 1
        self.render_intro()
        self.render_results()
        self.render_controls()



    def render_line(self, text, extra_margin=0):
        line = self.font.render(text, 1, aux.Color.WHITE)
        x = self.viewer.w + self.DEFAULT_MARGIN + extra_margin
        y = self.line_num * self.line_spacing
        lint_pos = pygame.Rect(x, y, 20, 20)
        self.viewer.draw_text_box(line, lint_pos)
        self.line_num += 1

    def draw(self, v, **kwargs):
        # draw a black background for the side panel
        pygame.draw.rect(v._window, aux.Color.BLACK, self.panel_rect)
        v.draw_line((v.w, 0), (v.w, v.h), color=aux.Color.WHITE)
        self.display_ga_info()