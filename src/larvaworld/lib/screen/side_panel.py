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
        self.line_spacing = None


    def display_ga_info(self):
        m = self.viewer.model
        v = self.viewer
        best_gen=m.best_genome

        cur_t = aux.TimeUtil.current_time_millis()
        cum_t = math.floor((cur_t - m.start_total_time) / 1000)
        # gen_t = math.floor((cur_t - m.start_generation_time) / 1000)


        v.draw_line((v.width, 0), (v.width, v.height), color=aux.Color.WHITE)

        if pygame.font:
            font = pygame.font.Font(None, self.FONT_SIZE)
            self.line_num = 1
            self.line_spacing = self.LINE_SPACING_MAX if v.height > self.SCENE_HEIGHT_THRESHOLD else self.LINE_SPACING_MIN


            self.render_line(font, 'Total time: ' + aux.TimeUtil.format_time_seconds(cum_t))
            self.render_line(font, 'Generation: ' + str(m.generation_num))
            self.render_line(font, 'Population: ' + str(len(m.agents)) +'/'+ str(m.Nagents))
            self.render_line(font, 'Generation real-time: ' + aux.TimeUtil.format_time_seconds(m.generation_sim_time))
            self.render_line(font, '')

            if best_gen is not None :
                self.render_line(font, 'Max fitness: ' + str(round(best_gen.fitness, 2)))
                if best_gen.fitness_dict is not None :
                    for name, dic in best_gen.fitness_dict.items():
                        self.render_line(font, f'{name} error: ')
                        for short, ks in dic.items():
                            self.render_line(font, f'{short}: ' + str(np.round(ks, 2)), self.LEFT_MARGIN)
                self.render_line(font, 'Best genome: ')
                for k, p in m.space_dict.items():
                    self.render_line(font, f'{p.name}: {best_gen.gConf[k]}', self.LEFT_MARGIN)
            else :
                self.render_line(font, 'No best genome yet!')

            self.render_line(font, '')
            self.render_line(font, 'Controls:')
            self.render_line(font, 'S : save current genomes to file', self.LEFT_MARGIN)
            self.render_line(font, 'E : evaluate and plot best genome', self.LEFT_MARGIN)
            self.render_line(font, '+ : increase scene speed', self.LEFT_MARGIN)
            self.render_line(font, '- : decrase scene speed', self.LEFT_MARGIN)
            self.render_line(font, 'R : restart', self.LEFT_MARGIN)
            self.render_line(font, 'ESC : quit', self.LEFT_MARGIN)


    def render_line(self, font, text, extra_margin=0):
        line = font.render(text, 1, aux.Color.WHITE)
        x = self.viewer.width + self.DEFAULT_MARGIN + extra_margin
        y = self.line_num * self.line_spacing
        lint_pos = pygame.Rect(x, y, 20, 20)
        self.viewer._window.blit(line, lint_pos)
        self.line_num += 1
