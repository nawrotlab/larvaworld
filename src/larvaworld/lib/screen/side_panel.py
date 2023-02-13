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

    def __init__(self, viewer, space_dict):
        self.viewer = viewer
        self.Nagents = None
        self.generation_num = None
        self.best_genome = None
        # self.fitness_best_genome = None
        self.cum_t = None
        self.gen_t = None
        self.line_num = None
        self.line_spacing = None
        self.space_dict = space_dict

    def update_ga_data(self, generation_num, best_genome):
        self.generation_num = generation_num
        self.best_genome = best_genome
        # self.fitness_best_genome = fitness_best_genome

    def update_ga_population(self, Nalive, Nagents):
        self.Nalive = Nalive
        self.Nagents = Nagents

    def update_ga_time(self, cum_t, gen_t, gen_sim_t):
        self.cum_t = cum_t
        self.gen_t = gen_t
        self.gen_sim_t = gen_sim_t

    def display_ga_info(self):
        self.viewer.draw_line((self.viewer.width, 0), (self.viewer.width, self.viewer.height), color=aux.Color.WHITE)

        if pygame.font:
            font = pygame.font.Font(None, self.FONT_SIZE)
            self.line_num = 1
            self.line_spacing = self.LINE_SPACING_MAX if self.viewer.height > self.SCENE_HEIGHT_THRESHOLD else self.LINE_SPACING_MIN

            fitness_best = '-' if self.best_genome is None else str(round(self.best_genome.fitness, 2))


            self.render_line(font, 'Total time: ' + aux.TimeUtil.format_time_seconds(self.cum_t))
            self.render_line(font, 'Generation: ' + str(self.generation_num))
            self.render_line(font, 'Population: ' + str(self.Nalive) +'/'+ str(self.Nagents))

            self.render_line(font, 'Generation real-time: ' + aux.TimeUtil.format_time_seconds(self.gen_sim_t))
            self.render_line(font, '')
            self.render_line(font, 'Max fitness: ' + fitness_best)
            if self.best_genome is not None and self.best_genome.fitness_dict is not None :
                for name,dic in self.best_genome.fitness_dict.items():
                    self.render_line(font, f'{name} error: ')
                    for short, ks in dic.items():
                        self.render_line(font, f'{short}: ' + str(np.round(ks,2)), self.LEFT_MARGIN)
            self.render_line(font, 'Best genome: ' )
            if self.best_genome is not None :
                for k, p in self.space_dict.items():
                    self.render_line(font, f'{p.name}: {self.best_genome.gConf[k]}', self.LEFT_MARGIN)
            self.render_line(font, '')
            self.render_line(font, 'Controls:')
            self.render_line(font, 'S : save current genomes to file', self.LEFT_MARGIN)
            self.render_line(font, 'E : evaluate and plot best genome', self.LEFT_MARGIN)
            self.render_line(font, '+ : increase scene speed', self.LEFT_MARGIN)
            self.render_line(font, '- : decrase scene speed', self.LEFT_MARGIN)
            self.render_line(font, 'R : restart', self.LEFT_MARGIN)
            self.render_line(font, 'ESC : quit', self.LEFT_MARGIN)

    def display_info(self, object_to_place):
        self.viewer.draw_line((self.viewer.width, 0), (self.viewer.width, self.viewer.height), color=aux.Color.WHITE)

        if pygame.font:
            font = pygame.font.Font(None, self.FONT_SIZE)
            self.line_num = 1
            self.line_spacing = self.LINE_SPACING_MAX

            self.render_line(font, 'Controls:')
            self.render_line(font, 'Click left : add ' + object_to_place, self.LEFT_MARGIN)
            self.render_line(font, 'Click right : remove ' + object_to_place, self.LEFT_MARGIN)
            self.render_line(font, 'J : add a vehicle', self.LEFT_MARGIN)
            self.render_line(font, 'K : remove a vehicle', self.LEFT_MARGIN)
            # self.render_line(font, 'S : save current scene to file', self.LEFT_MARGIN)
            self.render_line(font, '+ : incrase scene speed', self.LEFT_MARGIN)
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
