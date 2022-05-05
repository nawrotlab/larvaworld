from typing import Tuple

import pygame
import math
import numpy as np

from lib.ga.util.color import Color
from lib.ga.util.time_util import TimeUtil


class SidePanel:
    FONT_SIZE = 30
    LINE_SPACING_MIN = 25
    LINE_SPACING_MAX = 35
    SCENE_HEIGHT_THRESHOLD = 700
    DEFAULT_MARGIN = 35
    LEFT_MARGIN = 30

    def __init__(self, scene):
        self.scene = scene
        self.screen = scene.screen
        self.Nagents = None
        self.generation_num = None
        self.best_genome = None
        self.fitness_best_genome = None
        self.cum_t = None
        self.gen_t = None
        self.line_num = None
        self.line_spacing = None

    def update_ga_data(self, generation_num, best_genome, fitness_best_genome):
        self.generation_num = generation_num
        self.best_genome = best_genome
        self.fitness_best_genome = fitness_best_genome

    def update_ga_population(self, Nagents):
        self.Nagents = Nagents

    def update_ga_time(self, cum_t, gen_t, gen_sim_t):
        self.cum_t = cum_t
        self.gen_t = gen_t
        self.gen_sim_t = gen_sim_t

    def display_ga_info(self, space_dict):
        pygame.draw.line(self.screen, Color.GRAY, (self.scene.width, 0), (self.scene.width, self.scene.height))

        if pygame.font:
            font = pygame.font.Font(None, self.FONT_SIZE)
            self.line_num = 1
            self.line_spacing = self.LINE_SPACING_MAX if self.scene.height > self.SCENE_HEIGHT_THRESHOLD else self.LINE_SPACING_MIN
            # if self.scene.height > self.SCENE_HEIGHT_THRESHOLD:
            #     self.line_spacing = self.LINE_SPACING_MAX
            # else:
            #     self.line_spacing = self.LINE_SPACING_MIN
            #
            fitness_best = '-' if self.best_genome is None else str(round(self.fitness_best_genome, 2))

            # if self.best_genome is None:
                # this happens only at the first generation
                # fitness_best = '-'
                # generation_num_best = '-'
            # else:
            #     fitness_best = str(round(self.fitness_best_genome, 2))
            #     generation_num_best = str(self.best_genome.generation_num)

            self.render_line(font, 'Total time: ' + TimeUtil.format_time_seconds(self.cum_t))
            self.render_line(font, 'Generation: ' + str(self.generation_num))
            self.render_line(font, 'Population: ' + str(self.Nagents))

            self.render_line(font, 'Generation real-time: ' + TimeUtil.format_time_seconds(self.gen_sim_t))
            self.render_line(font, '')
            self.render_line(font, 'Max fitness: ' + fitness_best)
            if self.best_genome is not None and self.best_genome.fitness_dict is not None :
                for short, ks in self.best_genome.fitness_dict.items():
                    self.render_line(font, f'{short}: ' + str(np.round(ks,2)), self.LEFT_MARGIN)
            self.render_line(font, 'Best genome: ' )
            for k, vs in space_dict.items():
                pkey = '-' if self.best_genome is None else str(self.best_genome.get(rounded=True)[k])
                self.render_line(font, f'{vs["name"]}: ' + pkey, self.LEFT_MARGIN)
            self.render_line(font, '')
            self.render_line(font, 'Controls:')
            self.render_line(font, 'S : save current genomes to file', self.LEFT_MARGIN)
            self.render_line(font, '+ : increase scene speed', self.LEFT_MARGIN)
            self.render_line(font, '- : decrase scene speed', self.LEFT_MARGIN)
            self.render_line(font, 'R : restart', self.LEFT_MARGIN)
            self.render_line(font, 'ESC : quit', self.LEFT_MARGIN)

    def display_info(self, object_to_place):
        pygame.draw.line(self.screen, Color.GRAY, (self.scene.width, 0), (self.scene.width, self.scene.height))

        if pygame.font:
            font = pygame.font.Font(None, self.FONT_SIZE)
            self.line_num = 1
            self.line_spacing = self.LINE_SPACING_MAX

            self.render_line(font, 'Controls:')
            self.render_line(font, 'Click left : add ' + object_to_place, self.LEFT_MARGIN)
            self.render_line(font, 'Click right : remove ' + object_to_place, self.LEFT_MARGIN)
            self.render_line(font, 'J : add a vehicle', self.LEFT_MARGIN)
            self.render_line(font, 'K : remove a vehicle', self.LEFT_MARGIN)
            self.render_line(font, 'S : save current scene to file', self.LEFT_MARGIN)
            self.render_line(font, '+ : incrase scene speed', self.LEFT_MARGIN)
            self.render_line(font, '- : decrase scene speed', self.LEFT_MARGIN)
            self.render_line(font, 'R : restart', self.LEFT_MARGIN)
            self.render_line(font, 'ESC : quit', self.LEFT_MARGIN)

    def render_line(self, font, text, extra_margin=0):
        line = font.render(text, 1, Color.WHITE)
        x = self.scene.width + self.DEFAULT_MARGIN + extra_margin
        y = self.line_num * self.line_spacing
        lint_pos = pygame.Rect(x, y, 20, 20)
        self.screen.blit(line, lint_pos)
        self.line_num += 1
