import math
import random
from math import sin, cos, pi
import pygame

import numpy as np

from lib.aux.ang_aux import rotate_around_center_multi
from lib.ga.exception.collision_exception import Collision
from lib.ga.util.color import Color
from lib.model.modules.locomotor import DefaultLocomotor
from lib.process.spatial import straightness_index



class LarvaShape:

    def __init__(self, model, color_fg, color_bg, pos=(0, 0), direction=None, length=0.005,
                 seg_ratio=None, Nsegs=2, shape='drosophila_larva'):
        if direction is None:
            direction = random.uniform(0, 2 * pi)
        self.direction = direction

        self.x = model.scene.width / 2 + pos[0] * model.arena_scale
        self.y = model.scene.height / 2 + pos[1] * model.arena_scale
        self.model = model
        self.length = length
        self.size = int(length * model.arena_scale)
        self.Nsegs = Nsegs
        from lib.conf.stored.aux_conf import body_dict
        from lib.aux.sim_aux import generate_seg_shapes, circle_to_polygon
        if seg_ratio is None:
            seg_ratio = np.array([1 / Nsegs] * Nsegs)
        base_vertices = generate_seg_shapes(Nsegs, seg_ratio=seg_ratio, points=body_dict[shape]['points'])
        seg_ps = [[0.5 + (-i + (Nsegs - 1) / 2) * seg_ratio[i], 0.5] for i in range(Nsegs)]
        seg_rots = [np.array([0.5 + (-i + (Nsegs - 2) / 2) * seg_ratio[i], 0.5]) for i in range(Nsegs)]
        self.seg_vertices = [[seg_ps[ii] + base_vertices[ii][0]][0] for ii in range(Nsegs)]

        self.seg2_mid = seg_rots[0]
        self.seg2_bases = [base_vertices[0][0] - self.seg2_mid + seg_ps[0],
                           base_vertices[1][0] - self.seg2_mid + seg_ps[1]]

        self.color_fg = color_fg
        self.color_bg = color_bg

    def draw(self, screen):
        surf = pygame.Surface((self.size, self.size))
        surf.fill(self.color_bg)
        surf.set_colorkey(self.color_bg)
        for vs in self.seg_vertices:
            pygame.draw.polygon(surf, self.color_fg, vs * self.size)
        rect = surf.get_rect()
        rect.center = (self.x, self.y)
        screen.blit(surf, rect)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def sim_length(self):
        return self.length


class LarvaRobot(LarvaShape):

    def __init__(self, unique_id, model, **kwargs):

        super().__init__(model, Color.random_color(127, 127, 127), Color.BLACK)
        self.unique_id = unique_id
        self.eval = {
            'b': [],
            'fov': [],
        }
        self.trajectory = []
        self.brain = DefaultLocomotor(dt=self.model.dt, offline=True, **kwargs)

    def step(self):
        """ updates x, y and direction """
        lin, ang, feed = self.brain.step(A_in=0, length=self.length)
        dst=self.brain.last_dist * self.model.arena_scale
        self.direction += ang * self.brain.dt
        self.x += np.cos(self.direction) * dst
        self.y += np.sin(self.direction) * dst
        self.seg_vertices[0] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[0], -self.direction)
        self.seg_vertices[1] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[1],
                                                                          -self.direction + self.brain.bend)
        self.eval['b'].append(self.brain.bend)
        self.eval['fov'].append(ang)
        self.trajectory.append(self.pos)

    def finalize(self, eval_shorts=['b', 'fov']):
        self.trajectory = np.array(self.trajectory) / self.model.arena_scale
        self.eval['b'] = np.rad2deg(self.eval['b'])
        self.eval['fov'] = np.rad2deg(self.eval['fov'])
        if 'foa' in eval_shorts:
            self.eval['foa'] = np.diff(self.eval['fov']) / self.brain.dt
        if 'tor5' in eval_shorts:
            self.eval['tor5'] = straightness_index(self.trajectory, int(5 / self.brain.dt / 2))
        if 'tor2' in eval_shorts:
            self.eval['tor2'] = straightness_index(self.trajectory, int(2 / self.brain.dt / 2))

    def print_xyd(self):
        """ prints the x,y position and direction """
        print("x = " + str(self.x) + " " + "y = " + str(self.y))
        print("direction = " + str(self.direction))

    def sense_and_act(self):
        self.step()

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x + (self.size / 2), self.y + (self.size / 2), 50, 50)
            screen.blit(text, text_pos)

class ObstacleLarvaRobot(LarvaRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.sensor_delta_direction = 0.44
        # self.sensor_saturation_value= 0.44
        # self.sensor_error= 0.44
        # self.sensor_max_distance= 0.44
        self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None

    def sense_and_act(self):
        if not self.collision_with_object:
            try:
                self.left_motor_controller.sense_and_act()
                self.right_motor_controller.sense_and_act()
                self.speed_left_wheel = self.left_motor_controller.get_actuator_value()
                self.speed_right_wheel = self.right_motor_controller.get_actuator_value()
                self.step()
            except Collision:
                self.collision_with_object = True
                self.speed_left_wheel = 0
                self.speed_right_wheel = 0
        else:
            # a collision has already occured
            self.speed_left_wheel = 0
            self.speed_right_wheel = 0

    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, screen):
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw()
            self.right_motor_controller.sensor.draw()

        # call super method to draw the robot
        super().draw(screen)
