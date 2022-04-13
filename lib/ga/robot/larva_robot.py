import math
import random
from math import sin, cos, pi
import pygame
import pandas as pd
import numpy as np

from lib.aux.ang_aux import rotate_around_center_multi
from lib.ga.exception.collision_exception import Collision
from lib.ga.util.color import Color
from lib.model.modules.locomotor import DefaultLocomotor
from lib.process.aux import detect_turns, process_epochs, detect_strides, fft_max, detect_pauses
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

        # print(self.seg_vertices[0][0])
        # raise

        self.seg2_mid = seg_rots[0]
        self.seg2_bases = [base_vertices[0][0] - self.seg2_mid + seg_ps[0],
                           base_vertices[1][0] - self.seg2_mid + seg_ps[1]]
        # self.seg_vertices[0] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[0], -self.direction)
        # self.seg_vertices[1] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[1],
        #                                                                   -self.direction)


        self.color_fg = color_fg
        self.color_bg = color_bg

    @ property
    def head_pos(self):
        dx,dy=(self.seg_vertices[0][0]-0.5)*self.size
        return [self.x + dx, self.y + dy]

    def draw(self, screen):
        s=self.size
        surf = pygame.Surface((s, s))
        surf.fill(self.color_bg)
        surf.set_colorkey(self.color_bg)
        for vs in self.seg_vertices:
            pygame.draw.polygon(surf, self.color_fg, vs * s)
        pygame.draw.line(surf, Color.RED, (s/2, s/2),
                         ((0.5 + np.cos(self.direction)) *s, (0.5 + np.sin(self.direction)) * s))
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
            'v': [],
        }
        self.trajectory = []
        self.brain = DefaultLocomotor(dt=self.model.dt, offline=True, **kwargs)

    def update_pose(self):
        dst = self.brain.last_dist * self.model.arena_scale
        self.direction += self.brain.ang_vel * self.brain.dt
        self.x += np.cos(self.direction) * dst
        self.y += np.sin(self.direction) * dst

    def update_vertices(self):
        self.seg_vertices[0] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[0], -self.direction)
        self.seg_vertices[1] = self.seg2_mid + rotate_around_center_multi(self.seg2_bases[1],
                                                                          -self.direction + self.brain.bend)

    def store(self):
        self.eval['b'].append(self.brain.bend)
        self.eval['fov'].append(self.brain.ang_vel)
        self.eval['v'].append(self.brain.lin_vel)
        self.trajectory.append(self.pos)

    def finalize(self, eval_shorts=['b', 'fov']):
        dt=self.brain.dt
        self.trajectory = np.array(self.trajectory) / self.model.arena_scale
        self.eval['v'] = np.array(self.eval['v'])
        self.eval['b'] = np.rad2deg(self.eval['b'])
        self.eval['fov'] = np.rad2deg(self.eval['fov'])
        if 'bv' in eval_shorts:
            self.eval['bv'] = np.diff(self.eval['b']) / dt
        if 'a' in eval_shorts:
            self.eval['a'] = np.diff(self.eval['v']) / dt
        if 'foa' in eval_shorts:
            self.eval['foa'] = np.diff(self.eval['fov']) / dt
        if 'tor5' in eval_shorts:
            self.eval['tor5'] = straightness_index(self.trajectory, int(5 / dt / 2))
        if 'tor2' in eval_shorts:
            self.eval['tor2'] = straightness_index(self.trajectory, int(2 / dt / 2))
        if 'tor20' in eval_shorts:
            self.eval['tor20'] = straightness_index(self.trajectory, int(20 / dt / 2))
        if 'tur_fou' in eval_shorts:
            a_fov = pd.Series(self.eval['fov'])
            Lturns, Rturns = detect_turns(a_fov, dt)

            Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a_fov, Lturns, dt)
            Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a_fov, Rturns, dt)
            self.eval['tur_fou'] = np.abs(np.concatenate([Lamps, Ramps]))
            self.eval['tur_t'] = np.concatenate([Ldurs, Rdurs])
            self.eval['tur_fov_max'] = np.abs(np.concatenate([Lmaxs, Rmaxs]))
        if 'run_t' in eval_shorts:
            a_sv=pd.Series(self.eval['v']/self.length)
            fv=fft_max(a_sv, dt, fr_range=(1.0, 2.5), return_amps=False)
            strides, runs, run_counts = detect_strides(a_sv, dt, fr=fv, return_extrema=False)
            # strides1, stride_durs, stride_slices, stride_dsts, stride_idx, stride_maxs = process_epochs(a_sv, strides,dt)
            pauses = detect_pauses(a_sv, dt, runs=runs)
            pauses1, pause_durs, pause_slices, pause_dsts, pause_idx, pause_maxs = process_epochs(a_sv, pauses, dt)
            runs1, run_durs, run_slices, run_dsts, run_idx, run_maxs = process_epochs(a_sv, runs, dt)
            self.eval['run_d'] = run_dsts
            self.eval['run_t'] = run_durs
            self.eval['pau_t'] = pause_durs

    def print_xyd(self):
        """ prints the x,y position and direction """
        print("x = " + str(self.x) + " " + "y = " + str(self.y))
        print("direction = " + str(self.direction))

    def sense_and_act(self):
        lin, ang, feed = self.brain.step(A_in=0, length=self.length)
        self.update_pose()
        self.update_vertices()
        self.store()

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x + (self.size / 2), self.y + (self.size / 2), 50, 50)
            screen.blit(text, text_pos)

class ObstacleLarvaRobot(LarvaRobot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None

    def sense_and_act(self):
        if not self.collision_with_object:
            try:
                self.left_motor_controller.sense_and_act()
                self.right_motor_controller.sense_and_act()
                Rtorque = self.left_motor_controller.get_actuator_value()
                Ltorque = self.right_motor_controller.get_actuator_value()
                # print(int(Rtorque), int(Ltorque), int(self.brain.ang_vel))
                torque=Ltorque-Rtorque
                if Ltorque+Rtorque>2 :
                    self.brain.intermitter.interrupt_locomotion()
                self.brain.ang_vel+=torque*self.brain.dt
                # self.step()
                # print(int(Rtorque), int(Ltorque), int(self.brain.ang_vel))
            except Collision:
                self.collision_with_object = True
                self.brain.intermitter.interrupt_locomotion()
                # self.step()
                # self.speed_left_wheel = 0
                # self.speed_right_wheel = 0
            lin, ang, feed = self.brain.step(A_in=0, length=self.length)
            self.update_pose()
            self.update_vertices()
            self.store()
        else:
            pass
            # a collision has already occured
            # self.speed_left_wheel = 0
            # self.speed_right_wheel = 0


    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, screen):
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw(pos=self.head_pos, direction=-self.direction)
            self.right_motor_controller.sensor.draw(pos=self.head_pos, direction=-self.direction)

        # call super method to draw the robot
        super().draw(screen)
