import math
import random
import pygame
import pandas as pd
import numpy as np

from lib.ga.exception.collision_exception import Collision
from lib.ga.util.color import Color
from lib.model.modules.locomotor import DefaultLocomotor
from lib.model.body.body import LarvaShape
from lib.process.aux import detect_turns, process_epochs, detect_strides, fft_max, detect_pauses
from lib.process.spatial import straightness_index


class LarvaRobot(LarvaShape):

    def __init__(self, unique_id, model, direction=None, Nsegs=2, x=None, y=None, **kwargs):
        if x is None and y is None:
            x, y = model.scene.width / 2, model.scene.height / 2
        super().__init__(Nsegs=Nsegs, scaling_factor=model.scaling_factor, initial_orientation=direction,
                         initial_pos=(x, y), default_color=Color.random_color(127, 127, 127))
        self.direction = self.initial_orientation
        self.x, self.y = self.initial_pos
        self.model = model
        self.size = self.sim_length

        self.segs = self.generate_segs(self.initial_orientation, seg_positions=self.seg_positions)
        self.unique_id = unique_id
        self.eval = {
            'b': [],
            'fov': [],
            'v': [],
        }
        self.dst = 0
        self.trajectory = []
        self.brain = DefaultLocomotor(dt=self.model.dt, offline=True, **kwargs)

    def draw(self, screen):
        for seg in self.segs:
            for vs in seg.vertices:
                pygame.draw.polygon(screen, seg.color, vs)

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def update_pose(self):
        self.dst = self.brain.last_dist * self.model.scaling_factor
        self.direction += self.brain.ang_vel * self.brain.dt
        self.x += np.cos(self.direction) * self.dst
        self.y += np.sin(self.direction) * self.dst

    def update_vertices(self):
        hp = self.pos + np.array([np.cos(self.direction), np.sin(self.direction)]) * self.seg_lengths[0] / 2
        self.head.set_pose(hp, self.direction)
        self.head.update_vertices(hp, self.direction)
        if self.Nsegs == 2:
            new_or = self.direction - self.brain.bend
            new_p = self.pos + np.array([-np.cos(new_or), -np.sin(new_or)]) * self.seg_lengths[1] / 2
            self.tail.set_pose(new_p, new_or)
            self.tail.update_vertices(new_p, new_or)
        else:
            raise NotImplemented

    def store(self):
        self.eval['b'].append(self.brain.bend)
        self.eval['fov'].append(self.brain.ang_vel)
        self.eval['v'].append(self.brain.lin_vel)
        self.trajectory.append(self.pos / self.model.scaling_factor)

    def finalize(self, eval_shorts=['b', 'fov']):
        dt = self.brain.dt
        self.trajectory = np.array(self.trajectory)
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
            a_sv = pd.Series(self.eval['v'] / self.real_length)
            fv = fft_max(a_sv, dt, fr_range=(1.0, 2.5), return_amps=False)
            strides, runs, run_counts = detect_strides(a_sv, dt, fr=fv, return_extrema=False)
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

    def step(self):
        lin, ang, feed = self.brain.step(A_in=0, length=self.real_length)
        self.update_pose()
        self.update_vertices()
        self.store()

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
        self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None
        # self.wheel_radius = wheel_radius

    def sense_and_act(self):
        if not self.collision_with_object:
            try:
                self.left_motor_controller.sense_and_act()
                self.right_motor_controller.sense_and_act()
                Rtorque = self.left_motor_controller.get_actuator_value()
                Ltorque = self.right_motor_controller.get_actuator_value()

                self.brain.turner.neural_oscillator.E_r += Rtorque
                self.brain.turner.neural_oscillator.E_l += Ltorque
                self.step()
            except Collision:
                self.collision_with_object = True
                self.brain.intermitter.interrupt_locomotion()
        else:
            pass

    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, screen):
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw(pos=self.olfactor_pos, direction=-self.direction)
            self.right_motor_controller.sensor.draw(pos=self.olfactor_pos, direction=-self.direction)

        # call super method to draw the robot
        super().draw(screen)
