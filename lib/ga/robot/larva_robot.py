import math
import random
import pygame
import pandas as pd
import numpy as np

from lib.aux.dictsNlists import AttrDict
from lib.conf.base.par import getPar
from lib.ga.exception.collision_exception import Collision
from lib.ga.robot.actuator import Actuator
from lib.ga.robot.motor_controller import MotorController
from lib.ga.sensor.proximity_sensor import ProximitySensor
from lib.ga.util.color import Color
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain
from lib.process.aux import detect_turns, process_epochs, detect_strides, fft_max, detect_pauses, \
    compute_interference_solo
from lib.process.spatial import straightness_index


class LarvaRobot(BodySim):

    def __init__(self, unique_id, model, larva_pars, orientation=0, pos=(0, 0), **kwargs):

        super().__init__(model=model, pos=pos, orientation=orientation, default_color=Color.random_color(127, 127, 127),
                         **larva_pars.physics, **larva_pars.body, **larva_pars.Box2D_params)

        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False
        self.unique_id = unique_id
        self.eval = AttrDict.from_nested_dicts({
            'b': [],
            'fov': [],
            'v': [],
        })
        self.x, self.y = model.screen_pos(self.initial_pos)
        self.pos = self.initial_pos
        self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

    def draw(self, screen):
        for seg in self.segs:
            for vs in seg.vertices:
                pygame.draw.polygon(screen, seg.color, vs)

    @property
    def sim_pos(self):
        return np.array([self.x, self.y])

    @property
    def direction(self):
        return self.head.get_orientation()

    @property
    def olfactor_pos_ga(self):
        o0 = self.head.get_orientation()
        k = np.array([math.cos(o0), math.sin(o0)])
        p = self.pos + k * self.real_length * self.seg_ratio[0]
        return p

    # def print_xyd(self):
    #     """ prints the x,y position and direction """
    #     print("x = " + str(self.x) + " " + "y = " + str(self.y))
    #     print("direction = " + str(self.direction))

    def position_body_ga(self,lin_vel, ang_vel):
        hp0, o0 = self.head.get_pose()
        o1 = o0 + ang_vel * self.model.dt
        k1 = np.array([math.cos(o1), math.sin(o1)])
        dxy = k1 * self.dst
        self.pos += dxy
        sim_dxy = dxy * self.model.scaling_factor
        self.x += sim_dxy[0]
        self.y += sim_dxy[1]
        hp1 = self.sim_pos + k1 * self.seg_lengths[0] / 2
        self.head.update_all(hp1, o1, lin_vel, ang_vel)

        self.position_rest_of_body(o0=o0, pos=self.sim_pos, o1=o1)

    def step(self):
        self.cum_dur += self.model.dt
        self.Nticks += 1

        self.restore_body_bend(self.dst, self.real_length)
        self.lin_activity, self.ang_activity, self.feeder_motion = self.brain.step(pos=self.olfactor_pos_ga)
        lin_vel, ang_vel = self.get_vels(self.lin_activity, self.ang_activity, self.head.get_angularvelocity())
        self.dst = lin_vel * self.model.dt
        self.cum_dst += self.dst

        self.position_body_ga(lin_vel=lin_vel, ang_vel=ang_vel)
        self.trajectory.append(tuple(self.pos))

        self.eval.b.append(self.body_bend)
        self.eval.fov.append(ang_vel)
        self.eval.v.append(lin_vel)

    def sense_and_act(self):
        self.step()

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x + (self.sim_length / 2), self.y + (self.sim_length / 2), 50, 50)
            screen.blit(text, text_pos)

    def finalize(self, eval_shorts=['b', 'fov'], step_data=False):
        if self.finalized :
            return
        dt = self.brain.dt
        self.trajectory = np.array(self.trajectory)[:self.Nticks, :]
        self.eval.v = np.array(self.eval.v)
        self.eval.sv = self.eval.v / self.real_length
        self.eval.b = np.rad2deg(self.eval.b)
        self.eval.fov = np.rad2deg(self.eval.fov)
        if 'bv' in eval_shorts:
            self.eval.bv = np.diff(self.eval.b, prepend=[np.nan]) / dt
        if 'ba' in eval_shorts:
            self.eval.ba = np.diff(self.eval.bv, prepend=[np.nan]) / dt
        if 'a' in eval_shorts:
            self.eval.a = np.diff(self.eval.v, prepend=[np.nan]) / dt
        if 'foa' in eval_shorts:
            self.eval.foa = np.diff(self.eval.fov, prepend=[np.nan]) / dt
        if 'tor5' in eval_shorts:
            self.eval.tor5 = straightness_index(self.trajectory, int(5 / dt / 2), match_shape=False)
        if 'tor2' in eval_shorts:
            self.eval.tor2 = straightness_index(self.trajectory, int(2 / dt / 2), match_shape=False)
        if 'tor1' in eval_shorts:
            self.eval.tor1 = straightness_index(self.trajectory, int(1 / dt / 2), match_shape=False)
        if 'tor10' in eval_shorts:
            self.eval.tor10 = straightness_index(self.trajectory, int(10 / dt / 2), match_shape=False)
        if 'tor20' in eval_shorts:
            self.eval.tor20 = straightness_index(self.trajectory, int(20 / dt / 2), match_shape=False)
        if 'tur_fou' in eval_shorts:
            a_fov = pd.Series(self.eval.fov)
            Lturns, Rturns = detect_turns(a_fov, dt)

            Ldurs, Lamps, Lmaxs = process_epochs(a_fov, Lturns, dt, return_idx=False)
            Rdurs, Ramps, Rmaxs = process_epochs(a_fov, Rturns, dt, return_idx=False)
            self.eval.tur_fou = np.abs(np.concatenate([Lamps, Ramps]))
            self.eval.tur_t = np.concatenate([Ldurs, Rdurs])
            self.eval.tur_fov_max = np.abs(np.concatenate([Lmaxs, Rmaxs]))
        if 'run_t' in eval_shorts:
            a_sv = pd.Series(self.eval.sv)
            fv = fft_max(a_sv, dt, fr_range=(1.0, 2.5), return_amps=False)
            strides, runs, run_counts = detect_strides(a_sv, dt, fr=fv, return_extrema=False)
            pauses = detect_pauses(a_sv, dt, runs=runs)
            pause_durs, pause_dsts, pause_maxs = process_epochs(a_sv, pauses, dt, return_idx=False)
            run_durs, run_dsts, run_maxs = process_epochs(a_sv, runs, dt, return_idx=False)
            self.eval.run_d = run_dsts
            self.eval.run_t = run_durs
            self.eval.pau_t = pause_durs

        if step_data:
            ps = getPar(list(self.eval.keys()), to_return=['d'])[0]
            s = pd.DataFrame(index=np.arange(self.Nticks), columns=ps)
            for p, (k, vs) in zip(ps, self.eval.items()):
                if vs.shape[0] == self.Nticks:
                    s[p] = vs
                self.eval[k] = vs[~np.isnan(vs)]
        else:
            for k, vs in self.eval.items():
                self.eval[k] = vs[~np.isnan(vs)]

        self.finalized=True

    def interference_curves(self, **kwargs):
        fov_curve, sv_curve = compute_interference_solo(self.eval.sv, self.eval.fov, self.model.dt, **kwargs)
        return {'fov': fov_curve, 'sv': sv_curve}


class ObstacleLarvaRobot(LarvaRobot):
    def __init__(self, larva_pars, **kwargs):
        self.sensorimotor_kws = larva_pars.sensorimotor
        larva_pars.pop('sensorimotor', None)
        super().__init__(larva_pars=larva_pars, **kwargs)
        # self.collision_with_object = False
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.build_sensorimotor(**self.sensorimotor_kws)

    def build_sensorimotor(self, sensor_delta_direction, sensor_saturation_value, obstacle_sensor_error,
                           sensor_max_distance,
                           motor_ctrl_coefficient, motor_ctrl_min_actuator_value):

        S_kws = {
            'saturation_value': sensor_saturation_value,
            'error': obstacle_sensor_error,
            'max_distance': sensor_max_distance * self.sim_length,
            'scene': self.model.scene,
            'collision_distance': 0.1 * self.sim_length,
        }

        M_kws = {
            'coefficient': motor_ctrl_coefficient,
            'min_actuator_value': motor_ctrl_min_actuator_value,
        }

        Lsens = ProximitySensor(self, delta_direction=sensor_delta_direction, **S_kws)
        Rsens = ProximitySensor(self, delta_direction=-sensor_delta_direction, **S_kws)
        Lact = Actuator()
        Ract = Actuator()
        Lmot = MotorController(sensor=Lsens, actuator=Lact, **M_kws)
        Rmot = MotorController(sensor=Rsens, actuator=Ract, **M_kws)

        self.set_left_motor_controller(Lmot)
        self.set_right_motor_controller(Rmot)

    def sense_and_act(self):
        if not self.collision_with_object:
            try:
                self.left_motor_controller.sense_and_act(pos=self.olfactor_pos, direction=-self.direction)
                self.right_motor_controller.sense_and_act(pos=self.olfactor_pos, direction=-self.direction)
                Rtorque = self.left_motor_controller.get_actuator_value()
                Ltorque = self.right_motor_controller.get_actuator_value()
                # dRL=Rtorque-Ltorque
                # ang=self.head.get_angularvelocity()
                # self.head.set_ang_vel(ang-dRL*self.model.dt)
                # if dRL!=0 :
                #
                #     print(dRL*self.model.dt, ang)
                self.brain.locomotor.turner.neural_oscillator.E_r += Rtorque * self.model.dt
                self.brain.locomotor.turner.neural_oscillator.E_l += Ltorque * self.model.dt
                # if dRL>0 :
                #     self.brain.locomotor.turner.neural_oscillator.E_r += np.abs(dRL)
                # else :
                #     self.brain.locomotor.turner.neural_oscillator.E_l += np.abs(dRL)
                self.step()
            except Collision:
                self.collision_with_object = True
                self.brain.locomotor.intermitter.interrupt_locomotion()
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
