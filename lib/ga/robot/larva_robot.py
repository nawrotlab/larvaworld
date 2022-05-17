import math
import random
import pygame
import pandas as pd
import numpy as np

from lib.aux.ang_aux import rear_orientation_change, wrap_angle_to_0
from lib.aux.dictsNlists import AttrDict
from lib.conf.base.par import getPar
from lib.ga.exception.collision_exception import Collision

from lib.ga.robot.motor_controller import MotorController, Actuator
from lib.ga.sensor.proximity_sensor import ProximitySensor
from lib.ga.util.color import Color
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain
from lib.process.aux import detect_turns, process_epochs, detect_strides, fft_max, detect_pauses, \
    compute_interference_solo
from lib.process.spatial import straightness_index


class LarvaOffline:
    def __init__(self, unique_id, model, larva_pars, orientation=0, pos=(0, 0), **kwargs):
        self.model = model
        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False
        self.unique_id = unique_id
        self.eval = AttrDict.from_nested_dicts({
            'b': [],
            'fov': [],
            'rov': [],
            'v': [],
        })

        self.pos = pos
        self.orientation = orientation
        self.rear_orientation = orientation
        self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

        self.x, self.y = (0,0)
        self.real_length = larva_pars.body.initial_length
        self.lin_damping = larva_pars.physics.lin_damping
        self.ang_damping = larva_pars.physics.ang_damping
        self.body_spring_k = larva_pars.physics.body_spring_k
        self.torque_coef = larva_pars.physics.torque_coef
        self.bend_correction_coef = larva_pars.physics.bend_correction_coef
        self.trajectory = [self.pos]
        self.lin_vel = 0
        self.ang_vel = 0
        self.body_bend = 0
        self.body_bend_errors = 0
        self.Nsegs = larva_pars.body.Nsegs
        self.torque = 0
        self.cum_dur = 0

        self.cum_dst = 0.0
        self.dst = 0.0





    def step(self):
        dt = self.model.dt
        self.cum_dur += dt
        self.Nticks += 1


        lin_new, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        if lin_new != 0:
            self.lin_vel = lin_new
        else:
            self.lin_vel *= (1 - self.lin_damping * dt)
        self.torque = self.torque_coef * ang* self.brain.locomotor.cur_ang_suppression

        self.ang_vel += (-self.ang_damping * self.ang_vel - self.body_spring_k * self.body_bend + self.torque) * dt
        d_or=self.ang_vel * dt
        if np.abs(d_or)>np.pi:
            self.body_bend_errors += 1
        self.orientation = (self.orientation + d_or) % (2 * np.pi)
        # self.ang_vel *= self.brain.locomotor.cur_ang_suppression
        self.dst = self.lin_vel * dt
        self.rear_orientation_change = rear_orientation_change(self.body_bend, self.dst, self.real_length,
                                                               correction_coef=self.bend_correction_coef)
        self.rear_orientation=(self.rear_orientation+self.rear_orientation_change)%(2*np.pi)
        self.body_bend = wrap_angle_to_0(self.orientation-self.rear_orientation)
        # self.body_bend = self.check_bend_error(self.body_bend + self.ang_vel * dt- self.rear_orientation_change)
        # self.restore_bend()
        self.cum_dst += self.dst

        # self.orientation = (self.orientation + d_or)%(2*np.pi)
        k1 = np.array([math.cos(self.orientation), math.sin(self.orientation)])
        # dxy = k1 * self.dst
        self.pos += k1 * self.dst

        self.trajectory.append(tuple(self.pos))

        rov=self.rear_orientation_change/dt
        # print(self.body_bend,rov,self.ang_vel, self.lin_vel)
        self.eval.b.append(self.body_bend)
        self.eval.fov.append(self.ang_vel)
        self.eval.rov.append(rov)
        self.eval.v.append(self.lin_vel)

        # self.x, self.y = self.model.scene._transform(self.pos)
        # self.olfactor_screen_pos = self.model.scene._transform(self.olfactor_pos)

    def sense_and_act(self):
        self.step()

    # @property
    # def direction(self):
    #     return self.head.get_orientation()
    def finalize(self, eval_shorts=['b', 'fov', 'rov'], step_data=False):
        if self.finalized:
            return
        dt = self.brain.dt
        self.trajectory = np.array(self.trajectory)[:self.Nticks, :]
        self.eval.v = np.array(self.eval.v)
        self.eval.sv = self.eval.v / self.real_length
        self.eval.b = np.rad2deg(self.eval.b)
        self.eval.fov = np.rad2deg(self.eval.fov)
        self.eval.rov = np.rad2deg(self.eval.rov)
        if 'bv' in eval_shorts or 'rov' in eval_shorts:
            self.eval.bv = np.diff(self.eval.b, prepend=[np.nan]) / dt
            # self.eval.rov = self.eval.fov - self.eval.bv
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
            ps = getPar(list(self.eval.keys()))
            s = pd.DataFrame(index=np.arange(self.Nticks), columns=ps)
            for p, (k, vs) in zip(ps, self.eval.items()):
                if vs.shape[0] == self.Nticks:
                    s[p] = vs
                self.eval[k] = vs[~np.isnan(vs)]
        else:
            for k, vs in self.eval.items():
                self.eval[k] = vs[~np.isnan(vs)]

        self.finalized = True

    def interference_curves(self, **kwargs):
        fov_curve, sv_curve, foa_curve, rov_curve = compute_interference_solo(self.eval.sv, self.eval.fov,
                                                                              self.eval.foa, self.eval.rov,
                                                                              self.model.dt, **kwargs)
        return {'foa': foa_curve, 'fov': fov_curve, 'sv': sv_curve, 'rov': rov_curve}


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
            'rov': [],
            'v': [],
        })
        self.pos = self.initial_pos
        self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

        self.x, self.y = self.model.scene._transform(self.pos)

    def draw(self, scene):
        for seg in self.segs:
            for vs in seg.vertices:
                scene.draw_polygon(vs, filled=True, color=seg.color)
                # pygame.draw.polygon(screen, seg.color, vs)


    @property
    def direction(self):
        return self.head.get_orientation()

    def complete_step(self):
        self.Nticks += 1
        self.eval.b.append(self.body_bend)
        self.eval.rov.append(self.rear_orientation_change/self.model.dt)
        self.eval.fov.append(self.head.get_angularvelocity())
        self.eval.v.append(self.head.get_linearvelocity())

        self.x, self.y = self.model.scene._transform(self.pos)

    def sense_and_act(self):
        self.step()

    def draw_label(self, screen):
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x + (self.sim_length / 2), self.y + (self.sim_length / 2), 50, 50)
            screen.blit(text, text_pos)

    def finalize(self, eval_shorts=['b', 'fov', 'rov'], step_data=False):
        if self.finalized:
            return
        dt = self.brain.dt
        self.trajectory = np.array(self.trajectory)[:self.Nticks, :]
        self.eval.v = np.array(self.eval.v)
        self.eval.sv = self.eval.v / self.real_length
        self.eval.b = np.rad2deg(self.eval.b)
        self.eval.fov = np.rad2deg(self.eval.fov)
        self.eval.rov = np.rad2deg(self.eval.rov)
        if 'bv' in eval_shorts:
            self.eval.bv = np.diff(self.eval.b, prepend=[np.nan]) / dt
            # self.eval.rov = self.eval.fov - self.eval.bv
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
            ps = getPar(list(self.eval.keys()))
            s = pd.DataFrame(index=np.arange(self.Nticks), columns=ps)
            for p, (k, vs) in zip(ps, self.eval.items()):
                if vs.shape[0] == self.Nticks:
                    s[p] = vs
                self.eval[k] = vs[~np.isnan(vs)]
        else:
            for k, vs in self.eval.items():
                self.eval[k] = vs[~np.isnan(vs)]

        self.finalized = True

    def interference_curves(self, **kwargs):
        fov_curve, sv_curve, foa_curve, rov_curve = compute_interference_solo(self.eval.sv, self.eval.fov,
                                                                              self.eval.foa, self.eval.rov,
                                                                              self.model.dt, **kwargs)
        return {'foa': foa_curve, 'fov': fov_curve, 'sv': sv_curve, 'rov': rov_curve}


class ObstacleLarvaRobot(LarvaRobot):
    def __init__(self, larva_pars, **kwargs):
        self.sensorimotor_kws = larva_pars.sensorimotor
        larva_pars.pop('sensorimotor', None)
        super().__init__(larva_pars=larva_pars, **kwargs)
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.build_sensorimotor(**self.sensorimotor_kws)

    def build_sensorimotor(self, sensor_delta_direction, sensor_saturation_value, obstacle_sensor_error,
                           sensor_max_distance,
                           motor_ctrl_coefficient, motor_ctrl_min_actuator_value):
        S_kws = {
            'saturation_value': sensor_saturation_value,
            'error': obstacle_sensor_error,
            'max_distance': int(self.model.scene._scale[0, 0] * sensor_max_distance * self.real_length),
            # 'max_distance': sensor_max_distance * self.real_length,
            'scene': self.model.scene,
            'collision_distance': int(self.model.scene._scale[0, 0] * self.real_length / 5),
            # 'collision_distance': 0.1 * self.real_length,
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
            pos = self.model.scene._transform(self.olfactor_pos)
            try:

                self.left_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                self.right_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                Ltorque = self.left_motor_controller.get_actuator_value()
                Rtorque = self.right_motor_controller.get_actuator_value()
                dRL = Rtorque - Ltorque
                if dRL > 0:
                    self.brain.locomotor.turner.neural_oscillator.E_r += dRL * self.model.dt
                else:
                    self.brain.locomotor.turner.neural_oscillator.E_l -= dRL * self.model.dt
                # ang=self.head.get_angularvelocity()
                # self.head.set_ang_vel(ang-dRL*self.model.dt)
                # if dRL!=0 :
                #
                #     print(dRL*self.model.dt, ang)
                # self.brain.locomotor.turner.neural_oscillator.E_r += Rtorque * self.model.dt
                # self.brain.locomotor.turner.neural_oscillator.E_l += Ltorque * self.model.dt
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

    def draw(self, scene):
        # pos = self.olfactor_pos
        pos = self.model.scene._transform(self.olfactor_pos)
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw(pos=pos, direction=self.direction)
            self.right_motor_controller.sensor.draw(pos=pos, direction=self.direction)

        # call super method to draw the robot
        super().draw(scene)
