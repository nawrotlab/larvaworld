import math

import numpy as np

from lib.aux.ang_aux import rear_orientation_change, wrap_angle_to_0
from lib.model.body.controller import PhysicsController
from lib.model.modules.brain import DefaultBrain


class LarvaOffline:
    def __init__(self, unique_id, model, larva_pars, orientation=0, pos=(0, 0), **kwargs):
        self.model = model
        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False
        self.unique_id = unique_id


        self.pos = pos
        self.fo = orientation
        self.ro = orientation
        self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

        self.x, self.y = (0, 0)
        self.real_length = larva_pars.body.initial_length

        self.controller = PhysicsController(**larva_pars.physics)
        self.trajectory = [self.pos]
        self.lin_vel = 0
        self.ang_vel = 0
        self.body_bend = 0
        self.body_bend_errors = 0
        self.Nsegs = larva_pars.body.Nsegs
        self.torque = 0
        self.cum_dur = 0
        self.rear_orientation_change = 0
        self.cum_dst = 0.0
        self.dst = 0.0

    def step(self):
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        self.lin_vel, self.ang_vel = self.controller.get_vels(lin, ang, self.ang_vel, self.lin_vel,
                                                              self.body_bend, dt=self.model.dt,
                                                              ang_suppression=self.brain.locomotor.cur_ang_suppression)

        d_or = self.ang_vel * dt
        if np.abs(d_or) > np.pi:
            self.body_bend_errors += 1
        self.fo = (self.fo + d_or) % (2 * np.pi)
        self.dst = self.lin_vel * dt
        self.rear_orientation_change = rear_orientation_change(self.body_bend, self.dst, self.real_length,
                                       correction_coef=self.controller.bend_correction_coef)
        self.ro = (self.ro + self.rear_orientation_change) % (2 * np.pi)
        self.body_bend = wrap_angle_to_0(self.fo - self.ro)
        self.cum_dst += self.dst
        k1 = np.array([math.cos(self.fo), math.sin(self.fo)])
        self.pos += k1 * self.dst

        self.trajectory.append(tuple(self.pos))
        self.complete_step()

    def complete_step(self):
        self.model.engine.step_df[self.Nticks, self.unique_id, :]=[self.body_bend,self.ang_vel, self.rear_orientation_change/self.model.dt,
                                                                   self.lin_vel, self.pos[0],self.pos[1]]
        self.Nticks += 1



    def sense_and_act(self):
        self.step()