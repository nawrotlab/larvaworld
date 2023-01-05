import math

import numpy as np

from lib.model.agents._larva import LarvaMotile
from lib.model.agents.body import LarvaBody0
from lib import aux





class LarvaOffline(LarvaBody0, LarvaMotile):
    def __init__(self, unique_id, model, larva_pars, orientation=0, pos=(0, 0), group='', odor=None,
                 default_color=None, life_history=None, **kwargs):
        LarvaMotile.__init__(self, physics=larva_pars.physics, unique_id=unique_id, model=model, pos=pos,
                     odor=odor, group=group, default_color=default_color, life_history=life_history, energetics=larva_pars.energetics,
                         brain=larva_pars.brain)

        LarvaBody0.__init__(self, model=model, pos=self.pos, orientation=orientation, default_color=self.default_color,
                           **larva_pars.body)

        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False
        # self.unique_id = unique_id


        # self.pos = pos
        self.fo = orientation
        self.ro = orientation
        # self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

        self.x, self.y = (0, 0)
        # self.real_length = larva_pars.body.initial_length

        # self.trajectory = [self.pos]
        self.lin_vel = 0
        self.ang_vel = 0
        self.body_bend = 0
        # self.body_bend_errors = 0
        # self.negative_speed_errors = 0
        # self.Nsegs = larva_pars.body.Nsegs
        self.torque = 0
        # self.body_bend_errors = 0
        # self.negative_speed_errors = 0
        # self.border_go_errors = 0
        # self.border_turn_errors = 0
        # # self.Nangles_b = int(self.Nangles + 1 / 2)
        # # self.spineangles = [0.0] * self.Nangles
        # #
        # # self.mid_seg_index = int(self.Nsegs / 2)
        # self.rear_orientation_change = 0
        # # self.compute_body_bend()
        # self.cum_dur = 0
        #
        # self.cum_dst = 0.0
        # self.dst = 0.0
        # self.backward_motion = True

    def step(self):
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        self.lin_vel, self.ang_vel = self.get_vels(lin, ang, self.ang_vel, self.lin_vel,
                                                              self.body_bend, dt=self.model.dt,
                                                              ang_suppression=self.brain.locomotor.cur_ang_suppression)

        ang_vel_min, ang_vel_max=(-np.pi + self.body_bend) / self.model.dt, (np.pi + self.body_bend) / self.model.dt
        if self.ang_vel<ang_vel_min:
            self.ang_vel=ang_vel_min
            self.body_bend_errors+=1
        elif self.ang_vel > ang_vel_max:
            self.ang_vel = ang_vel_max
            self.body_bend_errors += 1

        d_or = self.ang_vel * dt
        self.fo = (self.fo + d_or) % (2 * np.pi)
        self.dst = self.lin_vel * dt
        self.rear_orientation_change = aux.rear_orientation_change(self.body_bend, self.dst, self.real_length,
                                       correction_coef=self.bend_correction_coef)
        self.ro = (self.ro + self.rear_orientation_change) % (2 * np.pi)
        self.body_bend = aux.wrap_angle_to_0(self.fo - self.ro)
        self.cum_dst += self.dst
        k1 = np.array([math.cos(self.fo), math.sin(self.fo)])
        self.pos += k1 * self.dst

        self.trajectory.append(tuple(self.pos))
        self.complete_step()

    def complete_step(self):
        if self.lin_vel<0:
            self.negative_speed_errors+=1
            self.lin_vel=0
        self.Nticks += 1

    @property
    def collect(self):
        return [self.body_bend,self.ang_vel, self.rear_orientation_change/self.model.dt,
                                                                   self.lin_vel, self.pos[0],self.pos[1]]

    def sense_and_act(self):
        self.step()