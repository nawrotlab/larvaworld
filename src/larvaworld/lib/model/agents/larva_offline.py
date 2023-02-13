import numpy as np

from larvaworld.lib.model.agents._larva import LarvaMotile
from larvaworld.lib import aux





class LarvaOffline(LarvaMotile):
    def __init__(self, larva_pars, **kwargs):
        super().__init__(**larva_pars, **kwargs)
        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False

        self.fo = self.orientation
        self.ro = self.orientation

        self.xx, self.yy = self.model.viewer._transform(self.pos)

        self.lin_vel = 0
        self.ang_vel = 0
        self.body_bend = 0
        self.torque = 0

    def step(self):
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        self.lin_vel, self.ang_vel = self.get_vels(lin, ang, self.ang_vel, self.body_bend, dt=self.model.dt,
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
        k1 = np.array([np.cos(self.fo), np.sin(self.fo)])
        self.pos += k1 * self.dst

        self.trajectory.append(tuple(self.pos))
        self.complete_step()

    def complete_step(self):
        if self.lin_vel<0:
            self.negative_speed_errors+=1
            self.lin_vel=0
        self.Nticks += 1
        self.xx, self.yy = self.model.viewer._transform(self.pos)

    @property
    def collect(self):
        return [self.body_bend,self.ang_vel, self.rear_orientation_change/self.model.dt,
                                                                   self.lin_vel, self.pos[0],self.pos[1]]

    def sense_and_act(self):
        self.step()
