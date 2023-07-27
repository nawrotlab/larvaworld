import numpy as np

from larvaworld.lib.model.agents._larva_sim import LarvaSim
from larvaworld.lib import aux





class LarvaOffline(LarvaSim):
    def __init__(self, larva_pars,genome=None,  **kwargs):
        super().__init__(**larva_pars, **kwargs)

        self.genome = genome


        self.fo = self.orientation
        self.ro = self.orientation


        self.lin_vel = 0
        self.ang_vel = 0

    def step(self):
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        self.lin_vel = lin * self.lin_vel_coef
        self.ang_vel = self.compute_ang_vel(ang, ang_vel=self.ang_vel, dt=self.model.dt,bend=self.body_bend)

        ang_vel_min, ang_vel_max=(-np.pi + self.body_bend) / self.model.dt, (np.pi + self.body_bend) / self.model.dt
        if self.ang_vel<ang_vel_min:
            self.ang_vel=ang_vel_min
            self.body_bend_errors+=1
        elif self.ang_vel > ang_vel_max:
            self.ang_vel = ang_vel_max
            self.body_bend_errors += 1

        self.fo = (self.fo + self.ang_vel * dt) % (2 * np.pi)
        self.dst = self.lin_vel * dt
        delta_ro = self.compute_delta_rear_angle(self.body_bend, self.dst, self.length)

        self.ro = (self.ro + delta_ro) % (2 * np.pi)
        self.body_bend = aux.wrap_angle_to_0(self.fo - self.ro)
        self.cum_dst += self.dst
        k1 = np.array([np.cos(self.fo), np.sin(self.fo)])
        self.pos += k1 * self.dst

        self.trajectory.append(tuple(self.pos))






