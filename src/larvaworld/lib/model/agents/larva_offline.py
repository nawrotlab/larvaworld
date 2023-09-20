import numpy as np

from larvaworld.lib.model.agents.larva_robot import LarvaRobot
# from larvaworld.lib.model.agents._larva_sim import LarvaSim
from larvaworld.lib import aux



__all__ = [
    'LarvaOffline',
]

__displayname__ = 'Offline larva'

class LarvaOffline(LarvaRobot):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        self.fo = self.orientation
        self.ro = self.orientation


        # self.lin_vel = 0
        # self.ang_vel = 0

    def step(self):
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.real_length)
        self.lin_vel = lin * self.lin_vel_coef
        self.ang_vel = self.compute_ang_vel(ang)

        ang_vel_min, ang_vel_max=(-np.pi + self.body_bend) / self.model.dt, (np.pi + self.body_bend) / self.model.dt
        if self.ang_vel<ang_vel_min:
            self.ang_vel=ang_vel_min
        elif self.ang_vel > ang_vel_max:
            self.ang_vel = ang_vel_max

        self.fo = (self.fo + self.ang_vel * dt) % (2 * np.pi)
        self.dst = self.lin_vel * dt
        delta_ro = self.compute_delta_rear_angle(self.body_bend, self.dst, self.length)

        self.ro = (self.ro + delta_ro) % (2 * np.pi)
        self.body_bend = aux.wrap_angle_to_0(self.fo - self.ro)
        self.cum_dst += self.dst
        k1 = np.array([np.cos(self.fo), np.sin(self.fo)])

        self.set_position(tuple(self.pos + k1 * self.dst))
        self.set_orientation(self.fo)
        self.set_angularvelocity(self.ang_vel)
        self.set_linearvelocity(self.lin_vel)

        self.trajectory.append(self.pos)






