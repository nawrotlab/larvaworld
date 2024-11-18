import numpy as np

from ... import util
from .larva_robot import LarvaRobot

__all__ = ["LarvaOffline"]

__displayname__ = "Offline agent"


class LarvaOffline(LarvaRobot):
    """
    Subclass of LarvaRobot that simulates the behavior of a larva in an offline environment.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fo = self.orientation
        self.ro = self.orientation

    def step(self):
        """
        Perform a single simulation step for the larva agent.

        This method updates the agent's state based on the current model time step.
        It calculates the linear and angular velocities, updates the agent's orientation
        and position, and records the trajectory.
        """
        dt = self.model.dt
        self.cum_dur += dt

        lin, ang, feed = self.brain.locomotor.step(A_in=0, length=self.length)
        self.lin_vel = lin * self.lin_vel_coef
        self.ang_vel = self.compute_ang_vel(ang)

        ang_vel_min, ang_vel_max = (
            (-np.pi + self.body_bend) / self.model.dt,
            (np.pi + self.body_bend) / self.model.dt,
        )
        if self.ang_vel < ang_vel_min:
            self.ang_vel = ang_vel_min
        elif self.ang_vel > ang_vel_max:
            self.ang_vel = ang_vel_max

        self.fo = (self.fo + self.ang_vel * dt) % (2 * np.pi)
        self.dst = self.lin_vel * dt
        delta_ro = self.compute_delta_rear_angle(self.body_bend, self.dst, self.length)

        self.ro = (self.ro + delta_ro) % (2 * np.pi)
        self.body_bend = util.wrap_angle_to_0(self.fo - self.ro)
        self.cum_dst += self.dst
        k1 = np.array([np.cos(self.fo), np.sin(self.fo)])

        self.set_position(tuple(self.pos + k1 * self.dst))
        self.set_orientation(self.fo)
        self.set_angularvelocity(self.ang_vel)
        self.set_linearvelocity(self.lin_vel)

        self.trajectory.append(self.pos)
