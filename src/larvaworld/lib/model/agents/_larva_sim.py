import math

import numpy as np
from shapely import geometry

from larvaworld.lib import aux
from larvaworld.lib.model import agents
from larvaworld.lib.model.agents.segmented_body import BaseController



class LarvaSim(agents.LarvaMotile, BaseController):
    def __init__(self, physics, Box2D_params, **kwargs):
        BaseController.__init__(self, **physics)
        agents.LarvaMotile.__init__(self,**kwargs)
        self.body_bend_errors = 0
        self.negative_speed_errors = 0
        self.border_go_errors = 0
        self.border_turn_errors = 0

        self.collision_with_object = False



    # def compute_ang_vel(self, torque, v):
    #     return v + (-self.ang_damping * v - self.body_spring_k * self.body_bend + torque) * self.model.dt

    def prepare_motion(self, lin, ang):
        lin_vel = lin * self.lin_vel_coef
        if self.ang_mode == 'torque':
            ang_vel = self.compute_ang_vel(ang,ang_vel=self.head.get_angularvelocity(), dt=self.model.dt, bend=self.body_bend)
        elif self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
        else:
            raise
        lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
        self.position_body(lin_vel, ang_vel)


    @property
    def border_collision(self):
        if len(self.model.borders) == 0:
            return False
        else:
            x,y=self.pos
            p0 = geometry.Point(self.pos)
            d0 = self.sim_length / 4
            oM = self.head.get_orientation()
            p1 = geometry.Point(
                x + math.cos(oM) * d0,
                y + math.sin(oM) * d0)

            sensor_ray = p0, p1

            min_dst, nearest_obstacle = aux.detect_nearest_obstacle(self.model.borders, sensor_ray, p0)

            if min_dst is None:
                return False
            else:
                return True

    def assess_collisions(self, lin_vel, ang_vel):
        if not self.model.larva_collisions:
            ids = self.model.detect_collisions(self.unique_id)
            larva_collision = False if len(ids) == 0 else True
        else:
            larva_collision = False
        if larva_collision:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * np.pi / 10
            return lin_vel, ang_vel
        res = self.border_collision
        d_ang = np.pi / 20
        if not res:
            return lin_vel, ang_vel
        else:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * d_ang
            return lin_vel, ang_vel

    # @profile
    def position_body(self, lin_vel, ang_vel):
        dt=self.model.dt
        sf = self.model.scaling_factor
        hp0, ho0 = self.head.get_pose()
        hr0 = self.head.rear_end
        l0 = self.head.length
        A0,A1=self.valid_Dbend_range(0)
        fov0,fov1 = A0 / dt, A1 / dt


        if ang_vel < fov0:
            ang_vel = fov0
            self.body_bend_errors += 1
        elif ang_vel > fov1:
            ang_vel = fov1
            self.body_bend_errors += 1

        if not self.model.space.torus :
            tank = self.model.space.polygon
            d, ang_vel, lin_vel,hp1, ho1, turn_err, go_err = aux.position_head_in_tank(hr0, ho0, l0, fov0,fov1, ang_vel, lin_vel, dt, tank, sf=sf)

            self.border_turn_errors+=turn_err
            self.border_go_errors+=go_err
        else:
            ho1 = ho0 + ang_vel * dt
            k = np.array([np.cos(ho1), np.sin(ho1)])
            d = lin_vel * dt
            hp1 = hr0 + k * (d * sf + l0 / 2)
        self.head.update_all(hp1, ho1, lin_vel, ang_vel)
        self.dst = d



        if self.Nsegs > 1:
            delta_ro = self.compute_delta_rear_angle(self.body_bend, self.dst, self.length)
            d_or = delta_ro / (self.Nsegs - 1)
            for i, seg in enumerate(self.segs[1:]):
                o1 = seg.get_orientation() + d_or
                k = np.array([np.cos(o1), np.sin(o1)])
                p1 = self.segs[i].rear_end - k * seg.length / 2
                seg.update_poseNvertices(p1, o1)

        self.set_position(tuple(self.global_midspine_of_body))
        self.set_orientation(self.head.get_orientation())
        self.trajectory.append(self.get_position())
        self.orientation_trajectory.append(self.get_orientation())
        self.model.space.move_to(self, np.array(self.get_position()))
        self.cum_dst += self.dst
        self.compute_body_bend()

    def valid_Dbend_range(self, idx=0):
        if self.Nsegs > idx + 1:
            dang = aux.wrap_angle_to_0(self.segs[idx + 1].get_orientation() - self.segs[idx].get_orientation())
        else:
            dang = 0
        return (-np.pi + dang), (np.pi + dang)
