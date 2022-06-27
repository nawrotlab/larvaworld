import math
import random

import numpy as np

from lib.aux.ang_aux import rear_orientation_change, wrap_angle_to_0
from lib.aux.sim_aux import inside_polygon
from lib.ga.geometry.point import Point
from lib.ga.geometry.util import detect_nearest_obstacle, radar_tuple
from lib.model.body.body import LarvaBody
from lib.aux.ang_aux import angle_dif

class PhysicsController :
    def __init__(self, lin_vel_coef=1.0, ang_vel_coef=1.0, lin_force_coef=1.0, torque_coef=0.5,
                 lin_mode='velocity', ang_mode='torque', body_spring_k=1.0, bend_correction_coef=1.0,
                 lin_damping=1.0, ang_damping=1.0):
        self.lin_damping = lin_damping
        self.ang_damping = ang_damping
        self.body_spring_k = body_spring_k
        self.bend_correction_coef = bend_correction_coef
        self.lin_mode = lin_mode
        self.ang_mode = ang_mode
        self.lin_vel_coef = lin_vel_coef
        self.ang_vel_coef = ang_vel_coef
        self.lin_force_coef = lin_force_coef
        self.torque_coef = torque_coef

    def compute_ang_vel(self, k, c, torque, v, b,dt, I=1):
        dtI=dt/I
        return v + (-c * v - k * b + torque) * dtI

    def get_vels(self, lin, ang, prev_ang_vel, prev_lin_vel, bend,dt, ang_suppression):
        if self.lin_mode == 'velocity':
            if lin!=0:
                lin_vel = lin * self.lin_vel_coef
            else :
                lin_vel = 0# prev_lin_vel*(1-self.lin_damping*dt)
        else:
            raise ValueError(f'Linear mode {self.lin_mode} not implemented for non-physics simulation')
        if self.ang_mode == 'torque':
            torque =ang * self.torque_coef
            # self.torque =ang * self.torque_coef*ang_suppression
            ang_vel = self.compute_ang_vel(torque=torque,
                                           v=prev_ang_vel, b=bend,
                                           c=self.ang_damping,  k=self.body_spring_k,dt=dt)
        elif self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
        # ang_vel*=ang_suppression
        lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
        ang_vel*=ang_suppression
        return lin_vel, ang_vel

    def assess_collisions(self,lin_vel, ang_vel):
        return lin_vel, ang_vel


class BodyManager(LarvaBody):
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

class BodyReplay(BodyManager):
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

class BodySim(BodyManager, PhysicsController):
    def __init__(self, physics,density=300.0, **kwargs):

        PhysicsController.__init__(self,**physics)
        BodyManager.__init__(self,density=density, **kwargs)


        self.head_contacts_ground = True
        self.trajectory = [self.initial_pos]

        self.body_bend = 0
        self.body_bend_errors = 0
        self.Nangles_b = int(self.Nangles + 1 / 2)
        self.spineangles = [0.0]*self.Nangles
        self.rear_orientation_change = 0
        self.compute_body_bend()
        self.mid_seg_index = int(self.Nsegs / 2)

        self.cum_dur = 0

        self.cum_dst = 0.0
        self.dst = 0.0
        self.backward_motion = True

    def detect_food(self, pos):
        t0 = []
        item, foodtype = None, None
        if self.brain.locomotor.feeder is not None or self.brain.toucher is not None:
            grid = self.model.food_grid
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_cell_value(cell) > 0:
                    item, foodtype = cell, grid.unique_id

            else:
                valid = [a for a in self.model.get_food() if a.amount > 0]
                accessible_food = [a for a in valid if a.contained(pos)]
                if accessible_food:
                    food = random.choice(accessible_food)
                    self.resolve_carrying(food)
                    item, foodtype = food, food.group
        return item, foodtype


    def Box2D_kinematics(self,lin, ang):
        self.compute_body_bend()
        if self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
            self.segs[0].set_ang_vel(ang_vel)
            if self.Nsegs > 1:
                for i in np.arange(1, self.mid_seg_index, 1):
                    self.segs[i].set_ang_vel(ang_vel / i)
        elif self.ang_mode == 'torque':
            torque = ang * self.torque_coef
            self.segs[0]._body.ApplyTorque(torque, wake=True)


        # Linear component
        # Option : Apply to single body segment
        # We get the orientation of the front segment and compute the linear vector
        # target_segment = self.get_head()
        # lin_vec = self.compute_new_lin_vel_vector(target_segment)
        #
        # # From web : Impulse = Force x 1 Sec in Box2D
        # if self.mode == 'impulse':
        #     imp = lin_vec / target_segment.get_Box2D_mass()
        #     target_segment._body.ApplyLinearImpulse(imp, target_segment._body.worldCenter, wake=True)
        # elif self.mode == 'force':
        #     target_segment._body.ApplyForceToCenter(lin_vec, wake=True)
        # elif self.mode == 'velocity':
        #     # lin_vec = lin_vec * target_segment.get_Box2D_mass()
        #     # Use this with gaussian crawler
        #     # target_segment.set_lin_vel(lin_vec * self.lin_coef, local=False)
        #     # Use this with square crawler
        #     target_segment.set_lin_vel(lin_vec, local=False)
        #     # pass

        # Option : Apply to all body segments. This allows to control velocity for any Npoints. But it has the same shitty visualization as all options
        # for seg in [self.segs[0]]:
        for seg in self.segs:
            if self.lin_mode == 'impulse':
                impulse = lin * self.lin_vel_coef * seg.get_world_facing_axis() / seg.get_Box2D_mass()
                seg._body.ApplyLinearImpulse(impulse, seg._body.worldCenter, wake=True)
            elif self.lin_mode == 'force':
                force = lin * self.lin_force_coef * seg.get_world_facing_axis()
                seg._body.ApplyForceToCenter(force, wake=True)
            elif self.lin_mode == 'velocity':
                vel = lin * self.lin_vel_coef * seg.get_world_facing_axis()
                seg.set_lin_vel(vel, local=False)

    def step(self):
        self.cum_dur += self.model.dt
        pos = self.olfactor_pos
        self.food_detected, self.current_foodtype = self.detect_food(pos)
        lin, ang, self.feeder_motion = self.brain.step(pos, reward= self.food_detected is not None)




        # Trying restoration for any number of segments
        # if self.Nsegs == 1:
        # if self.Nsegs > 0:
        #     # Angular component
        #     # Restore body bend due to forward motion of the previous step
        #     # pass
        #     # ... apply the torque against the restorative powers to the body,
        #     # to update the angular velocity (for the physics engine) and the body_bend (for body state calculations) ...
        # else:
        #     # Default mode : apply torque
        #     # self.get_head()._body.ApplyTorque(self.torque, wake=True)
        #     pass
        if self.model.Box2D:
            self.Box2D_kinematics(lin, ang)
        else:
            lin_vel, ang_vel = self.get_vels(lin, ang, self.head.get_angularvelocity(),self.head.get_linearvelocity(),
                                             self.body_bend, dt=self.model.dt,
                                             ang_suppression=self.brain.locomotor.cur_ang_suppression)
            # lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)

            self.position_body(lin_vel=lin_vel, ang_vel=ang_vel)
            self.compute_body_bend()
            self.trajectory.append(self.pos)
        self.complete_step()



    def complete_step(self):
        if not self.model.Box2D :
            self.model.space.move_agent(self, self.pos)
        self.update_larva()
        for o in self.carried_objects:
            o.pos = self.pos

    # Using the forward Euler method to compute the next theta and theta'

    '''Here we implement the lateral oscillator as described in Wystrach(2016) :
    We use a,b,c,d parameters to be able to generalize. In the paper a=1, b=2*z, c=k, d=0

    Quoting  : where z =n / (2* sqrt(k*g) defines the damping ratio, with n the damping force coefficient, 
    k the stiffness coefficient of a linear spring and g the muscle gain. We assume muscles on each side of the body 
    work against each other to change the heading and thus, in this two-dimensional model, the net torque produced is 
    taken to be the difference in spike rates of the premotor neurons E_L(t)-E_r(t) driving the muscles on each side. 

    Later : a level of damping, for which we have chosen an intermediate value z =0.5
    In the table of parameters  : k=1

    So a=1, b=1, c=n/4g=1, d=0 
    '''


    def update_trajectory(self):
        last_pos = self.trajectory[-1]
        if self.model.Box2D:
            self.pos = self.global_midspine_of_body
        self.dst = np.sqrt(np.sum(np.array(self.pos - last_pos) ** 2))
        self.cum_dst += self.dst
        self.trajectory.append(self.pos)

    def set_head_contacts_ground(self, value):
        self.head_contacts_ground = value


    def position_body(self, lin_vel, ang_vel, tank_contact=True):
        self.dst = lin_vel * self.model.dt
        self.cum_dst += self.dst
        self.rear_orientation_change = rear_orientation_change(self.body_bend, self.dst, self.real_length,
                                                               correction_coef=self.bend_correction_coef)


        hp0, ho0 = self.head.get_pose()
        if tank_contact :
            hr0 = self.global_rear_end_of_head
            ang_vel, ho1, hr1, hp1 = self.assess_tank_contact(ang_vel, ho0, self.dst, hr0, hp0, self.model.dt, self.seg_lengths[0])
        else :
            d_or = ang_vel * self.model.dt
            if np.abs(d_or) > np.pi:
                self.body_bend_errors += 1
            ho1 = ho0 + d_or
            k = np.array([math.cos(ho1), math.sin(ho1)])
            hp1 = hp0 + k * self.dst

        self.head.update_all(hp1, ho1, lin_vel, ang_vel)
        if self.Nsegs>1 :
            for i, (seg, l) in enumerate(zip(self.segs[1:], self.seg_lengths[1:])):
                self.position_seg(seg, d_or=self.rear_orientation_change / (self.Nsegs - 1),
                                  front_end_pos=self.get_global_rear_end_of_seg(seg_index=i),seg_length=l)
        self.pos = self.global_midspine_of_body

    def position_seg(self, seg, d_or, front_end_pos, seg_length):
        p0, o0 = seg.get_pose()
        o1 = o0 + d_or
        k = np.array([math.cos(o1), math.sin(o1)])
        p1 = front_end_pos - k * seg_length / 2
        seg.update_poseNvertices(p1, o1)

    def compute_body_bend(self):
        self.spineangles = [angle_dif(self.segs[i].get_orientation(), self.segs[i + 1].get_orientation(), in_deg=False) for i in range(self.Nangles)]
        self.body_bend = wrap_angle_to_0(sum(self.spineangles[:self.Nangles_b]))



    @property
    def border_collision(self):
        if len(self.model.border_walls) == 0:
            return False
        else:
            x,y=self.pos
            p0=Point(x,y)
            d0 = self.sim_length / 4
            oM = self.head.get_orientation()
            sensor_ray = radar_tuple(p0=p0, angle=oM, distance=d0)
            min_dst, nearest_obstacle = detect_nearest_obstacle(self.model.border_walls, sensor_ray,p0)


            if min_dst is None:
                return False
            else :
                return True

    @property
    def border_collision3(self):
        if len(self.model.border_lines) == 0:
            return False
        else:
            p0 = self.olfactor_point
            # p0 = Point(self.olfactor_pos[0],self.olfactor_pos[1])
            # p0 = Point(self.pos)
            d0 = self.sim_length / 4
            # shape = self.head.get_shape()
            for l in self.model.border_lines :
                # print(list(l.coords))
                # raise
                if p0.distance(l) < d0:
                # if l.distance(shape)< d0:
                    return True
            return False

    @property
    def border_collision2(self):
        simple=True
        # print(self.radius, self.sim_length)
        # raise
        if len(self.model.border_lines) == 0 :
            return False
        else :
            from lib.aux.shapely_aux import distance, distance_multi
            oM=self.head.get_orientation()
            oL=oM+np.pi/3
            oR=oM-np.pi/3
            p0 = self.pos
            # p0 = tuple(self.get_sensor_position('olfactor'))
            # p0 = tuple(self.olfactor_pos)
            # p0=Point(p0)
            d0=self.sim_length/3
            dM = distance_multi(point=p0, angle=oM, ways=self.model.border_lines, max_distance=d0)
            # print(dM)
            if dM is not None :
                if simple :
                    return True
                dL = distance_multi(point=p0, angle=oL, ways=self.model.border_lines, max_distance=d0)
                dR = distance_multi(point=p0, angle=oR, ways=self.model.border_lines, max_distance=d0)
                if dL is None and dR is None :
                    return 'M'
                elif dL is None and dR is not None :
                    if dR<dM :
                        return 'RRM'
                    else :
                        return 'MR'
                elif dR is None and dL is not None :
                    if dL<dM :
                        return 'LLM'
                    else :
                        return 'ML'
                elif dR is not None and dL is not None:
                    if dL<dR :
                        return 'LLM'
                    else :
                        return 'RRM'
            else :
                return False

            # radarM = radar_line(p0, oM, d0)
            # radarL = radar_line(p0, oL, d0)
            # radarR = radar_line(p0, oR, d0)
            # interM,interL,interR = [], [], []
            # dM0,dL0,dR0 = d0*2, d0*2, d0*2
            # for l in self.model.border_lines :
            #     dM=radarM.intersection(l)
            #     dR=radarR.intersection(l)
            #     dL=radarL.intersection(l)
            #     if not dM.is_empty :
            #         interM.append(p0.distance(dM))
            #     if not dR.is_empty :
            #         interR.append(p0.distance(dR))
            #     if not dL.is_empty :
            #         interL.append(p0.distance(dL))
            # if len(interM)>0 :
            #     dM0=np.min(interM)
            # if len(interL)>0 :
            #     dL0=np.min(interL)
            # if len(interR)>0 :
            #     dR0=np.min(interR)
            # dst = np.min([dM0, dL0, dR0])
            # if dst>=d0 :
            #     return False
            #
            # side = np.argmin([dM0, dL0, dR0])
            # if side==1:
            #     return 'L'
            # elif side==2:
            #     return 'R'
            # elif side == 0:
            #     if dL0<dR0 :
            #         return 'ML'
            #     else :
            #         return 'MR'

            # # olfactor_point=self.olfactor_point
            # # shape=self.head.get_shape()
            # # print(p0, self.pos)
            # # print(np.rad2deg([oM,oL,oR]))
            # # raise
            # for l in self.model.border_lines :
            #     from lib.aux.shapely_aux import distance
            #     dM=distance(point=p0, angle=oM, way=l, max_distance=d0)
            #     dL=distance(point=p0, angle=oL, way=l, max_distance=d0)
            #     dR=distance(point=p0, angle=oR, way=l, max_distance=d0)
            #
            #     # raise
            #     if dM is not None or dL is not None or dR is not None:
                    # print(dd)
                # if shape.distance(l)<self.radius/5:
                # if olfactor_point.distance(l)<self.radius/5:
                # if l.intersects(shape):
                #     return True
            # return False


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
        res=self.border_collision
        d_ang = np.pi / 20
        if not res :
            return lin_vel, ang_vel
        elif res==True :
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * d_ang
            return lin_vel, ang_vel
        if 'M' in res :
            lin_vel = 0
        if 'RR' in res :
            ang_vel +=2*d_ang
        elif 'R' in res :
            ang_vel +=d_ang
        if 'LL' in res :
            ang_vel -=2*d_ang
        elif 'L' in res :
            ang_vel -=d_ang

        return lin_vel, ang_vel

    def assess_tank_contact(self, ang_vel, o0, d, hr0, hp0, dt, l0):
        # a0 = self.spineangles[0] if len(self.spineangles) > 0 else 0.0
        # ang_vel0 = np.clip(ang_vel, a_min=-np.pi - a0 / dt, a_max=(np.pi - a0) / dt)

        def avoid_border(ang_vel, counter, dd=0.01):
            if math.isinf(ang_vel):
                ang_vel = 1.0
            if any([ss not in self.get_sensors() for ss in ['L_front', 'R_front']]):
                counter += 1
                ang_vel *= -(1 + dd * counter)
                return ang_vel, counter
            else:
                s = self.sim_length / 1000
                L, R = self.get_sensor_position('L_front'), self.get_sensor_position('R_front')
                Ld, Rd = self.model.tank_polygon.exterior.distance(Point(L)), self.model.tank_polygon.exterior.distance(
                    Point(R))
                Ld, Rd = Ld / s, Rd / s
                LRd = Ld - Rd
                ang_vel += dd * LRd
                return ang_vel, counter

        def check_in_tank(ang_vel, o0, d, hr0, l0):
            o1 = o0 + ang_vel * dt
            k1 = np.array([math.cos(o1), math.sin(o1)])
            dxy = k1 * self.dst
            sim_dxy = dxy * self.model.scaling_factor
            # k = np.array([math.cos(o1), math.sin(o1)])
            # dxy = k * d
            if self.Nsegs > 1:
                hr1 = hr0 + sim_dxy
                hp1 = hr1 + k1 * l0 / 2
                # hf1 = hp1 + k1 * l0 / 2
            else:
                hr1 = None
                hp1 = hp0 + sim_dxy
                # hf1 = hp1 + k1 * (self.sim_length / 2)
            in_tank = inside_polygon(points=[hp1 + k1 * l0 / 2], tank_polygon=self.model.tank_polygon)
            return in_tank, o1, hr1, hp1

        in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)
        counter = -1
        while not in_tank:
            ang_vel, counter = avoid_border(ang_vel, counter)
            try :
                in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)
            except :
                pass
        # print(counter)
        # if counter > 0:
            # print(counter)
            # ang_vel = np.abs(ang_vel) * np.sign(ang_vel0)
        return ang_vel, o1, hr1, hp1

    # def wind_obstructed(self, wind_direction):
    #     from lib.aux.ang_aux import line_through_point
    #     ll=line_through_point(self.pos, wind_direction, np.max(self.model.arena_dims))
    #     return any([l.intersects(ll) for l in self.model.border_lines])


