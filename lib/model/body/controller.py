import abc
import math
import random
import time

import numpy as np

from lib.aux.ang_aux import restore_bend, restore_bend_2seg
from lib.aux.sim_aux import inside_polygon
from lib.ga.geometry.point import Point
from lib.ga.geometry.util import detect_nearest_obstacle, radar_tuple
from lib.model.body.body import LarvaBody
from lib.aux.ang_aux import angle_dif

class BodyManager(LarvaBody):
    def __init__(self, model, pos, orientation, **kwargs) :
        super().__init__(model=model, pos=pos, orientation=orientation, **kwargs)

class BodyReplay(BodyManager):
    def __init__(self, model, pos, orientation, **kwargs) :
        super().__init__(model=model, pos=pos, orientation=orientation, **kwargs)

class BodySim(BodyManager):
    def __init__(self, model, pos, orientation,density=300.0,
                 lin_vel_coef=1.0, ang_vel_coef=1.0, lin_force_coef=1.0, torque_coef=1.0,
                 lin_mode='velocity', ang_mode='torque', body_spring_k=1.0, bend_correction_coef=1.0,
                 lin_damping=1.0, ang_damping=1.0, **kwargs):
        self.lin_damping = lin_damping
        self.ang_damping = ang_damping

        super().__init__(model=model, pos=pos, orientation=orientation, density=density, **kwargs)

        self.body_spring_k = body_spring_k
        self.bend_correction_coef = bend_correction_coef

        self.head_contacts_ground = True
        self.trajectory = [self.initial_pos]
        self.lin_activity = 0
        self.ang_activity = 0
        # self.ang_vel = 0
        self.body_bend = 0
        self.body_bend_errors = 0
        self.Nangles_b = int(self.Nangles + 1 / 2)
        self.spineangles = [0.0]*self.Nangles
        self.compute_body_bend()
        self.torque = 0
        self.mid_seg_index = int(self.Nsegs / 2)

        self.cum_dur = 0

        self.cum_dst = 0.0
        self.dst = 0.0

        self.lin_mode = lin_mode
        self.ang_mode = ang_mode

        # Cheating calibration (this is to get a step_to_length around 0.3 for gaussian crawler with amp=1 and std=0.05
        # applied on TAIL of l3-sized larvae with interval -0.05.
        # Basically this means for a 7-timestep window only one value is 1 and all others nearly 0)
        # Will become obsolete when we have a definite answer to :
        # is relative_to_length displacement_per_contraction a constant? How much is it?
        # if self.Npoints == 6:
        #     self.lin_coef = 3.2  # 2.7 for tail, 3.2 for head applied velocity
        # elif self.Npoints == 1:
        #     self.lin_coef = 1.7
        # elif self.Npoints == 11:
        #     self.lin_coef = 4.2
        # else:
        #     self.lin_coef = 1.5 + self.Npoints / 4

        self.lin_vel_coef = lin_vel_coef
        self.ang_vel_coef = ang_vel_coef
        self.lin_force_coef = lin_force_coef
        self.torque_coef = torque_coef
        self.backward_motion = True



        # from lib.conf.par import pargroups
        # from lib.conf.par import AgentCollector
        # g=pargroups['full']
        # self.collector=AgentCollector(g,self)
    def detect_food(self, pos):
        t0 = []
        # t0.append(time.time())
        item, foodtype = None, None
        if self.brain.locomotor.feeder is not None or self.touch_sensors is not None:
            # prev_item = self.food_detected
            grid = self.model.food_grid
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_cell_value(cell) > 0:
                    item, foodtype = cell, grid.unique_id

            else:
                # t0.append(time.time())
                valid = [a for a in self.model.get_food() if a.amount > 0]

                accessible_food = [a for a in valid if a.contained(pos)]
                # t0.append(time.time())
                if accessible_food:
                    food = random.choice(accessible_food)
                    self.resolve_carrying(food)
                    item, foodtype = food, food.group
                # t0.append(time.time())
            # self.food_found = True if (prev_item is None and item is not None) else False
            # self.food_missed = True if (prev_item is not None and item is None) else False
        # t0.append(time.time())
        # print(np.array(np.diff(t0) * 1000000).astype(int))
        return item, foodtype

    # def get_velocities(self):
    #     pass

    def Box2D_kinematics(self):
        if self.ang_mode == 'velocity':
            ang_vel = self.ang_activity * self.ang_vel_coef
            ang_vel = self.compute_ang_vel(v=ang_vel, c=0,  k=self.body_spring_k, b=self.body_bend)
            self.segs[0].set_ang_vel(ang_vel)
            if self.Nsegs > 1:
                for i in np.arange(1, self.mid_seg_index, 1):
                    self.segs[i].set_ang_vel(ang_vel / i)
        elif self.ang_mode == 'torque':
            self.torque = self.ang_activity * self.torque_coef
            # self.segs[0]._body.ApplyAngularImpulse(self.torque, wake=True)
            self.segs[0]._body.ApplyTorque(self.torque, wake=True)
            # if self.Nsegs > 1:
            #     for i in np.arange(1, self.mid_seg_index, 1):
            #         self.segs[i]._body.ApplyTorque(self.torque / i, wake=True)

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
                temp_lin_vec_amp = self.lin_activity * self.lin_vel_coef
                impulse = temp_lin_vec_amp * seg.get_world_facing_axis() / seg.get_Box2D_mass()
                seg._body.ApplyLinearImpulse(impulse, seg._body.worldCenter, wake=True)
            elif self.lin_mode == 'force':
                lin_force_amp = self.lin_activity * self.lin_force_coef
                force = lin_force_amp * seg.get_world_facing_axis()
                seg._body.ApplyForceToCenter(force, wake=True)
            elif self.lin_mode == 'velocity':
                lin_vel_amp = self.lin_activity * self.lin_vel_coef
                vel = lin_vel_amp * seg.get_world_facing_axis()
                seg.set_lin_vel(vel, local=False)

    def get_vels(self, lin, ang, prev_ang_vel, ang_suppression):
        if self.lin_mode == 'velocity':
            lin_vel = lin * self.lin_vel_coef
        else:
            raise ValueError(f'Linear mode {self.lin_mode} not implemented for non-physics simulation')
        if self.ang_mode == 'torque':
            self.torque =ang * self.torque_coef
            ang_vel = self.compute_ang_vel(torque=self.torque,
                                           v=prev_ang_vel, b=self.body_bend,
                                           c=self.ang_damping,  k=self.body_spring_k)
        elif self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
        ang_vel*=ang_suppression
        return lin_vel, ang_vel

    def step(self):
        self.cum_dur += self.model.dt
        self.restore_body_bend(self.dst, self.real_length)
        pos = self.olfactor_pos
        self.food_detected, self.current_foodtype = self.detect_food(pos)
        self.lin_activity, self.ang_activity, self.feeder_motion = self.brain.step(pos, reward= self.food_detected is not None)




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
            self.Box2D_kinematics()
        else:
            lin_vel, ang_vel = self.get_vels(self.lin_activity, self.ang_activity, self.head.get_angularvelocity(),
                                             self.brain.locomotor.cur_ang_suppression)
            lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
            self.dst = lin_vel * self.model.dt
            self.cum_dst += self.dst
            self.position_body(lin_vel=lin_vel, ang_vel=ang_vel)
            self.trajectory.append(self.pos)
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

    # Update 4.1.2020 : Setting b=0 because it is a substitute of the angular damping of the environment
    def compute_ang_vel(self, k, c, torque, v, b, I=1):
        dtI=self.model.dt/I
        return v + (-c * v - k * b + torque) * dtI

    # def compute_ang_vel2(self, k, c, torque, v, b, I=1):
    #     dtI = self.model.dt / I
    #     dv = (-c * v - k * b) * dtI
    #     v0 = v + dv
    #     if v0 * v < 0:
    #         v0 = 0
    #     v1 = v0 + torque * dtI
    #
    #     return v1

    def restore_body_bend(self,d,l):
        if not self.model.Box2D:
            if self.Nsegs == 2:
                self.spineangles[0] = restore_bend_2seg(self.spineangles[0], d, l,
                                                                        correction_coef=self.bend_correction_coef)
            else:
                self.spineangles = restore_bend(self.spineangles, d, l, self.Nsegs,
                                                           correction_coef=self.bend_correction_coef)
        else :
            self.compute_spineangles()
        self.compute_body_bend()

    def update_trajectory(self):
        last_pos = self.trajectory[-1]
        if self.model.Box2D:
            self.pos = self.global_midspine_of_body
        self.dst = np.sqrt(np.sum(np.array(self.pos - last_pos) ** 2))
        self.cum_dst += self.dst
        self.trajectory.append(self.pos)


    def set_head_contacts_ground(self, value):
        self.head_contacts_ground = value


    def position_body(self, lin_vel, ang_vel):
        hp0, o0 = self.head.get_pose()
        hr0 = self.global_rear_end_of_head
        ang_vel, o1, hr1, hp1 = self.assess_tank_contact(ang_vel, o0, self.dst, hr0, hp0, self.model.dt, self.seg_lengths[0])

        self.head.update_all(hp1, o1, lin_vel, ang_vel)
        self.position_rest_of_body(o0=o0, pos=hr1, o1=o1)
        self.pos = self.global_midspine_of_body if self.Nsegs != 2 else hr1



    def position_rest_of_body(self, o0, pos, o1):
        N = self.Nsegs
        if N == 1:
            pass
        else:
            if N == 2:
                self.spineangles[0]=self.check_bend_error(self.spineangles[0]+ o1-o0)

                o2 = (o1 - self.spineangles[0])% (np.pi * 2)
                k2 = np.array([math.cos(o2), math.sin(o2)])
                p2 = pos - k2 * self.seg_lengths[1] / 2
                self.tail.update_poseNvertices(p2, o2)
            else:
                bend_per_spineangle = (o1-o0) / (N / 2)
                for i, (seg, l) in enumerate(zip(self.segs[1:], self.seg_lengths[1:])):
                    if i == 0:
                        global_p = pos
                        previous_seg_or = o1
                    else:
                        global_p = self.get_global_rear_end_of_seg(seg_index=i)
                        previous_seg_or = self.segs[i].get_orientation()
                    if i + 1 <= N / 2:
                        self.spineangles[i] = self.check_bend_error(self.spineangles[i] + bend_per_spineangle)
                    new_or = (previous_seg_or - self.spineangles[i])% (np.pi * 2)
                    kk = np.array([math.cos(new_or), math.sin(new_or)])
                    new_p = global_p - kk * l / 2
                    seg.update_poseNvertices(new_p, new_or)
            self.compute_body_bend()

    def compute_spineangles(self):
        if self.Nangles==1:
            self.spineangles =[angle_dif(self.head.get_orientation(), self.tail.get_orientation(), in_deg=False)]
        else :
            self.spineangles = [angle_dif(self.segs[i].get_orientation(), self.segs[i + 1].get_orientation(), in_deg=False) for i in range(self.Nangles)]

    def compute_body_bend(self):
        self.body_bend = sum(self.spineangles[:self.Nangles_b])

    def check_bend_error(self, a):
        if np.abs(a) > np.pi:
            self.body_bend_errors += 1
            a = (a + np.pi) % (np.pi * 2) - np.pi
        return a

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
                # no obstacle detected
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
            if self.touch_sensors is None or any([ss not in self.get_sensors() for ss in ['L_front', 'R_front']]):
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


