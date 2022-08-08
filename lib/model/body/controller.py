import math
import random

import numpy as np

from lib.model.body.body import LarvaBody
from lib.aux import dictsNlists as dNl, ang_aux, sim_aux, shapely_aux


class PhysicsController:
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

    def compute_ang_vel(self, k, c, torque, v, b, dt, I=1):
        dtI = dt / I
        return v + (-c * v - k * b + torque) * dtI

    def get_vels(self, lin, ang, prev_ang_vel, prev_lin_vel, bend, dt, ang_suppression):
        if self.lin_mode == 'velocity':
            if lin != 0:
                lin_vel = lin * self.lin_vel_coef
            else:
                lin_vel = 0  # prev_lin_vel*(1-self.lin_damping*dt)
        else:
            raise ValueError(f'Linear mode {self.lin_mode} not implemented for non-physics simulation')
        if self.ang_mode == 'torque':
            torque = ang * self.torque_coef
            # self.torque =ang * self.torque_coef*ang_suppression
            ang_vel = self.compute_ang_vel(torque=torque,
                                           v=prev_ang_vel, b=bend,
                                           c=self.ang_damping, k=self.body_spring_k, dt=dt)
        elif self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
        # ang_vel*=ang_suppression
        lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
        ang_vel *= ang_suppression
        return lin_vel, ang_vel

    def assess_collisions(self, lin_vel, ang_vel):
        return lin_vel, ang_vel


class BodyManager(LarvaBody):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BodyReplay(BodyManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BodySim(BodyManager, PhysicsController):
    def __init__(self, physics, density=300.0, **kwargs):

        PhysicsController.__init__(self, **physics)
        BodyManager.__init__(self, density=density, **kwargs)

        self.head_contacts_ground = True
        self.trajectory = [self.initial_pos]

        self.body_bend = 0
        self.body_bend_errors = 0
        self.negative_speed_errors = 0
        self.border_go_errors = 0
        self.border_turn_errors = 0
        self.Nangles_b = int(self.Nangles + 1 / 2)
        self.spineangles = [0.0] * self.Nangles
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

    def Box2D_kinematics(self, lin, ang):
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
        lin, ang, self.feeder_motion = self.brain.step(pos, reward=self.food_detected is not None)

        if self.model.Box2D:
            self.Box2D_kinematics(lin, ang)
        else:
            # print(self.brain.locomotor.cur_ang_suppression, self.head.get_angularvelocity(),self.head.get_linearvelocity())
            lin_vel, ang_vel = self.get_vels(lin, ang, self.head.get_angularvelocity(), self.head.get_linearvelocity(),
                                             self.body_bend, dt=self.model.dt,
                                             ang_suppression=self.brain.locomotor.cur_ang_suppression)
            # print(lin_vel, ang_vel)
            # print()
            self.position_body(lin_vel=lin_vel, ang_vel=ang_vel, dt=self.model.dt)
            self.compute_body_bend()
            self.trajectory.append(self.pos)
        self.complete_step()

    def complete_step(self):
        if self.head.get_linearvelocity() < 0:
            self.negative_speed_errors += 1
            self.head.set_lin_vel(0)
        if not self.model.Box2D:
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

    # def go_forward(self, lin_vel, k, hf01,dt, tank,scaling_factor=1, delta=0.00011, counter=0, border_go_errors=0):
    #     if np.isnan(lin_vel) or counter>100 :
    #         border_go_errors += 1
    #         return 0, 0, hf01
    #     d = lin_vel * dt
    #     dxy = k * d * scaling_factor
    #     hf1 = hf01 + dxy
    #
    #     if not sim_aux.inside_polygon([hf01], tank):
    #         lin_vel -= delta
    #         if lin_vel < 0:
    #             return 0, 0, hf01
    #         counter += 1
    #         return self.go_forward(lin_vel, k, hf01, delta, counter,border_go_errors)
    #     else:
    #         return lin_vel, d, hf1, border_go_errors
    #
    # def turn_head(self, ang_vel, hr0, ho0, l0, ang_range,dt, tank, delta=np.pi / 90, counter=0, border_turn_errors=0):
    #     def get_hf(ho):
    #         kk = np.array([math.cos(ho), math.sin(ho)])
    #         hf = hr0 + kk * l0
    #         return kk, hf
    #     if np.isnan(ang_vel) or counter>100:
    #         border_turn_errors+=1
    #         k0, hf00 = get_hf(ho0)
    #         return 0, ho0, k0, hf00
    #     ho1 = ho0 + ang_vel * dt
    #     k, hf01 = get_hf(ho1)
    #     if not sim_aux.inside_polygon([hf01], tank):
    #         if counter == 0:
    #             delta *= np.sign(ang_vel)
    #         ang_vel -= delta
    #
    #         if ang_vel < ang_range[0]:
    #             ang_vel = ang_range[0]
    #             delta = np.abs(delta)
    #         elif ang_vel > ang_range[1]:
    #             ang_vel = ang_range[1]
    #             delta -= np.abs(delta)
    #         counter += 1
    #
    #         return self.turn_head(ang_vel, hr0, ho0, l0, ang_range, delta, counter, border_turn_errors)
    #     else:
    #         return ang_vel, ho1, k, hf01, border_turn_errors

    def position_body(self, lin_vel, ang_vel,dt, tank_contact=True):
        sf = self.model.scaling_factor
        hp0, ho0 = self.head.get_pose()
        hr0 = self.global_rear_end_of_head





        l0 = self.seg_lengths[0]
        A0,A1=self.valid_Dbend_range(0,ho0)
        ang_range = A0 / dt, A1 / dt


        if ang_vel < A0 / dt:
            ang_vel = A0 / dt
            self.body_bend_errors += 1
            # print(f'{self.body_bend_errors}---------------')
        elif ang_vel > A1 / dt:
            ang_vel = A1 / dt
            self.body_bend_errors += 1
            # print(f'{self.body_bend_errors}++++++++++++++++')
        # if lin_vel<0 :
        #     print(self.unique_id, 000000000)
        if tank_contact:
            tank = self.model.tank_polygon


            d, ang_vel, lin_vel,hp1, ho1, turn_err, go_err = sim_aux.position_head_in_tank(hr0, ho0, l0, ang_range, ang_vel, lin_vel, dt, tank, sf=sf)
            self.border_turn_errors+=turn_err
            self.border_go_errors+=go_err
            # ang_vel, ho1, k, hf01, border_turn_errors = sim_aux.turn_head(ang_vel, hr0, ho0, l0, ang_range=ang_range, dt=dt, tank=tank)
            # lin_vel, d, hf1,border_go_errors = sim_aux.go_forward(lin_vel, k, hf01, dt=dt, tank=tank, scaling_factor=scaling_factor)
            # print(lin_vel)
        else:
            ho1 = ho0 + ang_vel * dt
            k = np.array([math.cos(ho1), math.sin(ho1)])
            d = lin_vel * dt
            hp1 = hr0 + k * (d * sf + l0 / 2)
        self.head.update_all(hp1, ho1, lin_vel, ang_vel)
        self.dst = d
        self.cum_dst += self.dst
        self.rear_orientation_change = ang_aux.rear_orientation_change(self.body_bend, self.dst, self.real_length,
                                                                       correction_coef=self.bend_correction_coef)

        # if lin_vel<0 :
        #     print(self.unique_id, 11111111111111111)

        if self.Nsegs > 1:
            for i, (seg, l) in enumerate(zip(self.segs[1:], self.seg_lengths[1:])):
                self.position_seg(seg, d_or=self.rear_orientation_change / (self.Nsegs - 1),
                                  front_end_pos=self.get_global_rear_end_of_seg(seg_index=i), seg_length=l)
        self.pos = self.global_midspine_of_body

        # hp1=
        # self.dst=d
        #
        #
        #
        #
        #
        # d_or = ang_vel * self.model.dt
        # ho1=ho0+d_or
        # k = np.array([math.cos(ho1), math.sin(ho1)])
        # hf1=hr0+k*self.seg_lengths[0]
        #
        #
        #
        # hp00=ang_aux.rotate_around_point(hp0, d_or,hr0)
        #
        #
        # hp1,ho1,hf1=self.position_head(hp0, ho0,lin_vel, ang_vel)
        #
        # # dst = lin_vel * self.model.dt
        #
        # if tank_contact :
        #     counter=-1
        #     while not self.in_tank([hp1])
        #     while not self.in_tank([hf1]):
        #         if math.isinf(ang_vel):
        #             ang_vel = 1.0
        #         print(counter)
        #         counter+=1
        #         ang_vel += 0.01*counter
        #         if counter>100:
        #             ho0+=np.pi/4
        #             counter=1
        #             ang_vel=0.01
        #         hp1, ho1, hf1 = self.position_head(hp0, ho0, lin_vel, ang_vel)
        #         # if counter>100:
        #         #     ho0+=np.pi
        #         #     counter=0
        #         # ho0+=np.pi/180 * counter
        #         # lin_vel*=0.95
        #         # self.position_head(hp0, ho0, lin_vel, ang_vel)
        # # else :
        # self.head.update_all(hp1, ho1, lin_vel, ang_vel)
        #     # hr0 = self.global_rear_end_of_head
        #     # ang_vel, ho1, dst, hr1, hp1 = self.assess_tank_contact(ang_vel, ho0, dst, hr0, hp0, self.model.dt, self.seg_lengths[0])
        #
        # # else :
        # #     d_or = ang_vel * self.model.dt
        # #
        # #     ho1 = ho0 + d_or
        # #     k = np.array([math.cos(ho1), math.sin(ho1)])
        # #     hp1 = hp0 + k * dst
        # # self.dst = dst
        # self.cum_dst += self.dst
        # self.rear_orientation_change = ang_aux.rear_orientation_change(self.body_bend, self.dst, self.real_length,
        #                                                                correction_coef=self.bend_correction_coef)
        #
        # # self.head.update_all(hp1, ho1, lin_vel, ang_vel)

    def position_seg(self, seg, d_or, front_end_pos, seg_length):
        p0, o0 = seg.get_pose()
        o1 = o0 + d_or
        k = np.array([math.cos(o1), math.sin(o1)])
        p1 = front_end_pos - k * seg_length / 2
        seg.update_poseNvertices(p1, o1)

    def compute_body_bend(self):
        self.spineangles = [
            ang_aux.angle_dif(self.segs[i].get_orientation(), self.segs[i + 1].get_orientation(), in_deg=False) for i in
            range(self.Nangles)]
        self.body_bend = ang_aux.wrap_angle_to_0(sum(self.spineangles[:self.Nangles_b]))

    @property
    def border_collision(self):
        if len(self.model.border_walls) == 0:
            return False
        else:
            x, y = self.pos
            p0 = shapely_aux.Point(x, y)
            d0 = self.sim_length / 4
            oM = self.head.get_orientation()
            sensor_ray = shapely_aux.radar_tuple(p0=p0, angle=oM, distance=d0)
            min_dst, nearest_obstacle = shapely_aux.detect_nearest_obstacle(self.model.border_walls, sensor_ray, p0)

            if min_dst is None:
                return False
            else:
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
            for l in self.model.border_lines:
                # print(list(l.coords))
                # raise
                if p0.distance(l) < d0:
                    # if l.distance(shape)< d0:
                    return True
            return False

    @property
    def border_collision2(self):
        simple = True
        # print(self.radius, self.sim_length)
        # raise
        if len(self.model.border_lines) == 0:
            return False
        else:
            from lib.aux.shapely_aux import distance, distance_multi
            oM = self.head.get_orientation()
            oL = oM + np.pi / 3
            oR = oM - np.pi / 3
            p0 = self.pos
            # p0 = tuple(self.get_sensor_position('olfactor'))
            # p0 = tuple(self.olfactor_pos)
            # p0=Point(p0)
            d0 = self.sim_length / 3
            dM = distance_multi(point=p0, angle=oM, ways=self.model.border_lines, max_distance=d0)
            # print(dM)
            if dM is not None:
                if simple:
                    return True
                dL = distance_multi(point=p0, angle=oL, ways=self.model.border_lines, max_distance=d0)
                dR = distance_multi(point=p0, angle=oR, ways=self.model.border_lines, max_distance=d0)
                if dL is None and dR is None:
                    return 'M'
                elif dL is None and dR is not None:
                    if dR < dM:
                        return 'RRM'
                    else:
                        return 'MR'
                elif dR is None and dL is not None:
                    if dL < dM:
                        return 'LLM'
                    else:
                        return 'ML'
                elif dR is not None and dL is not None:
                    if dL < dR:
                        return 'LLM'
                    else:
                        return 'RRM'
            else:
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
        res = self.border_collision
        d_ang = np.pi / 20
        if not res:
            return lin_vel, ang_vel
        elif res == True:
            lin_vel = 0
            ang_vel += np.sign(ang_vel) * d_ang
            return lin_vel, ang_vel
        if 'M' in res:
            lin_vel = 0
        if 'RR' in res:
            ang_vel += 2 * d_ang
        elif 'R' in res:
            ang_vel += d_ang
        if 'LL' in res:
            ang_vel -= 2 * d_ang
        elif 'L' in res:
            ang_vel -= d_ang

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
                Ld, Rd = self.model.tank_polygon.exterior.distance(
                    shapely_aux.Point(L)), self.model.tank_polygon.exterior.distance(
                    shapely_aux.Point(R))
                Ld, Rd = Ld / s, Rd / s
                LRd = Ld - Rd
                ang_vel += dd * LRd
                return ang_vel, counter

        def check_in_tank(ang_vel, o0, d, hr0, l0):
            o1 = o0 + ang_vel * dt
            k1 = np.array([math.cos(o1), math.sin(o1)])
            dxy = k1 * d
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
            points = [hp1 + k1 * l0 / 2]
            # print(points)
            in_tank = sim_aux.inside_polygon(points=points, tank_polygon=self.model.tank_polygon)
            return in_tank, o1, hr1, hp1

        in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)
        counter = -1
        while not in_tank:
            # o0 += np.pi/180
            counter += 1
            ang_vel *= -(1 + 0.01 * counter)
            print(counter)
            # ang_vel, counter = avoid_border(ang_vel, counter)
            if counter > 1000:
                #     o0+=np.pi
                #     d=0
                ang_vel = 0.01
                counter = 0
                o0 -= np.pi
            in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)
            # except :
            #     pass
        # print(counter)
        # if counter > 0:
        # print(counter)
        # ang_vel = np.abs(ang_vel) * np.sign(ang_vel0)
        return ang_vel, o1, d, hr1, hp1

    # @ property
    def in_tank(self, ps):
        # hp, ho = self.head.get_pose()
        # k = np.array([math.cos(ho), math.sin(ho)])
        # hf=hp + k * self.seg_lengths[0] / 2
        # hr=hp - k * self.seg_lengths[0] / 2
        # ps=[hf, hp]
        return sim_aux.inside_polygon(points=ps, tank_polygon=self.model.tank_polygon)

    # def wind_obstructed(self, wind_direction):
    #     from lib.aux.ang_aux import line_through_point
    #     ll=line_through_point(self.pos, wind_direction, np.max(self.model.arena_dims))
    #     return any([l.intersects(ll) for l in self.model.border_lines])
