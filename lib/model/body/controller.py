import abc
import math
import numpy as np
from shapely.geometry import LineString, Polygon, Point

import lib.aux.ang_aux
import lib.aux.sim_aux
from lib.model.body.body import LarvaBody


class BodyManager(LarvaBody):
    def __init__(self, model, pos, orientation, **kwargs) :
        super().__init__(model=model, pos=pos, orientation=orientation, **kwargs)

class BodyReplay(BodyManager):
    def __init__(self, model, pos, orientation, **kwargs) :
        super().__init__(model=model, pos=pos, orientation=orientation, **kwargs)

class BodySim(BodyManager):
    def __init__(self, model, orientation,density=300.0,
                 lin_vel_coef=1.0, ang_vel_coef=None, lin_force_coef=None, torque_coef=1.0,
                 lin_mode='velocity', ang_mode='torque', body_spring_k=1.0, bend_correction_coef=1.0,
                 lin_damping=1.0, ang_damping=1.0, **kwargs):
        self.lin_damping = lin_damping
        self.ang_damping = ang_damping

        super().__init__(model=model, pos=self.pos, orientation=orientation, density=density, **kwargs)

        self.body_spring_k = body_spring_k
        self.bend_correction_coef = bend_correction_coef

        self.head_contacts_ground = True
        self.trajectory = [self.pos]
        self.lin_activity = 0
        self.ang_activity = 0
        self.ang_vel = 0
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

    def step(self):
        self.restore_body_bend()
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
            if self.ang_mode == 'velocity':
                ang_vel = self.ang_activity * self.ang_vel_coef
                ang_vel = self.compute_ang_vel(v=ang_vel, z=0)
                self.segs[0].set_ang_vel(ang_vel)
                if self.Nsegs > 1:
                    for i in np.arange(1, self.mid_seg_index, 1):
                        self.segs[i].set_ang_vel(ang_vel / i)
            elif self.ang_mode == 'torque':
                self.torque = self.ang_activity * self.torque_coef*10000
                self.segs[0]._body.ApplyTorque(self.torque, wake=True)
                if self.Nsegs > 1:
                    for i in np.arange(1, self.mid_seg_index, 1):
                        self.segs[i]._body.ApplyTorque(self.torque / i, wake=True)


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
        else:
            if self.lin_mode == 'velocity':
                lin_vel_amp = self.lin_activity * self.lin_vel_coef
            else:
                raise ValueError(f'Linear mode {self.lin_mode} not implemented for non-physics simulation')
            if self.ang_mode == 'torque':
                self.torque = self.ang_activity * self.torque_coef
                # The damping free mode is much closer to the experimental histogram of body bends (singlelarva_turns.ipynb)
                # On the other hand even a minor damping of 0.01 produces a two-top distribution at -20,20 (test_turner.py)
                # So I m gonna explore the no damping case.
                # TODO Attention, the experimental distribution is from a larva constantly striding but my findings are on stationary turner component
                #  I should explore whether interference corrects the dist even when damping is present
                # UPdate : 0 damping does not fix the two-pick (though makes it a bit better).Interference neither.
                # But maybe I cn raise the torque coef of 0.07 becuse two_osc reach -20,20 and interference drops it to -10,10.
                ang_vel = self.compute_ang_vel(torque=self.torque,
                                               v=self.get_head().get_angularvelocity(),
                                               z=self.ang_damping)
            elif self.ang_mode == 'velocity':
                ang_vel = self.ang_activity * self.ang_vel_coef
                ang_vel = self.compute_ang_vel(v=ang_vel, z=self.ang_damping)

            self.step_no_physics(lin_vel=lin_vel_amp, ang_vel=ang_vel)

        for o in self.carried_objects:
            o.pos = self.pos

        # print(self.unique_id, self.cum_dst)

    def compute_new_lin_vel_vector(self, target_segment):
        # Option 1 : Create the linear velocity from orientation.
        # This was the default. But it seems because of numerical issues it doesn't generate the expected vector,
        # which results in some angular velocity  when linear velocity is applied.
        # I haven't figured out when and why that happens
        # orientation = target_segment.get_normalized_orientation()
        # orientation = target_segment.get_orientation()
        # lin_vec = b2Vec2(self.lin_activity * np.cos(orientation),
        #                  self.lin_activity * np.sin(orientation))

        # Option 2 : Just retrieve the current lin_velocity vec
        # Update : Doesn't work because linear velocity can be zero
        # Trying to integrate the two options

        # if target_segment.get_linearvelocity_vec() != b2Vec2(0,0) :
        #     previous_lin_velocity_vec = target_segment.get_linearvelocity_vec()
        #     previous_lin_velocity_amp = target_segment.get_linearvelocity_amp()
        #     previous_lin_velocity_unit_vec = previous_lin_velocity_vec / previous_lin_velocity_amp
        #     lin_vec = self.lin_activity * previous_lin_velocity_unit_vec
        # else :
        #     orientation = target_segment.get_orientation()
        #     # orientation = target_segment.get_normalized_orientation()
        #     lin_vec = b2Vec2(self.lin_activity * np.cos(orientation),
        #                      self.lin_activity * np.sin(orientation))
        lin_vec = self.lin_activity * target_segment.get_world_facing_axis()

        return lin_vec

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
    def compute_ang_vel(self, torque=0.0, v=0.0, z=0.0):

        k = self.body_spring_k
        b = self.body_bend
        new_v = v + (-z * v - k * b + torque) * self.model.dt
        # if new_v * v<0:
        #     return 0.0
        # else:
        #     return new_v
        # if new_v>0 :
        #     print('dd')
        return new_v

    def restore_body_bend(self):
        self.compute_spineangles()
        d, l = self.dst, self.sim_length
        if not self.model.Box2D:
            if self.Nsegs == 2:
                self.spineangles[0] = lib.aux.ang_aux.restore_bend_2seg(self.spineangles[0], d, l,
                                                                        correction_coef=self.bend_correction_coef)
            else:
                self.spineangles = lib.aux.ang_aux.restore_bend(self.spineangles, d, l, self.Nsegs,
                                                                correction_coef=self.bend_correction_coef)
        self.compute_body_bend()

    # def set_lin_activity(self, value):
    #     self.lin_activity = value

    # def set_ang_activity(self, value):
    #     self.ang_activity = value


    def update_trajectory(self):
        last_pos = self.trajectory[-1]
        if self.model.Box2D:
            self.pos = self.get_global_midspine_of_body()
        self.dst = np.sqrt(np.sum(np.array(self.pos - last_pos) ** 2))
        self.cum_dst += self.dst
        self.trajectory.append(self.pos)


    def set_head_contacts_ground(self, value):
        self.head_contacts_ground = value

    def step_no_physics(self, lin_vel, ang_vel):

        # print()
        # print(np.round(fun.comp_bearing([self.pos[0]], [self.pos[1]],  np.rad2deg(self.get_head().get_orientation()), loc=(0.2,0.2), in_deg=True))[0])
        # self.body_bend += self.dt * ang_velocity
        # self.body_bend = np.clip(self.body_bend, a_min=-np.pi, a_max=np.pi)

        # BIO : Translate motor signal to behavior (how much to turn, how much to move)
        # distance = motor_vector[0] * self.max_speed
        # self.header = (self.header + motor_vector[1] * math.pi / 2) % (2 * math.pi)

        # COUNTER
        # self.total_distance += distance

        # TECH : Move the agent
        # Compute orientation
        dt = self.model.dt
        a0 = self.spineangles[0] if len(self.spineangles) > 0 else 0.0
        head = self.get_head()
        hp0, o0 = head.get_pose()
        hr0 = self.get_global_rear_end_of_head()
        # print(self.unique_id)


        border_collision = any([l.intersects(self.segs[0].get_shape()) for l in self.model.border_lines]) if len(self.model.border_lines) > 0 else False
        if not self.model.larva_collisions:
            ids=self.model.detect_collisions(self.unique_id)
            larva_collision=False if len(ids)==0 else True
        else:
            larva_collision = False
        if border_collision or larva_collision:
            lin_vel = 0
            ang_vel += np.sign(ang_vel)*np.pi/10



        d = lin_vel * dt
        ang_vel0=np.clip(ang_vel, a_min=-np.pi - a0 / dt, a_max=(np.pi - a0) / dt)

        def avoid_border(ang_vel, counter, dd=0.01):

            if self.touch_sensors is None or any([ss not in self.get_sensors() for ss in ['L_front', 'R_front']]):
                counter += 1
                ang_vel *= -(1 + dd * counter)
                return ang_vel, counter
            else :
                s=self.sim_length/1000
                L,R=self.get_sensor_position('L_front'), self.get_sensor_position('R_front')
                Ld, Rd=self.model.tank_polygon.exterior.distance(Point(L)), self.model.tank_polygon.exterior.distance(Point(R))
                Ld, Rd=Ld/s,Rd/s
                LRd=Ld-Rd
                ang_vel += dd * LRd
                return ang_vel, counter


        def check_in_tank(ang_vel, o0, d, hr0) :

            o1 = o0 + ang_vel * dt
            # print(o1,o0,ang_vel,dt)
            k = np.array([math.cos(o1), math.sin(o1)])
            dxy = k * d
            if self.Nsegs > 1:
                hr1 = hr0 + dxy
                hp1 = hr1 + k * self.seg_lengths[0] / 2
                hf1 = hr1 + k * self.seg_lengths[0]
            else:
                hr1 = None
                hp1 = hp0 + dxy
                hf1 = hp1 + k * (self.sim_length / 2)
            hf1_ok, hp1_ok = lib.aux.sim_aux.inside_polygon(points=[hf1, hp1], tank_polygon=self.model.tank_polygon)
            in_tank = all([hf1_ok, hp1_ok])
            return in_tank, o1, hr1, hp1

        in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0)
        counter = -1
        while not in_tank :
            ang_vel, counter=avoid_border(ang_vel, counter)
            in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0)

        if counter>0:
            # print(counter)
            ang_vel = np.abs(ang_vel)*np.sign(ang_vel0)
        head.set_pose(hp1, o1)
        head.update_vertices(hp1, o1)
        if self.Nsegs > 1:
            self.position_rest_of_body(o1-o0, head_rear_pos=hr1, head_or=o1)
        self.pos = self.get_global_midspine_of_body() if self.Nsegs != 2 else hr1
        self.model.space.move_agent(self, self.pos)
        head.set_lin_vel(lin_vel)
        head.set_ang_vel(ang_vel)
        self.dst = d
        self.cum_dst += d
        self.trajectory.append(self.pos)

    def position_rest_of_body(self, d_orientation, head_rear_pos, head_or):
        N = self.Nsegs
        if N == 1:
            pass
        else:
            if N == 2:
                seg = self.segs[1]
                self.spineangles[0] += d_orientation
                new_or = head_or - self.spineangles[0]
                seg.set_orientation(new_or)
                new_p = head_rear_pos + np.array([-np.cos(new_or), -np.sin(new_or)]) * self.seg_lengths[1] / 2
                seg.set_position(new_p)
                seg.update_vertices(new_p, new_or)
            else:
                bend_per_spineangle = d_orientation / (N / 2)
                for i, (seg, l) in enumerate(zip(self.segs[1:], self.seg_lengths[1:])):
                    if i == 0:
                        global_p = head_rear_pos
                        previous_seg_or = head_or
                    else:
                        global_p = self.get_global_rear_end_of_seg(seg_index=i)
                        previous_seg_or = self.segs[i].get_orientation()
                    if i + 1 <= N / 2:
                        self.spineangles[i] += bend_per_spineangle
                    new_or = previous_seg_or - self.spineangles[i]
                    seg.set_orientation(new_or)
                    new_p = global_p + np.array([-np.cos(new_or), -np.sin(new_or)]) * l / 2
                    seg.set_position(new_p)
                    seg.update_vertices(new_p, new_or)
            self.compute_body_bend()

    def compute_spineangles(self):
        seg_ors = [seg.get_orientation() for seg in self.segs]
        self.spineangles = [lib.aux.ang_aux.angle_dif(seg_ors[i], seg_ors[i + 1], in_deg=False) for i in range(self.Nangles)]

    def compute_body_bend(self):
        self.body_bend = sum(self.spineangles[:self.Nangles_b])
