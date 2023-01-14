import math
import random
from copy import deepcopy
import numpy as np


from lib import aux
from lib.model.agents._agent import LarvaworldAgent


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model, pos=None, radius=None, default_color='black', **kwargs):

        if unique_id is None:
            unique_id = model.next_id(type='Larva')
        super().__init__(unique_id=unique_id, model=model, default_color=default_color, pos=pos, radius=radius,
                         **kwargs)
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))
        self.carried_objects = []

    def update_color(self, default_color,dic, mode='lin'):

        color = deepcopy(default_color)
        if mode == 'lin':
            if dic.stride_id or dic.run_id:
                color = np.array([0, 150, 0])
            elif dic.pause_id:
                color = np.array([255, 0, 0])
            elif dic.feed_id:
                color = np.array([0, 0, 255])
        elif mode == 'ang':
            if dic.Lturn_id:
                color[2] = 150
            elif dic.Rturn_id:
                color[2] = 50
        return color

    @property
    def dt(self):
        return self.model.dt



    @property
    def x(self):
        return self.pos[0] / self.model.scaling_factor

    @property
    def y(self):
        return self.pos[1] / self.model.scaling_factor

    @property
    def x0(self):
        return self.initial_pos[0] / self.model.scaling_factor

    @property
    def y0(self):
        return self.initial_pos[1] / self.model.scaling_factor


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


class LarvaMotile(Larva, PhysicsController):
    def __init__(self, physics,energetics,brain,life_history=None,  **kwargs):
        Larva.__init__(self, **kwargs)

        PhysicsController.__init__(self, **physics)

        self.build_energetics(energetics, life_history=life_history)
        self.brain = self.build_brain(brain)
        if self.deb is not None:
            self.deb.set_intermitter(self.brain.locomotor.intermitter)
        self.reset_feeder()
        self.cum_dur = 0

        self.cum_dst = 0.0
        self.dst = 0.0
        self.backward_motion = True



        self.trajectory = [self.initial_pos]

        # self.body_bend = 0
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





    def detect_food(self, pos):
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


    def update_larva(self):
        self.current_V_eaten, self.feed_success = self.feed(self.food_detected, self.feeder_motion)
        self.update_foraging_dict(self.current_foodtype, self.current_V_eaten)
        self.run_energetics(self.current_V_eaten)
        if self.brain.locomotor.intermitter is not None:
            self.brain.locomotor.intermitter.update(food_present=self.food_detected, feed_success=self.feed_success)

        for o in self.carried_objects:
            o.pos = self.pos


    def feed(self, source, motion):

        def get_max_V_bite():
            return self.brain.locomotor.feeder.V_bite * self.V  # ** (2 / 3)

        if motion:
            a_max = get_max_V_bite()
            if source is not None:
                grid = self.model.food_grid
                if grid:
                    V = -grid.add_cell_value(source, -a_max)
                else:
                    V = source.subtract_amount(a_max)
                self.feed_success_counter += 1
                self.amount_eaten += V * 1000
                return V, 1
            else:
                return 0, -1
        else:
            return 0, 0

    def reset_feeder(self):
        self.food_detected, self.feeder_motion, self.current_V_eaten, self.current_foodtype, self.feed_success = None, False, None, 0, 0
        self.cum_food_detected, self.feed_success_counter, self.amount_eaten = 0, 0, 0
        self.foraging_dict = aux.AttrDict({id: {action: 0 for action in ['on_food_tr', 'sf_am']} for id in
                                           self.model.foodtypes.keys()})
        try:
            self.brain.feeder.reset()
        except:
            pass


    def build_energetics(self, energetic_pars, life_history):
        from lib.model.DEB.deb import DEB
        if energetic_pars is not None:
            pDEB = energetic_pars.DEB
            pGUT = energetic_pars.gut
            dt = pDEB.DEB_dt
            if dt is None:
                dt = self.model.dt
            self.temp_cum_V_eaten = 0
            self.f_exp_coef = np.exp(-pDEB.f_decay * dt)
            self.deb = DEB(id=self.unique_id, steps_per_day=24 * 6, gut_params=pGUT, **pDEB)
            self.deb.grow_larva(**life_history)
            self.deb_step_every = int(dt / self.model.dt)
            self.deb.set_steps_per_day(int(24 * 60 * 60 / dt))
            self.real_length = self.deb.Lw * 10 / 1000
            self.real_mass = self.deb.Ww
            self.V = self.deb.V
            # print(self.real_length)
        else:
            self.deb = None
            self.V = None
            self.real_mass = None
            self.real_length = None

    def run_energetics(self, V_eaten):
        if self.deb is not None:
            self.temp_cum_V_eaten += V_eaten
            if self.model.Nticks % self.deb_step_every == 0:
                X_V = self.temp_cum_V_eaten
                if X_V > 0:
                    self.deb.f += self.deb.gut.k_abs
                self.deb.f *= self.f_exp_coef
                self.deb.run(X_V=X_V)
                self.temp_cum_V_eaten = 0
                self.real_length = self.deb.Lw * 10 / 1000
                self.real_mass = self.deb.Ww
                self.V = self.deb.V
                self.adjust_body_vertices()

    def build_brain(self, conf):
        if conf.nengo:
            from lib.model.modules.nengobrain import NengoBrain
            return NengoBrain(agent=self, conf=conf, dt=self.model.dt)
        else:
            from lib.model.modules.brain import DefaultBrain
            return DefaultBrain(agent=self, conf=conf, dt=self.model.dt)

    def get_feed_success(self, t):
        return self.feed_success





    @property
    def on_food_dur_ratio(self):
        return self.cum_food_detected * self.model.dt / self.cum_dur if self.cum_dur != 0 else 0

    @property
    def on_food(self):
        return self.food_detected is not None

    def get_on_food(self, t):
        return self.on_food

    @property
    def scaled_amount_eaten(self):
        return self.amount_eaten / self.get_real_mass()

    def resolve_carrying(self, food):
        if food.can_be_carried and food not in self.carried_objects:
            if food.is_carried_by is not None:
                prev_carrier = food.is_carried_by
                if prev_carrier == self:
                    return
                prev_carrier.carried_objects.remove(food)
                prev_carrier.brain.olfactor.reset_all_gains()
            food.is_carried_by = self
            self.carried_objects.append(food)
            if self.model.experiment == 'capture_the_flag':
                self.brain.olfactor.set_gain(self.gain_for_base_odor, self.base_odor_id)
            elif self.model.experiment == 'keep_the_flag':
                carrier_group = self.group
                carrier_group_odor_id = self.odor_id
                opponent_group = aux.LvsRtoggle(carrier_group)
                opponent_group_odor_id = f'{opponent_group}_odor'
                for f in self.model.get_flies():
                    if f.group == carrier_group:
                        f.brain.olfactor.set_gain(f.gain_for_base_odor, opponent_group_odor_id)
                    else:
                        f.brain.olfactor.set_gain(0.0, carrier_group_odor_id)
                self.brain.olfactor.set_gain(-self.gain_for_base_odor, opponent_group_odor_id)


    def update_foraging_dict(self, foodtype, current_V_eaten):
        if foodtype is not None:
            self.foraging_dict[foodtype].on_food_tr += self.model.dt
            self.foraging_dict[foodtype].sf_am += current_V_eaten
            self.cum_food_detected += int(self.on_food)

    def finalize_foraging_dict(self):
        for id, vs in self.foraging_dict.items():
            vs.on_food_tr /= self.cum_dur
            vs.sf_am /= self.V
        return self.foraging_dict


    def update_trajectory(self):
        last_pos = self.trajectory[-1]
        if self.model.Box2D:
            self.pos = self.global_midspine_of_body
        self.dst = np.sqrt(np.sum(np.array(self.pos - last_pos) ** 2))
        self.cum_dst += self.dst
        self.trajectory.append(self.pos)

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






    @property
    def border_collision(self):
        if len(self.model.borders) == 0:
            return False
        else:
            x, y = self.pos
            p0 = aux.Point(x, y)
            d0 = self.sim_length / 4
            oM = self.head.get_orientation()
            sensor_ray = aux.radar_tuple(p0=p0, angle=oM, distance=d0)
            min_dst, nearest_obstacle = aux.detect_nearest_obstacle(self.model.borders, sensor_ray, p0)

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

                if p0.distance(l) < d0:
                    return True
            return False

    @property
    def border_collision2(self):
        simple = True

        if len(self.model.border_lines) == 0:
            return False
        else:
            oM = self.head.get_orientation()
            oL = oM + np.pi / 3
            oR = oM - np.pi / 3
            p0 = self.pos
            d0 = self.sim_length / 3
            dM = aux.min_dst_to_lines_along_vector(point=p0, angle=oM, target_lines=self.model.border_lines, max_distance=d0)
            if dM is not None:
                if simple:
                    return True
                dL = aux.min_dst_to_lines_along_vector(point=p0, angle=oL, target_lines=self.model.border_lines, max_distance=d0)
                dR = aux.min_dst_to_lines_along_vector(point=p0, angle=oR, target_lines=self.model.border_lines, max_distance=d0)
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
                    aux.Point(L)), self.model.tank_polygon.exterior.distance(
                    aux.Point(R))
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
            else:
                hr1 = None
                hp1 = hp0 + sim_dxy
            points = [hp1 + k1 * l0 / 2]
            in_tank = aux.inside_polygon(points=points, tank_polygon=self.model.tank_polygon)
            return in_tank, o1, hr1, hp1

        in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)
        counter = -1
        while not in_tank:
            # o0 += np.pi/180
            counter += 1
            ang_vel *= -(1 + 0.01 * counter)

            if counter > 1000:
                #     o0+=np.pi
                #     d=0
                ang_vel = 0.01
                counter = 0
                o0 -= np.pi
            in_tank, o1, hr1, hp1 = check_in_tank(ang_vel, o0, d, hr0, l0)

        return ang_vel, o1, d, hr1, hp1

    # @ property
    def in_tank(self, ps):
        # hp, ho = self.head.get_pose()
        # k = np.array([math.cos(ho), math.sin(ho)])
        # hf=hp + k * self.seg_lengths[0] / 2
        # hr=hp - k * self.seg_lengths[0] / 2
        # ps=[hf, hp]
        return aux.inside_polygon(points=ps, tank_polygon=self.model.tank_polygon)
