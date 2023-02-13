import random
from copy import deepcopy
import numpy as np


from larvaworld.lib import aux
from larvaworld.lib.model.agents import LarvaworldAgent, LarvaBody, PhysicsController


class Larva(LarvaworldAgent):
    def __init__(self, unique_id, model,orientation, **kwargs):

        if unique_id is None:
            unique_id = model.next_id(type='Larva')
        super().__init__(unique_id=unique_id, model=model,**kwargs)
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))
        self.carried_objects = []
        self.trajectory = [self.initial_pos]

        self.initial_orientation = orientation
        self.orientation = self.initial_orientation

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





class LarvaMotile(LarvaBody,Larva, PhysicsController):
    def __init__(self, model,body, physics,energetics,brain,life_history=None,Box2D_params={},orientation=None, pos=None,  **kwargs):
        if orientation is None:
            orientation = random.uniform(0, 2 * np.pi)
        if pos is None:
            pos = (0.0, 0.0)
        Larva.__init__(self,model=model, pos = pos, orientation = orientation, **kwargs)

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
        LarvaBody.__init__(self, model=self.model,pos=self.pos, orientation=self.orientation, **body, **Box2D_params)
        self.radius=self.sim_length/2









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
        from larvaworld.lib.model.deb.deb import DEB
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
            from larvaworld.lib.model.modules.nengobrain import NengoBrain
            return NengoBrain(agent=self, conf=conf, dt=self.model.dt)
        else:
            from larvaworld.lib.model.modules.brain import DefaultBrain
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

    def update_behavior_dict(self):
        d = aux.AttrDict(self.null_behavior_dict.copy())
        inter = self.brain.locomotor.intermitter
        if inter is not None:
            s, f, p, r = inter.active_bouts
            d.stride_id = s is not None
            d.feed_id = f is not None
            d.pause_id = p is not None
            d.run_id = r is not None
            d.stride_stop = inter.stride_stop

        orvel = self.front_orientation_vel
        if orvel > 0:
            d.Lturn_id = True
        elif orvel < 0:
            d.Rturn_id = True
        color = self.update_color(self.default_color, d)
        self.set_color([color] * self.Nsegs)






    def step(self):
        self.cum_dur += self.model.dt
        pos = self.olfactor_pos
        self.food_detected, self.current_foodtype = self.detect_food(pos)
        lin, ang, self.feeder_motion = self.brain.step(pos, reward=self.food_detected is not None)

        if self.model.Box2D:
            self.Box2D_kinematics(lin, ang)
        else:
            lin_vel, ang_vel = self.get_vels(lin, ang, self.head.get_angularvelocity(),
                                             self.body_bend, dt=self.model.dt,
                                             ang_suppression=self.brain.locomotor.cur_ang_suppression)
            self.position_body(lin_vel=lin_vel, ang_vel=ang_vel, dt=self.model.dt)
            self.compute_body_bend()
            self.trajectory.append(self.pos)
        self.complete_step()

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