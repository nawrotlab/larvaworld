import random
import time

import numpy as np

import lib.aux.sim_aux
from lib.aux.dictsNlists import AttrDict
from lib.model.agents._larva import Larva
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain
from lib.model.modules.nengobrain import NengoBrain
from lib.model.DEB.deb import DEB


class LarvaSim(BodySim, Larva):
    def __init__(self, unique_id, model, pos, orientation, larva_pars, odor, group='', default_color=None,
                 life_history=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, pos=pos,
                       odor=odor, group=group, default_color=default_color)
        self.build_energetics(larva_pars.energetics, life_history=life_history)
        BodySim.__init__(self, model=model, orientation=orientation, **larva_pars.physics, **larva_pars.body,
                         **larva_pars.Box2D_params,**kwargs)

        self.brain = self.build_brain(larva_pars.brain)
        if self.deb is not None:
            self.deb.set_intermitter(self.brain.locomotor.intermitter)
        self.reset_feeder()
        self.radius = self.sim_length / 2

        self.food_detected, self.feeder_motion, self.current_V_eaten, self.current_foodtype, self.feed_success = None, False, None, 0,0
        self.cum_food_detected = 0
        self.foraging_dict = AttrDict.from_nested_dicts({id: {action: 0 for action in ['on_food_tr', 'sf_am']} for id in
                              self.model.foodtypes.keys()})

    def update_larva(self):
        # t0 = []
        # t0.append(time.time())

        # pos = self.olfactor_pos
        # self.food_detected, foodtype = self.detect_food(pos)
        # t0.append(time.time())



        # t0.append(time.time())
        self.current_V_eaten, self.feed_success = self.feed(self.food_detected, self.feeder_motion)
        # t0.append(time.time())
        self.update_foraging_dict(self.current_foodtype, self.current_V_eaten)
        self.run_energetics(self.current_V_eaten)
        self.update_behavior()

        # t0.append(time.time())
        # print(np.array(np.diff(t0) * 1000000).astype(int))

    def feed(self, source, motion):

        if motion:
            a_max = self.get_max_V_bite()
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
        self.feed_success_counter = 0
        self.amount_eaten = 0
        self.feeder_motion = False
        # try:
        #     self.max_V_bite = self.get_max_V_bite()
        # except:
        #     self.max_V_bite = None
        try:
            self.brain.feeder.reset()
        except:
            pass

    def get_max_V_bite(self):
        return self.brain.locomotor.feeder.V_bite * self.V  # ** (2 / 3)

    def build_energetics(self, energetic_pars, life_history):
        if energetic_pars is not None:
            pDEB=energetic_pars.DEB
            pGUT=energetic_pars.gut
            dt=pDEB.DEB_dt
            if dt is None:
                dt=self.model.dt
            self.temp_cum_V_eaten = 0
            self.f_exp_coef = np.exp(-pDEB.f_decay * dt)
            self.deb = DEB(id=self.unique_id, steps_per_day=24*6,gut_params=pGUT, **pDEB)
            self.deb.grow_larva(**life_history)
            self.deb_step_every = int(dt / self.model.dt)
            self.deb.set_steps_per_day(int(24 * 60 * 60 / dt))
            self.real_length = self.deb.Lw * 10 / 1000
            self.real_mass = self.deb.Ww
            self.V = self.deb.V

        else:
            self.deb = None
            self.V = None
            self.real_mass = None
            self.real_length = None

    def build_brain(self, conf):
        if conf.nengo:
            return NengoBrain(agent=self, modules=conf.modules, conf=conf)
        else:
            return DefaultBrain(agent=self, modules=conf.modules, conf=conf)

    def run_energetics(self, V_eaten):
        if self.deb is not None:
            self.temp_cum_V_eaten += V_eaten
            if self.model.Nticks % self.deb_step_every == 0:
                X_V = self.temp_cum_V_eaten
                if X_V > 0:
                    self.deb.f += self.deb.absorption
                self.deb.f *= self.f_exp_coef
                self.deb.run(X_V=X_V)
                self.temp_cum_V_eaten = 0
                self.real_length = self.deb.Lw * 10 / 1000
                self.real_mass = self.deb.Ww
                self.V = self.deb.V
                self.adjust_body_vertices()

    def get_feed_success(self, t):
        return self.feed_success

    def update_behavior_dict(self):
        d = AttrDict.from_nested_dicts(self.null_behavior_dict.copy())
        inter = self.brain.locomotor.intermitter
        if inter is not None:
            s, f, p = inter.active_bouts
            d.stride_id = s is not None
            d.feed_id = f is not None
            d.pause_id = p is not None
            d.stride_stop = inter.stride_stop

        orvel = self.front_orientation_vel
        if orvel > 0:
            d.Lturn_id = True
        elif orvel < 0:
            d.Rturn_id = True
        color = self.update_color(self.default_color, d)
        self.set_color([color] * self.Nsegs)

    @property
    def on_food_dur_ratio(self):
        return self.cum_food_detected * self.model.dt / self.cum_dur if self.cum_dur != 0 else 0

    @property
    def on_food(self):
        return self.food_detected is not None

    def get_on_food(self, t):
        return self.on_food

    @property
    def front_orientation(self):
        return np.rad2deg(self.head.get_normalized_orientation())

    @property
    def front_orientation_unwrapped(self):
        return np.rad2deg(self.head.get_orientation())

    @property
    def rear_orientation_unwrapped(self):
        return np.rad2deg(self.tail.get_orientation())

    @property
    def rear_orientation(self):
        return np.rad2deg(self.tail.get_normalized_orientation())

    @property
    def bend(self):
        # return self.body_bend
        return np.rad2deg(self.body_bend)

    @property
    def bend_vel(self):
        return np.rad2deg(self.body_bend_vel)

    @property
    def bend_acc(self):
        return np.rad2deg(self.body_bend_acc)

    @property
    def front_orientation_vel(self):
        return np.rad2deg(self.head.get_angularvelocity())

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
                opponent_group = lib.aux.sim_aux.LvsRtoggle(carrier_group)
                opponent_group_odor_id = f'{opponent_group}_odor'
                for f in self.model.get_flies():
                    if f.group == carrier_group:
                        f.brain.olfactor.set_gain(f.gain_for_base_odor, opponent_group_odor_id)
                    else:
                        f.brain.olfactor.set_gain(0.0, carrier_group_odor_id)
                self.brain.olfactor.set_gain(-self.gain_for_base_odor, opponent_group_odor_id)

    def update_behavior(self):
        # Paint the body to visualize effector state
        if self.model.color_behavior:
            self.update_behavior_dict()
        if self.brain.locomotor.intermitter is not None :
            self.brain.locomotor.intermitter.update(food_present=self.food_detected, feed_success=self.feed_success)

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
