import random
import time

import numpy as np

import lib.aux.sim_aux
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
        self.build_energetics(larva_pars['energetics'], life_history=life_history)
        BodySim.__init__(self, model=model, orientation=orientation, **larva_pars['physics'], **larva_pars['body'],
                         **kwargs)
        self.brain = self.build_brain(larva_pars['brain'])
        if self.energetics:
            self.deb.intermitter = self.brain.intermitter

        self.reset_feeder()
        self.radius = self.sim_length / 2

        self.food_detected, self.feeder_motion, self.current_V_eaten, self.feed_success = None, False, 0, 0
        # self.food_missed, self.food_found = False, False
        self.cum_food_detected = 0
        self.foraging_dict = {id: {action: 0 for action in ['on_food_tr', 'sf_am']} for id in
                              self.model.foodtypes.keys()}
        # self.foraging_dict= {action :{id: [0] for id in self.model.foodtypes} for action in ['detection', 'consumption']}

    def compute_next_action(self):
        # t0 = []
        # t0.append(time.time())
        self.cum_dur += self.model.dt
        pos = self.olfactor_pos
        self.food_detected, foodtype = self.detect_food(pos)
        # t0.append(time.time())


        self.lin_activity, self.ang_activity, self.feeder_motion = self.brain.run(pos)
        # t0.append(time.time())
        self.current_V_eaten, self.feed_success = self.feed(self.food_detected, self.feeder_motion)
        # t0.append(time.time())
        self.update_foraging_dict(foodtype, self.current_V_eaten)
        self.run_energetics(self.current_V_eaten)
        self.update_behavior()

        # t0.append(time.time())
        # print(np.array(np.diff(t0) * 1000000).astype(int))
    def detect_food(self, pos):
        t0 = []
        # t0.append(time.time())
        item, foodtype = None, None
        if self.brain.feeder is not None or self.touch_sensors is not None:
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
        return self.brain.feeder.V_bite * self.V  # ** (2 / 3)

    def build_energetics(self, energetic_pars, life_history):
        if not hasattr(self, 'real_mass'):
            self.real_mass = None
        if not hasattr(self, 'real_length'):
            self.real_length = None

        self.V = None

        if energetic_pars is not None:
            self.energetics = True
            self.temp_cum_V_eaten = 0
            self.f_exp_coef = np.exp(-energetic_pars['f_decay'] * energetic_pars['DEB_dt'])
            steps_per_day = 24 * 6
            cc = {
                'id': self.unique_id,
                'steps_per_day': steps_per_day,
                'hunger_gain': energetic_pars['hunger_gain'],
                'hunger_as_EEB': energetic_pars['hunger_as_EEB'],
                'V_bite': energetic_pars['V_bite'],
                'absorption': energetic_pars['absorption'],
                'species': energetic_pars['species'],
                # 'substrate': self.model.food_grid.substrate,
                # 'substrate': life['substrate'],
                # 'substrate_type': life['substrate_type'],
                # 'intermitter': self.brain.intermitter,
            }
            self.deb = DEB(**cc)

            self.deb.grow_larva(**life_history)
            if energetic_pars['DEB_dt'] is None:
                self.deb_step_every = 1
                self.deb.set_steps_per_day(int(24 * 60 * 60 / self.model.dt))
            else:
                self.deb_step_every = int(energetic_pars['DEB_dt'] / self.model.dt)
                self.deb.set_steps_per_day(int(24 * 60 * 60 / energetic_pars['DEB_dt']))
            self.deb.assimilation_mode = energetic_pars['assimilation_mode']
            self.real_length = self.deb.Lw * 10 / 1000
            self.real_mass = self.deb.Ww
            self.V = self.deb.V

        else:
            self.energetics = False
            self.deb = None

    def build_brain(self, conf):
        modules = conf['modules']
        if conf['nengo']:
            brain = NengoBrain(agent=self, modules=modules, conf=conf)
        else:
            brain = DefaultBrain(agent=self, modules=modules, conf=conf)
        return brain

    def run_energetics(self, V_eaten):
        if self.energetics:
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
                # self.max_V_bite = self.get_max_V_bite()

    def get_feed_success(self, t):
        return self.feed_success

    def update_behavior_dict(self):
        d = self.null_behavior_dict.copy()
        inter = self.brain.intermitter
        if inter is not None:
            s, f, p = inter.active_bouts
            d['stride_id'] = s is not None
            d['feed_id'] = f is not None
            d['pause_id'] = p is not None
            d['stride_stop'] = inter.stride_stop

        orvel = self.front_orientation_vel
        if orvel > 0:
            d['Lturn_id'] = True
        elif orvel < 0:
            d['Rturn_id'] = True
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
                # if self.model.experiment=='flag' :
                #     prev_carrier.brain.olfactor.reset_gain(prev_carrier.base_odor_id)
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
                        # f.brain.olfactor.set_gain(0.0, 'Flag odor')
                    else:
                        f.brain.olfactor.set_gain(0.0, carrier_group_odor_id)
                        # f.brain.olfactor.reset_gain('Flag odor')
                self.brain.olfactor.set_gain(-self.gain_for_base_odor, opponent_group_odor_id)

    def update_behavior(self):
        # Paint the body to visualize effector state
        if self.model.color_behavior:
            self.update_behavior_dict()
        # print(self.deb.hunger, self.deb.e)
        self.brain.intermitter.update(food_present=self.food_detected, feed_success=self.feed_success)

    def update_foraging_dict(self, foodtype, current_V_eaten):
        if foodtype is not None:
            self.foraging_dict[foodtype]['on_food_tr'] += self.model.dt
            self.foraging_dict[foodtype]['sf_am'] += current_V_eaten
            self.cum_food_detected += int(self.on_food)

    def finalize_foraging_dict(self):
        for id, vs in self.foraging_dict.items():
            vs['on_food_tr'] /= self.cum_dur
            vs['sf_am'] /= self.V
        return self.foraging_dict
