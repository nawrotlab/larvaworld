import random

import numpy as np

from lib.aux import functions as fun
from lib.model.agents._larva import Larva
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain
from lib.model.modules.nengobrain import NengoBrain
from lib.model.DEB.deb import DEB


class LarvaSim(BodySim, Larva):
    def __init__(self, unique_id, model, pos, orientation, larva_pars,odor, group='', default_color=None,life=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, pos=pos,
                       odor=odor, group=group, default_color=default_color)
        # try:
        #     larva_pars['brain']['olfactor_params']['odor_dict'] = self.update_odor_dicts(
        #         larva_pars['brain']['olfactor_params']['odor_dict'])
        # except:
        #     pass
        self.brain = self.build_brain(larva_pars['brain'])
        self.build_energetics(larva_pars['energetics'], life=life)
        BodySim.__init__(self, model=model, orientation=orientation, **larva_pars['physics'],**larva_pars['body'], **kwargs)
        # print(larva_pars['body'])
        self.reset_feeder()
        self.radius = self.sim_length / 2

        self.food_detected, self.feeder_motion, self.current_V_eaten, self.feed_success = None, False, 0, None
        self.food_missed, self.food_found = False, False

    # def update_odor_dicts(self, odor_dict):  #
    #
    #     temp = {'mean': 0.0, 'std': 0.0}
    #     food_odor_ids = fun.unique_list(
    #         [s.odor_id for s in self.model.get_food() + [self] if s.odor_id is not None])
    #     if odor_dict is None:
    #         odor_dict = {}
    #         # odor_dict = {odor_id: temp for odor_id in food_odor_ids}
    #     for odor_id in food_odor_ids:
    #         if odor_id not in list(odor_dict.keys()):
    #             odor_dict[odor_id] = temp
    #     return odor_dict

    def compute_next_action(self):
        self.cum_dur += self.model.dt
        pos = self.olfactor_pos
        self.detect_food(pos)
        self.lin_activity, self.ang_activity, self.feeder_motion = self.brain.run(pos)
        self.current_V_eaten, self.feed_success = self.feed(self.food_detected, self.feeder_motion)
        self.run_energetics(self.current_V_eaten)
        self.update_behavior()


    def detect_food(self, pos):
        if self.brain.feeder is not None:
            prev_item = self.food_detected
            item, q = None, None
            grid = self.model.food_grid
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_cell_value(cell) > 0:
                    item, q = cell, grid.quality
            else:
                valid = [a for a in self.model.get_food() if a.amount > 0]
                accessible_food = [a for a in valid if a.contained(pos)]
                if accessible_food:
                    food = random.choice(accessible_food)
                    self.resolve_carrying(food)
                    item, q = food, food.quality
            self.food_found = True if (prev_item is None and item is not None) else False
            self.food_missed = True if (prev_item is not None and item is None) else False
            self.food_detected=item

    def feed(self, source, motion):
        a_max = self.max_V_bite
        if motion:
            if source is not None:
                grid = self.model.food_grid
                if grid:
                    V = -grid.add_cell_value(source, -a_max)
                else:
                    V = source.subtract_amount(a_max)
                self.feed_success_counter += 1
                self.amount_eaten += V * 1000
                return V, True
            else:
                return 0, False
        else:
            return 0, None

    def reset_feeder(self):
        self.feed_success_counter = 0
        self.amount_eaten = 0
        self.feeder_motion = False
        try:
            self.max_V_bite = self.get_max_V_bite()
        except:
            self.max_V_bite = None
        try:
            self.brain.feeder.reset()
        except:
            pass

    def get_max_V_bite(self):
        return self.brain.feeder.V_bite * self.V  # ** (2 / 3)

    def build_energetics(self, energetic_pars, life=None):
        self.real_length = None
        self.real_mass = None
        self.V = None

        # p_am=260
        if energetic_pars is not None:
            self.energetics = True
            if energetic_pars['deb_on']:
                self.temp_cum_V_eaten = 0
                self.temp_mean_f = []
                self.f_exp_coef = np.exp(-energetic_pars['f_decay'] * self.model.dt)
                steps_per_day = 24 * 60
                cc = {
                    'id': self.unique_id,
                    'steps_per_day': steps_per_day,
                    'hunger_gain': energetic_pars['hunger_gain'],
                    'hunger_as_EEB': energetic_pars['hunger_as_EEB'],
                    'V_bite': self.brain.feeder.V_bite,
                    'absorption': energetic_pars['absorption'],
                    'substrate_quality': life['substrate_quality'],
                    'substrate_type': life['substrate_type'],
                    'intermitter': self.brain.intermitter,
                }
                self.deb = DEB(**cc)

                self.deb.grow_larva(**life)
                # self.deb.grow_larva(hours_as_larva=self.model.hours_as_larva, epochs=self.model.epochs)
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
                self.deb = None
                self.food_to_biomass_ratio = 0.3
        else:
            self.energetics = False

    def build_brain(self, conf):
        modules = conf['modules']
        if conf['nengo']:
            brain = NengoBrain(agent=self, modules=modules, conf=conf)
        else:
            brain = DefaultBrain(agent=self, modules=modules, conf=conf)
        return brain

    def run_energetics(self, V_eaten):
        if self.energetics:
            if self.deb :
                f = self.deb.f
                if V_eaten>0:
                    f += self.deb.absorption
                    # f += food_quality * self.deb.absorption
                f *= self.f_exp_coef
                self.temp_cum_V_eaten += V_eaten
                self.temp_mean_f.append(f)
                if self.model.Nticks % self.deb_step_every == 0:
                    self.deb.run(f=np.mean(self.temp_mean_f), X_V=self.temp_cum_V_eaten)
                    self.temp_cum_V_eaten = 0
                    self.temp_mean_f = []

                self.real_length = self.deb.Lw * 10 / 1000
                self.real_mass = self.deb.Ww
                self.V = self.deb.V
                self.adjust_body_vertices()

            else:
                if V_eaten>0:
                    self.real_mass += V_eaten * self.food_to_biomass_ratio
                    self.adjust_shape_to_mass()
                    self.adjust_body_vertices()
                    self.V = self.real_length ** 3
            self.max_V_bite = self.get_max_V_bite()




    def update_behavior_dict(self):
        d = self.null_behavior_dict.copy()
        inter=self.brain.intermitter
        if inter is not None :
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
        self.set_color([color]*self.Nsegs)

    @property
    def front_orientation(self):
        return np.rad2deg(self.get_head().get_normalized_orientation())

    @property
    def front_orientation_unwrapped(self):
        return np.rad2deg(self.get_head().get_orientation())

    @property
    def rear_orientation_unwrapped(self):
        return np.rad2deg(self.get_tail().get_orientation())

    @property
    def rear_orientation(self):
        return np.rad2deg(self.get_tail().get_normalized_orientation())

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
        return np.rad2deg(self.get_head().get_angularvelocity())

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
                opponent_group = fun.LvsRtoggle(carrier_group)
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

