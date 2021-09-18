import random

import numpy as np
from nengo import Simulator

from lib.aux import functions as fun
from lib.model.agents._larva import Larva
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain
from lib.model.modules.nengobrain import NengoBrain
from lib.model.DEB.deb import DEB


class LarvaSim(BodySim, Larva):
    def __init__(self, unique_id, model, pos, orientation, larva_pars, group='', default_color=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, pos=pos,
                       **larva_pars['odor'], group=group, default_color=default_color)
        try:
            larva_pars['brain']['olfactor_params']['odor_dict'] = self.update_odor_dicts(
                larva_pars['brain']['olfactor_params']['odor_dict'])
        except:
            pass
        self.brain = self.build_brain(larva_pars['brain'])
        self.build_energetics(larva_pars['energetics'])
        BodySim.__init__(self, model=model, orientation=orientation, **larva_pars['physics'],
                         **larva_pars['body'], **kwargs)
        self.reset_feeder()
        self.radius = self.sim_length / 2

        self.food_detected, self.feeder_motion, self.current_V_eaten, self.feed_success = None, False,0, False

    def update_odor_dicts(self, odor_dict):  #

        temp = {'mean': 0.0, 'std': 0.0}
        food_odor_ids = fun.unique_list(
            [s.odor_id for s in self.model.get_food() + [self] if s.odor_id is not None])
        if odor_dict is None:
            odor_dict = {}
            # odor_dict = {odor_id: temp for odor_id in food_odor_ids}
        for odor_id in food_odor_ids:
            if odor_id not in list(odor_dict.keys()):
                odor_dict[odor_id] = temp
        return odor_dict

    def compute_next_action(self):
        self.cum_dur += self.model.dt
        pos = self.get_olfactor_position()
        self.food_detected, food_quality = self.detect_food(pos)
        self.lin_activity, self.ang_activity, self.feeder_motion = self.brain.run(pos)
        self.current_V_eaten, self.feed_success = self.feed(self.food_detected, self.feeder_motion)
        if self.energetics:
            self.run_energetics(self.food_detected, self.feed_success, self.current_V_eaten, food_quality)
        # Paint the body to visualize effector state
        if self.model.color_behavior:
            self.update_behavior_dict()
        # else:
        #     self.set_color([self.default_color] * self.Nsegs)

    def detect_food(self, pos):

        if self.brain.feeder is not None:
            grid = self.model.food_grid
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_cell_value(cell) > 0:
                    return cell, grid.quality
            else:
                valid = [a for a in self.model.get_food() if a.amount > 0]
                accessible_food = [a for a in valid if a.contained(pos)]
                if accessible_food:
                    food = random.choice(accessible_food)
                    self.resolve_carrying(food)
                    return food, food.quality
        return None, None

    def feed(self, source, motion):
        a_max = self.max_V_bite
        if motion and source is not None:
            grid = self.model.food_grid
            if grid :
                V=-grid.add_cell_value(source, -a_max)
            else :
                V = source.subtract_amount(a_max)
            self.feed_success_counter += 1
            self.amount_eaten += V*1000
            return V, True
        else:
            return 0, False

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
        # print(self.V*10**6)
        return self.brain.feeder.V_bite * self.V  # ** (2 / 3)

    def build_energetics(self, energetic_pars):
        self.real_length = None
        self.real_mass = None
        self.V = None

        # p_am=260
        if energetic_pars is not None:
            self.energetics = True
            if energetic_pars['deb_on']:
                self.temp_cum_V_eaten =0
                self.temp_mean_f =[]
                self.hunger_as_EEB = energetic_pars['hunger_as_EEB']
                self.f_exp_coef = np.exp(-energetic_pars['f_decay'] * self.model.dt)
                steps_per_day = 24 * 60
                cc = {
                    'id': self.unique_id,
                    'steps_per_day': steps_per_day,
                    'hunger_gain': energetic_pars['hunger_gain'],
                    'V_bite' : self.brain.feeder.V_bite,
                    'absorption': energetic_pars['absorption'],
                    'substrate_quality': self.model.substrate_quality,
                }
                if self.hunger_as_EEB:
                    self.deb = DEB(base_hunger=self.brain.intermitter.base_EEB, **cc)
                else:
                    self.deb = DEB(**cc)
                self.deb.grow_larva(hours_as_larva=self.model.hours_as_larva, epochs=self.model.epochs)
                if energetic_pars['DEB_dt'] is None :
                    self.deb_step_every=1
                    self.deb.set_steps_per_day(int(24 * 60 * 60 / self.model.dt))
                else :
                    self.deb_step_every = int(energetic_pars['DEB_dt']/ self.model.dt)
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

    def build_brain(self, brain):
        modules = brain['modules']
        if brain['nengo']:
            brain = NengoBrain()
            brain.setup(agent=self, modules=modules, conf=brain)
            brain.build(brain.nengo_manager, olfactor=brain.olfactor)
            brain.sim = Simulator(brain, dt=0.01)
            brain.Nsteps = int(self.model.dt / brain.sim.dt)
        else:
            brain = DefaultBrain(agent=self, modules=modules, conf=brain)
        return brain

    def run_energetics(self, food_detected, feed_success, V_eaten, food_quality):
        if self.deb:
            f = self.deb.f
            if feed_success:
                f += food_quality * self.deb.absorption
            f *= self.f_exp_coef
            self.temp_cum_V_eaten +=V_eaten
            self.temp_mean_f.append(f)
            if self.model.Nticks % self.deb_step_every == 0:
                self.deb.run(f=np.mean(self.temp_mean_f), X_V=self.temp_cum_V_eaten)
                self.temp_cum_V_eaten =0
                self.temp_mean_f=[]

            self.real_length = self.deb.Lw * 10 / 1000
            self.real_mass = self.deb.Ww
            self.V = self.deb.V

            if food_detected is None:
                self.brain.intermitter.EEB *= self.brain.intermitter.EEB_exp_coef
            else:
                if self.hunger_as_EEB:
                    self.brain.intermitter.EEB = self.deb.hunger
                else:
                    self.brain.intermitter.EEB = self.brain.intermitter.base_EEB
                if self.brain.intermitter.feeder_reocurrence_as_EEB :
                    self.brain.intermitter.feeder_reoccurence_rate=self.brain.intermitter.EEB
            self.adjust_body_vertices()

        else:
            if feed_success:
                self.real_mass += V_eaten * self.food_to_biomass_ratio
                self.adjust_shape_to_mass()
                self.adjust_body_vertices()
                self.V = self.real_length ** 3
        self.max_V_bite = self.get_max_V_bite()

    def update_behavior_dict(self):
        behavior_dict = self.null_behavior_dict.copy()
        if self.brain.modules['crawler'] and self.brain.crawler.active():
            behavior_dict['stride_id'] = True
            if self.brain.crawler.complete_iteration:
                behavior_dict['stride_stop'] = True
        if self.brain.modules['intermitter'] and self.brain.intermitter.active():
            behavior_dict['pause_id'] = True
        if self.brain.modules['feeder'] and self.brain.feeder.active():
            behavior_dict['feed_id'] = True
        orvel = self.get_head().get_angularvelocity()
        if orvel > 0:
            behavior_dict['Lturn_id'] = True
        elif orvel < 0:
            behavior_dict['Rturn_id'] = True
        color = self.update_color(self.default_color, behavior_dict)
        self.set_color([color for seg in self.segs])

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
