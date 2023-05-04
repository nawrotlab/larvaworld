import random
from copy import deepcopy
import numpy as np
import param

from larvaworld.lib import reg, aux
from larvaworld.lib.model.agents import LarvaworldAgent,Source


class Larva(LarvaworldAgent):
    def __init__(self, model,unique_id=None, orientation=None, **kwargs):
        if unique_id is None:
            unique_id = model.next_id(type='Larva')
        super().__init__(unique_id=unique_id, model=model,**kwargs)
        self.initial_pos = self.pos
        self.trajectory = [self.initial_pos]
        if orientation is None:
            orientation = random.uniform(0, 2 * np.pi)
        self.initial_orientation = orientation
        self.orientation = self.initial_orientation
        self.behavior_pars = ['stride_stop', 'stride_id', 'pause_id', 'feed_id', 'Lturn_id', 'Rturn_id']
        self.null_behavior_dict = dict(zip(self.behavior_pars, [False] * len(self.behavior_pars)))
        self.carried_objects = []
        self.cum_dur = 0

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


class LarvaMotile(Larva):
    def __init__(self, brain, energetics, life_history, **kwargs):
        super().__init__(**kwargs)
        self.brain = self.build_brain(brain)
        self.build_energetics(energetics, life_history=life_history)
        self.food_detected, self.feeder_motion = None, False
        self.cum_food_detected, self.amount_eaten = 0, 0

    def build_brain(self, conf):
        if conf.nengo:
            from larvaworld.lib.model.modules.nengobrain import NengoBrain
            return NengoBrain(agent=self, conf=conf, dt=self.model.dt)
        else:
            from larvaworld.lib.model.modules.brain import DefaultBrain
            return DefaultBrain(agent=self, conf=conf, dt=self.model.dt)




    def feed(self, source, motion):

        def get_max_V_bite():
            return self.brain.locomotor.feeder.V_bite * self.V*1000  # ** (2 / 3)

        if motion:
            a_max = get_max_V_bite()
            if source is not None:
                grid = self.model.food_grid
                if grid:
                    V = -grid.add_cell_value(source, -a_max)
                else:
                    V = source.subtract_amount(a_max)

                return V
            else:
                return 0
        else:
            return 0



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
            try:
                self.deb.set_intermitter(self.brain.locomotor.intermitter)
            except:
                pass
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

    def get_feed_success(self, t):
        if self.feeder_motion :
            if self.on_food :
                return 1
            else:
                return -1
        else:
            return 0


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
        return self.amount_eaten / self.real_mass

    def resolve_carrying(self, food):
        if food is None or not isinstance(food, Source):
            return
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
                carrier_group_odor_id = self.odor.id
                opponent_group = aux.LvsRtoggle(carrier_group)
                opponent_group_odor_id = f'{opponent_group}_odor'
                for f in self.model.agents:
                    if f.group == carrier_group:
                        f.brain.olfactor.set_gain(f.gain_for_base_odor, opponent_group_odor_id)
                    else:
                        f.brain.olfactor.set_gain(0.0, carrier_group_odor_id)
                self.brain.olfactor.set_gain(-self.gain_for_base_odor, opponent_group_odor_id)

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

    def sense(self):
        pass

    # @profile
    def step(self):
        self.cum_dur += self.model.dt
        self.sense()
        pos = self.olfactor_pos

        if self.model.space.accessible_sources :
            self.food_detected = self.model.space.accessible_sources[self]
            # if self.food_detected is not None :
            #     print(self.unique_id, self.food_detected.unique_id)
        elif self.brain.locomotor.feeder  or self.brain.toucher:
            self.food_detected = aux.sense_food(pos, sources=self.model.sources, grid=self.model.food_grid,
                                        radius=self.radius)
        self.resolve_carrying(self.food_detected)

        lin, ang, self.feeder_motion = self.brain.step(pos, length=self.real_length, on_food=self.on_food)
        V = self.feed(self.food_detected, self.feeder_motion)
        self.amount_eaten += V * 1000
        self.cum_food_detected += int(self.on_food)
        self.run_energetics(V)


        for o in self.carried_objects:
            o.pos = self.pos

        self.prepare_motion(lin=lin, ang=ang)

    def prepare_motion(self, lin, ang):
        pass
        # Overriden by subclasses

    def complete_step(self):
        pass




