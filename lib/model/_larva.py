import random
import numpy as np
from copy import deepcopy
from nengo import Simulator

from lib.aux import functions as fun
from lib.model import Larva, BodyReplay, BodySim, DEB, DefaultBrain, NengoBrain


class LarvaReplay(Larva, BodyReplay):
    def __init__(self, unique_id, model, length=5, data=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, radius=length / 2, **kwargs)

        self.chunk_ids = None
        self.trajectory = []
        self.color = deepcopy(self.default_color)
        self.sim_length = length

        N = len(data.index.unique().values)
        Nmid = self.model.Npoints
        Ncon = self.model.Ncontour
        Nangles = self.model.Nangles
        Nors = self.model.Nors
        Nsegs = self.model.draw_Nsegs
        mid_pars = self.model.mid_pars
        con_pars = self.model.con_pars
        cen_pars = self.model.cen_pars
        pos_pars = self.model.pos_pars
        ang_pars = self.model.angle_pars
        or_pars = self.model.or_pars
        mid_dim = [N, Nmid, 2]
        con_dim = [N, Ncon, 2]

        self.mid_ar = data[mid_pars].values.reshape(mid_dim) if Nmid > 0 else np.ones(mid_dim) * np.nan
        self.con_ar = data[con_pars].values.reshape(con_dim) if Ncon > 0 else np.ones(con_dim) * np.nan
        self.cen_ar = data[cen_pars].values if len(cen_pars) == 2 else np.ones([N, 2]) * np.nan
        self.pos_ar = data[pos_pars].values if len(pos_pars) == 2 else np.ones([N, 2]) * np.nan
        self.ang_ar = data[ang_pars].values if Nangles > 0 else np.ones([N, Nangles]) * np.nan
        self.or_ar = data[or_pars].values if Nors > 0 else np.ones([N, Nors]) * np.nan
        self.bend_ar = data['bend'].values if 'bend' in data.columns else np.ones(N) * np.nan
        self.front_or_ar = data['front_orientation'].values if 'front_orientation' in data.columns else np.ones(
            N) * np.nan

        vp_beh = [p for p in self.behavior_pars if p in self.model.pars]
        self.beh_ar = np.zeros([N, len(self.behavior_pars)], dtype=bool)
        for i, p in enumerate(self.behavior_pars):
            if p in vp_beh:
                self.beh_ar[:, i] = np.array([not v for v in np.isnan(data[p].values).tolist()])
        self.pos = self.pos_ar[0]
        if Nsegs is not None:
            # FIXME Here the sim_length is not divided by 1000 because all xy coords are in mm
            BodyReplay.__init__(self, model, pos=self.pos, orientation=self.or_ar[0][0],
                                initial_length=self.sim_length, length_std=0, Nsegs=Nsegs, interval=0)
        self.data = data

    def read_step(self, i):
        self.midline = self.mid_ar[i].tolist()
        self.vertices = self.con_ar[i]
        self.cen_pos = self.cen_ar[i]
        self.pos = self.pos_ar[i]
        self.trajectory = self.pos_ar[:i, :].tolist()
        self.angles = self.ang_ar[i]
        self.orients = self.or_ar[i]
        self.beh_dict = dict(zip(self.behavior_pars, self.beh_ar[i, :].tolist()))

        self.front_orientation = self.front_or_ar[i]
        self.bend = self.bend_ar[i]

        for p in ['front_orientation_vel']:
            setattr(self, p, self.data[p].values[i] if p in self.data.columns else np.nan)
        # self.front_orientation_vel = self.front_or_vel_ar[i]

    def step(self):
        step = self.model.active_larva_schedule.steps
        self.read_step(step)
        if not np.isnan(self.pos).any():
            self.model.space.move_agent(self, self.pos)
        if self.model.color_behavior:
            self.color = self.update_color(self.default_color, self.beh_dict)
        else:
            self.color = self.default_color
        if self.model.draw_Nsegs is not None:
            segs = self.segs
            if len(self.midline) == len(segs) + 1:
                for i, seg in enumerate(segs):
                    pos = [np.nanmean([self.midline[i][j], self.midline[i + 1][j]]) for j in [0, 1]]
                    o = np.deg2rad(self.orients[i])
                    seg.set_position(pos)
                    seg.set_orientation(o)
                    seg.update_vertices(pos, o)
            # elif len(segs) == 2 and len(self.orients) == 1 and len(self.angles) == 1:
            elif len(segs) == 2:
                l1, l2 = [self.sim_length * r for r in self.seg_ratio]
                x, y = self.pos
                h_or = np.deg2rad(self.front_orientation)
                b_or = np.deg2rad(self.front_orientation - self.bend)
                p_head = np.array(fun.rotate_around_point(origin=[x, y], point=[l1 + x, y], radians=-h_or))
                p_tail = np.array(fun.rotate_around_point(origin=[x, y], point=[l2 + x, y], radians=np.pi - b_or))
                pos1 = [np.nanmean([p_head[j], [x, y][j]]) for j in [0, 1]]
                pos2 = [np.nanmean([p_tail[j], [x, y][j]]) for j in [0, 1]]
                segs[0].set_position(pos1)
                segs[0].set_orientation(h_or)
                segs[0].update_vertices(pos1, h_or)
                segs[1].set_position(pos2)
                segs[1].set_orientation(b_or)
                segs[1].update_vertices(pos2, b_or)
                self.midline = np.array([p_head, self.pos, p_tail])

    def draw(self, viewer):
        if self.model.draw_contour:
            if self.model.draw_Nsegs is not None:
                for seg in self.segs:
                    seg.set_color(self.color)
                    seg.draw(viewer)
            elif len(self.vertices) > 0:
                viewer.draw_polygon(self.vertices, filled=True, color=self.color)
        # return
        # viewer.draw_polygon(self.get_shape().boundary.coords, self.color, filled=True, width=self.radius / 10)
        # return
        if self.model.draw_centroid:
            if not np.isnan(self.cen_pos).any():
                pos = self.cen_pos
            elif not np.isnan(self.pos).any():
                pos = self.pos
            else:
                pos = None
            if pos is not None:
                viewer.draw_circle(radius=self.radius / 2, position=pos, filled=True, color=self.color,
                                   width=self.radius / 3)
        if self.model.draw_midline and self.model.Npoints > 1:
            if not np.isnan(self.midline[0]).any():
                viewer.draw_polyline(self.midline, color=(0, 0, 255), closed=False, width=.07)
                for i, seg_pos in enumerate(self.midline):
                    c = 255 * i / (len(self.midline) - 1)
                    color = (c, 255 - c, 0)
                    viewer.draw_circle(radius=.07, position=seg_pos, filled=True, color=color, width=.01)
        if self.selected:
            if len(self.vertices) > 0 and not np.isnan(self.vertices).any():
                viewer.draw_polygon(self.vertices, filled=False, color=self.model.selection_color,
                                    width=self.radius / 5)
            elif not np.isnan(self.pos).any():
                viewer.draw_circle(radius=self.radius, position=self.pos,
                                   filled=False, color=self.model.selection_color, width=self.radius / 3)

    def set_color(self, color):
        self.color = color


class LarvaSim(BodySim, Larva):
    def __init__(self, unique_id, model, pos, orientation, larva_pars, group='', default_color=None, **kwargs):
        Larva.__init__(self, unique_id=unique_id, model=model, pos=pos,
                       **larva_pars['odor'], group=group, default_color=default_color)
        # print(list(larva_pars['brain'].keys()))
        # FIXME : Get rid of this
        try:
            larva_pars['brain']['olfactor_params']['odor_dict'] = self.update_odor_dicts(
                larva_pars['brain']['olfactor_params']['odor_dict'])
        except:
            pass
        self.brain = self.build_brain(larva_pars['brain'])
        self.build_energetics(larva_pars['energetics'])
        BodySim.__init__(self, model=model, orientation=orientation, **larva_pars['physics'],
                         **larva_pars['body'], **kwargs)
        self.build_gut(self.V)

        self.reset_feeder()
        self.radius = self.sim_length / 2

        self.food_detected, self.feeder_motion, self.current_amount_eaten, self.feed_success = None, False, 0, False


    def update_odor_dicts(self, odor_dict):  #

        temp = {'mean': 0.0, 'std': 0.0}
        food_odor_ids = fun.unique_list(
            [s.get_odor_id() for s in self.model.get_food() + [self] if s.get_odor_id() is not None])
        if odor_dict is None:
            odor_dict = {}
            # odor_dict = {odor_id: temp for odor_id in food_odor_ids}
        for odor_id in food_odor_ids:
            if odor_id not in list(odor_dict.keys()):
                odor_dict[odor_id] = temp
        return odor_dict

    def compute_next_action(self):

        self.cum_dur += self.model.dt

        self.food_detected, food_quality = self.detect_food()

        lin, ang, self.feeder_motion = self.brain.run()
        self.set_ang_activity(ang)
        self.set_lin_activity(lin)
        self.current_amount_eaten, self.feed_success = self.feed()

        if self.energetics:
            self.run_energetics(self.food_detected, self.feed_success, self.current_amount_eaten, food_quality)

        # Paint the body to visualize effector state
        if self.model.color_behavior:
            self.update_behavior_dict()
        else:
            self.set_color([self.default_color]*self.Nsegs)


    def detect_food(self):
        if self.brain.feeder is not None:
            # radius = self.brain.feeder.feed_radius * self.sim_length
            pos = self.get_olfactor_position()
            grid = self.model.food_grid
            if grid:
                cell = grid.get_grid_cell(pos)
                if grid.get_cell_value(cell) > 0:
                    return cell, grid.quality
                # else:
                #     return False, None, None
            else:
                accessible_food = [a for a in self.model.get_food() if (a.contained(pos) and a.amount > 0)]
                # accessible_food = fun.agents_spatial_query(pos=pos, radius=radius,agent_list=self.model.get_food())
                if accessible_food:
                    food = random.choice(accessible_food)
                    self.resolve_carrying(food)
                    return food, food.quality
                # else:
                #     return False, None, None
        # else:
        return None, None

    def feed(self):
        a_max = self.max_feed_amount
        source=self.food_detected
        if self.feeder_motion and source is not None and self.empty_gut_M >= a_max:
            grid = self.model.food_grid
            amount = -grid.add_cell_value(source, -a_max) if grid else source.subtract_amount(a_max)
            self.feed_success_counter += 1
            self.amount_eaten += amount
            self.update_gut(amount)
            return amount, True
        else:
            return 0, True

    def reset_feeder(self):
        self.feed_success_counter = 0
        self.amount_eaten = 0
        self.feeder_motion = False
        try:
            self.max_feed_amount = self.compute_max_feed_amount()
        except:
            self.max_feed_amount = None
        try:
            self.brain.feeder.reset()
        except:
            pass

    def compute_max_feed_amount(self):
        return self.brain.feeder.max_feed_amount_ratio * self.V ** (2 / 3)

    def build_energetics(self, energetic_pars):
        self.real_length = None
        self.real_mass = None
        self.V = None

        # p_am=260
        if energetic_pars is not None:
            self.energetics = True
            if energetic_pars['deb_on']:
                self.hunger_affects_balance = energetic_pars['hunger_affects_balance']
                self.absorption = energetic_pars['absorption']  # /60
                self.f_decay = energetic_pars['f_decay']
                self.f_exp_coef = np.exp(-self.f_decay * self.model.dt)
                steps_per_day = 24 * 60
                if self.hunger_affects_balance:
                    self.deb = DEB(steps_per_day=steps_per_day, base_hunger=self.brain.intermitter.base_EEB,
                                   hunger_sensitivity=energetic_pars['hunger_sensitivity'])
                else:
                    self.deb = DEB(steps_per_day=steps_per_day,
                                   hunger_sensitivity=energetic_pars['hunger_sensitivity'])
                self.deb.reach_stage('larva')
                self.deb.advance_larva_age(hours_as_larva=self.model.hours_as_larva, f=self.model.deb_base_f,
                                           starvation_hours=self.model.deb_starvation_hours)
                self.deb.steps_per_day = int(24 * 60 * 60 / self.model.dt)
                self.real_length = self.deb.get_real_L()
                self.real_mass = self.deb.get_W()
                self.V = self.deb.get_V()
                # p_am=self.deb.p_am

            else:
                self.deb = None
                self.food_to_biomass_ratio = 0.3
        else:
            self.energetics = False

    def build_gut(self, V):
        self.gut_M_ratio = 0.11
        self.gut_food_M = 0
        self.empty_gut_M = 0
        self.gut_product_M = 0
        self.amount_absorbed = 0
        self.filled_gut_ratio = 0
        # self.digestion_c = 80
        self.absorption = 0.3

        self.gut_M = self.gut_M_ratio * V

    def update_gut(self, amount_eaten):
        self.gut_M = self.gut_M_ratio * self.V
        self.empty_gut_M = self.gut_M - self.gut_food_M - self.gut_product_M
        # FIXME here I need to add the k_x but I don't know it
        # For V1-morphs ingestion rate is proportional to L**3 . Kooijman p.269. Didn't use it.
        # self.digestion_tau_unscaled = 24 * 60 * 60 / self.model.dt * self.gut_M_ratio * 550 / p_am
        # Trying to use μ_Ax=11.5 from p.272
        # self.digestion_tau_unscaled = 24*60*60/self.model.dt*self.gut_M_ratio*11.5/p_am
        # self.digestion_tau = self.digestion_tau_unscaled * self.V ** (1 / 3)

        self.gut_food_M += amount_eaten
        gut_food_dM = 0.001 * self.gut_food_M * self.model.dt
        # gut_food_dM = np.clip(self.gut_M/self.digestion_tau, 0, self.gut_food_M)
        # gut_food_dM = self.gut_food_M/self.digestion_tau
        self.gut_food_M -= gut_food_dM
        # print(gut_food_dM, self.gut_food_M)
        # self.gut_product_M += gut_food_dM
        absorbed_M = self.gut_product_M * self.absorption * self.model.dt
        # absorbed_M = 0
        self.gut_product_M += (gut_food_dM - absorbed_M)
        # self.empty_gut_M = self.gut_M - self.gut_food_M
        self.empty_gut_M = self.gut_M - self.gut_food_M - self.gut_product_M
        self.amount_absorbed += absorbed_M
        self.filled_gut_ratio = 1 - self.empty_gut_M / self.gut_M

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

    def run_energetics(self, food_detected, feed_success, amount_eaten, food_quality):
        if self.deb:
            f = self.deb.get_f()
            if feed_success:
                f += food_quality * self.absorption * amount_eaten / self.max_feed_amount
            f *= self.f_exp_coef
            self.deb.run(f=f)
            self.real_length = self.deb.get_real_L()
            self.real_mass = self.deb.get_W()
            self.V = self.deb.get_V()

            if food_detected is None:
                self.brain.intermitter.EEB *= self.brain.intermitter.EEB_exp_coef
            else:
                # h0=self.deb.base_hunger
                if self.hunger_affects_balance:
                    self.brain.intermitter.EEB=self.deb.base_hunger
                    # dh = self.deb.hunger - h0
                    # if dh > 0:
                    #     self.brain.intermitter.EEB = dh / (1 - h0) * (1 - h0) + h0
                    # else:
                    #     self.brain.intermitter.EEB = dh / h0 * h0 + h0
                else:
                    self.brain.intermitter.EEB = self.brain.intermitter.base_EEB
            self.adjust_body_vertices()
            self.max_feed_amount = self.compute_max_feed_amount()
        else:
            if feed_success:
                self.real_mass += amount_eaten * food_quality * self.food_to_biomass_ratio
                self.adjust_shape_to_mass()
                self.adjust_body_vertices()
                self.max_feed_amount = self.compute_max_feed_amount()
                self.V = self.get_real_length() ** 3

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
    def rear_orientation(self):
        return np.rad2deg(self.get_tail().get_normalized_orientation())

    @property
    def bend(self):
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
                carrier_group_odor_id = self.get_odor_id()
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
