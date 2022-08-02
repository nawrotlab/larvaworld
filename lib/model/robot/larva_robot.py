
from lib.aux.sim_aux import Collision


from lib.aux.color_util import Color
from lib.model.body.controller import BodySim
from lib.model.modules.brain import DefaultBrain


class LarvaRobot(BodySim):

    def __init__(self, unique_id, model, larva_pars, orientation=0, pos=(0, 0), **kwargs):

        super().__init__(model=model, pos=pos, orientation=orientation, default_color=Color.random_color(127, 127, 127),
                         physics=larva_pars.physics, **larva_pars.body, **larva_pars.Box2D_params)

        self.Nticks = 0
        self.finalized = False
        self.collision_with_object = False
        self.unique_id = unique_id

        self.pos = self.initial_pos
        # print(larva_pars.keys())
        # print()
        # print(larva_pars)
        # raise
        self.brain = DefaultBrain(dt=self.model.dt, conf=larva_pars.brain, agent=self)

        self.x, self.y = self.model.scene._transform(self.pos)


    def draw(self, scene):
        for seg in self.segs:
            for vs in seg.vertices:
                scene.draw_polygon(vs, filled=True, color=seg.color)

    @property
    def direction(self):
        return self.head.get_orientation()

    def complete_step(self):
        # if not self.model.engine.exclusion_mode :
        #     self.model.engine.step_df[self.Nticks, self.unique_id, :]=[self.body_bend,self.head.get_angularvelocity(),
        #                                                            self.rear_orientation_change/self.model.dt,
        #                                                            self.head.get_linearvelocity(), self.pos[0],self.pos[1]]

        self.x, self.y = self.model.scene._transform(self.pos)
        self.Nticks += 1

    @ property
    def collect(self):
        return [self.body_bend,self.head.get_angularvelocity(),
                                                                   self.rear_orientation_change/self.model.dt,
                                                                   self.head.get_linearvelocity(), self.pos[0],self.pos[1]]

    def sense_and_act(self):
        self.step()
        # print()
        # print(self.unique_id, self.Nticks)

    def draw_label(self, screen):
        import pygame
        if pygame.font and self.unique_id is not None:
            font = pygame.font.Font(None, 24)
            text = font.render(str(self.unique_id), 1, Color.YELLOW, Color.DARK_GRAY)
            text_pos = pygame.Rect(self.x + (self.sim_length / 2), self.y + (self.sim_length / 2), 50, 50)
            screen.blit(text, text_pos)





class ObstacleLarvaRobot(LarvaRobot):
    def __init__(self, larva_pars, **kwargs):
        self.sensorimotor_kws = larva_pars.sensorimotor
        larva_pars.pop('sensorimotor', None)
        super().__init__(larva_pars=larva_pars, **kwargs)
        self.left_motor_controller = None
        self.right_motor_controller = None
        self.build_sensorimotor(**self.sensorimotor_kws)

    def build_sensorimotor(self, sensor_delta_direction, sensor_saturation_value, obstacle_sensor_error,
                           sensor_max_distance,
                           motor_ctrl_coefficient, motor_ctrl_min_actuator_value):
        from lib.model.robot.motor_controller import MotorController, Actuator
        from lib.model.robot.sensor import ProximitySensor
        S_kws = {
            'saturation_value': sensor_saturation_value,
            'error': obstacle_sensor_error,
            'max_distance': int(self.model.scene._scale[0, 0] * sensor_max_distance * self.real_length),
            # 'max_distance': sensor_max_distance * self.real_length,
            'scene': self.model.scene,
            'collision_distance': int(self.model.scene._scale[0, 0] * self.real_length / 5),
            # 'collision_distance': 0.1 * self.real_length,
        }

        M_kws = {
            'coefficient': motor_ctrl_coefficient,
            'min_actuator_value': motor_ctrl_min_actuator_value,
        }

        Lsens = ProximitySensor(self, delta_direction=sensor_delta_direction, **S_kws)
        Rsens = ProximitySensor(self, delta_direction=-sensor_delta_direction, **S_kws)
        Lact = Actuator()
        Ract = Actuator()
        Lmot = MotorController(sensor=Lsens, actuator=Lact, **M_kws)
        Rmot = MotorController(sensor=Rsens, actuator=Ract, **M_kws)

        self.set_left_motor_controller(Lmot)
        self.set_right_motor_controller(Rmot)

    def sense_and_act(self):
        if not self.collision_with_object:
            pos = self.model.scene._transform(self.olfactor_pos)
            try:

                self.left_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                self.right_motor_controller.sense_and_act(pos=pos, direction=self.direction)
                Ltorque = self.left_motor_controller.get_actuator_value()
                Rtorque = self.right_motor_controller.get_actuator_value()
                dRL = Rtorque - Ltorque
                if dRL > 0:
                    self.brain.locomotor.turner.neural_oscillator.E_r += dRL * self.model.dt
                else:
                    self.brain.locomotor.turner.neural_oscillator.E_l -= dRL * self.model.dt
                # ang=self.head.get_angularvelocity()
                # self.head.set_ang_vel(ang-dRL*self.model.dt)
                # if dRL!=0 :
                #
                #     print(dRL*self.model.dt, ang)
                # self.brain.locomotor.turner.neural_oscillator.E_r += Rtorque * self.model.dt
                # self.brain.locomotor.turner.neural_oscillator.E_l += Ltorque * self.model.dt
                # if dRL>0 :
                #     self.brain.locomotor.turner.neural_oscillator.E_r += np.abs(dRL)
                # else :
                #     self.brain.locomotor.turner.neural_oscillator.E_l += np.abs(dRL)
                self.step()
            except Collision:
                self.collision_with_object = True
                self.brain.locomotor.intermitter.interrupt_locomotion()
        else:
            pass

    def set_left_motor_controller(self, left_motor_controller):
        self.left_motor_controller = left_motor_controller

    def set_right_motor_controller(self, right_motor_controller):
        self.right_motor_controller = right_motor_controller

    def draw(self, scene):
        # pos = self.olfactor_pos
        pos = self.model.scene._transform(self.olfactor_pos)
        # draw the sensor lines

        # in scene_loader a robot doesn't have sensors
        if self.left_motor_controller is not None:
            self.left_motor_controller.sensor.draw(pos=pos, direction=self.direction)
            self.right_motor_controller.sensor.draw(pos=pos, direction=self.direction)

        # call super method to draw the robot
        super().draw(scene)