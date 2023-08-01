
import param

from larvaworld.lib.param import PositiveNumber, PositiveInteger



class BaseController(param.Parameterized):
    lin_vel_coef = PositiveNumber(1.0, doc='Coefficient for translational velocity')
    ang_vel_coef = PositiveNumber(1.0, doc='Coefficient for angular velocity')
    lin_force_coef = PositiveNumber(1.0, doc='Coefficient for force')
    torque_coef = PositiveNumber(0.5, doc='Coefficient for torque')
    body_spring_k = PositiveNumber(1.0, doc='Torsional spring constant for body bending')
    bend_correction_coef = PositiveNumber(1.0, doc='Bend correction coefficient')
    lin_damping = PositiveNumber(1.0, doc='Translational damping coefficient')
    ang_damping = PositiveNumber(1.0, doc='Angular damping coefficient')
    lin_mode = param.Selector(objects=['velocity', 'force', 'impulse'], doc='Mode of translational motion generation')
    ang_mode = param.Selector(objects=['torque','velocity'], doc='Mode of angular motion generation')


    def compute_ang_vel(self, amp, ang_vel, dt, bend):
        torque = amp * self.torque_coef
        return ang_vel + (-self.ang_damping * ang_vel - self.body_spring_k * bend + torque) * dt

    def compute_delta_rear_angle(self, bend, dst, length):
        k0 = 2 * dst * self.bend_correction_coef / length
        if 0 <= k0 < 1:
            return bend * k0
        elif 1 <= k0:
            return bend
        elif k0 < 0:
            return 0

