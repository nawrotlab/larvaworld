import param

class PhysicsController(param.Parameterized):
    lin_vel_coef = param.Number(default=1.0, doc='Coefficient for translational velocity')
    ang_vel_coef = param.Number(default=1.0, doc='Coefficient for angular velocity')
    lin_force_coef = param.Number(default=1.0, doc='Coefficient for force')
    torque_coef = param.Number(default=0.5, doc='Coefficient for torque')
    body_spring_k = param.Number(default=1.0, doc='Torsional spring constant for body bending')
    bend_correction_coef = param.Number(default=1.0, doc='Bend correction coefficient')
    lin_damping = param.Number(default=1.0, doc='Translational damping coefficient')
    ang_damping = param.Number(default=1.0, doc='Angular damping coefficient')
    # lin_mode = param.Selector(default='velocity', objects=['velocity', 'force'], doc='Mode of translational motion generation')
    ang_mode = param.Selector(default='torque', objects=['velocity', 'torque'], doc='Mode of angular motion generation')

    def compute_ang_vel(self, torque, v, b, dt):
        return v + (-self.ang_damping * v - self.body_spring_k * b + torque) * dt

    def get_vels(self, lin, ang, prev_ang_vel, bend, dt, ang_suppression):
        lin_vel = lin * self.lin_vel_coef
        if self.ang_mode == 'torque':
            ang_vel = self.compute_ang_vel(torque=ang * self.torque_coef,v=prev_ang_vel, b=bend,dt=dt)
        elif self.ang_mode == 'velocity':
            ang_vel = ang * self.ang_vel_coef
        lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
        ang_vel *= ang_suppression
        return lin_vel, ang_vel

    def assess_collisions(self, lin_vel, ang_vel):
        return lin_vel, ang_vel

# class PhysicsController2:
#     def __init__(self, lin_vel_coef=1.0, ang_vel_coef=1.0, lin_force_coef=1.0, torque_coef=0.5,
#                  lin_mode='velocity', ang_mode='torque', body_spring_k=1.0, bend_correction_coef=1.0,
#                  lin_damping=1.0, ang_damping=1.0):
#         self.lin_damping = lin_damping
#         self.ang_damping = ang_damping
#         self.body_spring_k = body_spring_k
#         self.bend_correction_coef = bend_correction_coef
#         self.lin_mode = lin_mode
#         self.ang_mode = ang_mode
#         self.lin_vel_coef = lin_vel_coef
#         self.ang_vel_coef = ang_vel_coef
#         self.lin_force_coef = lin_force_coef
#         self.torque_coef = torque_coef
#
#     def compute_ang_vel(self, k, c, torque, v, b, dt, I=1):
#         dtI = dt / I
#         return v + (-c * v - k * b + torque) * dtI
#
#     def get_vels(self, lin, ang, prev_ang_vel, bend, dt, ang_suppression):
#         if self.lin_mode == 'velocity':
#             if lin != 0:
#                 lin_vel = lin * self.lin_vel_coef
#             else:
#                 lin_vel = 0  # prev_lin_vel*(1-self.lin_damping*dt)
#         else:
#             raise ValueError(f'Linear mode {self.lin_mode} not implemented for non-physics simulation')
#         if self.ang_mode == 'torque':
#             torque = ang * self.torque_coef
#             ang_vel = self.compute_ang_vel(torque=torque,
#                                            v=prev_ang_vel, b=bend,
#                                            c=self.ang_damping, k=self.body_spring_k, dt=dt)
#         elif self.ang_mode == 'velocity':
#             ang_vel = ang * self.ang_vel_coef
#         else :
#             raise ValueError (f'ang_mode {self.ang_mode} not implemented')
#         lin_vel, ang_vel = self.assess_collisions(lin_vel, ang_vel)
#         ang_vel *= ang_suppression
#         return lin_vel, ang_vel
#
#     def assess_collisions(self, lin_vel, ang_vel):
#         return lin_vel, ang_vel