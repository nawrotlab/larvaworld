import numpy as np
import matplotlib.pyplot as plt

from lib.conf.base.opt_par import getPar
from lib.conf.stored.conf import expandConf, kConfDict, loadRef
from lib.model.modules.brain import DefaultBrain
from pymdp.maths import spm_log_single as log_stable
def KL_divergence(q,p):
    return np.sum(q * (log_stable(q) - log_stable(p)))


length=0.004
dt = 0.1
dur = 100
N = int(dur / dt)
x = np.arange(0, dur, dt)

body_spring_k=0.02
def compute_ang_vel(body_bend,torque=0.0, v=0.0, z=2.5):
    return v + (-z * v - body_spring_k * body_bend + torque) * dt


def restore_bend_2seg(bend, d, l, correction_coef=1.4):
    k0 = 0.5 * l / correction_coef
    if 0 <= d < k0:
        return bend * (1 - d / k0)
    elif k0 <= d:
        return 0
    elif d < 0:
        return bend

y1s, y2s, y3s = [], [], []
dds=[]

d=loadRef('None.10_controls')
sv, fov, b = getPar(['sv', 'fov', 'b'], to_return='d')[0]
y10=d.get_par(sv, 'step')
y20=d.get_par(fov, 'step')
y30=d.get_par(b, 'step')


fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(8,16))
axs = axs.ravel()
ms=kConfDict('Brain')
for i,m in enumerate(ms) :
    print(f'================{m}==================')
    bend=0
    ang=0
    ds=[]
    b = expandConf(m, 'Brain')
    B=DefaultBrain(dt=dt, modules=b.modules, conf=b)
    y1, y2, y3 = [], [], []
    for i in range(N):
        crawler_out, turner_out, feed_motion = B.run(pos=None, length=1)
        d=crawler_out*dt*length
        ds.append(d)
        if i in [1,3] :
            ang=compute_ang_vel(body_bend=bend, torque=turner_out, v=ang, z=2.5)
        else :
            ang=turner_out
        bend+=ang*dt

        y1.append(crawler_out)
        y2.append(ang)
        y3.append(bend)
        bend = restore_bend_2seg(bend, d, length)
    dds.append(ds)
    y1s.append(np.array(y1))
    y2s.append(np.array(y2))
    y3s.append(np.array(y3))
    axs[0].plot(x, y1, label=m)
    axs[1].plot(x, y2, label=m)
    axs[2].plot(x, y3, label=m)
    axs[0].set_ylabel('forward speed (m/s)')
    axs[1].set_ylabel('angular speed (rad/s)')
    axs[2].set_ylabel('Bend (rad)')

axs[2].set_xlabel('time (sec)')

axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.show()

fig, ax = plt.subplots(1, 1)
for ds, m in zip(dds,ms) :
    ax.plot(x,np.cumsum(ds), label=m)
ax.legend()
ax.set_xlabel('time (sec)')
ax.set_ylabel('Pathlength (m)')
plt.show()

# y1s.append(y10)
# y2s.append(y20)
# ms.append('Exp')
for id, sdf in y10.groupby(level='AgentID'):
    y1=sdf.dropna().values
    y1s.append(y1)
for id, sdf in y20.groupby(level='AgentID'):
    y2 = sdf.dropna().values
    y2s.append(y2)
for id, sdf in y30.groupby(level='AgentID'):
    y3 = sdf.dropna().values
    y3s.append(y3)
    ms.append(None)
    # values, bin_edges = np.histogram(sdf.loc[count, 'vel_x'])

fig, axs = plt.subplots(3, 1, sharex=False, sharey=True, figsize=(8,15))
axs = axs.ravel()

w10 = np.ones_like(y10) / len(y10)
w20 = np.ones_like(y20) / len(y20)
w30 = np.ones_like(y30) / len(y30)
q10, bins10,_ =axs[0].hist(y10, label='Exp', weights=w10,histtype='step', bins=100, color='black')
q20, bins20,_ =axs[1].hist(y20, label='Exp', weights=w20,histtype='step', bins=100, color='black')
q30, bins30,_ =axs[2].hist(y30, label='Exp', weights=w30,histtype='step', bins=100, color='black')


for m, y1,y2, y3 in zip(ms, y1s,y2s, y2s) :
    color='grey' if m is None else None
    alpha=0 if m is None else 1
    # y2=np.abs(y2)
    w1 = np.ones_like(y1) / len(y1)
    w2 = np.ones_like(y2) / len(y2)
    w3 = np.ones_like(y3) / len(y3)
    q1, bins1,_ =axs[0].hist(y1, label=m, weights=w1,histtype='step', bins=100, color=color, alpha=alpha)
    q2, bins2,_ =axs[1].hist(y2, label=m, weights=w2,histtype='step', bins=100, color=color, alpha=alpha)
    q3, bins3,_ =axs[2].hist(y3, label=m, weights=w3,histtype='step', bins=100, color=color, alpha=alpha)
    print(m, KL_divergence(q1,q10), KL_divergence(q2,q20), KL_divergence(q3,q30))
    # print(m, KL_divergence(q2,q20))
axs[0].set_xlim((0,1))
axs[1].set_xlim((-500,500))
axs[2].set_xlim((-180,180))
axs[0].set_ylim((0,0.1))
axs[0].set_ylabel('forward speed (m/s)')
axs[1].set_ylabel('angular speed (rad/s)')
axs[2].set_ylabel('Bend (rad)')
axs[0].set_ylabel('probability')
axs[1].set_ylabel('probability')
axs[2].set_ylabel('probability')
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.subplots_adjust()
plt.show()
