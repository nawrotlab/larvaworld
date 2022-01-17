import numpy as np
import matplotlib.pyplot as plt

from lib.conf.base.dtypes import null_dict
from lib.model.modules.crawler import Crawler
from lib.model.modules.turner import Turner


raise

# Turner modules

T = null_dict('turner')

Tno_noise = null_dict('turner', activation_noise=0.0, noise=0.0)

Tsin = null_dict('turner',
                 mode='sinusoidal',
                 initial_amp=15.0,
                 amp_range=[0.0, 50.0],
                 initial_freq=0.3,
                 freq_range=[0.1, 1.0],
                 noise=0.15,
                 activation_noise=0.5,
                 )
Tsin_no_noise = null_dict('turner',
                          mode='sinusoidal',
                          initial_amp=15.0,
                          amp_range=[0.0, 50.0],
                          initial_freq=0.3,
                          freq_range=[0.1, 1.0],
                          noise=0.0,
                          activation_noise=0.0,
                          )

T_dict = {
    'neural': T,
    'neural*': Tno_noise,
    'sinusoidal': Tsin,
    'sinusoidal*': Tsin_no_noise,
}

C=null_dict('crawler')
C_no_noise=null_dict('crawler', noise=0.0)
Ccon = null_dict('crawler', waveform='constant', initial_amp=0.0012)

Ccon_no_noise = null_dict('crawler', waveform='constant', initial_amp=0.0012, noise=0.0)

C_dict = {
    'default': C,
    'default*': C_no_noise,
    'constant': Ccon,
    'constant*': Ccon_no_noise,
}


nT = len(T_dict)
nC = len(C_dict)
T_labels=list(T_dict.keys())
C_labels=list(C_dict.keys())

length=null_dict('body')['initial_length']
dt = 0.1
dur = 1000
N = int(dur / dt)
x = np.arange(0, dur, dt)

fig, axs = plt.subplots(nT, 1, sharex=True, sharey=True)
axs = axs.ravel()
T_ys=[]
for ii, (label, conf) in enumerate(T_dict.items()):
    t = Turner(dt=dt, **conf)
    y = []
    for i in range(N):
        a = t.step()
        y.append(a)
    T_ys.append(y)
    axs[ii].plot(x, y, label=label)
    axs[ii].set_ylabel('Turner output (-)')

axs[-1].set_xlabel('time (sec)')

# plt.legend()
plt.show()

fig, axs = plt.subplots(nC, 1, sharex=True, sharey=True)
axs = axs.ravel()
C_ys=[]
for ii, (label, conf) in enumerate(C_dict.items()):
    c = Crawler(dt=dt, **conf)
    y = []
    for i in range(N):
        a = c.step(length=length)
        y.append(a)
    C_ys.append(y)
    axs[ii].plot(x, y, label=label)
    axs[ii].set_ylabel('Crawler output (-)')

axs[-1].set_xlabel('time (sec)')

# plt.legend()
plt.show()

from scipy.fft import fft, fftfreq
fig, axs = plt.subplots(2, 1, sharex=True)
axs=axs.ravel()
xf = fftfreq(N, dt)[:N//2]
ax=axs[0]
for jj,y in enumerate(T_ys) :
    yf = fft(y)
    yf=2.0/N * np.abs(yf[0:N//2])
    #yf = 1000 * yf / np.sum(yf)
    #yf=moving_average(yf, n=21)
    ax.plot(xf, yf, label=T_labels[jj])
ax.set_ylabel('Magnitude')
ax.legend()
ax=axs[1]
for jj,y in enumerate(C_ys) :
    yf = fft(y)
    yf=2.0/N * np.abs(yf[0:N//2])
    #yf = 1000 * yf / np.sum(yf)
    #yf=moving_average(yf, n=21)
    ax.plot(xf, yf, label=C_labels[jj])
ax.set_ylabel('Magnitude')
ax.set_xlim([0,2])
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.legend()
plt.show()

