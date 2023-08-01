import numpy as np

from larvaworld.lib.aux import nam


def fft_max(a, dt, fr_range=(0.0, +np.inf), return_amps=False):
    """
    Power-spectrum of signal.

    Compute the power spectrum of a signal and its dominant frequency within some range.

    Parameters
    ----------
    a : array
        1D np.array : signal timeseries
    dt : float
        Timestep of the timeseries
    fr_range : Tuple[float,float]
        Frequency range allowed. Default is (0.0, +np.inf)
    return_amps: bool
        whether to return the whole array of frequency powers

    Returns
    -------
    yf : array
        Array of computed frequency powers.
    fr : float
        Dominant frequency within range.

    """
    from scipy.fft import fft
    a = np.nan_to_num(a)
    N = len(a)
    xf = np.fft.fftfreq(N, dt)[:N // 2]
    yf = fft(a, norm="ortho")
    yf = 2.0 / N * np.abs(yf[:N // 2])
    yf = 1000 * yf / np.sum(yf)

    fr_min,fr_max=fr_range
    xf_trunc = xf[(xf >= fr_min) & (xf <= fr_max)]
    yf_trunc = yf[(xf >= fr_min) & (xf <= fr_max)]
    fr = xf_trunc[np.argmax(yf_trunc)]
    if return_amps:
        return fr, yf
    else:
        return fr


def fft_freqs(s, e, c):
    v, fov = nam.vel(['', nam.orient('front')])
    fv, fsv, ffov = nam.freq([v, nam.scal(v), fov])

    try:
        e[fv] = s[v].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(1.0, 2.5))
        e[fsv] = e[fv]
    except:
        pass
    try:
        e[ffov] = s[fov].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(0.1, 0.8))
        e['turner_input_constant'] = (e[ffov] / 0.024) + 5
    except:
        pass

def get_freq(d, par, fr_range=(0.0, +np.inf)):
    s, e, c = d.step_data, d.endpoint_data, d.config
    e[nam.freq(par)] = s[par].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=fr_range)


