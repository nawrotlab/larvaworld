"""
Intermittency modules alternating between activity and pause.

Larval behaviour is organized into bouts: stretches of crawling or feeding
separated by pauses. These modules generate that alternation, either by
sampling the bout durations from fitted distributions or by running a
branching process, and gate the crawler, turner and feeder accordingly.
"""

from __future__ import annotations
from typing import Any

import os

import numpy as np
import pandas as pd
import param

from ... import reg, util
from ...param import OptionalPositiveNumber, PositiveNumber
from ...util import nam
from .oscillator import Timer

__all__: list[str] = [
    "Intermitter",
    "OfflineIntermitter",
    "BranchIntermitter",
    "FittedIntermitter",
    "get_EEB_poly1d",
    "get_EEB_time_fractions",
]

default_bout_distros = util.AttrDict(
    {
        "turn_dur": {"range": [0.25, 3.25], "name": "exponential", "beta": 4.70099},
        "turn_amp": {
            "range": [0.00042, 305.7906],
            "name": "lognormal",
            "mu": 2.08133,
            "sigma": 1.26487,
        },
        "turn_vel_max": {
            "range": [0.00031, 2752.80403],
            "name": "levy",
            "mu": -1.22434,
            "sigma": 11.93284,
        },
        "run_dur": {
            "range": [0.44, 114.25],
            "name": "lognormal",
            "mu": 1.148,
            "sigma": 1.11329,
        },
        "run_dst": {
            "range": [0.00022, 0.14457],
            "name": "levy",
            "mu": 0.00017,
            "sigma": 0.00125,
        },
        "pause_dur": {"range": [0.12, 15.94], "name": "exponential", "beta": 1.01503},
        "run_count": {
            "range": [1, 142],
            "name": "lognormal",
            "mu": 1.39115,
            "sigma": 1.14667,
        },
    }
)


class Intermitter(Timer):
    """
    Intermitter module for run/pause/feed behavioral state control.

    Manages transitions between locomotor states (running, pausing, feeding)
    using stochastic bout generators. Controls exploitation-exploration
    balance (EEB) for feeding decisions and tracks behavioral statistics.

    Attributes:
        EEB: Exploitation-exploration balance (0=exploit, 1=explore)
        EEB_decay: Exponential decay of EEB when no food detected
        crawl_freq: Default crawling frequency (Hz)
        feed_freq: Default feeding frequency (Hz)
        run_mode: Generation mode ('stridechain' or 'exec')
        feeder_reoccurence_rate: Feed reoccurrence probability
        feed_bouts: Whether feeding epochs are generated
        pause_dist: Temporal distribution params for pause epochs
        stridechain_dist: Stride-number distribution for run epochs
        run_dist: Temporal distribution for run epochs
        cur_state: Current behavioral state ('exec', 'pause', or 'feed')

    Example:
        >>> intermitter = Intermitter(EEB=0.3, crawl_freq=1.42, feed_freq=2.0)
        >>> state = intermitter.step(stride_completed=True, on_food=False)
    """

    EEB = param.Magnitude(
        0.0,
        step=0.01,
        label="exploitation-exploration balance",
        doc="The baseline exploitation-exploration balance. 0 means only exploitation, 1 only exploration.",
    )
    EEB_decay = PositiveNumber(
        1.0,
        softmax=2.0,
        doc="The exponential decay coefficient of the exploitation-exploration balance when no food is detected.",
    )
    crawl_freq = PositiveNumber(
        10 / 7, bounds=(0.5, 3.0), doc="The default crawling frequency."
    )
    feed_freq = PositiveNumber(
        2.0, bounds=(1.0, 3.0), doc="The default feeding frequency."
    )
    run_mode = param.Selector(
        default="stridechain",
        objects=["stridechain", "exec"],
        doc="The generation mode of the crawling epochs.",
    )
    feeder_reoccurence_rate = OptionalPositiveNumber(
        softmax=1.0,
        label="feed reoccurence",
        doc="The default reoccurence rate of the feeding motion.",
    )
    feed_bouts = param.Boolean(False, doc="Whether feeding epochs are generated.")
    pause_dist = param.Dict(
        default=None, doc="The temporal distribution of pause epochs."
    )
    stridechain_dist = param.Dict(
        default=None, doc="The stride-number distribution of run epochs (stridechains)."
    )
    run_dist = param.Dict(default=None, doc="The temporal distribution of run epochs.")

    def __init__(self, **kwargs: Any) -> None:
        """Build the intermitter and its bout-duration generators.

        Args:
            **kwargs: Intermittency parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        if self.feeder_reoccurence_rate is None:
            self.feeder_reoccurence_rate = self.EEB
            self.use_EEB = True
        else:
            self.use_EEB = False

        self.reset()
        self.cur_state = None

        if self.run_mode == "stridechain":
            if self.stridechain_dist is None or self.stridechain_dist.range is None:
                self.stridechain_dist = default_bout_distros.run_count
            self.stridechain_min, self.stridechain_max = self.stridechain_dist.range
            self.stridechain_generator = reg.BoutGenerator(
                **self.stridechain_dist, dt=1
            )
            self.run_generator = None

        elif self.run_mode == "exec":
            if self.run_dist is None or self.run_dist.range is None:
                self.run_dist = default_bout_distros.run_dur
            self.stridechain_min, self.stridechain_max = self.run_dist.range
            self.run_generator = reg.BoutGenerator(**self.run_dist, dt=self.dt)
            self.stridechain_generator = None
        else:
            raise ValueError("None of stidechain or exec distribution exist")

        if self.pause_dist is None or self.pause_dist.range is None:
            self.pause_dist = default_bout_distros.pause_dur
        self.pau_min, self.pau_max = (np.array(self.pause_dist.range) / self.dt).astype(
            int
        )
        self.pause_generator = reg.BoutGenerator(**self.pause_dist, dt=self.dt)

        self.Nstrides = 0
        self.Nstridechains = 0
        self.Nruns = 0
        self.Npauses = 0
        self.Nfeeds = 0
        self.Nfeedchains = 0
        self.Nfeeds_success = 0
        self.Nfeeds_fail = 0
        self.base_EEB = self.EEB

        self.exp_Nstrides = None
        self.cur_Nstrides = 0
        self.exp_Trun = None
        self.exp_Tpause = None
        self.cur_Nfeeds = None

        self.stridechain_lengths = []
        self.stridechain_durs = []
        self.feedchain_lengths = []
        self.feedchain_durs = []
        self.pause_durs = []
        self.run_durs = []
        self.feed_durs = []
        self.stride_durs = []

    @property
    def pause_completed(self) -> bool:
        """Whether the current pause has run past its sampled duration."""
        t = self.exp_Tpause
        return t is not None and self.t > t

    @property
    def run_completed(self) -> bool:
        """Whether the current run has run past its sampled duration."""
        t = self.exp_Trun
        return t is not None and self.t > t

    @property
    def stridechain_completed(self) -> bool:
        """Whether the current run has reached its sampled stride count."""
        n = self.exp_Nstrides
        return n is not None and self.cur_Nstrides > n

    def alternate_crawlNpause(self, stride_completed: bool = False) -> None:
        """Switch between running and pausing when the current bout ends.

        A run ends either after its sampled duration or after its sampled
        number of strides, depending on which the configuration generates.

        Args:
            stride_completed: Whether a stride completed this timestep.
        """
        if stride_completed:
            self.cur_Nstrides += 1

        if self.stridechain_completed or self.run_completed:
            self.interrupt_locomotion()

        elif self.pause_completed:
            self.trigger_locomotion()

    @property
    def feed_repeated(self) -> bool:
        """Draw whether another feeding motion follows the last one.

        The repeat probability is the configured reoccurrence rate, or the
        exploitation balance when the intermitter is driven by it.
        """
        r = self.feeder_reoccurence_rate if not self.use_EEB else self.EEB
        return np.random.random() < r

    def alternate_exploreNexploit(
        self, feed_motion: bool = False, on_food: bool = False
    ) -> None:
        """Switch between exploring and exploiting a food patch.

        Args:
            feed_motion: Whether a feeding motion completed this timestep.
            on_food: Whether the agent currently sits on food.
        """
        if feed_motion:
            assert self.cur_Nfeeds is not None
            self.Nfeeds += 1
            if not on_food:
                self.Nfeeds_fail += 1
                self.trigger_locomotion()
            else:
                self.Nfeeds_success += 1
                if self.feed_repeated:
                    self.cur_Nfeeds += 1
                else:
                    self.trigger_locomotion()
        elif on_food and self.cur_Nfeeds is None and np.random.random() <= self.EEB:
            self.cur_Nfeeds = 1
            self.register()
            self.cur_state = "feed"
            # self.reset()

    def register(self, bout: str | None = None) -> None:
        """Record the bout that is ending into the running statistics.

        Args:
            bout: The bout type to register. Defaults to the current state.
        """
        if bout is None:
            if self.cur_state is not None:
                bout = self.cur_state
            else:
                return
        dur = self.t
        self.ticks = 0
        if bout == "exec":
            if self.stridechain_generator is None and self.run_generator is not None:
                bout = "run"
            elif self.stridechain_generator is not None and self.run_generator is None:
                bout = "stridechain"

        if bout in ["feedchain", "feed"]:
            self.Nfeedchains += 1
            self.feedchain_lengths.append(self.cur_Nfeeds)
            self.feedchain_durs.append(dur)
            self.cur_Nfeeds = None

        elif bout == "stridechain":
            self.Nstridechains += 1
            self.stridechain_lengths.append(self.cur_Nstrides)
            self.stridechain_durs.append(dur)
            self.exp_Nstrides = None
            self.cur_Nstrides = 0

        elif bout == "run":
            self.Nruns += 1
            self.run_durs.append(dur)
            self.exp_Trun = None

        elif bout == "pause":
            self.Npauses += 1
            self.pause_durs.append(dur)
            self.exp_Tpause = None

    def update_state(
        self,
        stride_completed: bool = False,
        feed_motion: bool = False,
        on_food: bool = False,
    ) -> str | None:
        if self.cur_state is None:
            self.trigger_locomotion()
        if self.feed_bouts:
            self.alternate_exploreNexploit(feed_motion, on_food)
        self.alternate_crawlNpause(stride_completed)
        return self.cur_state

    def step(self, **kwargs: Any) -> str | None:
        """Advance the intermitter by one timestep.

        Args:
            **kwargs: Forwarded to the state update: whether a stride or
                feeding motion completed, and whether the agent is on food.

        Returns:
            The behavioural state after the update.
        """
        self.count_time()
        return self.update_state(**kwargs)

    def generate_stridechain(self) -> int:
        """Sample the number of strides in the next run."""
        return self.stridechain_generator.sample()

    def generate_run(self) -> float:
        """Sample the duration of the next run, in seconds."""
        return self.run_generator.sample()

    def interrupt_locomotion(self) -> None:
        """End the current run and begin a pause.

        Called by the sensors to cut a run short, for instance when the odor
        gradient turns unfavourable. Does nothing unless a run is under way.
        """
        if not self.cur_state == "exec":
            return
        self.register()
        self.exp_Tpause = self.generate_pause()
        self.cur_state = "pause"

    def trigger_locomotion(self, force: bool = False) -> None:
        """End the current bout and begin a run.

        Args:
            force: When True, restart the run even if one is already under way.
        """
        if not force and self.cur_state == "exec":
            return
        self.register()
        if self.stridechain_generator is not None:
            self.exp_Nstrides = self.generate_stridechain()
        elif self.run_generator is not None:
            self.exp_Trun = self.generate_run()
        self.cur_Nstrides = 0
        self.cur_state = "exec"
        self.ticks = 0

    def generate_pause(self) -> float:
        """Sample the duration of the next pause, in seconds."""
        return self.pause_generator.sample()

    def build_dict(self) -> dict[str, Any]:
        """Summarize the bouts observed over the run.

        Returns:
            The recorded bout durations and counts, together with the fraction
            of total time spent in each behavioural state.
        """
        cum_t = nam.cum("t")
        d = {}
        total_t = self.total_t
        d[cum_t] = total_t
        d[nam.num("tick")] = int(self.total_ticks)
        for c0 in ["feed", "stride"]:
            c = nam.chain(c0)
            Nc0, _ = nam.num([c0, c])
            l = nam.length(c)
            d[l] = [int(ll) for ll in getattr(self, f"{l}s")]
            d[Nc0] = int(sum(d[l]))
            d[nam.mean(nam.freq(c0))] = d[Nc0] / total_t if total_t else 0.0

        for c in ["feedchain", "stridechain", "pause"]:
            t = nam.dur(c)
            d[t] = getattr(self, f"{t}s")
            d[nam.num(c)] = len(d[t])
            cum_chunk_t = nam.cum(t)
            d[cum_chunk_t] = np.sum(d[t])
            d[nam.dur_ratio(c)] = d[cum_chunk_t] / total_t if total_t else 0.0
        return d

    def save_dict(
        self, path: str | None = None, dic: dict[str, Any] | None = None
    ) -> None:
        """Write the bout statistics to disk.

        Args:
            path: Destination path. Defaults to the configured output path.
            dic: The statistics to write. Defaults to freshly built ones.
        """
        if dic is None:
            dic = self.build_dict()
        if path is not None:
            file = f"{path}/intermitter_dict.txt"
            if not os.path.exists(path):
                os.makedirs(path)
            util.save_dict(dic, file)

    @property
    def active_bouts(self) -> tuple[int | None, int | None, float | None, float | None]:
        """The targets of the bouts currently under way.

        Returns:
            The expected stride count, the current feed count, and the expected
            pause and run durations. Entries are None where no such bout is
            active.
        """
        return self.exp_Nstrides, self.cur_Nfeeds, self.exp_Tpause, self.exp_Trun

    @property
    def mean_feed_freq(self) -> float:
        """The mean feeding frequency over the run, in Hz."""
        return self.Nfeeds / self.total_t if self.total_t else 0.0


class OfflineIntermitter(Intermitter):
    """
    Offline intermitter with fixed-frequency stride/feed detection.

    Extends Intermitter with tick-based stride and feed detection
    at fixed intervals (offline mode, no real-time physics).

    Example:
        >>> offline_int = OfflineIntermitter(EEB=0.5, crawl_freq=1.5, dt=0.1)
        >>> state = offline_int.step(on_food=True)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Build the offline intermitter and its fixed tick intervals.

        Args:
            **kwargs: Intermittency parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.crawl_ticks = np.round(1 / (self.crawl_freq * self.dt)).astype(int)
        if self.feed_freq is None:
            self.feed_ticks = None
        else:
            self.feed_ticks = np.round(1 / (self.feed_freq * self.dt)).astype(int)

    def step(
        self,
        stride_completed: bool | None = None,
        feed_motion: bool | None = None,
        on_food: bool = True,
    ) -> str | None:
        self.count_time()
        if feed_motion is None:
            if self.feed_ticks is None:
                feed_motion = False
            else:
                feed_motion = (
                    self.cur_state == "feed" and self.ticks % self.feed_ticks == 0
                )
        if stride_completed is None:
            stride_completed = (
                self.cur_state == "exec" and self.ticks % self.crawl_ticks == 0
            )
        return self.update_state(stride_completed, feed_motion, on_food)


class BranchIntermitter(Intermitter):
    """
    Branch intermitter with critical dynamics for pause/run generation.

    Extends Intermitter using exponential (exp_bout) and critical
    (critical_bout) distributions for more realistic behavioral branching.
    No feeding bouts in this mode.

    Attributes:
        feed_bouts: Fixed to False (no feeding)
        beta: Beta coefficient for exponential bout distribution
        c: Critical parameter for criticality function
        sigma: ISING branching coefficient

    Example:
        >>> branch_int = BranchIntermitter(beta=4.7, c=0.7, sigma=1.0, dt=0.1)
        >>> state = branch_int.step(stride_completed=True)
    """

    feed_bouts = param.Boolean(False, readonly=True)
    #: Matches `util.exp_bout`'s own default -- `generate_stridechain` feeds
    #: this straight into that function, so an unset `beta` should resolve
    #: to the same fallback `exp_bout` itself would use.
    beta = OptionalPositiveNumber(
        default=0.01, doc="The beta coefficient for the exponential function"
    )
    c = PositiveNumber(default=0.7, doc="The c parameter for the criticality function")
    sigma = PositiveNumber(default=1.0, doc="The ISING branching coef.")

    def generate_stridechain(self) -> int:
        """Sample a run length from an exponential branching process."""
        return util.exp_bout(
            beta=self.beta, tmax=self.stridechain_max, tmin=self.stridechain_min
        )

    def generate_pause(self) -> float:
        """Sample a pause duration from a critical branching process.

        The critical process yields the heavy-tailed pause durations observed
        experimentally, unlike an exponential one.
        """
        return (
            util.critical_bout(
                c=self.c, sigma=self.sigma, N=1000, tmax=self.pau_max, tmin=self.pau_min
            )
            * self.dt
        )


class FittedIntermitter(OfflineIntermitter):
    """
    Fitted intermitter using reference dataset parameters.

    Constructs OfflineIntermitter from stored reference dataset
    configurations (crawl/feed frequencies, bout distributions).

    Args:
        refID: Reference dataset ID to load intermitter config from
        **kwargs: Override parameters (optional)

    Example:
        >>> fitted_int = FittedIntermitter(refID='exploration')
        >>> state = fitted_int.step(on_food=False)
    """

    def __init__(self, refID: str, **kwargs: Any) -> None:
        """Build an intermitter from a reference dataset's fitted bouts.

        Args:
            refID: The reference dataset whose fitted bout distributions and
                frequencies configure this intermitter.
            **kwargs: Parameters overriding the stored ones.
        """
        c = reg.conf.Ref.getRef(refID)["intermitter"]
        stored_conf = {
            "crawl_freq": c["crawl_freq"],
            "feed_freq": c["feed_freq"],
            "dt": c["dt"],
            "stridechain_dist": c["stride"]["best"],
            "pause_dist": c["pause"]["best"],
            "feeder_reoccurence_rate": c["feeder_reoccurence_rate"],
        }
        stored_conf.update(kwargs)
        stored_conf["feed_bouts"] = (
            True if stored_conf["feed_freq"] is not None else False
        )
        super().__init__(**stored_conf)


def get_EEB_poly1d(max_dur: float = 60 * 60, **kws: Any) -> np.poly1d:
    """
    Compute polynomial fit of EEB vs mean feeding frequency.

    Simulates intermitter across EEB range (0-1) and fits polynomial
    to map feeding frequency back to EEB parameter.

    Args:
        max_dur: Maximum simulation duration (seconds)
        **kws: Intermitter configuration keyword arguments

    Returns:
        Polynomial (degree 5) mapping feed frequency to EEB

    Example:
        >>> poly = get_EEB_poly1d(crawl_freq=1.42, feed_freq=2.0, dt=0.1)
        >>> eeb_estimate = poly(0.15)  # For feed_freq=0.15
    """
    EEBs = np.arange(0, 1.05, 0.05)
    ms = []
    for EEB in EEBs:
        M = OfflineIntermitter(EEB=EEB, **kws)
        while M.total_t < max_dur:
            M.step()
        ms.append(M.mean_feed_freq)
    z = np.poly1d(np.polyfit(np.array(ms), EEBs, 5))
    return z


def get_EEB_time_fractions(
    refID: str | None = None,
    dt: float | None = None,
    max_dur: float = 60 * 60,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Compute time fractions for behavioral states across EEB range.

    Simulates intermitter across EEB values (0-1) and computes
    duration ratios for crawl/pause/feed states. Returns DataFrame
    for analysis and visualization.

    Args:
        refID: Reference dataset ID for intermitter config (optional)
        dt: Time step override (optional)
        max_dur: Maximum simulation duration (seconds)
        **kwargs: Intermitter configuration if refID not provided

    Returns:
        DataFrame with EEB values and corresponding time fraction ratios

    Example:
        >>> df = get_EEB_time_fractions(refID='exploration', dt=0.1)
        >>> print(df[['EEB', 'crawl ratio', 'pause ratio']])
    """
    if refID is not None:
        kws = reg.conf.Ref.getRef(refID)["intermitter"]
    else:
        kws = kwargs
    if dt is not None:
        kws["dt"] = dt
    rts = {
        f"{q} ratio": nam.dur_ratio(p)
        for p, q in zip(
            ["stridechain", "pause", "feedchain"], ["crawl", "pause", "feed"]
        )
    }
    EEBs = np.round(np.arange(0, 1, 0.02), 2)
    data = []
    for EEB in EEBs:
        M = OfflineIntermitter(EEB=EEB, **kws)
        while M.total_t < max_dur:
            M.step()
        dic = M.build_dict()
        entry = {
            "EEB": EEB,
            **{k: np.round(dic[v], 2) for k, v in rts.items()},
            nam.mean(nam.freq("feed")): M.mean_feed_freq,
        }
        data.append(entry)
    df = pd.DataFrame.from_records(data=data)
    return df
