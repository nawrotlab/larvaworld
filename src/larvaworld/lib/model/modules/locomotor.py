from __future__ import annotations
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Locomotor'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.modules import Locomotor'",
        DeprecationWarning,
        stacklevel=2,
    )
from ...param import ClassAttr, NestedConf
from .module_modes import moduleDB as MD
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .crawler import Crawler
    from .feeder import Feeder
    from .turner import Turner
    from .intermitter import Intermitter
    from .crawl_bend_interference import Interference as _Interference

__all__: list[str] = [
    "Locomotor",
]


class Locomotor(NestedConf):
    """
    Locomotor module orchestrating agent movement control.

    Coordinates multiple behavioral modules (crawler, turner, feeder,
    intermitter) to produce realistic larva locomotion patterns with
    peristaltic crawling, turning, feeding, and run/pause transitions.

    Attributes:
        interference: Crawl-bend coupling module (attenuates turning during crawling)
        intermitter: Behavioral intermittency module (controls run/pause/feed states)
        feeder: Feeding behavior module (head-sweeping motions)
        turner: Body-bending module (directional changes)
        crawler: Peristaltic crawling module (forward locomotion)
        dt: Simulation time step (seconds)

    Args:
        conf: Locomotor configuration dict with module configs:
              - 'crawler': Crawler module config (or None)
              - 'turner': Turner module config (or None)
              - 'feeder': Feeder module config (or None)
              - 'intermitter': Intermitter module config (or None)
              - 'interference': Interference module config (or None)
        dt: Simulation time step in seconds (default: 0.1)
        **kwargs: Additional keyword arguments passed to parent class

    Returns:
        Tuple of (linear_velocity, angular_velocity, feed_flag) from step()

    Example:
        >>> locomotor = Locomotor(conf=loco_conf, dt=0.1)
        >>> lin_vel, ang_vel, feeding = locomotor.step(A_in=0.5, length=2.0, on_food=False)
    """

    interference = ClassAttr(
        class_=MD.parent_class("interference"),
        default=None,
        doc="The crawl-bend coupling module",
    )
    intermitter = ClassAttr(
        class_=MD.parent_class("intermitter"),
        default=None,
        doc="The behavioral intermittency module",
    )
    feeder = ClassAttr(
        class_=MD.parent_class("feeder"), default=None, doc="The feeding module"
    )
    turner = ClassAttr(
        class_=MD.parent_class("turner"), default=None, doc="The body-bending module"
    )
    crawler = ClassAttr(
        class_=MD.parent_class("crawler"),
        default=None,
        doc="The peristaltic crawling module",
    )

    def __init__(self, conf: Any, dt: float = 0.1, **kwargs: Any) -> None:
        self.dt: float = dt
        kwargs.update(MD.build_locomodules(conf=conf, dt=dt))
        super().__init__(**kwargs)

    def on_new_pause(self) -> None:
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_run(self) -> None:
        if self.crawler:
            self.crawler.start_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_feed(self) -> None:
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.start_effector()

    def step_intermitter(self, **kwargs: Any) -> None:
        if self.intermitter:
            pre_state = self.intermitter.cur_state
            cur_state = self.intermitter.step(**kwargs)
            if pre_state != "pause" and cur_state == "pause":
                self.on_new_pause()
            elif pre_state != "exec" and cur_state == "exec":
                self.on_new_run()
            elif pre_state != "feed" and cur_state == "feed":
                self.on_new_feed()
            # print(cur_state)

    @property
    def stride_completed(self) -> bool:
        if self.crawler:
            return self.crawler.complete_iteration
        else:
            return False

    @property
    def feed_motion(self) -> bool:
        if self.feeder:
            return self.feeder.complete_iteration
        else:
            return False

    def step(
        self, A_in: float = 0, length: float = 1, on_food: bool = False
    ) -> tuple[float, float, bool]:
        C, F, T, If = self.crawler, self.feeder, self.turner, self.interference
        if If:
            If.cur_attenuation = 1
        if F:
            F.step()
            if F.active and If:
                If.check_module(F, "Feeder")
        if C:
            lin = C.step() * length
            if C.active and If:
                If.check_module(C, "Crawler")
        else:
            lin = 0
        self.step_intermitter(
            stride_completed=self.stride_completed,
            feed_motion=self.feed_motion,
            on_food=on_food,
        )

        if T:
            if If:
                cur_att_in, cur_att_out = If.apply_attenuation(If.cur_attenuation)
            else:
                cur_att_in, cur_att_out = 1, 1
            ang = T.step(A_in=A_in * cur_att_in) * cur_att_out
        else:
            ang = 0
        return lin, ang, self.feed_motion
