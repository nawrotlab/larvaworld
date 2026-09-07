"""
The locomotor, orchestrating the motion-generating modules.

Combines the crawler, turner, feeder, intermitter and their crawl-bend
coupling into a single module that produces the agent's linear and angular
velocity each timestep.
"""

from __future__ import annotations

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
        """Build the locomotor and its configured sub-modules.

        Args:
            conf: The model configuration naming the module modes to build.
            dt: The simulation timestep in seconds.
            **kwargs: Additional module instances overriding the built ones.
        """
        self.dt: float = dt
        kwargs.update(MD.build_locomodules(conf=conf, dt=dt))
        super().__init__(**kwargs)

    def on_new_pause(self) -> None:
        """Stop crawling and feeding on entering a pause."""
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_run(self) -> None:
        """Start crawling and stop feeding on entering a run."""
        if self.crawler:
            self.crawler.start_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_feed(self) -> None:
        """Stop crawling and start feeding on entering a feeding bout."""
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.start_effector()

    def step_intermitter(self, **kwargs: Any) -> None:
        """Advance the intermitter and apply any behavioural state change.

        Entering the pause, run or feeding state starts and stops the crawler
        and feeder accordingly.

        Args:
            **kwargs: Forwarded to the intermitter's step, carrying whether a
                stride or feeding motion completed and whether the agent is on
                food.
        """
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
        """Whether the crawler completed a stride this timestep."""
        if self.crawler:
            return self.crawler.complete_iteration
        else:
            return False

    @property
    def feed_motion(self) -> bool:
        """Whether the feeder completed a feeding motion this timestep."""
        if self.feeder:
            return self.feeder.complete_iteration
        else:
            return False

    def step(
        self, A_in: float = 0, length: float = 1, on_food: bool = False
    ) -> tuple[float, float, bool]:
        """Advance the locomotor by one timestep.

        Steps the feeder and crawler, letting each set the current crawl-bend
        attenuation, then advances the intermitter, then steps the turner with
        the attenuation applied to its input, its output, or both according to
        the coupling's suppression mode.

        Args:
            A_in: The activation arriving from the brain's sensors.
            length: The agent's body length, which scales the crawler's
                displacement.
            on_food: Whether the agent currently sits on food.

        Returns:
            The linear velocity, the angular velocity, and whether a feeding
            motion completed this timestep.
        """
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
