from ...param import ClassAttr, NestedConf
from .module_modes import moduleDB as MD

__all__ = [
    "Locomotor",
]


class Locomotor(NestedConf):
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

    def __init__(self, conf, dt=0.1, **kwargs):
        self.dt = dt
        kwargs.update(MD.build_locomodules(conf=conf, dt=dt))
        super().__init__(**kwargs)

    def on_new_pause(self):
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_run(self):
        if self.crawler:
            self.crawler.start_effector()
        if self.feeder:
            self.feeder.stop_effector()

    def on_new_feed(self):
        if self.crawler:
            self.crawler.stop_effector()
        if self.feeder:
            self.feeder.start_effector()

    def step_intermitter(self, **kwargs):
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
    def stride_completed(self):
        if self.crawler:
            return self.crawler.complete_iteration
        else:
            return False

    @property
    def feed_motion(self):
        if self.feeder:
            return self.feeder.complete_iteration
        else:
            return False

    def step(self, A_in=0, length=1, on_food=False):
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
