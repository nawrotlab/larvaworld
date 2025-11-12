from __future__ import annotations

import random

import numpy as np
from shapely import geometry
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .single_run import ExpRun

from .. import util

__all__: list[str] = [
    "get_exp_condition",
]


def get_exp_condition(exp: str) -> type[ExpCondition] | None:
    """
    Get experiment condition class for given experiment type.

    Maps experiment type strings to their corresponding condition
    checker classes for experiment termination logic.

    Args:
        exp: Experiment type identifier string. Supported values:
             - 'PItrain_mini': Preference training (mini version)
             - 'PItrain': Full preference training
             - 'catch_me': Chase/tag game
             - 'keep_the_flag': Flag possession game
             - 'capture_the_flag': Flag capture game

    Returns:
        ExpCondition subclass for the experiment, or None if
        experiment type is not recognized.

    Example:
        >>> CondClass = get_exp_condition('PItrain')
        >>> if CondClass:
        >>>     condition = CondClass(env=exp_env)
        >>>     if condition.check():
        >>>         print("Experiment complete!")
    """
    d = {
        "PItrain_mini": PrefTrainCondition,
        "PItrain": PrefTrainCondition,
        "catch_me": CatchMeCondition,
        "keep_the_flag": KeepFlagCondition,
        "capture_the_flag": CaptureFlagCondition,
    }
    return d[exp] if exp in d else None


class ExpCondition:
    """
    Base class for experiment completion conditions.

    Defines interface for checking if an experiment should terminate
    and provides utilities for UI state updates.
    """

    def __init__(self, env: ExpRun):
        """
        Initialize experiment condition checker.

        Args:
            env: ExpRun instance with agents, sources, and simulation state.

        Example:
            >>> condition = MyCondition(env=experiment_env)
            >>> if condition.check():
            >>>     print("Experiment complete!")
        """
        self.env = env

    def check(self) -> bool:
        """
        Check if experiment completion condition is met.

        Returns:
            True if experiment should end, False otherwise.

        Note:
            Subclasses must override this method with specific logic.
        """
        return False

    def set_state(self, text: str) -> None:
        try:
            self.env.screen_manager.screen_state.set_text(text)
        except:
            pass

    def flash_text(self, text: str) -> None:
        try:
            self.env.input_box.flash_text(text)
        except:
            pass

    @property
    def agents(self):
        return self.env.agents

    @property
    def sources(self):
        return self.env.sources


class PrefTrainCondition(ExpCondition):
    """
    Preference training experiment condition.

    Implements olfactory preference training protocol with alternating
    CS (conditioned stimulus) and UCS (unconditioned stimulus) phases,
    followed by test trials to measure preference index (PI).
    """

    def __init__(self, **kwargs):
        """
        Initialize preference training condition.

        Sets up odor sources, training counters, and peak intensities
        for CS/UCS stimuli in preference learning paradigm.

        Args:
            **kwargs: Passed to ExpCondition.__init__ (requires env).

        Attributes:
            peak_intensity: Maximum odor intensity (default 2.0).
            CS_counter: Number of CS training trials completed.
            UCS_counter: Number of UCS training trials completed.
            CS_sources: Food odor sources (conditioned).
            UCS_sources: Non-food odor sources (unconditioned).

        Example:
            >>> condition = PrefTrainCondition(env=exp_env)
            >>> condition.check()  # Returns True when all trials complete
        """
        super().__init__(**kwargs)
        self.peak_intensity = 2.0
        self.CS_counter = 0
        self.UCS_counter = 0
        self.CS_sources = [f for f in self.sources if f.odor.id == "CS"]
        self.UCS_sources = [f for f in self.sources if f.odor.id == "UCS"]

    def move_larvae_to_center(self):
        for a in self.env.agents:
            a.reset_larva_pose()

    def toggle_odors(self, CS_intensity=2.0, UCS_intensity=0.0):
        for f in self.CS_sources:
            f.odor.intensity = CS_intensity
            f.visible = True if CS_intensity > 0 else False
        for f in self.UCS_sources:
            f.odor.intensity = UCS_intensity
            f.visible = True if UCS_intensity > 0 else False

    def init_test(self):
        for f in self.CS_sources:
            if f.unique_id == "CS_r":
                self.CS_sources.remove(f)
                self.env.delete_agent(f)
        for f in self.UCS_sources:
            if f.unique_id == "UCS_l":
                self.UCS_sources.remove(f)
                self.env.delete_agent(f)

    def start_trial(self, on_food=True):
        c = self.peak_intensity
        if on_food:
            self.CS_counter += 1
            if self.CS_counter <= 3:
                self.flash_text(f"Training trial {self.CS_counter}")
                self.toggle_odors(c, 0.0)
                self.move_larvae_to_center()
            elif self.CS_counter == 4:
                self.flash_text("Test trial on food")
                self.init_test()
                self.toggle_odors(c, c)
                self.move_larvae_to_center()

        else:
            self.UCS_counter += 1
            if self.UCS_counter <= 3:
                self.flash_text(f"Starvation trial {self.UCS_counter}")
                self.toggle_odors(0.0, c)
                self.move_larvae_to_center()
            elif self.UCS_counter == 4:
                PI = util.comp_PI(
                    xs=[l.pos[0] for l in self.agents],
                    arena_xdim=self.env.space.dims[0],
                )
                sec = int(self.env.Nticks * self.env.dt)
                m, s = int(sec / 60), sec % 60
                print()
                print(f"Test trial on food ended at {m}:{s} with PI={PI}")
                self.flash_text("Test trial off food")
                self.toggle_odors(c, c)
                self.move_larvae_to_center()

    def check(self):
        for i, ep in enumerate(self.env.sim_epochs):
            if self.env.Nticks == ep["start"]:
                q = ep["substrate"]["quality"]
                if q == 0.0:
                    self.env.food_grid.empty_grid()
                    self.start_trial(on_food=False)
                elif q == 1.0:
                    self.env.food_grid.reset()
                    self.start_trial(on_food=True)
            elif self.env.Nticks == ep["stop"] and i == len(self.env.sim_epochs) - 1:
                PI = util.comp_PI(
                    xs=[l.pos[0] for l in self.agents],
                    arena_xdim=self.env.space.dims[0],
                )
                print()
                print(f"Test trial off food ended with PI={PI}")
                self.flash_text(f"Test trial off food PI={PI}")
                return True
        return False


class CatchMeCondition(ExpCondition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_target_group("Left" if random.uniform(0, 1) > 0.5 else "Right")
        for f in self.targets:
            f.brain.olfactor.gain = {id: -v for id, v in f.brain.olfactor.gain.items()}
        self.score = {self.target_group: 0.0, self.follower_group: 0.0}

    def set_target_group(self, group):
        self.target_group = group
        self.follower_group = "Right" if self.target_group == "Left" else "Left"
        self.targets = [f for f in self.agents if f.group == self.target_group]
        self.followers = [f for f in self.agents if f.group == self.follower_group]

    def check(self):
        if self.env.Nticks == 0:
            self.flash_text("Catch me")
        targets_pos = [f.get_position() for f in self.targets]
        for f in self.followers:
            if any(
                [
                    geometry.Point(f.get_position()).distance(geometry.Point(p))
                    < f.radius
                    for p in targets_pos
                ]
            ):
                self.set_target_group(f.group)
                for a in self.agents:
                    a.brain.olfactor.gain = {
                        id: -v for id, v in a.brain.olfactor.gain.items()
                    }
                break
        self.score[self.target_group] += self.env.dt
        for group, score in self.score.items():
            if score >= 20000.0:
                print(f"{group} group wins")
                return True
        self.set_state(
            f'L:{np.round(self.score["Left"], 1)} vs R:{np.round(self.score["Right"], 1)}'
        )
        return False


class KeepFlagCondition(ExpCondition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for f in self.sources:
            if f.unique_id == "Flag":
                self.flag = f
        self.l_t = 0
        self.r_t = 0

    def check(self):
        if self.env.Nticks == 0:
            self.flash_text("Keep the flag")
        dur = 180
        carrier = self.flag.is_carried_by
        if carrier is None:
            self.l_t = 0
            self.r_t = 0
        elif carrier.group == "Left":
            self.l_t += self.env.dt
            self.r_t = 0
            if self.l_t - dur > 0:
                print("Left group wins")
                return True
        elif carrier.group == "Right":
            self.r_t += self.env.dt
            self.l_t = 0
            if self.r_t - dur > 0:
                print("Right group wins")
                return True
        self.set_state(
            f"L:{np.round(dur - self.l_t, 2)} vs R:{np.round(dur - self.r_t, 2)}"
        )
        return False


class CaptureFlagCondition(ExpCondition):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for f in self.sources:
            if f.unique_id == "Flag":
                self.flag = f
            elif f.unique_id == "Left_base":
                self.l_base = f
            elif f.unique_id == "Right_base":
                self.r_base = f
        self.l_base_p = self.l_base.get_position()
        self.r_base_p = self.r_base.get_position()
        self.l_dst0 = self.flag.radius * 2 + self.l_base.radius * 2
        self.r_dst0 = self.flag.radius * 2 + self.r_base.radius * 2

    def check(self):
        def compute_dst(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        if self.env.Nticks == 0:
            self.flash_text("Capture the flag")
        flag_p = self.flag.get_position()
        l_dst = -self.l_dst0 + compute_dst(flag_p, self.l_base_p)
        r_dst = -self.r_dst0 + compute_dst(flag_p, self.r_base_p)
        l_dst = np.round(l_dst * 1000, 2)
        r_dst = np.round(r_dst * 1000, 2)
        if l_dst < 0:
            print("Left group wins")
            return True
        elif r_dst < 0:
            print("Right group wins")
            return True
        self.set_state(f"L:{l_dst} vs R:{r_dst}")
        return False
