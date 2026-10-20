"""Deterministic, per-environment random force-pulse scheduling (no simulator imports)."""

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PushChange:
    env_id: int
    force: float
    duration_steps: int  # zero for a pulse ending
    step: int


class RandomPushScheduler:
    """Emit force changes; hold each force until an explicit zero-force change.

    Call step() once per control step. reset() restarts selected environments'
    grace/wait timers; callers must also clear their applied force on reset.
    Independent RNGs keep one environment's sequence unchanged by other resets.
    """

    def __init__(self, env_ids, dt, seed, start_s, force_min, force_max,
                 duration_min_s, duration_max_s, wait_min_s, wait_max_s):
        values = (dt, start_s, force_min, force_max, duration_min_s,
                  duration_max_s, wait_min_s, wait_max_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Random-push parameters must be finite")
        if dt <= 0 or start_s < 0 or not 0 < force_min <= force_max:
            raise ValueError("Require dt > 0, start_s >= 0, and 0 < force_min <= force_max")
        if not 0 < duration_min_s <= duration_max_s:
            raise ValueError("Require 0 < duration_min_s <= duration_max_s")
        if not 0 < wait_min_s <= wait_max_s:
            raise ValueError("Require 0 < wait_min_s <= wait_max_s")
        self.dt = dt
        self.start_steps = math.ceil(start_s / dt)
        self.force_range = (force_min, force_max)
        self.duration_range = (duration_min_s, duration_max_s)
        self.wait_range = (wait_min_s, wait_max_s)
        self.tick = 0
        self._states = {}
        for env_id in sorted(set(env_ids)):
            if env_id < 0:
                raise ValueError("Environment IDs must be nonnegative")
            self._states[env_id] = {
                "rng": random.Random(seed + env_id * 1000003),
                "force": 0.0, "end": 0, "next": 0,
            }
        self.reset(self._states)

    def _sample_steps(self, rng, bounds):
        # Round up so a pulse/wait is never shorter than the sampled time.
        return max(1, math.ceil(rng.uniform(*bounds) / self.dt))

    def reset(self, env_ids):
        for env_id in env_ids:
            if env_id not in self._states:
                continue
            state = self._states[env_id]
            state["force"] = 0.0
            state["end"] = self.tick
            state["next"] = (self.tick + self.start_steps
                             + self._sample_steps(state["rng"], self.wait_range))

    def step(self):
        changes = []
        for env_id, state in self._states.items():
            if self.tick >= state["next"]:
                rng = state["rng"]
                magnitude = rng.uniform(*self.force_range)
                state["force"] = magnitude if rng.random() < 0.5 else -magnitude
                duration = self._sample_steps(rng, self.duration_range)
                state["end"] = self.tick + duration
                state["next"] = (state["end"]
                                 + self._sample_steps(rng, self.wait_range))
                changes.append(PushChange(env_id, state["force"], duration, self.tick))
            elif state["force"] != 0.0 and self.tick >= state["end"]:
                state["force"] = 0.0
                changes.append(PushChange(env_id, 0.0, 0, self.tick))
        self.tick += 1
        return changes
