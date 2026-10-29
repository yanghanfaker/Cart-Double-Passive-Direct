"""Explicit robosuite 1.4.0 OSC_POSE defaults, not a scene calibration.

Sources and version assumptions are documented in README.md. The returned
vectors are in the OSC world frame; callers must transform to Isaac root axes.
"""

import math


def decode_libero_action(action, previous_isaac_gripper=1.0):
    """Decode controller-domain actions AFTER OpenPI server unnormalization.

    Zero gripper input means keep the preceding command in this binary adapter.
    This does not reproduce robosuite's incremental finger actuator dynamics.
    """
    if len(action) != 7 or not all(math.isfinite(value) for value in action):
        raise ValueError("LIBERO action must contain seven finite values")
    if previous_isaac_gripper not in (-1.0, 1.0):
        raise ValueError("previous Isaac gripper must be -1 or +1")
    clipped = [max(-1.0, min(1.0, float(value))) for value in action]
    gripper = previous_isaac_gripper
    if clipped[6] != 0:
        gripper = -1.0 if clipped[6] > 0 else 1.0
    return {
        "profile": "robosuite-1.4.0-OSC_POSE-default",
        "input_action": list(action),
        "clipped_action": clipped,
        "clipped_indices": [i for i, (a, b) in enumerate(zip(action, clipped)) if a != b],
        "delta_world_m": [value * 0.05 for value in clipped[:3]],
        "delta_world_rad": [value * 0.5 for value in clipped[3:6]],
        "isaac_gripper": gripper,
    }


def mapping_test_plan():
    """Signed small commands on all axes, followed by open/close/reopen."""
    plan = []
    for index, axis in enumerate(("x", "y", "z", "rx", "ry", "rz")):
        amplitude = 0.1 if index < 3 else math.radians(2) / 0.5
        for sign, suffix in ((1, "positive"), (-1, "negative")):
            command = [0.0] * 6 + [-1.0]
            command[index] = sign * amplitude
            plan.append((axis + "_" + suffix, command))
    for name, value in (("open", -1.0), ("close", 1.0), ("reopen", -1.0)):
        plan.append((name, [0.0] * 6 + [value]))
    return plan


def stream_test_plan(axis, amplitude_scale=1.0):
    """One command per 50 ms: 5 mm/2 deg out, stop, back, stop.

    Commands are relative to the measured current pose, as in OSC set_goal;
    the nominal cumulative path is a reference, not an accumulated target.
    """
    axes = ("x", "y", "z", "rx", "ry", "rz")
    if axis not in axes:
        raise ValueError("invalid streaming axis")
    if isinstance(amplitude_scale, bool) or not math.isfinite(amplitude_scale) or not 1 <= amplitude_scale <= 2:
        raise ValueError("stream amplitude scale must be finite and in [1, 2]")
    index = axes.index(axis)
    amplitude = amplitude_scale * (0.005 / 20 / 0.05 if index < 3 else math.radians(2) / 20 / 0.5)
    plan = []
    for phase, count, sign in (("out", 20, 1), ("stop_out", 10, 0), ("back", 20, -1), ("stop_back", 20, 0)):
        for _ in range(count):
            action = [0.0] * 6 + [-1.0]
            action[index] = sign * amplitude
            plan.append((phase, action))
    return plan
