"""Small, version-independent configuration changes for actuator experiments."""

import math
import re

GAIN_PROFILES = {
    "baseline": (1.0, 1.0),
    "kp1p5": (1.5, 1.0),
    "kp1p5-damped": (1.5, math.sqrt(1.5)),
    "kp1p5-kd0p75": (1.5, 0.75),
    "kp1p5-kd0p5": (1.5, 0.5),
    "kp1p5-kd0p375": (1.5, 0.375),
    "kp1p5-kd0p25": (1.5, 0.25),
    "kp2p25-kd0p25": (2.25, 0.25),
    "kp2p25-kd0p375": (2.25, 0.375),
    "kp2p25-kd0p5": (2.25, 0.5),
    "kp3-kd0p25": (3.0, 0.25),
}
LIMIT_FIELDS = ("joint_effort_limit", "joint_velocity_limit", "actuator_effort_limit",
                "actuator_velocity_limit", "effort_limit", "velocity_limit", "effort_limit_sim", "velocity_limit_sim")


def scaled_gain(value, multiplier):
    if isinstance(value, dict):
        return {key: scaled_gain(item, multiplier) for key, item in value.items()}
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("gain must be a finite non-negative scalar or dictionary; USD-inherited gains cannot be scaled")
    return value * multiplier


def resolve_joint_gain(value, joint):
    if isinstance(value, dict):
        matches = [gain for pattern, gain in value.items() if re.fullmatch(pattern, joint)]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one gain match for {joint}")
        value = matches[0]
    return scaled_gain(value, 1.0)


def validate_joint_overrides(joint_overrides):
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    joint_overrides = {} if joint_overrides is None else joint_overrides
    if not isinstance(joint_overrides, dict) or set(joint_overrides) - set(arm_names):
        raise ValueError("joint overrides must be a dictionary restricted to panda_joint1 through panda_joint7")
    for joint, gains in joint_overrides.items():
        if not isinstance(gains, dict) or not gains or set(gains) - {"stiffness", "damping"}:
            raise ValueError(f"{joint}: expected stiffness and/or damping")
        for value in gains.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{joint}: per-joint gains must be finite positive scalars")
    return joint_overrides


def configure_gain_profile(actuators, profile, joint_overrides=None):
    if profile not in GAIN_PROFILES:
        raise ValueError("unknown gain profile")
    kp_factor, kd_factor = GAIN_PROFILES[profile]
    arm_names = [f"panda_joint{i}" for i in range(1, 8)]
    joint_overrides = validate_joint_overrides(joint_overrides)
    finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
    covered = set()
    plans = []
    report = []
    for name, actuator in actuators.items():
        patterns = actuator.joint_names_expr
        if isinstance(patterns, str):
            patterns = [patterns]
        matches = [joint for joint in arm_names + finger_names if any(re.fullmatch(pattern, joint) for pattern in patterns)]
        arm = [joint for joint in matches if joint in arm_names]
        covered.update(arm)
        original_kp, original_kd = actuator.stiffness, actuator.damping
        new_kp, new_kd = original_kp, original_kd
        has_overrides = any(joint in joint_overrides for joint in arm)
        if arm and (profile != "baseline" or has_overrides):
            if any(joint in finger_names for joint in matches):
                raise ValueError(f"{name}: mixed arm/finger actuator group; refusing to alter fingers")
            if not getattr(actuator.class_type, "is_implicit_model", False):
                raise ValueError(f"{name}: this experiment supports implicit joint drives only")
            new_kp = scaled_gain(original_kp, kp_factor)
            new_kd = scaled_gain(original_kd, kd_factor)
            if has_overrides:
                new_kp = {joint: joint_overrides.get(joint, {}).get("stiffness", resolve_joint_gain(new_kp, joint)) for joint in arm}
                new_kd = {joint: joint_overrides.get(joint, {}).get("damping", resolve_joint_gain(new_kd, joint)) for joint in arm}
        plans.append((actuator, new_kp, new_kd))
        report.append({
            "group": name, "model": actuator.class_type.__name__, "joint_patterns": list(patterns),
            "matched_arm_joints": arm, "original_stiffness": original_kp, "original_damping": original_kd,
            "experiment_stiffness": new_kp, "experiment_damping": new_kd,
            "configured_limits": {key: getattr(actuator, key) for key in LIMIT_FIELDS if hasattr(actuator, key)},
        })
    if set(arm_names) != covered:
        raise ValueError(f"actuator configuration does not cover the seven Panda arm joints: {sorted(covered)}")
    # Validate the whole configuration before changing any group.
    for actuator, new_kp, new_kd in plans:
        actuator.stiffness, actuator.damping = new_kp, new_kd
    return {"profile": profile, "kp_factor": kp_factor, "kd_factor": kd_factor,
            "joint_overrides": joint_overrides, "groups": report}
