"""Validate named per-joint experiments before starting any simulator process."""

import re

from gain_probe import GAIN_PROFILES, validate_joint_overrides


def validate_gain_cases(cases):
    if not isinstance(cases, list) or not cases:
        raise ValueError("gain cases must be a nonempty list")
    names, validated = set(), []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"name", "profile", "joint_overrides"}:
            raise ValueError("each case requires exactly name, profile, joint_overrides")
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) or name in names:
            raise ValueError("case names must be unique safe lowercase identifiers")
        if case["profile"] not in GAIN_PROFILES:
            raise ValueError("unknown base gain profile")
        overrides = validate_joint_overrides(case["joint_overrides"])
        validated.append(dict(case, joint_overrides=overrides))
        names.add(name)
    return validated
