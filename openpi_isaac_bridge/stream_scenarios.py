"""Bounded initial-position and amplitude conditions for deterministic streams."""
import math
import re


def validate_stream_scenarios(scenarios):
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("stream scenarios must be a nonempty list")
    seen = set()
    for s in scenarios:
        if not isinstance(s, dict) or set(s) != {"name", "start_offset_m", "amplitude_scale"}:
            raise ValueError("scenario requires name, start_offset_m, amplitude_scale")
        name, offset, scale = s["name"], s["start_offset_m"], s["amplitude_scale"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", name) or name in seen:
            raise ValueError("scenario names must be unique safe lowercase identifiers")
        if (not isinstance(offset, (list, tuple)) or len(offset) != 3 or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in offset) or
            math.hypot(*offset) > 0.025):
            raise ValueError("start offset must be three finite metres with norm <= 0.025")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 1 <= scale <= 2:
            raise ValueError("stream amplitude scale must be in [1, 2]")
        seen.add(name)
    return scenarios


def expand_stream_cases(cases, scenarios):
    scenarios = validate_stream_scenarios(scenarios)
    expanded = []
    for case in cases:
        for s in scenarios:
            name = case["name"] + "-" + s["name"]
            if len(name) > 80:
                raise ValueError("combined gain/scenario name exceeds 80 characters")
            expanded.append(dict(case, name=name, base_case=case["name"], stream_scenario=s))
    if len({c["name"] for c in expanded}) != len(expanded):
        raise ValueError("combined gain/scenario names collide")
    return expanded
