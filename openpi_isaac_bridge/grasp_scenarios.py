"""Explicit, bounded object-reset and rotation-rate conditions."""
import math
import re


def validate_grasp_scenarios(scenarios):
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("grasp scenarios must be a nonempty list")
    seen = set()
    for s in scenarios:
        if not isinstance(s, dict) or set(s) != {"name", "object_offset_xy_m", "object_yaw_deg", "seed", "orient_step_deg"}:
            raise ValueError("grasp scenario requires name, object_offset_xy_m, object_yaw_deg, seed, orient_step_deg")
        name = s["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,35}", name) or name in seen:
            raise ValueError("grasp scenario names must be unique safe lowercase identifiers")
        offset = s["object_offset_xy_m"]
        if not isinstance(offset, (list, tuple)) or len(offset)!=2 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in offset) or math.hypot(*offset)>0.04:
            raise ValueError("object XY offset must be finite, norm <= 0.04 m")
        for key, lo, hi in (("object_yaw_deg",-30,30),("orient_step_deg",0.1,0.5)):
            v = s[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
                raise ValueError(f"{key} outside [{lo}, {hi}]")
        if isinstance(s["seed"],bool) or not isinstance(s["seed"],int) or not 0<=s["seed"]<=1000000:
            raise ValueError("seed must be an integer in [0, 1000000]")
        seen.add(name)
    return scenarios


def expand_grasp_cases(cases, scenarios):
    validate_grasp_scenarios(scenarios)
    expanded = [dict(case, name=case["name"]+"-"+s["name"], base_case=case["name"], grasp_scenario=s)
                for case in cases for s in scenarios]
    if any(len(c["name"])>80 for c in expanded) or len({c["name"] for c in expanded})!=len(expanded):
        raise ValueError("combined grasp case names too long or collide")
    return expanded
