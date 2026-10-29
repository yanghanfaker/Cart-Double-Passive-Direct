"""Additional cumulative Cartesian stream screen; does not alter legacy tests."""

import math


def screen_cartesian_stream(axes):
    expected = {"x", "y", "z", "rx", "ry", "rz"}
    names, checks = [], []
    for axis in axes:
        name, gain = axis["axis"], axis.get("outbound_gain")
        names.append(name)
        finite = isinstance(gain, (float, int)) and not isinstance(gain, bool) and math.isfinite(gain)
        passed = axis.get("passed") is True and finite and 0.9 <= gain <= 1.1
        checks.append({"axis": name, "legacy_passed": axis.get("passed"),
                       "outbound_gain": gain, "passed": passed})
    complete = len(names) == 6 and set(names) == expected
    return {"complete": complete, "all_passed": complete and all(c["passed"] for c in checks),
            "outbound_gain_range": [0.9, 1.1], "axes": checks,
            "scope": "20-command outbound cumulative gain plus unchanged legacy tracking, return, stop drift and gripper checks; not per-command 50ms settling or OSC equivalence"}
