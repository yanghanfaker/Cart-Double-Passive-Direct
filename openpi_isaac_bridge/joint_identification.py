"""Joint-space step statistics; no inertia or pole fitting from coupled motion."""

import math


def summarize_joint_step(rows, joint_index, initial_positions, command_step_rad, joint_names):
    if not rows or not math.isfinite(command_step_rad) or command_step_rad == 0:
        raise ValueError("nonempty samples and a finite nonzero joint step are required")
    start_time = rows[0]["command_time_s"]
    samples = [{"time_s": r["time_s"] - start_time,
                "gain": (r["position_rad"][joint_index] - initial_positions[joint_index]) / command_step_rad}
               for r in rows]
    def sustained_time(predicate):
        last_bad = max((i for i, row in enumerate(samples) if not predicate(row)), default=-1)
        return samples[last_bad+1]["time_s"] if last_bad+1 < len(samples) else None
    gain50 = next((r["gain"] for r in samples if math.isclose(r["time_s"], 0.05, abs_tol=1e-8)), None)
    errors = [r["position_target_rad"][joint_index] - r["position_rad"][joint_index] for r in rows]
    peak = max(r["gain"] for r in samples)
    return {
        "joint": joint_names[joint_index], "step_rad": command_step_rad,
        "gain_at_50ms": gain50,
        "overshoot_pct": max(0, peak-1)*100,
        "peak_gain": peak,
        "t90_sustained_s": sustained_time(lambda r: r["gain"] >= 0.9),
        "settling_5pct_s": sustained_time(lambda r: abs(r["gain"]-1) <= 0.05),
        "rms_tracking_error_rad": math.sqrt(sum(e*e for e in errors)/len(errors)),
        "final_tracking_error_rad": errors[-1],
        "coupling_peak_rad": {name: max(abs(r["position_rad"][i]-initial_positions[i]) for r in rows)
                              for i, name in enumerate(joint_names) if i != joint_index},
        "checkpoints": [row for row in samples if any(math.isclose(row["time_s"], t, abs_tol=1e-8)
                        for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0))],
        "sample_count": len(rows),
        "scope": "one fixed joint target stepped while other arm position targets are held; residual dynamics remain coupled",
    }
