"""CPU-only summaries for fixed-target step responses."""


def summarize_response(trace):
    if not trace:
        raise ValueError("response trace is empty")
    gain_50ms = next((row["gain"] for row in trace if abs(row["time_s"] - 0.05) < 1e-8), None)
    def sustained_time(predicate):
        last_bad = max((i for i, row in enumerate(trace) if not predicate(row)), default=-1)
        return trace[last_bad + 1]["time_s"] if last_bad + 1 < len(trace) else None
    return {
        "gain_at_50ms": gain_50ms,
        "within_10pct_at_50ms": None if gain_50ms is None else 0.9 <= gain_50ms <= 1.1,
        "within_5pct_at_50ms": None if gain_50ms is None else 0.95 <= gain_50ms <= 1.05,
        "checkpoints": [{k: row[k] for k in (
            "step", "time_s", "gain", "position_error_m", "rotation_error_deg",
            "controller_target_vs_fixed_position_m", "controller_target_vs_fixed_rotation_deg",
        ) if k in row} for row in trace if row["step"] in (1, 2, 5, 10, 20, 60)],
        "final_gain": trace[-1]["gain"],
        "peak_gain": max(row["gain"] for row in trace),
        "overshoot_pct": max(0.0, max(row["gain"] for row in trace) - 1.0) * 100,
        "tail_gain_peak_to_peak": max(row["gain"] for row in trace[-10:]) - min(row["gain"] for row in trace[-10:]),
        "t90_sustained_s": sustained_time(lambda row: row["gain"] >= 0.9),
        "settling_5pct_s": sustained_time(lambda row: abs(row["gain"] - 1) <= 0.05),
        "final_position_error_m": trace[-1]["position_error_m"],
        "final_rotation_error_deg": trace[-1]["rotation_error_deg"],
    }
