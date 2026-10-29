"""Frozen, bounded action replay; no model calls or desired-pose accumulation."""
import hashlib
import json
import math
from pathlib import Path

from openpi_step_probe import bounded_openpi_action, validate_action_chunk

REPLAY_MODES = ("openpi-replay-translation", "openpi-replay-rotation", "openpi-replay-mixed")


def quaternion_distance_deg(first, second):
    """Shortest orientation distance, normalized to avoid float32 norm bias."""
    normalized = []
    for quaternion in (first, second):
        if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
            raise ValueError("expected finite xyzw quaternion")
        length = math.hypot(*quaternion)
        if not math.isclose(length, 1.0, abs_tol=.001):
            raise ValueError("expected approximately unit quaternion")
        normalized.append([v/length for v in quaternion])
    a, b = normalized
    if sum(x*y for x,y in zip(a,b)) < 0:
        b = [-v for v in b]
    # Stable even when the orientations are almost identical.
    return math.degrees(4*math.atan2(math.dist(a,b), math.hypot(*(x+y for x,y in zip(a,b)))))


def load_replay(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    actions = validate_action_chunk(data["actions"])
    if len(actions) != 20 or data.get("control_dt_s") != 0.05:
        raise ValueError("replay requires exactly 20 actions at 50 ms")
    pose = data["initial_pose_root"]
    if (len(pose["position_m"]) != 3 or len(pose["quat_xyzw"]) != 4
            or not all(math.isfinite(v) for v in pose["position_m"] + pose["quat_xyzw"])
            or abs(sum(v*v for v in pose["quat_xyzw"])-1) > .001):
        raise ValueError("invalid replay initial pose")
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def replay_mapping(action, mode):
    if mode not in REPLAY_MODES:
        raise ValueError("unknown replay component")
    mapped = bounded_openpi_action(action)
    # Remove a component AFTER applying the original vector-norm limits.
    # Retained components are therefore identical across all three runs.
    if mode == "openpi-replay-translation":
        mapped["bounded_delta_world_rad"] = [0.0]*3
    elif mode == "openpi-replay-rotation":
        mapped["bounded_delta_world_m"] = [0.0]*3
    mapped["gripper_policy"] = "forced open throughout deterministic replay"
    return mapped


def run_replay(data, mode, read_state, execute, announce=print):
    trace = []
    report = {"probe_completed": False, "completed_steps": 0, "requested_steps": 40,
              "source_commit": data["source_commit"], "fixture_sha256": data["sha256"],
              "component": mode, "trace": trace,
              "scope": "20 frozen increments followed by 20 zero increments; 50ms each; current-pose relative targets; gripper open; no inference"}
    try:
        state = read_state()
        expected = data["initial_pose_root"]
        if len(state["position_m"]) != 3 or not all(math.isfinite(v) for v in state["position_m"]):
            raise ValueError("expected finite initial position")
        error = math.dist(state["position_m"], expected["position_m"])
        angle = quaternion_distance_deg(state["quat_xyzw"], expected["quat_xyzw"])
        report["initial_position_error_m"] = error
        report["initial_rotation_error_deg"] = angle
        if error > .00025 or angle > .1:
            raise RuntimeError("reset pose differs from recorded source by >0.25mm or >0.1deg")
        for index in range(40):
            action = data["actions"][index] if index < 20 else [0.0]*7
            mapped = replay_mapping(action, mode)
            row = {"step": index+1, "phase": "replay" if index < 20 else "zero-input-tail",
                   "raw_source_action": action, "action_mapping": mapped, "executed": False}
            trace.append(row)
            row.update(execute(mapped, index))
            if "bounded_delta_root_m_rad" in row:
                for name, desired, actual in (
                    ("translation_projection_gain", row["bounded_delta_root_m_rad"][:3], row["actual_delta_root_m"]),
                    ("rotation_projection_gain", row["bounded_delta_root_m_rad"][3:], row["actual_delta_rotation_root_rad"]),
                ):
                    denominator = sum(v*v for v in desired)
                    row[name] = sum(a*b for a,b in zip(desired,actual))/denominator if denominator > 1e-20 else None
            row["executed"] = True
            report["completed_steps"] += 1
            announce(f"[OpenPI replay] {mode} {index+1}/40: {row['phase']}")
        report["probe_completed"] = True
        report["period_error_summary"] = {
            field: {"rms": math.sqrt(sum(row[field]**2 for row in trace[:20])/20),
                    "max": max(row[field] for row in trace[:20])}
            for field in ("period_position_error_m", "period_rotation_error_deg")
            if all(field in row for row in trace[:20])}
    except Exception as error:
        report.update(failure_reason=str(error), error_type=type(error).__name__)
        announce(f"[OpenPI replay] stopped: {type(error).__name__}: {error}")
    return report
