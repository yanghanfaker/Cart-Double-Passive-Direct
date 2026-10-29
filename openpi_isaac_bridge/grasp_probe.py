"""Deterministic ground-truth grasp diagnostic, independent of Isaac imports.

Finger obstruction is only a candidate. Success requires sustained object lift
and stable object-in-EE pose; neither a contact-force measurement nor VLA use.
"""
import math


class GraspAbort(Exception):
    pass


def angle_deg(a, b):
    dot = abs(sum(x*y for x, y in zip(a, b)))
    denom = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if not math.isfinite(denom) or denom < 1e-9:
        raise GraspAbort("invalid quaternion")
    return math.degrees(2*math.acos(min(1.0, dot/denom)))


def down_quat_from_object(object_quat):
    """Yaw-aligned top-down EE quaternion: Rz(object yaw) * Rx(pi), xyzw."""
    if len(object_quat)!=4 or not all(math.isfinite(v) for v in object_quat):
        raise ValueError("expected finite xyzw object quaternion")
    length = math.sqrt(sum(v*v for v in object_quat))
    if length<1e-9:
        raise ValueError("zero object quaternion")
    x,y,z,w = [v/length for v in object_quat]
    yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return [math.cos(yaw/2), math.sin(yaw/2), 0.0, 0.0], yaw


def run_grasp_probe(read_state, step_pose, announce=print, orient_step_deg=0.5):
    if not math.isfinite(orient_step_deg) or not 0.1<=orient_step_deg<=0.5:
        raise ValueError("orientation step must be in [0.1, 0.5] degrees")
    trace, phases = [], []
    initial = read_state()
    cube = list(initial["object_pos_root_m"])
    down, object_yaw = down_quat_from_object(initial["object_quat_root"])
    grasp = [cube[0], cube[1], cube[2]+0.005]
    above = [cube[0], cube[1], cube[2]+0.10]
    report = {"success": False, "initial_state": initial, "phases": phases, "trace": trace,
              "grasp_target_root_m": grasp, "grasp_quat_root_xyzw": down,
              "measured_object_yaw_rad": object_yaw, "orient_step_deg": orient_step_deg,
              "thresholds": {"position_m": 0.002, "rotation_deg": 2.0,
                "max_object_speed_m_s": 0.5, "min_held_lift_m": 0.035,
                "max_hold_relative_drift_m": 0.01, "max_hold_relative_rotation_deg": 10,
                "hold_duration_s": 2.0},
              "scope": "explicit object spawn; simulator ground-truth object pose; yaw-aligned top-down grasp and +5mm TCP height hypothesis; not visual/VLA or contact-force validation"}

    def sample(phase, target, quat, gripper):
        step_pose(target, quat, gripper, phase)
        state = dict(read_state())
        for key in ("ee_pos_root_m", "ee_quat_root", "object_pos_root_m", "object_quat_root",
                    "finger_positions_m", "object_in_ee_pos_m", "object_in_ee_quat"):
            if not all(math.isfinite(v) for v in state[key]):
                raise GraspAbort(f"nonfinite state: {key}")
        previous = trace[-1] if trace else initial
        speed = math.dist(state["object_pos_root_m"], previous["object_pos_root_m"])/0.05
        state.update(phase=phase, time_s=(len(trace)+1)*0.05,
                     target_pos_root_m=list(target), target_quat_root=list(quat), gripper_action=gripper,
                     position_error_m=math.dist(state["ee_pos_root_m"], target),
                     rotation_error_deg=angle_deg(state["ee_quat_root"], quat),
                     object_speed_estimate_m_s=speed,
                     object_lift_m=state["object_pos_root_m"][2]-cube[2])
        trace.append(state)
        if speed > 0.5:
            raise GraspAbort("object motion exceeded 0.5 m/s control-sampled estimate")
        if state["object_lift_m"] < -0.01:
            raise GraspAbort("object fell below its initial support height")
        if phase in ("orient", "above", "descend", "close") and math.dist(state["object_pos_root_m"], cube) > 0.02:
            raise GraspAbort("object moved more than 2 cm before commanded lift")
        return state

    def move(label, target, quat, limit, gripper=1.0):
        begin = len(trace)
        for _ in range(limit):
            sample(label, target, quat, gripper)
            tail = trace[-5:]
            if len(trace)-begin >= 5 and all(s["position_error_m"] <= 0.002 and s["rotation_error_deg"] <= 2 for s in tail):
                phases.append({"phase": label, "reached": True, "steps": len(trace)-begin})
                announce(f"[Grasp] {label}: reached in {len(trace)-begin} steps")
                return
        raise GraspAbort(f"{label} target did not settle")

    try:
        if len(initial["finger_positions_m"]) != 2 or not all(abs(q-0.04) <= 0.002 for q in initial["finger_positions_m"]):
            raise GraspAbort("fingers are not initially open")
        if initial["ee_pos_root_m"][2] < cube[2]+0.08:
            raise GraspAbort("initial EE is too low for orientation change above the object")
        move("orient", initial["ee_pos_root_m"], down, math.ceil(260*0.5/orient_step_deg))
        move("above", above, down, 400)
        move("descend", grasp, down, 200)
        for _ in range(80):
            sample("close", grasp, down, -1.0)
        tail = trace[-10:]
        fingers = tail[-1]["finger_positions_m"]
        stable = all(max(s["finger_positions_m"][i] for s in tail)-min(s["finger_positions_m"][i] for s in tail) <= 0.0005 for i in (0, 1))
        obstruction = all(q > 0.002 for q in fingers) and 0.006 <= sum(fingers) <= 0.07 and abs(fingers[0]-fingers[1]) <= 0.004
        centered = math.dist(tail[-1]["ee_pos_root_m"], tail[-1]["object_pos_root_m"]) <= 0.015
        close_ok = stable and obstruction and centered and all(s["position_error_m"]<=0.002 and s["rotation_error_deg"]<=2 for s in tail)
        report["close_check"] = {"stable_fingers": stable, "obstruction_candidate": obstruction,
                                  "object_near_tcp": centered, "passed": close_ok, "finger_positions_m": fingers}
        phases.append({"phase": "close", **report["close_check"]})
        announce(f"[Grasp] close: candidate={close_ok}; fingers={fingers}")
        if not close_ok:
            raise GraspAbort("close candidate check failed; lift was not attempted")
        closed_relative = list(tail[-1]["object_in_ee_pos_m"])
        lift = list(tail[-1]["ee_pos_root_m"])
        lift[2] += 0.05
        move("lift", lift, down, 200, -1.0)
        for _ in range(40):
            sample("hold", lift, down, -1.0)
        held = trace[-40:]
        minimum_lift = min(s["object_lift_m"] for s in held)
        drift = max(math.dist(s["object_in_ee_pos_m"], held[0]["object_in_ee_pos_m"]) for s in held)
        rot_drift = max(angle_deg(s["object_in_ee_quat"], held[0]["object_in_ee_quat"]) for s in held)
        transfer_drift = max(math.dist(s["object_in_ee_pos_m"], closed_relative) for s in trace if s["phase"] in ("lift", "hold"))
        hold_ok = (minimum_lift >= 0.035 and drift <= 0.01 and rot_drift <= 10 and transfer_drift <= 0.015
                   and all(all(q>0.002 for q in s["finger_positions_m"]) for s in held)
                   and all(s["position_error_m"]<=0.002 and s["rotation_error_deg"]<=2 for s in held))
        report.update(success=hold_ok, hold_check={"min_object_lift_m": minimum_lift,
            "relative_drift_m": drift, "relative_rotation_drift_deg": rot_drift,
            "max_relative_transfer_drift_m": transfer_drift,
            "max_object_speed_m_s": max(s["object_speed_estimate_m_s"] for s in held)})
        if not hold_ok:
            report["failure_reason"] = "lift/hold did not meet sustained height and relative-pose checks"
    except GraspAbort as error:
        report["failure_reason"] = str(error)
    report["recorded_controls"] = len(trace)
    announce(f"[Grasp] success={report['success']}; reason={report.get('failure_reason', 'none')}")
    return report
