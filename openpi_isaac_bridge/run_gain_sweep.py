"""Run isolated Isaac Lab actuator comparisons from ordinary system Python."""

import argparse
import hashlib
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from replay_probe import REPLAY_MODES, load_replay, replay_plan
from gain_probe import GAIN_PROFILES
from gain_cases import validate_gain_cases
from joint_identification import screen_joint_steps
from stream_screen import screen_cartesian_stream
from stream_scenarios import expand_stream_cases
from grasp_scenarios import expand_grasp_cases



def expand_openpi_replan_cases(cases, lengths):
    if not lengths or len(set(lengths))!=len(lengths) or any(isinstance(n,bool) or not isinstance(n,int) or n not in (1,5) for n in lengths):
        raise ValueError("choose unique replan lengths from 1 and 5")
    result=[dict(case,name=case["name"]+f"-chunk{n}",base_case=case["name"],openpi_condition={"replan_steps":n})
            for case in cases for n in lengths]
    if any(len(c["name"])>80 for c in result) or len({c["name"] for c in result})!=len(result):
        raise ValueError("expanded OpenPI case names too long or collide")
    return result


def capture_policy_server_log(source, folder):
    limit=128*1024
    with source.open("rb") as stream:
        content=stream.read(limit+1)
    prefix=content[:limit]
    (folder/"policy_server_startup.log").write_bytes(prefix)
    return {"source_path":str(source),"captured_bytes":len(prefix),"truncated":len(content)>limit,
            "sha256":hashlib.sha256(prefix).hexdigest(),"capture_time_utc":datetime.now(timezone.utc).isoformat(),
            "scope":"user-supplied log prefix; not independent proof that this process owns the policy endpoint"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/test/Robot/IsaacLab"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profiles", choices=tuple(GAIN_PROFILES), nargs="+")
    selection.add_argument("--gain-cases-json", type=Path, help="named base profiles and per-joint overrides")
    parser.add_argument("--record-jacobians", action="store_true", help="read-only actual IK Jacobian attribution for frozen replay")
    parser.add_argument("--record-physics", action="store_true", help="capture each existing physics substep")
    parser.add_argument("--joint-gains-json", type=Path, help="optional per-joint gains, applied after each profile")
    parser.add_argument("--stream-scenarios-json", type=Path, help="cross gain cases with bounded stream start positions/amplitudes")
    parser.add_argument("--grasp-scenarios-json", type=Path, help="cross gain cases with explicit object poses, seeds and orientation rates")
    parser.add_argument("--replay-actions-json", type=Path, default=Path(__file__).with_name("openpi_replay_actions.json"))
    parser.add_argument("--openpi-approach-long", action="store_true", help="explicit 400-control approach budget")
    parser.add_argument("--openpi-approach", action="store_true", help="up to 60 controls with segmented progress stops")
    parser.add_argument("--openpi-replan-steps", type=int, choices=(1,5), nargs="+", help="expand candidate into chunk-length comparisons")
    parser.add_argument("--policy-server-log", type=Path, help="optional server startup log prefix to preserve with results")
    parser.add_argument("--openpi-steps", type=int, default=20)
    parser.add_argument("--openpi-state-frame", choices=("world","root"), default="world")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube")
    parser.add_argument("--modes", choices=("step-response", "libero-stream", "joint-step", "grasp", "openpi-step", *REPLAY_MODES), nargs="+",
                        default=["step-response", "libero-stream"])
    args = parser.parse_args()
    if args.openpi_approach_long and not args.openpi_approach:
        parser.error("--openpi-approach-long requires --openpi-approach")
    if args.openpi_approach and (args.modes != ["openpi-step"] or args.openpi_replan_steps != [5]):
        parser.error("--openpi-approach requires --modes openpi-step --openpi-replan-steps 5")
    if (args.openpi_replan_steps or args.policy_server_log) and args.modes != ["openpi-step"]:
        parser.error("OpenPI chunk/log options require --modes openpi-step only")
    if args.policy_server_log:
        args.policy_server_log=args.policy_server_log.expanduser().resolve()
        if not args.policy_server_log.is_file():
            parser.error("server log not found; start the server with tee before running the probe")
    if args.record_jacobians:
        if any(mode not in REPLAY_MODES for mode in args.modes):
            parser.error("--record-jacobians requires frozen replay modes only")
        args.record_physics = True
    if not 1<=args.openpi_steps<=(400 if args.openpi_approach_long else 60 if args.openpi_approach else 20):
        parser.error("--openpi-steps must be 1..20, 1..60 with approach, or 1..400 with explicit long approach")
    if any(mode in args.modes for mode in ("joint-step", "grasp", "openpi-step", *REPLAY_MODES)):
        args.record_physics = True
    joint_overrides = (
        json.loads(args.joint_gains_json.expanduser().resolve().read_text(encoding="utf-8"))
        if args.joint_gains_json else None
    )
    if args.gain_cases_json and args.joint_gains_json:
        parser.error("--gain-cases-json already defines overrides; omit --joint-gains-json")
    if args.stream_scenarios_json and args.grasp_scenarios_json:
        parser.error("choose stream or grasp scenarios, not both")
    try:
        cases = validate_gain_cases(
            json.loads(args.gain_cases_json.read_text(encoding="utf-8")) if args.gain_cases_json else
            [{"name": profile, "profile": profile, "joint_overrides": joint_overrides}
             for profile in (args.profiles or ["baseline", "kp1p5", "kp1p5-damped"])])
        if args.stream_scenarios_json:
            if args.modes != ["libero-stream"]:
                parser.error("--stream-scenarios-json requires --modes libero-stream only")
            cases = expand_stream_cases(cases, json.loads(args.stream_scenarios_json.read_text(encoding="utf-8")))
        if args.grasp_scenarios_json:
            if args.modes != ["grasp"]:
                parser.error("--grasp-scenarios-json requires --modes grasp only")
            cases = expand_grasp_cases(cases, json.loads(args.grasp_scenarios_json.read_text(encoding="utf-8")))
    except ValueError as error:
        parser.error(str(error))
    if args.openpi_replan_steps:
        try:
            cases=expand_openpi_replan_cases(cases,args.openpi_replan_steps)
        except ValueError as error:
            parser.error(str(error))
    if args.openpi_approach:
        cases=[dict(case,name=case["name"]+("-approach-long" if args.openpi_approach_long else "-approach"),
                    openpi_condition={**case["openpi_condition"],"segmented_approach":True,"extended_approach":args.openpi_approach_long}) for case in cases]
        if any(len(case["name"])>80 for case in cases):
            parser.error("approach case name exceeds 80 characters")
    if any(mode in REPLAY_MODES for mode in args.modes):
        replay_data=load_replay(args.replay_actions_json)
        for mode in args.modes:
            if mode in REPLAY_MODES:
                replay_plan(replay_data,mode)
    root = args.isaaclab_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Isaac Lab directory not found: {root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "logs" / "openpi_gain_sweep" / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"output": str(output), "status": "running", "runs": [],
        "profiles": [case["name"] for case in cases], "cases": cases,
        "modes": args.modes, "record_physics": args.record_physics, "record_jacobians": args.record_jacobians,"openpi_approach":args.openpi_approach,"openpi_approach_long":args.openpi_approach_long,
        "joint_overrides": joint_overrides,
        "scope": "selected actuator gain profiles; no automatic selection or deployment; effort telemetry is approximate"}
    def save():
        (output / "sweep_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output.parent / "latest_gain_sweep.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save()
    script = Path(__file__).with_name("check_cameras.py").resolve()
    try:
        for case in cases:
            profile = case["name"]
            override_path = output / f"{profile}_joint_gains.json"
            if case["joint_overrides"]:
                override_path.write_text(json.dumps(case["joint_overrides"], indent=2), encoding="utf-8")
            for mode in args.modes:
                run_output = output / profile / mode
                run_output.mkdir(parents=True)
                extras=["--extra","viser"] + (["--extra","openpi"] if mode=="openpi-step" else [])
                command = ["uv", "run", *extras, "isaaclab", "-p", str(script),
                           "--viz", "viser", "--motion-test", mode, "--gain-profile", case["profile"],
                           "--output", str(run_output)]
                if args.record_physics:
                    command.append("--record-physics")
                if args.record_jacobians:
                    command.append("--record-jacobians")
                if mode=="openpi-step":
                    command.extend(["--server-host",args.server_host,"--server-port",str(args.server_port),
                                    "--prompt",args.prompt,"--openpi-steps",str(args.openpi_steps),
                                    "--openpi-state-frame",args.openpi_state_frame,
                                    "--openpi-replan-steps",str(case.get("openpi_condition",{}).get("replan_steps",1))])
                if args.openpi_approach:
                    command.append("--openpi-approach")
                if args.openpi_approach_long:
                    command.append("--openpi-approach-long")
                if mode in REPLAY_MODES:
                    command.extend(["--replay-actions-json",str(args.replay_actions_json.resolve())])
                if case["joint_overrides"]:
                    command.extend(["--joint-gains-json", str(override_path)])
                if case.get("stream_scenario"):
                    scenario = case["stream_scenario"]
                    command.extend(["--stream-start-offset", *map(str, scenario["start_offset_m"]),
                                    "--stream-amplitude-scale", str(scenario["amplitude_scale"])])
                if case.get("grasp_scenario"):
                    scenario = case["grasp_scenario"]
                    command.extend(["--grasp-object-offset", *map(str, scenario["object_offset_xy_m"]),
                                    "--grasp-object-yaw", str(scenario["object_yaw_deg"]),
                                    "--grasp-orient-step-deg", str(scenario["orient_step_deg"]),
                                    "--grasp-seed", str(scenario["seed"])])
                entry = {"profile": profile, "base_profile": case["profile"],
                         "joint_overrides": case["joint_overrides"], "command": command,
                         "mode": mode, "output": str(run_output), "status": "running"}
                if case.get("stream_scenario"):
                    entry.update(base_case=case["base_case"], stream_scenario=case["stream_scenario"])
                if case.get("grasp_scenario"):
                    entry.update(base_case=case["base_case"], grasp_scenario=case["grasp_scenario"])
                if case.get("openpi_condition"):
                    entry.update(base_case=case["base_case"],openpi_condition=case["openpi_condition"])
                manifest["runs"].append(entry)
                save()
                print(f"[Gain sweep] START {profile}/{mode}; log={run_output / 'console.log'}", flush=True)
                recent = deque(maxlen=60)
                with (run_output / "console.log").open("w", encoding="utf-8", buffering=1) as log:
                    with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, encoding="utf-8", errors="replace") as process:
                        for line in process.stdout:
                            log.write(line)
                            recent.append(line)
                            if any(tag in line for tag in ("[Actuator audit]", "[Step response]", "[LIBERO stream]", "[Physics trace]", "[Joint step]", "[Stream start]", "[Grasp]", "[OpenPI step]", "[OpenPI replay]", "[Jacobian bodies]", "[OpenPI segment]")):
                                print(line, end="", flush=True)
                        code = process.wait()
                entry["exit_code"] = code
                report_file = run_output / f"motion_{mode}.json"
                if code != 0 or not report_file.is_file():
                    entry["status"] = "failed"
                    manifest["status"] = "failed"
                    save()
                    print("".join(recent), file=sys.stderr)
                    print(f"[Gain sweep] STOP: inspect {run_output / 'console.log'}", file=sys.stderr)
                    return 1
                if args.policy_server_log:
                    entry["policy_server_log"]=capture_policy_server_log(args.policy_server_log,run_output)
                result = json.loads(report_file.read_text(encoding="utf-8"))
                entry["status"] = "completed"
                entry["joint_telemetry_summary"] = result["joint_telemetry_summary"]
                entry["actuator_audit"] = result["actuator_audit"]
                if args.record_physics:
                    entry["physics_joint_summary"] = result["physics_joint_summary"]
                if mode == "step-response":
                    entry["responses"] = [{key: phase[key] for key in (
                        "feedback", "axis", "gain_at_50ms", "within_10pct_at_50ms", "within_5pct_at_50ms",
                        "final_gain", "peak_gain", "overshoot_pct", "tail_gain_peak_to_peak",
                        "t90_sustained_s", "settling_5pct_s", "final_position_error_m", "final_rotation_error_deg",
                    )} for phase in result["phases"]]
                elif mode == "joint-step":
                    entry["joint_steps"] = result["joint_steps"]
                    entry["joint_step_screen"] = screen_joint_steps(result["joint_steps"])
                    print(f"[Joint screen] {profile}: complete={entry['joint_step_screen']['complete']}, "
                          f"all_passed={entry['joint_step_screen']['all_passed']}", flush=True)
                elif mode == "grasp":
                    entry["grasp_result"] = result["grasp_result"]
                    entry["grasp_setup"] = result["grasp_setup"]
                elif mode in REPLAY_MODES:
                    entry["replay_result"] = result["replay_result"]
                    entry["replay_setup"] = result["replay_setup"]
                elif mode == "openpi-step":
                    entry["openpi_result"] = result["openpi_result"]
                    entry["openpi_setup"] = result["openpi_setup"]
                else:
                    entry["all_stream_tests_passed"] = result["all_stream_tests_passed"]
                    # Preserve pose/target samples for diagnosing axis coupling
                    # and per-period errors after exporting just the summary.
                    entry["axes"] = result["axes"]
                    entry["stream_thresholds"] = result["thresholds"]
                    entry["stream_target_semantics"] = result["target_semantics"]
                    entry["stream_condition"] = {k: result[k] for k in ("stream_amplitude_scale", "stream_start") if k in result}
                    entry["cartesian_stream_screen"] = screen_cartesian_stream(result["axes"])
                    print(f"[Cartesian screen] {profile}: complete={entry['cartesian_stream_screen']['complete']}, "
                          f"all_passed={entry['cartesian_stream_screen']['all_passed']}", flush=True)
                save()
                print(f"[Gain sweep] DONE {profile}/{mode}", flush=True)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        save()
        raise
    manifest["status"] = "completed"
    save()
    print(f"[Gain sweep] Summary: {output / 'sweep_summary.json'}", flush=True)
    print(f"[Gain sweep] Latest summary: {output.parent / 'latest_gain_sweep.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
