"""Run isolated Isaac Lab actuator comparisons from ordinary system Python."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from gain_probe import GAIN_PROFILES
from gain_cases import validate_gain_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/home/test/Robot/IsaacLab"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profiles", choices=tuple(GAIN_PROFILES), nargs="+")
    selection.add_argument("--gain-cases-json", type=Path, help="named base profiles and per-joint overrides")
    parser.add_argument("--record-physics", action="store_true", help="capture each existing physics substep")
    parser.add_argument("--joint-gains-json", type=Path, help="optional per-joint gains, applied after each profile")
    parser.add_argument("--modes", choices=("step-response", "libero-stream", "joint-step"), nargs="+",
                        default=["step-response", "libero-stream"])
    args = parser.parse_args()
    if "joint-step" in args.modes:
        args.record_physics = True
    joint_overrides = (
        json.loads(args.joint_gains_json.expanduser().resolve().read_text(encoding="utf-8"))
        if args.joint_gains_json else None
    )
    if args.gain_cases_json and args.joint_gains_json:
        parser.error("--gain-cases-json already defines overrides; omit --joint-gains-json")
    try:
        cases = validate_gain_cases(
            json.loads(args.gain_cases_json.read_text(encoding="utf-8")) if args.gain_cases_json else
            [{"name": profile, "profile": profile, "joint_overrides": joint_overrides}
             for profile in (args.profiles or ["baseline", "kp1p5", "kp1p5-damped"])])
    except ValueError as error:
        parser.error(str(error))
    root = args.isaaclab_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Isaac Lab directory not found: {root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "logs" / "openpi_gain_sweep" / stamp
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"output": str(output), "status": "running", "runs": [],
        "profiles": [case["name"] for case in cases], "cases": cases,
        "modes": args.modes, "record_physics": args.record_physics,
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
                command = ["uv", "run", "--extra", "viser", "isaaclab", "-p", str(script),
                           "--viz", "viser", "--motion-test", mode, "--gain-profile", case["profile"],
                           "--output", str(run_output)]
                if args.record_physics:
                    command.append("--record-physics")
                if case["joint_overrides"]:
                    command.extend(["--joint-gains-json", str(override_path)])
                entry = {"profile": profile, "base_profile": case["profile"],
                         "joint_overrides": case["joint_overrides"], "command": command,
                         "mode": mode, "output": str(run_output), "status": "running"}
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
                            if any(tag in line for tag in ("[Actuator audit]", "[Step response]", "[LIBERO stream]", "[Physics trace]", "[Joint step]")):
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
                else:
                    entry["all_stream_tests_passed"] = result["all_stream_tests_passed"]
                    entry["axes"] = [{k: v for k, v in axis.items() if k != "trace"} for axis in result["axes"]]
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
