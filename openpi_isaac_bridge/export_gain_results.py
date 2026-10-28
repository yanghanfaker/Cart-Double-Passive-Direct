"""Prepare completed gain results for git; does not commit or push anything."""

import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest", type=Path, default=Path("/home/test/Robot/IsaacLab/logs/openpi_gain_sweep/latest_gain_sweep.json"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--include-physics-traces", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.latest.read_text(encoding="utf-8"))
    runs = data["runs"]
    modes = data.get("modes", ["step-response", "libero-stream"])
    if not modes or set(modes) - {"step-response", "libero-stream", "joint-step"}:
        raise SystemExit("Unknown test mode; no files exported")
    expected = {(profile, mode) for profile in data["profiles"] for mode in modes}
    if (data["status"] != "completed" or len(runs) != len(expected)
            or {(run["profile"], run["mode"]) for run in runs} != expected
            or any(run["status"] != "completed" for run in runs)):
        raise SystemExit("Incomplete or inconsistent gain sweep; no files exported")
    files = {"summary.json": args.latest.read_bytes()}
    for run in runs:
        folder = Path(run["output"])
        prefix = f'{run["profile"]}_{run["mode"]}'
        if Path(prefix).name != prefix or "/" in prefix or "\\" in prefix:
            raise ValueError("unsafe profile/mode in result manifest")
        files[prefix + "_actuators.json"] = (folder / "actuator_audit.json").read_bytes()
        if data.get("record_physics"):
            files[prefix + "_physics_joint_summary.json"] = (folder / "physics_joint_summary.json").read_bytes()
            if args.include_physics_traces:
                files[prefix + "_physics_joint_trace.jsonl.gz"] = gzip.compress(
                    (folder / "physics_joint_trace.jsonl").read_bytes(), mtime=0)
        elif args.include_physics_traces:
            raise SystemExit("This sweep did not enable physics tracing; no files exported")
    # Read all required sources first, so missing artifacts cannot leave a
    # partially exported new summary alongside stale artifacts.
    destination = args.repo_root.resolve() / "diagnostics" / "gain_sweep"
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (destination / name).write_bytes(content)
    print(f"Exported {len(files)} files to {destination}; review and git add/commit/push separately")


if __name__ == "__main__":
    main()
