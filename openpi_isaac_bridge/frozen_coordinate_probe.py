"""Frozen A/B/C state-coordinate sensitivity test; no simulation or action execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from frozen_observation_probe import analyze_chunk
from openpi_step_probe import read_server_metadata


CONDITIONS = ("A_original", "B_world_candidate", "C_world_and_tcp_candidate")
TRANSLATION = (-.66, 0., .912)
TCP_LOCAL_DELTA = (0., 0., -.0064)


def rotate_vector(rotvec, vector):
    """Rodrigues rotation; accepts signed/noncanonical axis-angle observations."""
    angle = math.hypot(*rotvec)
    if angle < 1e-12:
        return list(vector)
    axis = [v / angle for v in rotvec]
    cross = [axis[1]*vector[2]-axis[2]*vector[1],
             axis[2]*vector[0]-axis[0]*vector[2],
             axis[0]*vector[1]-axis[1]*vector[0]]
    dot = sum(a*b for a, b in zip(axis, vector))
    return [vector[i]*math.cos(angle)+cross[i]*math.sin(angle)+axis[i]*dot*(1-math.cos(angle)) for i in range(3)]


def state_condition(state, obj, condition):
    if condition not in CONDITIONS:
        raise ValueError("unknown condition")
    if len(state) != 8 or len(obj) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in [*state, *obj]):
        raise ValueError("expected finite eight-dimensional state and diagnostic object xyz")
    result = list(state)
    target = list(obj)
    if condition != CONDITIONS[0]:
        result[:3] = [v+t for v, t in zip(result[:3], TRANSLATION)]
        target = [v+t for v, t in zip(target, TRANSLATION)]
    if condition == CONDITIONS[2]:
        # Saved orientation describes the hand; source sensor offset has identity rotation.
        delta = rotate_vector(state[3:6], TCP_LOCAL_DELTA)
        result[:3] = [v+d for v, d in zip(result[:3], delta)]
    return result, target


def validate_source(source):
    if source.get("status") != "completed" or source.get("actions_executed") != 0 or len(source.get("snapshots", [])) != 4:
        raise ValueError("need completed four-snapshot frozen report with zero executed actions")
    names = set()
    for item in source["snapshots"]:
        name = item["name"]
        if name in names or not name or Path(name).name != name or any(c in name for c in "/\\"):
            raise ValueError("invalid or duplicate snapshot name")
        names.add(name)
        state_condition(item["state"], item["object_world_m_diagnostic_only"], CONDITIONS[2])
        setup = item["source_setup"]
        # The derivation requires this specific source point and orientation convention.
        if not isinstance(item["prompt"], str):
            raise ValueError("prompt must be text")
        for view in ("external", "wrist"):
            file = item["image_files"][view]
            if Path(file).name != file or any(c in file for c in "/\\") or not file.endswith(".png"):
                raise ValueError("unsafe image filename")
        # Retain setup verbatim for provenance; exact offset verified separately below.
        if not isinstance(setup, dict):
            raise ValueError("source setup missing")
        if setup.get("state_frame") != "world" or setup.get("ee_sensor_offset_m") != [0., 0., .1034]:
            raise ValueError("candidate requires original world-frame 103.4 mm sensor")
        if setup.get("rotation_vector_convention") != "official LIBERO _quat2axisangle; xyzw; no w>=0 sign canonicalization":
            raise ValueError("unexpected orientation encoding")
        base = setup.get("robot_root_pos_world_m", [])
        quat = setup.get("robot_root_quat_world", [])
        if len(base) != 3 or len(quat) != 4 or any(not math.isfinite(v) for v in base+quat) or math.hypot(*base) > 1e-5 or math.hypot(*quat[:3]) > 1e-5 or abs(abs(quat[3])-1) > 1e-5:
            raise ValueError("candidate requires captured near-identity Isaac base")


def digest_observation(obs):
    return hashlib.sha256(obs["observation/state"].tobytes()+obs["observation/image"].tobytes()+obs["observation/wrist_image"].tobytes()+obs["prompt"].encode()).hexdigest()


def query_snapshot(client, item, images, repeats, np, save):
    """Interleave A/B/C with rotating order; repeats characterize unpaired sampling noise."""
    records = []
    for repeat in range(repeats):
        order = CONDITIONS[repeat % 3:] + CONDITIONS[:repeat % 3]
        for condition in order:
            state, obj = state_condition(item["state"], item["object_world_m_diagnostic_only"], condition)
            obs = {"observation/state": np.asarray(state, dtype=np.float32),
                   "observation/image": images["external"].copy(),
                   "observation/wrist_image": images["wrist"].copy(), "prompt": item["prompt"]}
            before = digest_observation(obs)
            started = time.perf_counter()
            response = client.infer(obs)
            if digest_observation(obs) != before:
                raise RuntimeError("client mutated frozen input")
            chunk = response["actions"]
            if hasattr(chunk, "tolist"):
                chunk = chunk.tolist()
            row = {"repeat": repeat+1, "condition": condition, "input_sha256": before,
                   "state_sent_float32": obs["observation/state"].tolist(),
                   "inference_wall_s": time.perf_counter()-started,
                   **analyze_chunk(chunk, state, obj)}
            records.append(row)
            save(row)
            print(f'[Coordinate query] {item["name"]} {condition} repeat={repeat+1}; first5={row["first5"]["bounded_local_approach_projection_sum_mm"]:.3f} mm; whole={row["whole_chunk"]["bounded_local_approach_projection_sum_mm"]:.3f} mm; executed=0', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output-root", type=Path, default=root/"diagnostics/openpi_coordinate_probe")
    parser.add_argument("--repeats", type=int, choices=range(3, 10), default=5)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    args = parser.parse_args()
    source = args.source_report
    if source is None:
        paths = sorted((root/"diagnostics/openpi_frozen_observation").glob("*/report.json"))
        completed = [p for p in paths if json.loads(p.read_text())["status"] == "completed"]
        if not completed:
            parser.error("no completed frozen report; supply --source-report")
        source = completed[-1]
    source = source.resolve()
    payload = source.read_bytes()
    original = json.loads(payload)
    validate_source(original)
    import numpy as np
    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    folder = args.output_root.resolve()/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "actions_executed": 0, "policy_queries": 0,
              "source_report": str(source), "source_report_sha256": hashlib.sha256(payload).hexdigest(),
              "candidate": {"rotation_L_from_I": [[1,0,0],[0,1,0],[0,0,1]],
                  "translation_L_from_I_m": TRANSLATION, "tcp_hand_local_delta_m": TCP_LOCAL_DELTA,
                  "source_sensor_offset_m": .1034, "reference_site_offset_m": .097,
                  "orientation": "preserve hand rotation-vector values, not grip_site orientation",
                  "evidence_commit": "1f301b39520b8a751bfcfa5c3479303da1ef7d91",
                  "status": "single-pose candidate only; no multi-pose calibration or camera correction"},
              "scope": "input sensitivity only; unpaired stochastic queries; not rollout, grasp success or proof of alignment",
              "server_host": args.server_host, "server_port": args.server_port,
              "repeats_per_condition": args.repeats, "snapshots": []}
    def save():
        (folder/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    save()
    try:
        # Prepare and verify ALL input files before connecting or querying.
        prepared = []
        for item in original["snapshots"]:
            images = {}
            for view, name in item["image_files"].items():
                content = (source.parent/name).read_bytes()
                if hashlib.sha256(content).hexdigest() != item["image_sha256"][view]:
                    raise ValueError("source image SHA mismatch")
                with Image.open(source.parent/name) as picture:
                    image = np.asarray(picture).copy()
                if image.dtype != np.uint8 or image.shape != (224,224,3):
                    raise ValueError("expected saved 224x224 uint8 RGB")
                (folder/name).write_bytes(content)
                images[view] = image
            copied = {k:v for k,v in item.items() if k != "queries"}
            copied["queries"] = []
            report["snapshots"].append(copied)
            prepared.append((copied, images))
        client = WebsocketClientPolicy(host=args.server_host, port=args.server_port)
        report["server_metadata"] = read_server_metadata(client)
        for item, images in prepared:
            def append(row):
                item["queries"].append(row)
                report["policy_queries"] += 1
                save()
            query_snapshot(client, item, images, args.repeats, np, append)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
        print(f'[Coordinate query] Report: {folder/"report.json"}', flush=True)


if __name__ == "__main__":
    main()
