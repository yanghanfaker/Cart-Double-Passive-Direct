"""Query frozen saved OpenPI observations; never launch Isaac or execute actions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from openpi_step_probe import bounded_openpi_action, read_server_metadata, validate_action_chunk


def select_snapshots(manifest):
    runs=manifest.get("runs",[])
    if manifest.get("status")!="completed" or len(runs)!=3 or any(
        r.get("status")!="completed" or r.get("mode")!="openpi-step" for r in runs):
        raise ValueError("expected a completed three-run OpenPI result manifest")
    snapshots=[]
    for index,run in enumerate(runs):
        result=run["openpi_result"]; trace=result["trace"]
        if not trace or not result.get("stopped_by_monitor") or not run["openpi_setup"].get("extended_approach"):
            raise ValueError("expected three nonempty monitored long-approach runs")
        selected=[("initial",trace[0])] if index==0 else []
        selected.append((f"repeat{index+1}_last_before_action",trace[-1]))
        for label,row in selected:
            state=row.get("state_observed")
            if row.get("state_frame")!="world" or not isinstance(state,list) or len(state)!=8 or not all(
                isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in state):
                raise ValueError("expected a finite saved eight-dimensional world-frame state")
            images=row.get("image_files",[])
            if len(images)!=2 or any(Path(name).name!=name or "/" in name or "\\" in name or not name.endswith(".png") for name in images):
                raise ValueError("expected two safe saved policy image names")
            by_view={view:[name for name in images if name.endswith(f"_{view}.png")] for view in ("external","wrist")}
            if any(len(names)!=1 for names in by_view.values()):
                raise ValueError("need one external and one wrist image")
            obj=row.get("object_pos_world_m_for_diagnostic_only")
            if not isinstance(obj,list) or len(obj)!=3 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in obj):
                raise ValueError("expected diagnostic object position")
            snapshots.append({"name":label,"source_profile":run["profile"],"source_output":run["output"],
                "source_step":row["step"],"source_observation_used_for_inference":row["observation_used_for_inference"],
                "state":state,"image_files":{v:names[0] for v,names in by_view.items()},
                "object_world_m_diagnostic_only":obj,"prompt":run["openpi_setup"]["prompt"],
                "source_setup":run["openpi_setup"]})
    return snapshots


def analyze_chunk(chunk,state,object_pos):
    validate_action_chunk(chunk)
    if len(chunk)<5:
        raise ValueError("chunk must contain at least five actions")
    direction=[o-p for o,p in zip(object_pos,state[:3])]
    distance=math.hypot(*direction)
    if not math.isfinite(distance) or distance==0:
        raise ValueError("invalid diagnostic distance")
    unit=[v/distance for v in direction]
    mappings=[bounded_openpi_action(action) for action in chunk]
    def aggregate(items):
        return {"count":len(items),
            "decoded_translation_sum_mm":[1000*sum(m["delta_world_m"][i] for m in items) for i in range(3)],
            "bounded_translation_sum_mm":[1000*sum(m["bounded_delta_world_m"][i] for m in items) for i in range(3)],
            "decoded_local_approach_projection_sum_mm":1000*sum(sum(v*u for v,u in zip(m["delta_world_m"],unit)) for m in items),
            "bounded_local_approach_projection_sum_mm":1000*sum(sum(v*u for v,u in zip(m["bounded_delta_world_m"],unit)) for m in items),
            "translation_clamped":sum(m["delta_world_m_norm_scale"]<1 for m in items),
            "rotation_clamped":sum(m["delta_world_rad_norm_scale"]<1 for m in items),
            "close_requests":sum(m["requested_isaac_gripper"]<0 for m in items)}
    details=[]
    for index,mapped in enumerate(mappings):
        details.append({"action_index":index,"raw_action":chunk[index],"hypothetical_bounded_mapping":mapped,
            "decoded_translation_norm_mm":1000*math.hypot(*mapped["delta_world_m"]),
            "decoded_rotation_norm_deg":math.degrees(math.hypot(*mapped["delta_world_rad"])),
            "translation_retained_fraction":mapped["delta_world_m_norm_scale"],
            "rotation_retained_fraction":mapped["delta_world_rad_norm_scale"],
            "executed":False})
    return {"raw_action_chunk":chunk,"actions":details,"first5":aggregate(mappings[:5]),
        "whole_chunk":aggregate(mappings),"snapshot_distance_mm":distance*1000,
        "projection_note":"positive means towards object; fixed snapshot direction only, not a rollout or IK target; object truth is never policy input",
        "mapping_note":"reuse existing decoder and 1 mm/0.25 deg caps for numerical comparison only; no action including gripper is executed"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest",type=Path,default=Path("/home/test/Robot/IsaacLab/logs/openpi_gain_sweep/latest_gain_sweep.json"))
    parser.add_argument("--output-root",type=Path,default=Path(__file__).resolve().parent.parent/"diagnostics/openpi_frozen_observation")
    parser.add_argument("--repeats",type=int,choices=range(1,6),default=3)
    parser.add_argument("--server-host",default="127.0.0.1")
    parser.add_argument("--server-port",type=int,default=8000)
    parser.add_argument("--policy-server-log",type=Path)
    args=parser.parse_args()
    source=args.latest.expanduser().resolve(); source_bytes=source.read_bytes()
    manifest=json.loads(source_bytes); snapshots=select_snapshots(manifest)
    if args.policy_server_log and not args.policy_server_log.is_file():
        parser.error("server startup log not found")
    # These imports are intentionally delayed: pure validation tests need no Isaac, numpy or client.
    import numpy as np
    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    folder=args.output_root.expanduser().resolve()/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True,exist_ok=False)
    prepared=[]
    for snapshot in snapshots:
        item=dict(snapshot); item["image_files"]={}; images={}; hashes={}
        for view,name in snapshot["image_files"].items():
            path=Path(snapshot["source_output"])/name
            payload=path.read_bytes()
            with Image.open(path) as picture:
                image=np.asarray(picture).copy()
            if image.dtype!=np.uint8 or image.shape!=(224,224,3):
                raise ValueError("saved input must already be 224x224 uint8 RGB; no resize/flip/conversion is applied")
            dest=f'{snapshot["name"]}_{view}.png'; (folder/dest).write_bytes(payload)
            images[view]=image; hashes[view]=hashlib.sha256(payload).hexdigest()
            item["image_files"][view]=dest
        item["image_sha256"]=hashes; item["queries"]=[]
        prepared.append((item,images))
    report={"status":"running","actions_executed":0,"source_manifest":str(source),
        "source_manifest_sha256":hashlib.sha256(source_bytes).hexdigest(),
        "source_scope":"three saved long-approach runs; last snapshots precede their last executed action and may originally have been cached-step diagnostics",
        "server_host":args.server_host,"server_port":args.server_port,"repeats_per_snapshot":args.repeats,
        "input_semantics":"identical images, eight-dimensional world-frame state and original prompt within each snapshot; no simulation steps, state changes or gripper commands",
        "geometry_status":{"isaac_tcp_offsets":"copied from each source_setup; offset difference does not establish training TCP alignment",
            "libero_training_tcp_registration":"unverified; no reference geometry supplied",
            "actual_camera_extrinsics_and_intrinsics":"not recorded in source manifest; cannot calibrate from these saved images alone"},
        "snapshots":[item for item,_ in prepared]}
    if args.policy_server_log:
        payload=args.policy_server_log.read_bytes()[:128*1024]
        (folder/"policy_server_startup.log").write_bytes(payload)
        report["policy_server_log"]={"sha256":hashlib.sha256(payload).hexdigest(),"captured_bytes":len(payload),
            "truncated":args.policy_server_log.stat().st_size>len(payload),
            "scope":"user-supplied prefix; not independent endpoint ownership proof"}
    def save():
        (folder/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    save()
    try:
        client=WebsocketClientPolicy(host=args.server_host,port=args.server_port)
        report["server_metadata"]=read_server_metadata(client)
        for item,images in prepared:
            state=np.asarray(item["state"],dtype=np.float32)
            for repeat in range(args.repeats):
                observation={"observation/state":state.copy(),"observation/image":images["external"].copy(),
                    "observation/wrist_image":images["wrist"].copy(),"prompt":item["prompt"]}
                digest=lambda:hashlib.sha256(observation["observation/state"].tobytes()+observation["observation/image"].tobytes()+observation["observation/wrist_image"].tobytes()+observation["prompt"].encode()).hexdigest()
                before=digest(); started=time.perf_counter(); response=client.infer(observation)
                elapsed=time.perf_counter()-started
                if digest()!=before:
                    raise RuntimeError("client mutated frozen input arrays or prompt")
                chunk=response["actions"]
                if hasattr(chunk,"tolist"):chunk=chunk.tolist()
                query={"repeat":repeat+1,"input_sha256":before,"inference_wall_s":elapsed,
                    **analyze_chunk(chunk,item["state"],item["object_world_m_diagnostic_only"])}
                item["queries"].append(query);save()
                first=query["first5"]
                print(f'[Frozen query] {item["name"]} repeat={repeat+1}; first5 projected approach={first["bounded_local_approach_projection_sum_mm"]:.3f} mm; bounded Z={first["bounded_translation_sum_mm"][2]:.3f} mm; close={first["close_requests"]}/5; executed=0',flush=True)
        report["status"]="completed"
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error))
        raise
    finally:
        save(); print(f"[Frozen query] Report: {folder/'report.json'}",flush=True)


if __name__=="__main__":
    main()
