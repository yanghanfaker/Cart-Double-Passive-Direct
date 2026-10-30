"""Controller-domain action checks for an explicitly bounded model-motion probe."""
import math
import time
import json

from libero_mapping import decode_libero_action


def read_server_metadata(client):
    """Read cached handshake diagnostics; never turn metadata into instructions."""
    note = "server supplied; does not independently verify checkpoint or normalization statistics"
    getter = getattr(client,"get_server_metadata",None)
    if not callable(getter):
        return {"status":"unsupported","note":note}
    try:
        metadata = getter()
        if not isinstance(metadata,dict):
            raise ValueError("server metadata must be a dictionary")
        metadata = json.loads(json.dumps(metadata,default=str,allow_nan=False))
        return {"status":"reported" if metadata else "empty","value":metadata,"note":note}
    except Exception as error:
        return {"status":"unavailable","error":str(error),"note":note}


def libero_state_vector(position, quat_xyzw, fingers):
    """Match the official LIBERO example's signed quaternion -> axis-angle.

    Do not canonicalize to w>=0. Input fingers are the two nonnegative Isaac
    Panda positions; robosuite 1.4 Panda's second joint uses the opposite sign.
    """
    if len(position)!=3 or len(quat_xyzw)!=4 or len(fingers)!=2 or not all(math.isfinite(v) for v in list(position)+list(quat_xyzw)+list(fingers)):
        raise ValueError("expected finite position3, xyzw quaternion4 and finger positions2")
    if abs(sum(v*v for v in quat_xyzw)-1)>0.001:
        raise ValueError("policy state quaternion must be approximately unit length")
    if any(q < -0.0001 or q > 0.041 for q in fingers):
        raise ValueError("expected Isaac Panda finger positions near [0, 0.04] metres")
    w=max(-1.0,min(1.0,quat_xyzw[3]))
    den=math.sqrt(1-w*w)
    rotation=[0.0]*3 if math.isclose(den,0.0) else [v*2*math.acos(w)/den for v in quat_xyzw[:3]]
    return list(position)+rotation+[fingers[0],-fingers[1]]


def validate_action_chunk(chunk):
    if not isinstance(chunk, list) or not chunk:
        raise ValueError("expected nonempty action chunk")
    if any(not isinstance(row, list) or len(row)!=7 or
           not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row)
           for row in chunk):
        raise ValueError("expected finite N by 7 controller-domain actions")
    return chunk


def bounded_openpi_action(action):
    decoded = decode_libero_action(action)
    result = dict(decoded)
    for field,limit in (("delta_world_m",0.001),("delta_world_rad",math.radians(0.25))):
        vector=decoded[field];length=math.hypot(*vector)
        factor=min(1.0,limit/length) if length else 1.0
        result["bounded_"+field]=[v*factor for v in vector]
        result[field+"_norm_scale"]=factor
    result["requested_isaac_gripper"]=decoded["isaac_gripper"]
    result["applied_isaac_gripper"]=1.0
    result["gripper_policy"]="forced open for first model-motion diagnostic"
    return result


def preview_images(trace, segment_steps=None):
    if not trace:
        return []
    selected=[trace[0]]+[r for r in trace if segment_steps and r["step"]%segment_steps==0]+[trace[-1]]
    return list(dict.fromkeys(name for row in selected for name in row["image_files"]))


class ApproachStopped(Exception):
    pass


def run_openpi_steps(observe, infer, execute, steps=20, announce=print, replan_steps=1, approach_monitor=None, extended_approach=False):
    if extended_approach and approach_monitor is None:
        raise ValueError("extended approach requires an explicit progress monitor")
    maximum=(400 if extended_approach else 60) if approach_monitor is not None else 20
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=maximum:
        raise ValueError(f"model-motion probe requires 1 to {maximum} steps")
    if isinstance(replan_steps,bool) or not isinstance(replan_steps,int) or replan_steps not in (1,5,10):
        raise ValueError("replan_steps must be 1, 5 or 10")
    if approach_monitor is not None and replan_steps not in (5,10):
        raise ValueError("segmented approach requires replan_steps=5 or 10")
    trace=[]
    report={"probe_completed":False,"requested_steps":steps,"completed_steps":0,"trace":trace,
            "replan_steps":replan_steps,"inference_requests":0,"validated_inferences":0,
            "scope":f"execute first {replan_steps} actions per fresh inference; capped at 1 mm/0.25 deg, gripper open; not grasp or real-time 20 Hz validation"}
    chunk=None
    try:
        if approach_monitor is not None and approach_monitor.stop_reason:
            raise ApproachStopped(approach_monitor.stop_reason)
        for index in range(steps):
            obs,metadata=observe(index)
            new_inference=index % replan_steps == 0
            row=dict(metadata,step=index+1,executed=False,execution_attempted=False,
                     inference_performed=new_inference,observation_used_for_inference=new_inference,
                     chunk_action_index=index % replan_steps,chunk_observation_step=(index//replan_steps)*replan_steps+1)
            row["state_observed"]=row.get("state_sent")
            if not new_inference:
                row["state_sent"]=None
            trace.append(row)
            if new_inference:
                report["inference_requests"]+=1
                started=time.perf_counter()
                chunk=infer(obs)["actions"]
                row["inference_wall_s"]=time.perf_counter()-started
                if hasattr(chunk,"tolist"):
                    chunk=chunk.tolist()
                validate_action_chunk(chunk)
                if len(chunk)<replan_steps:
                    raise ValueError("returned action chunk is shorter than requested replan length")
                report["validated_inferences"]+=1
            else:
                row["inference_wall_s"]=None
            row["chunk_id"]=report["validated_inferences"]
            row["raw_action_chunk"]=chunk
            action=chunk[row["chunk_action_index"]]
            row["selected_raw_action"]=action
            bounded=bounded_openpi_action(action)
            row["action_mapping"]=bounded
            row["execution_attempted"]=True
            row.update(execute(bounded,index))
            row["executed"]=True
            report["completed_steps"]+=1
            timing=f"{row['inference_wall_s']:.3f}s" if new_inference else "cached"
            announce(f"[OpenPI step] {index+1}/{steps}: infer={timing}; "
                     f"chunk={row['chunk_id']} action={row['chunk_action_index']}; "
                     f"model_gripper={action[6]:.4f}; applied_gripper=open")
            if approach_monitor is not None:
                row["approach_diagnostic"]=approach_monitor.update(row)
                checkpoint=row["approach_diagnostic"]["checkpoint"]
                if checkpoint:
                    announce(f"[OpenPI segment] step={index+1}; distance={checkpoint['end_distance_m']*1000:.2f}mm; "
                             f"segment_progress={checkpoint['distance_reduction_m']*1000:.2f}mm")
                if approach_monitor.stop_reason:
                    raise ApproachStopped(approach_monitor.stop_reason)
        report["probe_completed"]=True
    except ApproachStopped as error:
        report.update(stopped_by_monitor=True,termination_reason=str(error),failure_reason=str(error))
        announce(f"[OpenPI segment] controlled stop after {report['completed_steps']} steps: {error}")
    except Exception as error:
        report["failure_reason"]=str(error)
        report["error_type"]=type(error).__name__
        announce(f"[OpenPI step] stopped: {type(error).__name__}: {error}")
    if approach_monitor is not None:
        report["approach_summary"]=approach_monitor.summary()
    report["preview_images"]=preview_images(trace,20 if approach_monitor is not None else None)
    return report
