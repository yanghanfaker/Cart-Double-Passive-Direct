"""Controller-domain action checks for an explicitly bounded model-motion probe."""
import math
import time

from libero_mapping import decode_libero_action


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


def preview_images(trace):
    if not trace:
        return []
    return list(dict.fromkeys(name for row in (trace[0],trace[-1]) for name in row["image_files"]))


def run_openpi_steps(observe, infer, execute, steps=20, announce=print):
    if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=20:
        raise ValueError("model-motion probe requires 1 to 20 steps")
    trace=[]
    report={"probe_completed":False,"requested_steps":steps,"completed_steps":0,"trace":trace,
            "scope":"one first action per fresh inference, capped at 1 mm/0.25 deg, gripper forced open; not grasp or real-time 20 Hz validation"}
    try:
        for index in range(steps):
            obs,metadata=observe(index)
            row=dict(metadata,step=index+1,executed=False,execution_attempted=False)
            trace.append(row)
            started=time.perf_counter()
            chunk=infer(obs)["actions"]
            row["inference_wall_s"]=time.perf_counter()-started
            if hasattr(chunk,"tolist"):
                chunk=chunk.tolist()
            row["raw_action_chunk"]=validate_action_chunk(chunk)
            bounded=bounded_openpi_action(chunk[0])
            row["action_mapping"]=bounded
            row["execution_attempted"]=True
            row.update(execute(bounded,index))
            row["executed"]=True
            report["completed_steps"]+=1
            announce(f"[OpenPI step] {index+1}/{steps}: infer={row['inference_wall_s']:.3f}s; "
                     f"model_gripper={chunk[0][6]:.4f}; applied_gripper=open")
        report["probe_completed"]=True
    except Exception as error:
        report["failure_reason"]=str(error)
        report["error_type"]=type(error).__name__
        announce(f"[OpenPI step] stopped: {type(error).__name__}: {error}")
    report["preview_images"]=preview_images(trace)
    return report
