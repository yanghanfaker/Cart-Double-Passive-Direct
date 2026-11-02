"""Static native FK at the stopped segment's joint pose; no action steps."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
from geometry_audit import pose_matrix
from libero_reference_audit import source_targets
from multipose_geometry import rotation_angle
from native_libero_rollout import save_json


def read(p):return json.loads(Path(p).read_bytes())
def sha(b):return hashlib.sha256(b).hexdigest()


def requested_geometry(g,native):
    result=copy.deepcopy(g);indices={n:i for i,n in enumerate(g['joint_names'])}
    for k,v in enumerate(native['robot0_joint_pos'],1):result['joint_positions_rad'][indices[f'panda_joint{k}']]=v
    fingers=native['robot0_gripper_qpos']
    result['finger_positions_isaac_m']=[fingers[0],-fingers[1]]
    source_targets(result)
    return result


def rotation(q):return np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':q}))[:3,:3]


def analyze(g,settings,ref):
    if ref['status']!='completed' or ref['sim_time_before_forward']!=ref['sim_time_after_capture']:
        raise ValueError('completed zero-time native FK capture required')
    native=settings['native_initial'];R=np.array(settings['calibration']['world_from_isaac_rotation']);t=np.array(settings['calibration']['world_from_isaac_translation_m'])
    hand=np.array(pose_matrix(g['sensor_body_world']));isaac=R@(hand[:3,3]+hand[:3,:3]@np.array([0,0,.097]))+t
    dynamic=np.array(native['robot0_eef_pos']);static=np.array(ref['observation']['robot0_eef_pos'])
    arm=[a for a in ref['joint_assignments'] if 'finger' not in a['joint']]
    if len(arm)!=7 or max(abs(a['actual']-b) for a,b in zip(arm,native['robot0_joint_pos']))>1e-10:
        raise ValueError('native static joints differ from requested native pose')
    joints=dict(zip(g['joint_names'],g['joint_positions_rad']))
    jdelta=np.array([joints[f'panda_joint{k}'] for k in range(1,8)])-native['robot0_joint_pos']
    if np.max(abs(jdelta))>.001:raise ValueError('Isaac source joints not matched')
    matrices={'isaac':R@hand[:3,:3],'native_dynamic':rotation(native['robot0_eef_quat']),
              'native_static':rotation(ref['observation']['robot0_eef_quat'])}
    def difference(a,b):
        delta=a-b
        return {'vector_mm':(delta*1000).tolist(),'norm_mm':float(np.linalg.norm(delta))*1000}
    return {'status':'completed','actions_executed':0,'new_policy_queries':0,'native_fk_time_advanced_s':0,
        'isaac_minus_dynamic':difference(isaac,dynamic),'static_minus_dynamic':difference(static,dynamic),
        'isaac_minus_static':difference(isaac,static),'positions_L_m':{'isaac':isaac.tolist(),'native_dynamic':dynamic.tolist(),'native_static':static.tolist()},
        'rotations_deg':{'isaac_vs_dynamic':rotation_angle(matrices['native_dynamic'].T@matrices['isaac']),
            'static_vs_dynamic':rotation_angle(matrices['native_dynamic'].T@matrices['native_static']),
            'isaac_vs_static':rotation_angle(matrices['native_static'].T@matrices['isaac'])},
        'isaac_native_joint_difference_max_rad':float(np.max(abs(jdelta))),
        'scope':'native static FK after direct joint assignment versus recorded dynamic observation and saved Isaac static capture; no action replay; scene/object images are not matched',
        'interpretation':'static-vs-dynamic includes observation refresh/kinematic consistency; remaining Isaac-vs-static includes model, TCP and registration differences plus small joint initialization residual; not a unique causal diagnosis'}


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--segment-manifest',type=Path);p.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/native_pose_refresh');args=p.parse_args()
    source=args.segment_manifest or next((f for f in sorted((repo/'diagnostics/isaac_native_segment').glob('*/run.json'),reverse=True) if read(f).get('status')=='failed'),None)
    if source is None:p.error('stopped Isaac segment source required')
    settings=read(source.parent/'segment_settings.json');geometries=list((source.parent/'capture').glob('*/geometry_audit.json'))
    if len(geometries)!=1:raise ValueError('expected one saved Isaac geometry')
    g=read(geometries[0]);stopped=read(geometries[0].parent/'online_registered_result.json')
    if stopped['error']!='native/Isaac initial pose mismatch' or stopped['native_controls_executed']!=0:raise ValueError('requires zero-action initial-pose stop')
    requested=requested_geometry(g,settings['native_initial'])
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    (folder/'saved_isaac_geometry.json').write_bytes(geometries[0].read_bytes());save_json(folder/'isaac_source.json',requested)
    save_json(folder/'segment_settings.json',settings);(folder/'source_segment_run.json').write_bytes(source.read_bytes())
    report={'status':'preflight','source_segment_sha256':sha(source.read_bytes()),'saved_geometry_sha256':sha(geometries[0].read_bytes()),'actions_executed':0,'new_policy_queries':0}
    try:
        from run_multipose_geometry import preflight_reference
        python=(args.openpi_root/'examples/libero/.venv/bin/python').absolute()
        report['preflight']=preflight_reference(python,args.openpi_root.absolute(),folder)
        command=[str(python),str(Path(__file__).with_name('libero_reference_audit.py').absolute()),'--worker',str(folder),'--suite','libero_spatial','--task-id','0','--seed','7']
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.update(PYTHONPATH=str(args.openpi_root/'third_party/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy')
        report.update(status='capturing',command=command);save_json(folder/'run.json',report)
        print(f'[Native pose refresh] Output: {folder}',flush=True)
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as child:
                try:code=child.wait(timeout=180)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('native static FK timeout')
        if code:raise RuntimeError('native capture failed; inspect reference.json and worker.log')
        result=analyze(g,settings,read(folder/'reference.json'));save_json(folder/'analysis.json',result)
        report.update(status='completed',position_differences={k:result[k] for k in ('isaac_minus_dynamic','static_minus_dynamic','isaac_minus_static')});return 0
    except Exception as error:report.update(status='failed',error=str(error));return 1
    finally:save_json(folder/'run.json',report);print(f"[Native pose refresh] {report['status']}: {folder/'run.json'}",flush=True)


if __name__=='__main__':raise SystemExit(main())
