"""Five archived native commands through the capped Isaac bridge, no contact."""
import argparse
import copy
import os
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from native_libero_replay import load_source
from native_libero_rollout import save_json
from clean_scene_probe import validate_clean_scene
from registered_replay_probe import target_from_action
from replay_probe import quaternion_distance_deg
from geometry_audit import multiply


def read(p): return json.loads(Path(p).read_bytes())
def sha(b): return hashlib.sha256(b).hexdigest()


def use_static_reference(settings, manifest):
    manifest=Path(manifest);root=manifest.parent;m=read(manifest)
    if m['status']!='completed' or m['actions_executed']!=0 or m['new_policy_queries']!=0:raise ValueError('completed static refresh required')
    payloads={name:(root/name).read_bytes() for name in ('reference.json','segment_settings.json','saved_isaac_geometry.json','source_segment_run.json','analysis.json')}
    if sha(payloads['saved_isaac_geometry.json'])!=m['saved_geometry_sha256'] or sha(payloads['source_segment_run.json'])!=m['source_segment_sha256']:
        raise ValueError('static refresh source SHA mismatch')
    old=json.loads(payloads['segment_settings.json']);ref=json.loads(payloads['reference.json'])
    for key in ('raw_actions','native_initial','native_segment','calibration','source_native_report_sha256','expected_joint_gains'):
        if old[key]!=settings[key]:raise ValueError('static reference belongs to different segment: '+key)
    from native_pose_refresh import analyze
    # Independently recompute the saved decomposition before using its reference.
    analysis=analyze(json.loads(payloads['saved_isaac_geometry.json']),old,ref)
    stored=json.loads(payloads['analysis.json'])
    for key in ('isaac_minus_dynamic','static_minus_dynamic','isaac_minus_static'):
        if not np.allclose(analysis[key]['vector_mm'],stored[key]['vector_mm'],atol=1e-8,rtol=0):raise ValueError('static analysis disagrees')
    result=copy.deepcopy(settings);result['native_dynamic_initial']=copy.deepcopy(settings['native_initial'])
    # prepare() initially shares this object with native_segment[0]['before'].
    # Break that alias before replacing the initialization comparison pose.
    result['native_initial']=copy.deepcopy(settings['native_initial'])
    for key in ('robot0_eef_pos','robot0_eef_quat'):
        result['native_initial'][key]=ref['observation'][key]
    result['initial_reference']='static native FK at recorded arm joints; original dynamic pose retained separately'
    result['initial_velocity_note']='Isaac starts held near zero velocity; original native segment was moving; this is an execution diagnostic, not matched dynamic state replay'
    result['static_reference_sha256']=sha(payloads['reference.json']);result['static_refresh_manifest_sha256']=sha(manifest.read_bytes())
    return result,{**payloads,'run.json':manifest.read_bytes()}


def prepare(native, scene, calibration):
    checks=read(native.parent/'replay_checks.json')
    if checks['status']!='completed' or checks['archived_responses_consumed']!=55 or not all(x['exact_match'] for x in checks['input_checks']):raise ValueError('verified native replay required')
    m,reports,responses=load_source(native)
    # Episode 0 query 8 is the first query containing an executed closure.
    r=reports[0];close=next(x for x in r['trace'] if x['close_requested'])
    q=next(x for x in r['queries'] if x['query']==close['query'])
    segment=[x for x in r['trace'] if x['query']==q['query']]
    if len(segment)!=5:raise ValueError('complete five-step source segment required')
    first=segment[0]['before'];actions=next(x['actions'][:5] for x in responses if x['episode']==0 and x['query']==q['query'])
    if not any(a[6]>0 for a in actions):raise ValueError('source segment must include closure')
    config=json.loads(json.dumps(scene));config.update(name='native-episode0-first-close-query',arm_positions_rad=first['robot0_joint_pos'],
        object_xy_world_m=[.7,.25],reference_eef_position_world_m=first['robot0_eef_pos'])
    validate_clean_scene(config)
    if not calibration['candidate_geometry_within_thresholds']:raise ValueError('validated registration required')
    for a,b in [('reference_world_rotation','world_from_isaac_rotation'),('reference_world_translation_m','world_from_isaac_translation_m')]:
        if not np.allclose(config[a],calibration[b],atol=1e-12,rtol=0):raise ValueError('scene calibration mismatch')
    return config,{'mode':'native-segment','calibration':calibration,'native_initial':first,'native_segment':segment,
        'source_native_bddl_sha256':m.get('bddl_sha256'),
        'raw_actions':actions,'source_native_report_sha256':sha(native.read_bytes()),'source_episode':0,'source_query':q['query'],
        'scope':'same raw commands, matched joint start; capped Isaac IK and binary gripper versus recorded native OSC; cube moved away, no contact; not controller equivalence'}


def run_segment(env,base,arm,settings,output,tensor):
    from openpi_grasp_runtime import GraspRuntime
    output=Path(output);report={'status':'preparing','policy_queries':0,'native_controls_executed':0,'trace':[],
        'scope':settings['scope'],'settings':settings}
    def save():save_json(output/'online_registered_result.json',report)
    save();runtime=None
    try:
        if read(output/'joint_gains.json')!=settings['expected_joint_gains']:raise RuntimeError('gains differ from source')
        runtime=GraspRuntime(env,base,arm,settings,output,tensor,report,save)
        first=runtime.state();native=settings['native_initial'];R=np.array(settings['calibration']['world_from_isaac_rotation']);t=np.array(settings['calibration']['world_from_isaac_translation_m'])
        error=float(np.linalg.norm(R@first['tcp_world_m']+t-native['robot0_eef_pos']))
        from geometry_audit import pose_matrix
        from multipose_geometry import rotation_angle
        I=np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':first['hand_quat_world_xyzw']}))[:3,:3]
        L=np.array(pose_matrix({'position_m':[0,0,0],'quat_xyzw':native['robot0_eef_quat']}))[:3,:3]
        rotation=rotation_angle(L.T@R@I)
        report.update(initial=first,initial_tcp_error_m=error,initial_rotation_error_deg=rotation,
            initial_joint_error_rad=float(np.max(np.abs(np.array(first['joint_positions_rad'])-native['robot0_joint_pos']))))
        save()
        if error>.0005 or rotation>.1 or report['initial_joint_error_rad']>.001:raise RuntimeError('native/Isaac initial pose mismatch')
        contact=settings.get('execution_case',{}).get('kind')=='contact-grasp'
        if min(first['finger_positions_m'])<.035:raise RuntimeError('need open gripper')
        if contact:
            if first['tcp_world_m'][2]-first['object_pos_world_m'][2]<.08:raise RuntimeError('contact orientation clearance below 8cm')
        elif math.dist(first['tcp_world_m'],first['object_pos_world_m'])<.12:raise RuntimeError('need distant cube')
        report.update(initial=first,initial_tcp_error_m=error,initial_rotation_error_deg=rotation);save()
        if 'execution_case' in settings:
            from execution_alignment_probe import run_alignment
            return run_alignment(runtime,R,settings,first,report,save)
        gripper=1.;target=None
        for i in range(25):
            before=runtime.state()
            if i<5:
                a=settings['raw_actions'][i];target=target_from_action(before['hand_position_world_m'],before['hand_quat_world_xyzw'],R,a)
                if a[6]!=0:gripper=-1. if a[6]>0 else 1.
            after=runtime.step(target,gripper,'close')
            if math.dist(after['tcp_world_m'],after['object_pos_world_m'])<.10 or math.dist(after['object_pos_world_m'],first['object_pos_world_m'])>.002:raise RuntimeError('no-contact condition violated')
            report['trace'].append({'step':i+1,'phase':'archived-action' if i<5 else 'last-target-hold','gripper_applied':gripper,'before':before,'after':after,
                'mapped_tcp_world_L_m':(R@after['tcp_world_m']+t).tolist(),
                'native_reference':settings['native_segment'][i] if i<5 else None})
            save()
        report.update(status='completed',final=runtime.state(),action_steps=5,hold_steps=20)
    except Exception as error:report.update(status='failed',error=str(error));raise
    finally:
        if runtime is not None:runtime.trace.close()
        save()


def main():
    repo=Path(__file__).resolve().parent.parent;p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-report',type=Path);p.add_argument('--scene-manifest',type=Path)
    p.add_argument('--pose-refresh-manifest',type=Path)
    p.add_argument('--execution-case-json',type=Path)
    p.add_argument('--pose-cache-root',type=Path)
    p.add_argument('--isaaclab-root',type=Path,default=Path('/home/test/Robot/IsaacLab'))
    p.add_argument('--output-root',type=Path,default=repo/'diagnostics/isaac_native_segment');args=p.parse_args()
    native=args.native_report or next((f for f in sorted((repo/'diagnostics/native_libero_replay').glob('*/report.json'),reverse=True) if read(f.parent/'replay_checks.json').get('status')=='completed'),None)
    scene=args.scene_manifest or next((f for f in sorted((repo/'diagnostics/near_descent_reference_ab').glob('*/run.json'),reverse=True) if read(f).get('status')=='completed'),None)
    if native is None or scene is None:p.error('completed native replay and near-descent source required')
    m=read(scene);cb=(scene.parent/'world_registration.json').read_bytes()
    if sha(cb)!=m['world_registration_sha256']:raise ValueError('calibration SHA mismatch')
    config,settings=prepare(native,read(scene.parent/m['cases'][0]['name']/'clean_scene_request.json'),json.loads(cb))
    settings['expected_joint_gains']=read((scene.parent/m['cases'][0]['result']).parent/'joint_gains.json')
    refresh=args.pose_refresh_manifest or next((f for f in sorted((repo/'diagnostics/native_pose_refresh').glob('*/run.json'),reverse=True) if read(f).get('status')=='completed'),None)
    if refresh is None:p.error('completed native static refresh required')
    settings,reference_files=use_static_reference(settings,refresh)
    pose_files=None
    if args.execution_case_json:
        from execution_alignment_probe import validate_case
        settings['execution_case']=validate_case(read(args.execution_case_json))
        if settings['execution_case']['kind']=='contact-grasp':
            from execution_contact import contact_scene
            config=contact_scene(config,settings)
            settings['scope']='deterministic cube contact/lift diagnostic with fixed arm gains and two finger candidates; no policy inference or native scene equivalence claim'
            if settings['execution_case'].get('hybrid_role')=='policy':
                settings.update(server_host='127.0.0.1',server_port=8000,prompt=settings['execution_case']['prompt'],
                    scope='live policy approach followed by explicit program refinement and verified grasp; cube proxy domain, not native LIBERO task')
        if 'pose_variant' in settings['execution_case']:
            from execution_generalization import prepare_pose
            if args.pose_cache_root is None:p.error('pose variant requires --pose-cache-root')
            settings,config,pose_files=prepare_pose(settings,config,settings['execution_case']['pose_variant'],args.pose_cache_root)
        if 'arm_profile' in settings['execution_case']:
            from execution_sync_cases import gain_plan
            audit_path=(scene.parent/m['cases'][0]['result']).parent/'actuator_audit.json'
            audit_bytes=audit_path.read_bytes()
            settings['source_joint_gains']=copy.deepcopy(settings['expected_joint_gains'])
            settings['source_arm_audit_sha256']=sha(audit_bytes)
            settings['expected_joint_gains'],settings['expected_runtime_arm_gains']=gain_plan(
                settings['source_joint_gains'],json.loads(audit_bytes),settings['execution_case']['arm_profile'])
    config['reference_eef_position_world_m']=settings['native_initial']['robot0_eef_pos']
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True)
    reference_folder=folder/'static_reference';reference_folder.mkdir()
    for name,payload in reference_files.items():(reference_folder/name).write_bytes(payload)
    if pose_files is not None:
        pose_folder=folder/'pose_reference';pose_folder.mkdir()
        for name,payload in pose_files.items():(pose_folder/name).write_bytes(payload)
    save_json(folder/'clean_scene_request.json',config);save_json(folder/'segment_settings.json',settings)
    command=[sys.executable,str(Path(__file__).with_name('run_geometry_audit.py')),'--isaaclab-root',str(args.isaaclab_root.absolute()),'--output-root',str(folder/'capture'),
        '--clean-scene-json',str(folder/'clean_scene_request.json'),'--online-registered-json',str(folder/'segment_settings.json')]
    if 'source_joint_gains' in settings:
        (folder/'source_arm_actuator_audit.json').write_bytes(audit_bytes)
        save_json(folder/'execution_joint_gains.json',settings['expected_joint_gains'])
        command+=['--joint-gains-json',str(folder/'execution_joint_gains.json')]
    report={'status':'running','command':command,'policy_queries':0};save_json(folder/'run.json',report)
    try:
        child_env=os.environ.copy();child_env.pop('VIRTUAL_ENV',None);child_env.pop('PYTHONPATH',None)
        code=subprocess.call(command,env=child_env)
        results=list((folder/'capture').glob('*/online_registered_result.json'))
        if code or len(results)!=1 or read(results[0])['status']!='completed':raise RuntimeError('segment failed; inspect capture console and result')
        report.update(status='completed',result=results[0].relative_to(folder).as_posix(),result_sha256=sha(results[0].read_bytes()));return 0
    except Exception as error:report.update(status='failed',error=str(error));return 1
    finally:save_json(folder/'run.json',report);print(f"[Isaac native segment] {report['status']}: {folder/'run.json'}",flush=True)


if __name__=='__main__':raise SystemExit(main())
