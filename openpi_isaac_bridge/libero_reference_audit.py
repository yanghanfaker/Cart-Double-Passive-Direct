"""Read installed LIBERO reference geometry; no policy calls or robot actions."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


MODULES=('numpy','PIL','yaml','mujoco','robosuite','libero','openpi_client','torch')
PREFLIGHT="""import importlib.util,importlib.metadata,json,platform,sys
modules={}
for name in %r:
 try:
  spec=importlib.util.find_spec(name);modules[name]={'available':spec is not None,'origin':None if spec is None else spec.origin,'locations':[] if spec is None else list(spec.submodule_search_locations or [])}
 except Exception as e:modules[name]={'available':False,'error':str(e)}
versions={}
for name in ('numpy','Pillow','PyYAML','mujoco','robosuite','libero','openpi-client','torch'):
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
print(json.dumps({'python':sys.executable,'version':sys.version,'machine':platform.machine(),'modules':modules,'versions':versions}))
""" % (MODULES,)


def source_targets(audit):
    if audit.get('status')!='completed':raise ValueError('Isaac geometry audit must be completed')
    values=dict(zip(audit['joint_names'],audit['joint_positions_rad']))
    names=[f'panda_joint{i}' for i in range(1,8)]
    targets=[values[name] for name in names]
    fingers=list(audit['finger_positions_isaac_m'])
    if len(fingers)!=2 or not all(math.isfinite(v) for v in targets+fingers):raise ValueError('invalid source joints')
    return targets,[fingers[0],-fingers[1]]


def ordered_joints(names,kind):
    suffix=r'joint([1-7])$' if kind=='arm' else r'finger_joint([12])$'
    size=7 if kind=='arm' else 2;indexed={}
    for name in names:
        match=re.search(suffix,name)
        if match:
            index=int(match.group(1))
            if index in indexed:raise ValueError('ambiguous joint suffix')
            indexed[index]=name
    if set(indexed)!=set(range(1,size+1)):raise ValueError(f'cannot resolve {kind} joints from {names}')
    return [indexed[i] for i in range(1,size+1)]


def capture(folder,suite_name,task_id,seed):
    """Worker only: imports native MuJoCo in an isolated subprocess."""
    report={'status':'starting','policy_queries':0,'action_steps':0,
      'scope':'installed LIBERO geometry reference, not benchmark evaluation or proof of training dependency versions'}
    def save():(folder/'reference.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();env=None
    try:
        spec=importlib.util.find_spec('libero')
        candidates=[Path(p)/'libero' for p in (spec.submodule_search_locations or [])]
        benchmark_root=next((p for p in candidates if (p/'bddl_files').is_dir() and (p/'assets').is_dir()),None)
        if benchmark_root is None:raise RuntimeError('installed LIBERO bddl_files/assets directory not found')
        # JSON is valid YAML. Keep first-import config isolated from ~/.libero.
        config_dir=folder/'libero_config';config_dir.mkdir(exist_ok=True)
        config={'benchmark_root':str(benchmark_root),'bddl_files':str(benchmark_root/'bddl_files'),
                'init_states':str(benchmark_root/'init_files'),'datasets':str(benchmark_root.parent/'datasets'),
                'assets':str(benchmark_root/'assets')}
        (config_dir/'config.yaml').write_text(json.dumps(config,indent=2),encoding='utf-8')
        os.environ['LIBERO_CONFIG_PATH']=str(config_dir)
        import numpy as np
        from PIL import Image
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client import image_tools
        from robosuite.utils.transform_utils import mat2quat
        import robosuite
        import inspect
        from replay_probe import quaternion_distance_deg
        from openpi_step_probe import libero_state_vector
        def safe(v):
            if v is None or isinstance(v,(str,bool,int,float)):return v
            if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
            if isinstance(v,(list,tuple)):return [safe(x) for x in v]
            if hasattr(v,'tolist'):return safe(v.tolist())
            return f'<{type(v).__name__}>'
        suite=benchmark.get_benchmark_dict()[suite_name]()
        if not 0<=task_id<suite.n_tasks:raise ValueError('task index outside selected suite')
        task=suite.get_task(task_id)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        report.update(status='creating_environment',suite=suite_name,task_id=task_id,seed=seed,
            task_description=str(task.language),bddl_file=str(bddl),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            initialization_note='task reset only; no demonstration initial-state files loaded',
            image_preprocessing='official OpenPI LIBERO example: observation image[::-1,::-1], resize/pad224, convert_to_uint8')
        save()
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(seed);env.reset()
        sim=env.sim;robot=env.robots[0]
        report['control_timing']={'control_frequency_hz':getattr(env.env,'control_freq',None),
            'robosuite_model_timestep_s':getattr(env.env,'model_timestep',None),
            'mujoco_timestep_s':float(sim.model.opt.timestep)}
        source=json.loads((folder/'isaac_source.json').read_text());arm_target,finger_target=source_targets(source)
        arm_names=ordered_joints(list(robot.robot_model.joints),'arm')
        finger_names=ordered_joints(list(robot.gripper.joints),'finger')
        assignments=[]
        for name,target in zip(arm_names+finger_names,arm_target+finger_target):
            address=sim.model.get_joint_qpos_addr(name);vel_address=sim.model.get_joint_qvel_addr(name)
            if isinstance(address,(tuple,list)) or isinstance(vel_address,(tuple,list)):raise ValueError('expected scalar hinge/slide joints')
            jid=sim.model.joint_name2id(name);limits=np.asarray(sim.model.jnt_range[jid]).tolist()
            if sim.model.jnt_limited[jid] and not limits[0]-1e-6<=target<=limits[1]+1e-6:raise ValueError(f'{name} target outside reference joint range')
            assignments.append({'joint':name,'qpos_address':int(address),'qvel_address':int(vel_address),'target':target,'limits':limits})
            sim.data.qpos[address]=target;sim.data.qvel[vel_address]=0
        time_before=float(sim.data.time);sim.forward()
        obs=env.env._get_observations(force_update=True)
        report['sim_time_before_forward']=time_before;report['sim_time_after_capture']=float(sim.data.time)
        if report['sim_time_after_capture']!=time_before:raise RuntimeError('reference capture unexpectedly advanced simulation time')
        report['joint_assignments']=assignments
        for item in assignments:item['actual']=float(sim.data.qpos[item['qpos_address']])
        report['joint_pose_note']='direct qpos assignment and forward kinematics; not a controller tracking result; suffix matching is not proof of identical physical zero/sign conventions'
        site=int(robot.eef_site_id);site_parent=int(sim.model.site_bodyid[site])
        eef_pos=np.asarray(sim.data.site_xpos[site]);eef_R=np.asarray(sim.data.site_xmat[site]).reshape(3,3)
        eef_q=mat2quat(eef_R)
        def body(name):
            bid=sim.model.body_name2id(name)
            return {'name':name,'id':int(bid),'position_world_m':np.asarray(sim.data.body_xpos[bid]).tolist(),
                    'rotation_world_matrix':np.asarray(sim.data.body_xmat[bid]).reshape(3,3).tolist()}
        root=body(robot.robot_model.root_body);hand=body(robot.robot_model.eef_name);gripper=body(robot.gripper.root_body)
        root_R=np.asarray(root['rotation_world_matrix']);hand_R=np.asarray(hand['rotation_world_matrix'])
        report.update(robot_root=root,robot_eef_body=hand,gripper_root=gripper,
            eef_site={'id':site,'name':sim.model.site_id2name(site),'parent_body':sim.model.body_id2name(site_parent),
                'local_position_m':np.asarray(sim.model.site_pos[site]).tolist(),'local_quat_mujoco_wxyz':np.asarray(sim.model.site_quat[site]).tolist(),
                'position_world_m':eef_pos.tolist(),'rotation_world_matrix':eef_R.tolist(),'quat_xyzw':np.asarray(eef_q).tolist(),
                'position_in_robot_root_m':(root_R.T@(eef_pos-np.asarray(root['position_world_m']))).tolist(),
                'position_in_eef_body_m':(hand_R.T@(eef_pos-np.asarray(hand['position_world_m']))).tolist(),
                'rotation_in_eef_body':(hand_R.T@eef_R).tolist()},
            observation={key:safe(obs[key]) for key in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot0_joint_pos') if key in obs})
        observed_pos=np.asarray(obs['robot0_eef_pos']);observed_q=np.asarray(obs['robot0_eef_quat']);fingers=np.asarray(obs['robot0_gripper_qpos'])
        report['observation_reference_checks']={'position_obs_vs_grip_site_error_m':float(np.linalg.norm(observed_pos-eef_pos)),
            'rotation_obs_vs_eef_body_deg':quaternion_distance_deg(observed_q.tolist(),np.asarray(mat2quat(hand_R)).tolist()),
            'rotation_obs_vs_grip_site_deg':quaternion_distance_deg(observed_q.tolist(),np.asarray(eef_q).tolist()),
            'note':'robosuite 1.4 reference uses grip-site position and EEF-body orientation; installed observable source saved for verification'}
        # Reuse the signed rotation-vector adapter; this conversion takes two nonnegative Isaac-style finger inputs.
        report['policy_state_world_float32']=np.asarray(libero_state_vector(observed_pos.tolist(),observed_q.tolist(),[float(fingers[0]),float(-fingers[1])]),dtype=np.float32).tolist()
        report['cameras']={}
        for name,key in [('agentview','agentview_image'),('robot0_eye_in_hand','robot0_eye_in_hand_image')]:
            cid=sim.model.camera_name2id(name);p=np.asarray(sim.data.cam_xpos[cid]);R=np.asarray(sim.data.cam_xmat[cid]).reshape(3,3)
            raw=np.asarray(obs[key])
            if raw.dtype!=np.uint8 or raw.shape!=(256,256,3):raise RuntimeError('expected 256x256 RGB observation')
            rotated=np.ascontiguousarray(raw[::-1,::-1])
            processed=image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated,224,224))
            files={}
            for label,image in [('observation256',raw),('rotated256',rotated),('policy224',processed)]:
                path=folder/f'{name}_{label}.png';Image.fromarray(image).save(path);files[label]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            fovy=float(sim.model.cam_fovy[cid]);f=128/math.tan(math.radians(fovy)/2)
            report['cameras'][name]={'id':int(cid),'parent_body':sim.model.body_id2name(int(sim.model.cam_bodyid[cid])),
                'position_world_m':p.tolist(),'rotation_world_opengl_matrix':R.tolist(),
                'rotation_world_ros_matrix':(R@np.diag([1,-1,-1])).tolist(),'position_in_robot_root_m':(root_R.T@(p-np.asarray(root['position_world_m']))).tolist(),
                'rotation_in_robot_root_opengl':(root_R.T@R).tolist(),'vertical_fov_deg':fovy,
                'pinhole_intrinsic_256':[[f,0,128],[0,f,128],[0,0,1]],'files':files,
                'intrinsic_note':'derived from MuJoCo vertical fovy and 256 square resolution; image orientation preprocessing recorded separately'}
        controller=robot.controller
        report['controller']={key:safe(getattr(controller,key,None)) for key in ('name','input_min','input_max','output_min','output_max','kp','damping_ratio','control_delta','policy_freq')}
        for label,model in [('robot',robot.robot_model),('gripper',robot.gripper)]:
            path=folder/f'{label}_model.xml';path.write_text(model.get_xml(),encoding='utf-8')
        (folder/'composed_scene.xml').write_text(env.env.model.get_xml(),encoding='utf-8')
        implementations=[]
        for obj in (OffScreenRenderEnv,env.env._get_observations,robot.setup_observables):
            try:implementations.append(inspect.getsource(obj))
            except (TypeError,OSError):implementations.append(f'source unavailable: {obj}')
        (folder/'reference_implementation.txt').write_text('\n\n'.join(implementations),encoding='utf-8')
        report.update(status='completed',robosuite_file=str(Path(robosuite.__file__).resolve()),
            registration_status='unverified: world/base/hand frame correspondence to Isaac still requires comparison; no automatic parameter changes')
        save()
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));save();raise
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    parser.add_argument('--isaac-audit',type=Path)
    parser.add_argument('--python',type=Path,help='optional existing LIBERO interpreter; no package installation')
    parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parent.parent/'diagnostics/libero_reference')
    parser.add_argument('--suite',default='libero_spatial');parser.add_argument('--task-id',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--mujoco-gl',default='egl',choices=('egl','glx','osmesa'))
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        capture(args.worker,args.suite,args.task_id,args.seed);return 0
    repo=Path(__file__).resolve().parent.parent
    sources=sorted((repo/'diagnostics/geometry_audit').glob('*/geometry_audit.json'))
    source=args.isaac_audit or next((p for p in reversed(sources) if json.loads(p.read_text()).get('status')=='completed'),None)
    if source is None:parser.error('no saved Isaac audit; pass --isaac-audit')
    payload=source.read_bytes();source_targets(json.loads(payload))
    folder=args.output_root.expanduser().resolve()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    (folder/'isaac_source.json').write_bytes(payload)
    env=os.environ.copy();third=args.openpi_root/'third_party/libero'
    env['PYTHONPATH']=str(third)+os.pathsep+env.get('PYTHONPATH','')
    env['MUJOCO_GL']=args.mujoco_gl;env['PYOPENGL_PLATFORM']=args.mujoco_gl if args.mujoco_gl!='glx' else 'glx'
    env.setdefault('PYNPUT_BACKEND','dummy')
    paths=[args.python] if args.python else [args.openpi_root/'examples/libero/.venv/bin/python',args.openpi_root/'.venv/bin/python',Path(sys.executable)]
    candidates=list(dict.fromkeys(str(p.expanduser().absolute()) for p in paths))
    report={'status':'preflight','isaac_source_sha256':hashlib.sha256(payload).hexdigest(),'python_candidates':[],
        'source_isaac_path':str(source),'render_backend':args.mujoco_gl,'pynput_backend':env['PYNPUT_BACKEND'],'package_installs':0,'policy_queries':0,
        'openpi_root':str(args.openpi_root),'third_party_libero':{'path':str(third),'exists':third.is_dir(),
            'inner_init_exists':(third/'libero/libero/__init__.py').is_file(),
            'assets_exist':(third/'libero/libero/assets').is_dir(),'bddl_exist':(third/'libero/libero/bddl_files').is_dir()}}
    def save():(folder/'run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    save();chosen=None
    for executable in candidates:
        item={'requested_python':executable}
        if not Path(executable).is_file():item.update(available=False,reason='interpreter not found')
        else:
            try:
                pre=subprocess.run([executable,'-c',PREFLIGHT],env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=20)
                item['exit_code']=pre.returncode
                item.update(json.loads(pre.stdout.strip().splitlines()[-1]))
                if all(item['modules'][name]['available'] for name in MODULES):chosen=executable
                item['stderr']=pre.stderr[-3000:]
            except Exception as error:item['error']=str(error)
        report['python_candidates'].append(item);save()
        if chosen:break
    if chosen is None:
        report['status']='dependency_missing';save()
        print(f'[LIBERO reference] Dependencies not ready; no environment created. Report: {folder/"run.json"}',flush=True)
        return 0
    report.update(status='capturing',chosen_python=chosen);save()
    command=[chosen,str(Path(__file__).resolve()),'--worker',str(folder),'--suite',args.suite,'--task-id',str(args.task_id),'--seed',str(args.seed)]
    report['command']=command
    try:
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as worker:
                try:code=worker.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait();report['status']='capture_timeout';return 1
        report['worker_exit_code']=code;reference=folder/'reference.json'
        report['status']='completed' if code==0 and reference.is_file() and json.loads(reference.read_text())['status']=='completed' else 'capture_failed'
        return 0 if report['status']=='completed' else 1
    finally:
        save();print(f'[LIBERO reference] {report["status"]}; report: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
