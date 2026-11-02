"""Cross candidate gains/fingers at three checked static native FK poses."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np

POSES={'center':{},'shoulder-up':{1:-.035,3:.025},'shoulder-down':{1:.035,3:-.025}}


def generalization_cases():
    result=[]
    for pose in POSES:
        def make(name,arm,finger,deg=.1,kind='combined'):
            c={'name':name,'kind':kind,'pose_variant':pose,'arm_profile':arm,'finger_profile':finger,
                'gripper_mode':'substep','cap_m':.002,'rotation_cap_deg':deg,'requires':[]}
            if finger=='kp1000-kd100':c['finger_armature']=1.
            if kind=='rotation':c.update(axis=0,sign=1,reference_mode='anchored_tcp',force_open=True,cap_m=.0005)
            return c
        result.append(make(f'general-{pose}-baseline','baseline','kp350-kd50'))
        for arm in ('d50','k1687p5-d50'):
            for finger in ('kp350-kd50','kp1000-kd100'):
                result.append(make(f'general-{pose}-{arm}-{finger}',arm,finger))
            c=make(f'general-{pose}-{arm}-rotation0.25',arm,'kp350-kd50',.25)
            c['requires']=[f'general-{pose}-{arm}-kp350-kd50'];result.append(c)
            result.append(make(f'general-{pose}-{arm}-rx0.25',arm,'kp350-kd50',.25,'rotation'))
    return result


def pose_request(settings,variant):
    from multipose_geometry import validate_arm_pose
    if variant not in POSES:raise ValueError('unknown pose variant')
    q=list(settings['native_initial']['robot0_joint_pos'])
    for index,delta in POSES[variant].items():q[index]+=delta
    validate_arm_pose(q)
    return {'status':'completed','joint_names':[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2'],
        'joint_positions_rad':q+[.04,.04],'finger_positions_isaac_m':[.04,.04],
        'pose_variant':variant,'source_native_report_sha256':settings['source_native_report_sha256'],
        'source_static_reference_sha256':settings['static_reference_sha256'],
        'calibration':settings['calibration'],
        'scope':'validated joint assignment request for native FK; not a measured Isaac geometry capture'}


def validate_reference(request,reference,bddl_sha):
    if reference.get('status')!='completed' or reference.get('policy_queries')!=0 or reference.get('action_steps')!=0:
        raise ValueError('completed zero-action native reference required')
    if reference['sim_time_before_forward']!=reference['sim_time_after_capture'] or reference['bddl_sha256']!=bddl_sha:
        raise ValueError('reference time or task differs')
    from libero_reference_audit import source_targets,ordered_joints
    q,fingers=source_targets(request);assignments=reference['joint_assignments']
    if len(assignments)!=9:raise ValueError('nine reference joint assignments required')
    names=[a['joint'] for a in assignments]
    if names!=ordered_joints(names,'arm')+ordered_joints(names,'finger'):raise ValueError('reference joint order differs')
    for a,target in zip(assignments,q+fingers):
        if not np.isfinite([a['target'],a['actual']]).all() or abs(a['target']-target)>1e-10 or abs(a['actual']-target)>1e-10:raise ValueError('reference joint assignment differs')
    obs=reference['observation']
    for key,size in (('robot0_eef_pos',3),('robot0_eef_quat',4),('robot0_joint_pos',7),('robot0_gripper_qpos',2)):
        v=np.asarray(obs[key]);
        if v.shape!=(size,) or not np.isfinite(v).all():raise ValueError('invalid static observation '+key)
    np.testing.assert_allclose(obs['robot0_joint_pos'],q,atol=1e-10,rtol=0)
    np.testing.assert_allclose(obs['robot0_gripper_qpos'],fingers,atol=1e-10,rtol=0)
    if abs(np.linalg.norm(obs['robot0_eef_quat'])-1)>.001:raise ValueError('invalid static orientation')
    return obs


def prepare_pose(settings,config,variant,cache_root,openpi_root=Path('/home/test/Robot/openpi')):
    from native_libero_rollout import save_json
    from run_multipose_geometry import preflight_reference
    request=pose_request(settings,variant);payload=json.dumps(request,sort_keys=True,allow_nan=False).encode()
    digest=hashlib.sha256(payload).hexdigest();folder=Path(cache_root)/variant
    folder.mkdir(parents=True,exist_ok=True);manifest=folder/'run.json'
    bddl_sha=settings.get('source_native_bddl_sha256')
    if not bddl_sha:raise ValueError('source task BDDL digest missing')
    if manifest.exists():
        m=json.loads(manifest.read_bytes())
        if m.get('status')!='completed' or m.get('request_sha256')!=digest:raise ValueError('pose cache failed or belongs to another source')
        if hashlib.sha256((folder/'reference.json').read_bytes()).hexdigest()!=m['reference_sha256']:raise ValueError('cached pose reference digest differs')
    else:
        m={'status':'preparing','request_sha256':digest,'actions_executed':0,'policy_queries':0}
        save_json(manifest,m);save_json(folder/'isaac_source.json',request)
        try:
            python=(openpi_root/'examples/libero/.venv/bin/python').absolute()
            m['preflight']=preflight_reference(python,openpi_root.absolute(),folder)
            command=[str(python),str(Path(__file__).with_name('libero_reference_audit.py').absolute()),
                '--worker',str(folder.absolute()),'--suite','libero_spatial','--task-id','0','--seed','7']
            m['command']=command;save_json(manifest,m)
            env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env.update(PYTHONPATH=str(openpi_root/'third_party/libero'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy')
            with (folder/'worker.log').open('w',encoding='utf-8') as log:
                with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as child:
                    try:code=child.wait(timeout=180)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();raise RuntimeError('native pose FK capture timeout')
            if code:raise RuntimeError('native pose FK capture failed; inspect '+str(folder/'worker.log'))
            ref=json.loads((folder/'reference.json').read_bytes());validate_reference(request,ref,bddl_sha)
            m.update(status='completed',reference_sha256=hashlib.sha256((folder/'reference.json').read_bytes()).hexdigest())
        except Exception as error:m.update(status='failed',error=str(error));raise
        finally:save_json(manifest,m)
    saved_request=json.loads((folder/'isaac_source.json').read_bytes())
    if saved_request!=request:raise ValueError('cached request file differs')
    ref=json.loads((folder/'reference.json').read_bytes());obs=validate_reference(request,ref,bddl_sha)
    result=copy.deepcopy(settings);scene=copy.deepcopy(config)
    result['original_pose_native_initial']=copy.deepcopy(settings['native_initial'])
    result['native_initial']={**copy.deepcopy(obs),'robot0_joint_vel':[0.]*7}
    result['pose_reference_sha256']=m['reference_sha256'];result['pose_request_sha256']=digest
    result['initial_reference']='new native static FK at declared perturbed joints; same original recorded action vectors used as diagnostic input'
    result['pose_variant']=variant
    result['scope']='counterfactual archived action input at declared static native-FK checked joint pose; no new policy inference, native dynamic equivalence, contact, or grasp success claim'
    scene.update(name='generalization-'+variant,arm_positions_rad=request['joint_positions_rad'][:7],
        reference_eef_position_world_m=obs['robot0_eef_pos'],object_xy_world_m=[.7,-.25])
    from clean_scene_probe import validate_clean_scene
    validate_clean_scene(scene)
    return result,scene,{name:(folder/name).read_bytes() for name in ('run.json','reference.json','isaac_source.json')}
