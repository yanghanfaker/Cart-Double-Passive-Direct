"""Native LIBERO reset/benchmark layout and rendered camera-direction diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET


def marked_xml(xml,cameras):
    """Only add world-fixed visual geoms and materials; no new dynamics or joints."""
    root=ET.fromstring(xml);world=root.find('worldbody');asset=root.find('asset')
    if world is None:raise ValueError('XML worldbody missing')
    if asset is None:asset=ET.SubElement(root,'asset')
    for view,markers in cameras.items():
        for m in markers:
            name=f'camera_probe_{view}_{m["id"]}'
            color=' '.join(str(x) for x in [*m['emissive_rgb'],1])
            ET.SubElement(asset,'material',name=name,rgba=color,emission='1',specular='0',shininess='0')
            ET.SubElement(world,'geom',name=name,type='sphere',pos=' '.join(map(str,m['world_position_m'])),
                size=str(m['radius_m']),material=name,contype='0',conaffinity='0',group='1')
    return ET.tostring(root,encoding='unicode')


def select_state(states,index,expected_length):
    import numpy as np
    if index<0 or index>=len(states):raise ValueError('episode index outside initial-state collection')
    state=states[index]
    if hasattr(state,'detach'):state=state.detach().cpu().numpy()
    state=np.asarray(state,dtype=float)
    if state.shape!=(expected_length,) or not np.isfinite(state).all():raise ValueError('invalid initial state shape or nonfinite values')
    return state.copy()


def save_views(folder,obs):
    import numpy as np
    from PIL import Image
    from openpi_client import image_tools
    folder.mkdir(parents=True,exist_ok=True);files={}
    for view in ('agentview','robot0_eye_in_hand'):
        raw=np.asarray(obs[view+'_image']).copy()
        if raw.shape!=(256,256,3) or raw.dtype!=np.uint8:raise ValueError('expected 256 RGB')
        rotated=np.ascontiguousarray(raw[::-1,::-1]);policy=image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated,224,224))
        files[view]={}
        for label,image in [('raw256',raw),('official_rotated256',rotated),('policy224',policy)]:
            p=folder/f'{view}_{label}.png';Image.fromarray(image).save(p)
            files[view][label]={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return files


def camera_data(sim):
    import numpy as np
    result={}
    for view in ('agentview','robot0_eye_in_hand'):
        cid=sim.model.camera_name2id(view);p=np.asarray(sim.data.cam_xpos[cid]).copy()
        R=np.asarray(sim.data.cam_xmat[cid]).reshape(3,3)@np.diag([1,-1,-1])
        f=128/math.tan(math.radians(float(sim.model.cam_fovy[cid]))/2)
        result[view]={'position_world_m':p.tolist(),'rotation_ros':R.tolist(),
            'K_256':[[f,0,128],[0,f,128],[0,0,1]],'fovy_deg':float(sim.model.cam_fovy[cid])}
    return result


def camera_differences(before,after):
    import numpy as np
    return {view:{key+'_max_abs_difference':float(np.max(np.abs(np.array(c[key])-np.array(after[view][key]))))
        for key in ('position_world_m','rotation_ros','K_256')} for view,c in before.items()}


def synchronized_observation(env):
    """Refresh derived kinematics at the integrated state without advancing time."""
    import numpy as np
    state=np.asarray(env.get_sim_state()).copy();before=camera_data(env.sim)
    env.sim.forward()
    observation=env.env._get_observations(force_update=True)
    np.testing.assert_array_equal(state,np.asarray(env.get_sim_state()))
    after=camera_data(env.sim)
    return observation,{'state_and_time_unchanged':True,'camera_refresh_differences':camera_differences(before,after)}


def capture_layout(env,obs,folder,label):
    import numpy as np
    from camera_reference_audit import camera_from_world
    from libero_reference_audit import ordered_joints
    from openpi_step_probe import libero_state_vector
    sim=env.sim;robot=env.robots[0];cameras=camera_data(sim)
    baseid=sim.model.body_name2id(robot.robot_model.root_body)
    p=np.asarray(sim.data.body_xpos[baseid]);R=np.asarray(sim.data.body_xmat[baseid]).reshape(3,3)
    def point(position):
        position=np.asarray(position)
        return {'position_world_m':position.tolist(),'position_robot_root_m':(R.T@(position-p)).tolist(),
            'ros_optical_projections':{view:{'uv_px':(uv:=camera_from_world(position,c['position_world_m'],c['rotation_ros'],c['K_256'])),
                'in_image':uv is not None and 0<=uv[0]<256 and 0<=uv[1]<256,
                'note':'body origin projection only, not visibility or segmentation'} for view,c in cameras.items()}}
    objects={str(name):point(sim.data.body_xpos[int(bid)]) for name,bid in env.env.obj_body_id.items()}
    sites={}
    for sid in range(sim.model.nsite):
        name=sim.model.site_id2name(sid)
        if name and 'table' in name and ('top' in name or 'surface' in name):sites[name]=point(sim.data.site_xpos[sid])
    joints={name:float(sim.data.qpos[sim.model.get_joint_qpos_addr(name)]) for name in ordered_joints(list(robot.robot_model.joints),'arm')+ordered_joints(list(robot.gripper.joints),'finger')}
    fingers=np.asarray(obs['robot0_gripper_qpos']).tolist()
    state=libero_state_vector(np.asarray(obs['robot0_eef_pos']).tolist(),np.asarray(obs['robot0_eef_quat']).tolist(),[fingers[0],-fingers[1]])
    return {'label':label,'sim_time_s':float(sim.data.time),'joint_positions':joints,
        'robot_root_world':{'position_m':p.tolist(),'rotation_matrix':R.tolist()},
        'eef_observation':point(obs['robot0_eef_pos']),'eef_hand_quat_xyzw':np.asarray(obs['robot0_eef_quat']).tolist(),
        'policy_state_float32':np.asarray(state,dtype=np.float32).tolist(),'objects':objects,'table_sites':sites,
        'objects_of_interest':list(env.obj_of_interest),'cameras':cameras,'images':save_views(folder,obs),
        'image_folder':folder.name,'layout_note':'reset and benchmark-init labels are distinct; object body origins are not common grasp surfaces'}


def run_worker(folder,args):
    report={'status':'starting','policy_queries':0,'model_actions_executed':0,'dummy_wait_steps':0,'layouts':[],
        'observation_synchronization':[]}
    def save():(folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();env=None
    try:
        spec=importlib.util.find_spec('libero')
        roots=[Path(p)/'libero' for p in (spec.submodule_search_locations or [])]
        benchmark_root=next(p for p in roots if (p/'bddl_files').is_dir() and (p/'assets').is_dir())
        config=folder/'libero_config';config.mkdir()
        (config/'config.yaml').write_text(json.dumps({'benchmark_root':str(benchmark_root),'bddl_files':str(benchmark_root/'bddl_files'),
            'assets':str(benchmark_root/'assets'),'init_states':str(benchmark_root/'init_files'),'datasets':str(benchmark_root.parent/'datasets')}),encoding='utf-8')
        os.environ['LIBERO_CONFIG_PATH']=str(config)
        import numpy as np
        import torch
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from camera_alignment_probe import marker_layout,analyze_markers
        suite=benchmark.get_benchmark_dict()[args.suite]();task=suite.get_task(args.task_id)
        bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        report.update(suite=args.suite,task_id=args.task_id,episode_index=args.episode_index,seed=args.seed,
            task_description=str(task.language),bddl=str(bddl),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest())
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(args.seed);obs=env.reset()
        obs,sync=synchronized_observation(env);report['observation_synchronization'].append({'stage':'default_reset',**sync})
        report['layouts'].append(capture_layout(env,obs,folder/'default_reset','default_reset'));save()
        init_path=Path(get_libero_path('init_states'))/task.problem_folder/task.init_states_file
        report['benchmark_init']={'path':str(init_path),'exists':init_path.is_file(),'status':'missing'}
        initial=None
        if init_path.is_file():
            report['benchmark_init']['sha256']=hashlib.sha256(init_path.read_bytes()).hexdigest()
            try:
                # Do not silently execute arbitrary pickle constructors in old pruned_init files.
                allowed=[np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]
                with torch.serialization.safe_globals(allowed):states=torch.load(init_path,map_location='cpu',weights_only=True)
                initial=select_state(states,args.episode_index,len(env.get_sim_state()))
                report['benchmark_init'].update(status='loaded_restricted',count=len(states),loader='torch weights_only=True with NumPy array/dtype allowlist')
            except Exception as error:
                report['benchmark_init'].update(status='load_failed',error_type=type(error).__name__,error=str(error))
        if initial is not None:
            obs=env.set_init_state(initial)
            obs,sync=synchronized_observation(env);report['observation_synchronization'].append({'stage':'benchmark_initial',**sync})
            report['layouts'].append(capture_layout(env,obs,folder/'benchmark_initial','benchmark_initial'));save()
            for _ in range(10):
                obs,_,_,_=env.step([0.]*6+[-1.]);report['dummy_wait_steps']+=1
            obs,sync=synchronized_observation(env);report['observation_synchronization'].append({'stage':'benchmark_after10wait',**sync})
            report['layouts'].append(capture_layout(env,obs,folder/'benchmark_after10wait','benchmark_after10wait'))
        report['benchmark_initialization_available']=initial is not None
        report['marker_source_layout']=report['layouts'][-1]['label'];save()
        obs,sync=synchronized_observation(env);report['observation_synchronization'].append({'stage':'before_markers',**sync});save()
        before=np.asarray(env.get_sim_state()).copy();cameras=camera_data(env.sim)
        markers={view:marker_layout(c['position_world_m'],c['rotation_ros'],c['K_256']) for view,c in cameras.items()}
        original_xml=env.env.model.get_xml();xml=marked_xml(original_xml,markers)
        (folder/'scene_before_markers.xml').write_text(original_xml,encoding='utf-8')
        (folder/'scene_with_markers.xml').write_text(xml,encoding='utf-8')
        dimensions=[int(env.sim.model.nq),int(env.sim.model.nv)];time_before=float(env.sim.data.time)
        env.reset_from_xml_string(xml);env.set_state(before);env.sim.forward()
        if [int(env.sim.model.nq),int(env.sim.model.nv)]!=dimensions:raise RuntimeError('marker insertion changed dynamic state dimensions')
        after=np.asarray(env.get_sim_state());np.testing.assert_array_equal(before,after)
        actual_cameras=camera_data(env.sim)
        report['marker_camera_restoration']={'tolerance_max_abs':1e-8,'before':cameras,'after':actual_cameras,
            'differences':camera_differences(cameras,actual_cameras),'state_exactly_restored':True};save()
        for view in cameras:
            for key in ('position_world_m','rotation_ros','K_256'):np.testing.assert_allclose(cameras[view][key],actual_cameras[view][key],atol=1e-8,rtol=0)
        marker_obs=env.env._get_observations(force_update=True)
        if float(env.sim.data.time)!=time_before:raise RuntimeError('marker render advanced restored simulation time')
        marker_folder=folder/'markers';files=save_views(marker_folder,marker_obs);results={}
        for view in cameras:
            raw=np.asarray(marker_obs[view+'_image']);rotated=np.ascontiguousarray(raw[::-1,::-1])
            results[view]={'markers':markers[view],'raw':analyze_markers(raw,markers[view]),
                'official_rotated':analyze_markers(rotated,markers[view]),'images':files[view]}
        report['marker_test']={'state_exactly_restored':True,'simulation_time_restored_s':time_before,'cameras':results,
            'all_marker_views_passed':all(v['raw']['passed'] and v['official_rotated']['passed'] for v in results.values()),
            'scope':'native LIBERO RGB-to-ROS direction and official rotated RGB-to-ROS direction; no policy queries'}
        if (folder/'isaac_source.json').is_file():
            old=json.loads((folder/'isaac_source.json').read_text());cal=json.loads((folder/'world_registration.json').read_text())
            from geometry_audit import pose_matrix
            oldhand=np.array(pose_matrix(old['sensor_body_world']));oldtcp=np.array(old['sensor_tcp_world']['position_m'])+oldhand[:3,:3]@np.array([0,0,-.0064])
            world_R=np.array(cal['world_from_isaac_rotation']);world_t=np.array(cal['world_from_isaac_translation_m'])
            q=dict(zip(old['joint_names'],old['joint_positions_rad']))
            from libero_reference_audit import ordered_joints
            layout_differences=[]
            for layout in report['layouts']:
                names=ordered_joints(list(layout['joint_positions']),'arm')
                layout_differences.append({'label':layout['label'],
                    'max_arm_joint_difference_rad':max(abs(q[f'panda_joint{k}']-layout['joint_positions'][names[k-1]]) for k in range(1,8)),
                    'mapped_tcp_position_difference_m':float(np.linalg.norm(world_R@oldtcp+world_t-np.array(layout['eef_observation']['position_world_m'])))})
            report['isaac_comparison']={'isaac_cube_world_m':old['object_root_world']['position_m'],
                'isaac_cube_mapped_to_libero_world_m':(world_R@np.array(old['object_root_world']['position_m'])+world_t).tolist(),
                'isaac_corrected_tcp_mapped_to_libero_world_m':(world_R@oldtcp+world_t).tolist(),
                'isaac_tabletop_height':'not captured; not inferred from cube center',
                'layouts':layout_differences,
                'note':'cube/bowl body origins are not equivalent geometric contact points; no scene parameters automatically changed'}
        report.update(status='completed',benchmark_initialization_available=initial is not None,
            limitations='Default reset is not benchmark initial state. No model executed. Missing/restricted init files remain explicit; no package or dataset download.')
        print('[LIBERO scene] init='+report['benchmark_init']['status']+'; '+json.dumps({k:{'raw':v['raw'].get('best_transform'),'official_rotated':v['official_rotated'].get('best_transform'),'passed':v['raw']['passed'] and v['official_rotated']['passed']} for k,v in results.items()}),flush=True)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--openpi-root',type=Path,default=Path('/home/test/Robot/openpi'))
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/libero_scene_probe')
    parser.add_argument('--suite',default='libero_spatial');parser.add_argument('--task-id',type=int,default=0)
    parser.add_argument('--episode-index',type=int,default=0);parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.task_id<0 or args.episode_index<0:parser.error('task and episode indices must be nonnegative')
    if args.worker:run_worker(args.worker,args);return 0
    folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    python=(args.openpi_root/'examples/libero/.venv/bin/python').absolute()
    if not python.is_file():parser.error('existing isolated reference Python not found')
    manifest={'status':'running','reference_python':str(python),'policy_queries':0,'package_installs':0}
    def save():(folder/'run.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    save()
    try:
        from run_multipose_geometry import preflight_reference
        manifest['preflight']=preflight_reference(python,args.openpi_root.absolute(),folder)
        aligned=sorted((repo/'diagnostics/camera_alignment').glob('*/run.json'))
        selected=next((p for p in reversed(aligned) if json.loads(p.read_text()).get('all_camera_tests_passed')),None)
        if selected:
            al=json.loads(selected.read_text());home=next(p for p in al['poses'] if p['name']=='home')
            src=(selected.parent/home['result']).with_name('geometry_audit.json')
            (folder/'isaac_source.json').write_bytes(src.read_bytes())
            ca=json.loads(Path(al['source_camera_audit']).read_text());mp=Path(ca['source_manifest'])
            (folder/'world_registration.json').write_bytes((mp.parent/'analysis.json').read_bytes())
            manifest['source_alignment']=str(selected)
            manifest['isaac_source_sha256']=hashlib.sha256((folder/'isaac_source.json').read_bytes()).hexdigest()
            manifest['world_registration_sha256']=hashlib.sha256((folder/'world_registration.json').read_bytes()).hexdigest()
        command=[str(python),str(Path(__file__).absolute()),'--worker',str(folder),'--suite',args.suite,
            '--task-id',str(args.task_id),'--episode-index',str(args.episode_index),'--seed',str(args.seed)]
        env=os.environ.copy();env.pop('VIRTUAL_ENV',None);env['PYTHONPATH']=str(args.openpi_root/'third_party/libero')
        env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYNPUT_BACKEND='dummy')
        manifest['command']=command;save();print(f'[LIBERO scene] Output: {folder}',flush=True)
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            with subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True) as worker:
                try:code=worker.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait();raise RuntimeError('native capture timeout; inspect worker.log')
        manifest['worker_exit_code']=code
        result=json.loads((folder/'report.json').read_text())
        if code!=0 or result['status']!='completed':raise RuntimeError('native capture failed; inspect worker.log')
        manifest.update(status='completed',all_marker_views_passed=result['marker_test']['all_marker_views_passed'],
            benchmark_initialization_available=result['benchmark_initialization_available'])
        return 0
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error));print(f'[LIBERO scene] STOP: {error}',flush=True);return 1
    finally:
        save();print(f'[LIBERO scene] Manifest: {folder/"run.json"}',flush=True)


if __name__=='__main__':raise SystemExit(main())
