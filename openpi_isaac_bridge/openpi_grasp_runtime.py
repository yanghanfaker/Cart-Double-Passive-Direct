"""Native adapter for registered OpenPI approach and gated grasp verification."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

from grasp_probe import GraspAbort
from openpi_grasp_logic import run_grasp_experiment,bounded_goal_target
from online_registered_probe import prepare_capture_observation,query_once
from registered_replay_probe import target_from_action
from replanning_probe import capture_round,feedback_change
from openpi_health import check_openpi_health


class GraspRuntime:
    def __init__(self,env,base,arm,settings,output,tensor,report,save):
        import numpy as np
        self.env=env;self.base=base;self.arm=arm;self.settings=settings;self.output=Path(output);self.tensor=tensor
        self.report=report;self.save=save;self.np=np;self.controls=0;self.phase_counts={};self.path=0.;self.last_phase=None
        self.query_index=0;self.prefix_index=0
        self.geometry=json.loads((self.output/'geometry_audit.json').read_bytes());g=self.geometry
        if not g['same_sensor_ik_body'] or not g['sensor_offset_closure']['within_0p1mm_0p05deg'] or not g['ik_offset_closure']['within_0p1mm_0p05deg']:
            raise RuntimeError('invalid hand/IK geometry')
        if g['sensor_body_offset']!={'position_m':[0.,0.,.1034],'quat_xyzw':[0.,0.,0.,1.]} or g['ik_body_offset']!={'position_m':[0.,0.,.107],'quat_xyzw':[0.,0.,0.,1.]}:raise RuntimeError('expected sensor103.4/IK107 identity offsets')
        if not arm.cfg.controller.use_relative_mode or arm.action_dim!=6 or not math.isclose(float(base.step_dt),.05,abs_tol=1e-8):raise RuntimeError('requires relative pose IK at 50ms')
        self.scale=float(arm.cfg.scale)
        if not math.isfinite(self.scale) or self.scale<=0:raise RuntimeError('invalid IK scale')
        self.robot=base.scene['robot'];self.hand_id=g['body_resolution']['sensor_body_ids'][0]
        self.joints,names=self.robot.find_joints('panda_joint[1-7]')
        if list(names)!=[f'panda_joint{k}' for k in range(1,8)]:raise RuntimeError('unexpected arm joint order')
        self.fingers,names=self.robot.find_joints('panda_finger.*')
        if list(names)!=['panda_finger_joint1','panda_finger_joint2']:raise RuntimeError('unexpected finger order')
        self.initial=self.state();self.previous=self.initial
        self.trace=(self.output/'motion.jsonl').open('a',encoding='utf-8')
        (self.output/'queries').mkdir();(self.output/'motion_images').mkdir()

    def state(self):
        from geometry_audit import inverse,multiply
        from kinematic_attribution import rotate
        tensor=self.tensor;robot=self.robot;obj=self.base.scene['object']
        hp=tensor(robot.data.body_pos_w)[0,self.hand_id].detach().cpu().tolist();hq=tensor(robot.data.body_quat_w)[0,self.hand_id].detach().cpu().tolist()
        tcp=(self.np.array(hp)+rotate(hq,[0,0,.097])).tolist()
        op=tensor(obj.data.root_pos_w)[0].detach().cpu().tolist();oq=tensor(obj.data.root_quat_w)[0].detach().cpu().tolist()
        state={'hand_position_world_m':hp,'hand_quat_world_xyzw':hq,'tcp_world_m':tcp,'object_pos_world_m':op,'object_quat_world_xyzw':oq,
            'object_in_tcp_pos_m':rotate(inverse(hq),(self.np.array(op)-tcp).tolist()),'object_in_tcp_quat_xyzw':multiply(inverse(hq),oq),
            'joint_positions_rad':tensor(robot.data.joint_pos)[0,self.joints].detach().cpu().tolist(),
            'joint_velocities_rad_s':tensor(robot.data.joint_vel)[0,self.joints].detach().cpu().tolist(),
            'finger_positions_m':tensor(robot.data.joint_pos)[0,self.fingers].detach().cpu().tolist()}
        if any(not self.np.isfinite(v).all() for v in state.values()):raise RuntimeError('nonfinite native grasp state')
        try:
            buffers=getattr(robot.actuators,'target_command',None)
            targets=buffers.position if buffers is not None else robot.data.joint_pos_target
            state['joint_position_targets_rad']=tensor(targets)[0,self.joints].detach().cpu().tolist()
        except (AttributeError,NotImplementedError):state['joint_position_targets_rad']=None
        if state['joint_position_targets_rad'] is not None and not self.np.isfinite(state['joint_position_targets_rad']).all():raise RuntimeError('nonfinite native joint targets')
        return state

    def emit(self,row):
        self.trace.write(json.dumps(row,allow_nan=False)+'\n');self.trace.flush()

    def step(self,target,gripper,phase,fixed_goal=None):
        import torch
        from geometry_audit import multiply,inverse
        from kinematic_attribution import rotate
        from replay_probe import quaternion_distance_deg
        from isaaclab.utils.math import compute_pose_error
        target={**target,'gripper_applied':float(gripper)}
        before=self.state();tensor=self.tensor;robot=self.robot;np=self.np
        rp=tensor(robot.data.root_pos_w)[0].detach().cpu().tolist();rq=tensor(robot.data.root_quat_w)[0].detach().cpu().tolist()
        tp=rotate(inverse(rq),(np.array(target['ik107_target_world_m'])-rp).tolist());tq=multiply(inverse(rq),target['hand_target_quat_world_xyzw'])
        cp,cq=self.arm._compute_frame_pose();cp=tensor(cp);cq=tensor(cq)
        pe,re=compute_pose_error(cp,cq,torch.tensor([tp],device=cp.device,dtype=cp.dtype),torch.tensor([tq],device=cq.device,dtype=cq.dtype),rot_error_type='axis_angle')
        if not bool(torch.isfinite(pe).all() and torch.isfinite(re).all()) or float(pe.norm())>.0015 or float(re.norm())>math.radians(.3):raise GraspAbort('IK correction outside local grasp envelope')
        values=(pe[0]/self.scale).detach().cpu().tolist()+(re[0]/self.scale).detach().cpu().tolist()+[float(gripper)]
        attempt={'event':'attempt','control':self.controls+1,'phase':phase,'before':before,'target':target,'fixed_goal':fixed_goal,
            'gripper_applied':gripper,'raw_isaac_action':values,'ik107_target_root_m':tp,'ik107_target_root_quat_xyzw':tq}
        self.emit(attempt);step=int(self.base.sim.get_physics_step_count());result=self.env.step(torch.tensor([values],dtype=torch.float32,device=self.base.device))
        self.controls+=1;self.phase_counts[phase]=self.phase_counts.get(phase,0)+1
        self.report.update(native_controls_executed=self.controls,native_phase_counts=self.phase_counts)
        try:after=self.state()
        except Exception as error:
            self.emit({'event':'result','control':self.controls,'phase':phase,'status':'state_read_failed','error':str(error)})
            self.save();raise
        delta=np.array(after['tcp_world_m'])-before['tcp_world_m'];self.path+=float(np.linalg.norm(delta))
        row={'event':'result','control':self.controls,'phase':phase,'after':after,
            'physics_substeps':int(self.base.sim.get_physics_step_count())-step,
            'tcp97_target_error_m':float(np.linalg.norm(np.array(after['tcp_world_m'])-target['tcp97_target_world_m'])),
            'hand_target_rotation_error_deg':quaternion_distance_deg(after['hand_quat_world_xyzw'],target['hand_target_quat_world_xyzw']),
            'total_tcp_displacement_m':float(np.linalg.norm(np.array(after['tcp_world_m'])-self.initial['tcp_world_m'])),
            'total_hand_rotation_deg':quaternion_distance_deg(after['hand_quat_world_xyzw'],self.initial['hand_quat_world_xyzw']),
            'tcp_path_length_m':self.path,'object_speed_estimate_m_s':math.dist(after['object_pos_world_m'],before['object_pos_world_m'])/.05}
        try:
            if bool(tensor(result[2]).any()) or bool(tensor(result[3]).any()):raise GraspAbort('environment ended during grasp')
            if row['physics_substeps']!=5:raise GraspAbort('unexpected physics substep count')
            if np.linalg.norm(delta)>.0015 or quaternion_distance_deg(after['hand_quat_world_xyzw'],before['hand_quat_world_xyzw'])>.3:raise GraspAbort('period motion exceeded 1.5mm/0.3deg')
            if row['total_tcp_displacement_m']>.35 or row['total_hand_rotation_deg']>90 or self.path>.85:raise GraspAbort('overall grasp motion envelope exceeded')
            q=tensor(robot.data.joint_pos)[0,self.joints];limits=tensor(robot.data.soft_joint_pos_limits)[0,self.joints]
            if not bool(((q>=limits[:,0]-.001)&(q<=limits[:,1]+.001)).all()):raise GraspAbort('soft joint limit violation')
            if row['object_speed_estimate_m_s']>.5:raise GraspAbort('object speed exceeded 0.5m/s control estimate')
            if phase in ('approach','close') and math.dist(after['object_pos_world_m'],self.initial['object_pos_world_m'])>.02:raise GraspAbort('object moved >2cm before lift')
            if after['object_pos_world_m'][2]<self.initial['object_pos_world_m'][2]-.01:raise GraspAbort('object fell below initial support')
            if phase=='approach' and min(after['finger_positions_m'])<.03:raise GraspAbort('approach gripper not open')
            if self.controls%20==0 or self.last_phase!=phase:self.save_images(phase)
            self.last_phase=phase;row['status']='checked'
        except Exception as error:
            row.update(status='stopped',error=str(error));raise
        finally:
            self.previous=after;self.emit(row)
            self.report.update(native_controls_executed=self.controls,native_phase_counts=self.phase_counts,tcp_path_length_m=self.path,last_state=after);self.save()
        return after

    def save_images(self,phase):
        from PIL import Image
        for name in ('external_camera','wrist_camera'):
            rgb=self.tensor(self.base.scene[name].data.output['rgb'])[0,...,:3].detach().cpu().numpy().copy()
            Image.fromarray(rgb).save(self.output/'motion_images'/f'{self.controls:04d}_{phase}_{name}.png')

    def step_model(self,action,gripper):
        s=self.state();target=target_from_action(s['hand_position_world_m'],s['hand_quat_world_xyzw'],self.settings['calibration']['world_from_isaac_rotation'],action)
        self.prefix_index+=1
        target.update(source_query=self.query_index,source_prefix_index=self.prefix_index,raw_model_action=list(action))
        return self.step(target,gripper,'approach')

    def step_goal(self,p,q,gripper,phase):
        s=self.state();target=bounded_goal_target(s['hand_position_world_m'],s['hand_quat_world_xyzw'],p,q)
        return self.step(target,gripper,phase,{'tcp_world_m':list(p),'hand_quat_world_xyzw':list(q)})


def run_openpi_grasp(env,base,arm,settings,output,tensor):
    import numpy as np
    from openpi_step_probe import read_server_metadata
    from geometry_audit import multiply
    from replay_probe import quaternion_distance_deg
    output=Path(output)
    maximum=settings.get('max_queries',200)
    if type(maximum) is not int or not 1<=maximum<=200:raise ValueError('max_queries must be 1..200')
    report={'status':'preparing','mode':'openpi-grasp','policy_queries':0,'inference_attempts':0,'native_controls_executed':0,
        'success':False,'queries':[],'control_trace':'motion.jsonl','max_queries':maximum,
        'scope':'OpenPI approach/close request; geometric close gate; program lift/hold validation; no contact-force measurement'}
    def save():(output/'online_registered_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    runtime=None;save()
    try:
        report['server_health']=check_openpi_health(settings['server_host'],settings['server_port']);save()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port']);report['server_metadata']=read_server_metadata(client)
        runtime=GraspRuntime(env,base,arm,settings,output,tensor,report,save)
        def query(index):
            row={'query':index,'status':'capturing'};report['queries'].append(row);save()
            archive=output/'queries'/f'query_{index:04d}.zip'
            with tempfile.TemporaryDirectory(prefix='openpi_grasp_capture_') as tmp:
                folder=Path(tmp)
                try:
                    geometry,step=capture_round(base,arm,folder,output,tensor,index)
                    obs,request,meta=prepare_capture_observation(folder,settings['calibration'],settings['prompt'],geometry)
                    row.update(input=meta,capture_physics_step=step,camera_frames={n:int(np.array(geometry['cameras'][n]['frame']).reshape(-1)[0]) for n in ('external_camera','wrist_camera')})
                    if index>1:row['feedback_change']=feedback_change(report['queries'][-2],row)
                    row['status']='querying';report['inference_attempts']+=1;save()
                    def received(response,elapsed):
                        report['policy_queries']+=1;row['inference_wall_s']=elapsed
                        chunk=response.get('actions') if isinstance(response,dict) else response
                        if hasattr(chunk,'tolist'):chunk=chunk.tolist()
                        try:(folder/'online_model_response.json').write_text(json.dumps({'actions':chunk},indent=2,allow_nan=False),encoding='utf-8')
                        except (TypeError,ValueError):(folder/'invalid_model_response.txt').write_text(repr(chunk),encoding='utf-8')
                        save()
                    request,_=query_once(client,obs,request,on_response=received)
                    row['post_query_physics_step']=int(base.sim.get_physics_step_count())
                    if row['post_query_physics_step']!=step:raise GraspAbort('physics advanced during inference')
                    s=runtime.state();R=np.array(request['world_rotation_L_from_I']);t=np.array(request['world_translation_L_from_I_m'])
                    if np.linalg.norm(R@np.array(s['tcp_world_m'])+t-request['source_state_float32'][:3])>.0005 or quaternion_distance_deg(multiply(request['registration_quat_xyzw'],s['hand_quat_world_xyzw']),request['source_mapped_hand_quat_xyzw'])>.1:
                        raise GraspAbort('live state differs from queried state')
                    if np.max(np.abs(np.array(s['joint_positions_rad'])-request['source_arm_positions_rad']))>.001:raise GraspAbort('live joints differ from queried state')
                    request['source_repeat']=index
                    runtime.query_index=index;runtime.prefix_index=0
                    (folder/'online_replay_request.json').write_text(json.dumps(request,indent=2,allow_nan=False),encoding='utf-8')
                    row['status']='validated';return request
                except Exception as error:
                    row.update(status='failed',error=str(error));raise
                finally:
                    (folder/'query_record.json').write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
                    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                        for file in sorted(folder.iterdir()):
                            if file.is_file():z.write(file,file.name)
                    row.update(archive=archive.relative_to(output).as_posix(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());save()
        report['status']='running';save()
        result=run_grasp_experiment(runtime.state,query,runtime.step_model,runtime.step_goal,max_queries=maximum)
        report.update(status='completed',success=result['success'],outcome=result['outcome'],grasp_result=result)
        runtime.save_images('final')
        return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        if runtime is not None:runtime.trace.close()
        save();print(f'[OpenPI grasp] Report: {output/"online_registered_result.json"}',flush=True)
