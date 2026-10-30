"""Small fixed recorded-action replay through explicit LIBERO/TCP/IK transforms."""
import json
import math
from pathlib import Path
from openpi_step_probe import validate_action_chunk

TRANSLATION_CAP=.0005
ROTATION_CAP=math.radians(.1)
MODEL_STEPS=5
SETTLE_STEPS=10


def validate_replay(request):
    import numpy as np
    from kinematic_attribution import normalized_quaternion
    from geometry_audit import pose_matrix
    if request.get('condition')!='flip_x':raise ValueError('replay must come from frozen horizontal-flip observation')
    validate_action_chunk(request['raw_action_chunk'])
    if len(request['raw_action_chunk'])<10:raise ValueError('source must contain at least ten validated actions')
    R=np.array(request['world_rotation_L_from_I']);t=np.array(request['world_translation_L_from_I_m'])
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
        raise ValueError('invalid world registration')
    if np.asarray(request['source_state_float32']).shape!=(8,) or not np.isfinite(request['source_state_float32']).all():raise ValueError('invalid source state')
    if np.asarray(request['source_arm_positions_rad']).shape!=(7,) or not np.isfinite(request['source_arm_positions_rad']).all():raise ValueError('invalid source arm positions')
    normalized_quaternion(request['source_mapped_hand_quat_xyzw'])
    normalized_quaternion(request['registration_quat_xyzw'])
    qR=np.asarray(pose_matrix({'position_m':[0,0,0],'quat_xyzw':request['registration_quat_xyzw']}))[:3,:3]
    if not np.allclose(qR,R,atol=1e-6):raise ValueError('registration quaternion and rotation disagree')
    return request


def target_from_action(hand_position,hand_quat_xyzw,world_rotation,action):
    import numpy as np
    from geometry_audit import multiply
    from kinematic_attribution import rotate
    validate_action_chunk([list(action)])
    R=np.array(world_rotation);clipped=np.clip(action,-1,1)
    dp=clipped[:3]*.05;dr=clipped[3:6]*.5
    for vector,cap in ((dp,TRANSLATION_CAP),(dr,ROTATION_CAP)):
        norm=np.linalg.norm(vector)
        if norm>cap:vector*=cap/norm
    delta=R.T@dp;rotation=R.T@dr;angle=float(np.linalg.norm(rotation))
    dq=[0,0,0,1] if angle<1e-14 else [*(rotation*math.sin(angle/2)/angle).tolist(),math.cos(angle/2)]
    q=multiply(dq,list(hand_quat_xyzw))
    tcp=np.asarray(hand_position)+rotate(hand_quat_xyzw,[0,0,.097])
    tcp_target=tcp+delta
    ik_target=tcp_target+rotate(q,[0,0,.010])
    return {'bounded_delta_L_m':dp.tolist(),'bounded_rotation_L_rad':dr.tolist(),
        'bounded_delta_I_m':delta.tolist(),'bounded_rotation_I_rad':rotation.tolist(),
        'tcp97_before_world_m':tcp.tolist(),'tcp97_target_world_m':tcp_target.tolist(),
        'ik107_target_world_m':ik_target.tolist(),'hand_target_quat_world_xyzw':q,
        'gripper_applied':1.0,'scope':'current-pose-relative TCP97 target; world-axis rotation; exact 10mm rigid point shift to IK107'}


def global_envelope(sample,reference):
    """Bound all chunks against the same original pose and object position."""
    import numpy as np
    from replay_probe import quaternion_distance_deg
    from kinematic_attribution import normalized_quaternion
    for data in (sample,reference):
        for key in ('tcp97_world_m','object_position_world_m'):
            if np.asarray(data[key]).shape!=(3,) or not np.isfinite(data[key]).all():raise ValueError('invalid global envelope reference')
        normalized_quaternion(data['hand_quat_world_xyzw'])
    result={'tcp_displacement_m':float(np.linalg.norm(np.array(sample['tcp97_world_m'])-reference['tcp97_world_m'])),
        'hand_rotation_deg':quaternion_distance_deg(sample['hand_quat_world_xyzw'],reference['hand_quat_world_xyzw']),
        'object_displacement_m':float(np.linalg.norm(np.array(sample['object_position_world_m'])-reference['object_position_world_m']))}
    if result['tcp_displacement_m']>.010 or result['hand_rotation_deg']>2 or result['object_displacement_m']>.002:
        raise RuntimeError('global replanning envelope exceeded (10mm/2deg; object 2mm)')
    return result


def run_registered_replay(env,base,arm,request,output,tensor,*,settle_steps=SETTLE_STEPS,global_reference=None):
    import numpy as np
    import torch
    import time
    from PIL import Image
    from geometry_audit import multiply,inverse
    from kinematic_attribution import rotate
    from replay_probe import quaternion_distance_deg
    from isaaclab.utils.math import compute_pose_error
    request=validate_replay(request);output=Path(output)
    if type(settle_steps) is not int or settle_steps not in (0,SETTLE_STEPS):raise ValueError('settle steps must be zero or ten')
    report={'status':'running','policy_queries':0,'source_condition':'flip_x','source_repeat':request['source_repeat'],
        'source_input_sha256':request['source_input_sha256'],'completed_action_periods':0,'completed_settle_periods':0,
        'caps':{'tcp_translation_m':TRANSLATION_CAP,'rotation_deg':.1},'trace':[],
        'scope':'fixed first five recorded actions, not new inference or feedback replanning; last absolute target held for ten periods',
        'model_source_timing':'recorded inference, no server calls during this run'}
    report['settle_steps_requested']=settle_steps
    if settle_steps==0:report['scope']='first five recorded actions only; no extra settling before next observation'
    if global_reference is not None:report['global_reference']=global_reference
    def save():(output/'registered_replay_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        geometry=json.loads((output/'geometry_audit.json').read_text())
        if not geometry['same_sensor_ik_body'] or geometry['sensor_body_offset']!={'position_m':[0.,0.,.1034],'quat_xyzw':[0.,0.,0.,1.]} or geometry['ik_body_offset']!={'position_m':[0.,0.,.107],'quat_xyzw':[0.,0.,0.,1.]}:
            raise RuntimeError('requires same hand body with exact sensor103.4/IK107 offsets and identity local rotations')
        if not geometry['sensor_offset_closure']['within_0p1mm_0p05deg'] or not geometry['ik_offset_closure']['within_0p1mm_0p05deg']:raise RuntimeError('internal TCP closure failed')
        if not arm.cfg.controller.use_relative_mode or arm.action_dim!=6 or not math.isclose(float(base.step_dt),.05,abs_tol=1e-8):raise RuntimeError('expected relative pose IK at 50ms')
        robot=base.scene['robot'];hand_index=geometry['body_resolution']['sensor_body_ids'][0]
        joint_ids,joint_names=robot.find_joints('panda_joint[1-7]')
        if list(joint_names)!=[f'panda_joint{k}' for k in range(1,8)]:raise RuntimeError('unexpected arm joint order')
        finger_ids,finger_names=robot.find_joints('panda_finger.*')
        if list(finger_names)!=['panda_finger_joint1','panda_finger_joint2']:raise RuntimeError('unexpected finger order')
        scale=float(arm.cfg.scale)
        if not math.isfinite(scale) or scale<=0:raise RuntimeError('invalid IK action scale')
        R=np.array(request['world_rotation_L_from_I']);t=np.array(request['world_translation_L_from_I_m'])
        def sample():
            hp=tensor(robot.data.body_pos_w)[0,hand_index].detach().cpu().tolist()
            hq=tensor(robot.data.body_quat_w)[0,hand_index].detach().cpu().tolist()
            values={'hand_position_world_m':hp,'hand_quat_world_xyzw':hq,
                'tcp97_world_m':(np.array(hp)+rotate(hq,[0,0,.097])).tolist(),
                'joint_positions_rad':tensor(robot.data.joint_pos)[0,joint_ids].detach().cpu().tolist(),
                'joint_velocities_rad_s':tensor(robot.data.joint_vel)[0,joint_ids].detach().cpu().tolist(),
                'object_position_world_m':tensor(base.scene['object'].data.root_pos_w)[0].detach().cpu().tolist(),
                'finger_positions_m':tensor(robot.data.joint_pos)[0,finger_ids].detach().cpu().tolist()}
            if any(not np.isfinite(v).all() for v in values.values()):raise RuntimeError('nonfinite runtime sample')
            try:
                buffers=getattr(robot.actuators,'target_command',None)
                targets=buffers.position if buffers is not None else robot.data.joint_pos_target
                values['joint_position_targets_rad']=tensor(targets)[0,joint_ids].detach().cpu().tolist()
            except (AttributeError,NotImplementedError):values['joint_position_targets_rad']=None
            if values['joint_position_targets_rad'] is not None and not np.isfinite(values['joint_position_targets_rad']).all():raise RuntimeError('nonfinite joint targets')
            return values
        initial=sample();initial_tcp=np.array(initial['tcp97_world_m']);initial_q=initial['hand_quat_world_xyzw']
        if global_reference is not None:report['initial_global_envelope']=global_envelope(initial,global_reference)
        source_state=np.array(request['source_state_float32']);registered_tcp=R@initial_tcp+t
        guard=float(np.linalg.norm(registered_tcp-source_state[:3]))
        registered_q=multiply(request['registration_quat_xyzw'],initial_q)
        orientation_guard=quaternion_distance_deg(registered_q,request['source_mapped_hand_quat_xyzw'])
        report.update(initial=initial,initial_state_position_guard_m=guard,initial_state_rotation_guard_deg=orientation_guard)
        if guard>.0005 or orientation_guard>.1:raise RuntimeError('live initialization differs from frozen source by >0.5mm/0.1deg')
        source_arms=np.array(request['source_arm_positions_rad'])
        if np.max(np.abs(source_arms-initial['joint_positions_rad']))>.001:raise RuntimeError('live arm joints differ from source by >0.001rad')
        direction=np.array(initial['object_position_world_m'])-initial_tcp
        if np.linalg.norm(direction)<.08:raise RuntimeError('source is too close to object for distant replay diagnostic')
        unit=direction/np.linalg.norm(direction)
        report['initial_object_distance_m']=float(np.linalg.norm(direction));report['translation_command_sum_I_m']=[0.,0.,0.];save()
        target=None
        for index in range(MODEL_STEPS+settle_steps):
            before=sample();phase='recorded-action' if index<MODEL_STEPS else 'fixed-target-settle'
            if index<MODEL_STEPS:
                target=target_from_action(before['hand_position_world_m'],before['hand_quat_world_xyzw'],R,request['raw_action_chunk'][index])
                report['translation_command_sum_I_m']=(np.array(report['translation_command_sum_I_m'])+target['bounded_delta_I_m']).tolist()
            root_p=tensor(robot.data.root_pos_w)[0].detach().cpu().tolist();root_q=tensor(robot.data.root_quat_w)[0].detach().cpu().tolist()
            target_root=rotate(inverse(root_q),(np.array(target['ik107_target_world_m'])-root_p).tolist())
            target_root_q=multiply(inverse(root_q),target['hand_target_quat_world_xyzw'])
            current_p,current_q=arm._compute_frame_pose();current_p=tensor(current_p);current_q=tensor(current_q)
            target_p=torch.tensor([target_root],dtype=current_p.dtype,device=current_p.device)
            target_q=torch.tensor([target_root_q],dtype=current_q.dtype,device=current_q.device)
            pe,re=compute_pose_error(current_p,current_q,target_p,target_q,rot_error_type='axis_angle')
            if not bool(torch.isfinite(pe).all() and torch.isfinite(re).all()) or float(pe.norm())>.0015 or float(re.norm())>math.radians(.3):raise RuntimeError('IK correction exceeds local diagnostic envelope')
            values=(pe[0]/scale).detach().cpu().tolist()+(re[0]/scale).detach().cpu().tolist()+[1.0]
            row={'control_index':index+1,'phase':phase,'before':before,'target':dict(target),
                'ik107_target_root_m':target_root,'ik107_target_root_quat_xyzw':target_root_q,'raw_isaac_action':values,
                'execution_attempted':True,'executed':False}
            report['trace'].append(row);save()
            physics_before=int(base.sim.get_physics_step_count())
            result=env.step(torch.tensor([values],dtype=torch.float32,device=base.device))
            row['executed']=True
            if index<MODEL_STEPS:report['completed_action_periods']+=1
            else:report['completed_settle_periods']+=1
            after=sample();row['after']=after;row['physics_substeps']=int(base.sim.get_physics_step_count())-physics_before
            if global_reference is not None:row['global_envelope']=global_envelope(after,global_reference)
            row['tcp97_target_error_m']=float(np.linalg.norm(np.array(after['tcp97_world_m'])-target['tcp97_target_world_m']))
            row['hand_target_rotation_error_deg']=quaternion_distance_deg(after['hand_quat_world_xyzw'],target['hand_target_quat_world_xyzw'])
            row['object_distance_m']=float(np.linalg.norm(np.array(after['object_position_world_m'])-after['tcp97_world_m']))
            for name in ('external_camera','wrist_camera'):
                rgb=tensor(base.scene[name].data.output['rgb'])[0,...,:3].detach().cpu().numpy().copy()
                Image.fromarray(rgb).save(output/f'registered_replay_{index+1:02d}_{name}.png')
            save()
            if bool(tensor(result[2]).any()) or bool(tensor(result[3]).any()):raise RuntimeError('environment reset during replay')
            if row['physics_substeps']!=round(.05/float(base.cfg.sim.dt)):raise RuntimeError('unexpected physics substep count')
            if np.linalg.norm(np.array(after['tcp97_world_m'])-initial_tcp)>.004 or quaternion_distance_deg(initial_q,after['hand_quat_world_xyzw'])>.7:raise RuntimeError('total replay envelope exceeded (4mm/0.7deg)')
            if np.linalg.norm(np.array(after['tcp97_world_m'])-before['tcp97_world_m'])>.0015 or quaternion_distance_deg(before['hand_quat_world_xyzw'],after['hand_quat_world_xyzw'])>.3:raise RuntimeError('period motion envelope exceeded (1.5mm/0.3deg)')
            if np.linalg.norm(np.array(after['object_position_world_m'])-initial['object_position_world_m'])>.002:raise RuntimeError('object moved >2mm during distant replay')
            if min(after['finger_positions_m'])<.03:raise RuntimeError('gripper did not remain open')
            q=tensor(robot.data.joint_pos)[0,joint_ids];limits=tensor(robot.data.soft_joint_pos_limits)[0,joint_ids]
            if not bool(torch.isfinite(q).all()) or not bool(((q>=limits[:,0]-.001)&(q<=limits[:,1]+.001)).all()):raise RuntimeError('nonfinite joint or soft joint limit violation')
            print(f'[Registered replay] {index+1:02d} {phase}; TCP target error={row["tcp97_target_error_m"]*1000:.3f}mm',flush=True)
            time.sleep(.05)
        final=sample();delta=np.array(final['tcp97_world_m'])-initial_tcp;command=np.array(report['translation_command_sum_I_m'])
        report.update(status='completed',final=final,actual_net_tcp_delta_I_m=delta.tolist(),
            actual_initial_direction_approach_mm=float(delta@unit*1000),command_initial_direction_approach_mm=float(command@unit*1000),
            final_tcp97_target_error_m=report['trace'][-1]['tcp97_target_error_m'],final_rotation_error_deg=report['trace'][-1]['hand_target_rotation_error_deg'],
            final_target_reached=report['trace'][-1]['tcp97_target_error_m']<=.00025 and report['trace'][-1]['hand_target_rotation_error_deg']<=.1)
        return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save();print(f'[Registered replay] Report: {output/"registered_replay_result.json"}',flush=True)
