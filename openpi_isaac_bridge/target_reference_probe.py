"""Frozen-stream A/B diagnostic of current-pose and accumulated TCP targets."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np
from geometry_audit import multiply
from kinematic_attribution import rotate
from registered_replay_probe import target_from_action,validate_replay
from replay_probe import quaternion_distance_deg


class TargetReference:
    def __init__(self,state,world_rotation):
        self.position=np.array(state['tcp_world_m'],dtype=float)
        self.quaternion=list(state['hand_quat_world_xyzw']);self.rotation=np.array(world_rotation)
        self.command_sum=np.zeros(3)

    def advance(self,action,actual,mode):
        if mode not in ('current','accumulated'):raise ValueError('unknown reference mode')
        actual_target=target_from_action(actual['hand_position_world_m'],actual['hand_quat_world_xyzw'],self.rotation,action)
        virtual_hand=self.position-rotate(self.quaternion,[0,0,.097])
        ideal=target_from_action(virtual_hand.tolist(),self.quaternion,self.rotation,action)
        self.position=np.array(ideal['tcp97_target_world_m']);self.quaternion=list(ideal['hand_target_quat_world_xyzw'])
        self.command_sum+=np.array(ideal['bounded_delta_I_m'])
        target=actual_target if mode=='current' else ideal
        return {**target,'reference_mode':mode,'raw_model_action':list(action),
            'scope':f'{mode} diagnostic TCP97 reference; world-axis rotations; exact target-pose shift to IK107'},ideal


def run_target_reference(env,base,arm,settings,output,tensor):
    from openpi_grasp_runtime import GraspRuntime
    mode=settings['reference_mode'];request=validate_replay(settings['replay_request']);steps=settings['steps']
    if mode not in ('current','accumulated') or steps not in (100,200) or len(request['raw_action_chunk'])!=steps:raise ValueError('expected 100 or 200 fixed actions and valid mode')
    if hashlib.sha256(json.dumps(request['raw_action_chunk'],separators=(',',':'),allow_nan=False).encode()).hexdigest()!=settings['fixed_stream_sha256']:raise ValueError('fixed stream SHA mismatch')
    output=Path(output)
    report={'status':'preparing','mode':'target-reference-ab','reference_mode':mode,'policy_queries':0,
        'action_controls':0,'hold_controls':0,'native_controls_executed':0,'steps_requested':steps,
        'scope':'same recorded stream; reference convention A/B only, not a new policy rollout or LIBERO equivalence claim',
        'fixed_stream_sha256':settings['fixed_stream_sha256']}
    def save():(output/'online_registered_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    runtime=None;trace=None;save()
    try:
        if json.loads((output/'joint_gains.json').read_bytes())!=settings['expected_joint_gains']:raise RuntimeError('gains differ from recorded source')
        runtime=GraspRuntime(env,base,arm,settings,output,tensor,report,save);initial=runtime.state()
        R=np.array(request['world_rotation_L_from_I']);t=np.array(request['world_translation_L_from_I_m'])
        position_guard=float(np.linalg.norm(R@np.array(initial['tcp_world_m'])+t-request['source_state_float32'][:3]))
        rotation_guard=quaternion_distance_deg(multiply(request['registration_quat_xyzw'],initial['hand_quat_world_xyzw']),request['source_mapped_hand_quat_xyzw'])
        if position_guard>.0005 or rotation_guard>.1 or np.max(np.abs(np.array(initial['joint_positions_rad'])-request['source_arm_positions_rad']))>.001:raise RuntimeError('initial pose differs from fixed source')
        if math.dist(initial['object_pos_world_m'],settings['source_object_position_world_m'])>.001:raise RuntimeError('object differs from fixed source')
        report.update(initial=initial,initial_position_guard_m=position_guard,initial_rotation_guard_deg=rotation_guard)
        reference=TargetReference(initial,R);trace=(output/'reference_trace.jsonl').open('w',encoding='utf-8');target=None;ideal=None
        action_residual=np.zeros(3);action_end=None
        report['status']='running';save()
        for index in range(steps+10):
            before=runtime.state();phase='approach' if index<steps else 'hold'
            if index<steps:target,ideal=reference.advance(request['raw_action_chunk'][index],before,mode)
            row={'control':index+1,'phase':'recorded-action' if index<steps else 'fixed-target-hold',
                'applied_tcp97_target_world_m':target['tcp97_target_world_m'],
                'ideal_tcp97_target_world_m':ideal['tcp97_target_world_m'],
                'ideal_hand_target_quat_world_xyzw':ideal['hand_target_quat_world_xyzw'],
                'execution_attempted':True,'executed':False}
            trace.write(json.dumps({'event':'reference',**row},allow_nan=False)+'\n');trace.flush()
            after=runtime.step(target,1.,phase)
            row.update(executed=True,after=after,
                ideal_position_error_m=float(np.linalg.norm(np.array(after['tcp_world_m'])-ideal['tcp97_target_world_m'])),
                applied_target_error_m=float(np.linalg.norm(np.array(after['tcp_world_m'])-target['tcp97_target_world_m'])),
                applied_rotation_error_deg=quaternion_distance_deg(after['hand_quat_world_xyzw'],target['hand_target_quat_world_xyzw']),
                ideal_rotation_error_deg=quaternion_distance_deg(after['hand_quat_world_xyzw'],ideal['hand_target_quat_world_xyzw']))
            if index<steps:
                report['action_controls']+=1
                action_residual+=np.array(target['tcp97_target_world_m'])-after['tcp_world_m'];action_end=after
            else:report['hold_controls']+=1
            trace.write(json.dumps({'event':'response',**row},allow_nan=False)+'\n');trace.flush();save()
            if math.dist(after['tcp_world_m'],initial['tcp_world_m'])>.12 or quaternion_distance_deg(after['hand_quat_world_xyzw'],initial['hand_quat_world_xyzw'])>25 or runtime.path>.15:
                raise RuntimeError('reference comparison exceeded 120mm/25deg or 150mm path envelope')
            if math.dist(after['object_pos_world_m'],initial['object_pos_world_m'])>.002 or min(after['finger_positions_m'])<.03 or math.dist(after['tcp_world_m'],after['object_pos_world_m'])<.08:
                raise RuntimeError('comparison must remain open-gripper and distant from unmoved object')
            if (index+1)%25==0:print(f'[Target reference] {mode} {index+1}/{steps+10}; ideal error={row["ideal_position_error_m"]*1000:.3f}mm',flush=True)
        final=runtime.state();command=reference.command_sum;actual=np.array(final['tcp_world_m'])-initial['tcp_world_m']
        action_actual=np.array(action_end['tcp_world_m'])-initial['tcp_world_m']
        report.update(status='completed',final=final,action_end=action_end,command_delta_sum_I_m=command.tolist(),
            actual_delta_I_m=actual.tolist(),action_end_actual_delta_I_m=action_actual.tolist(),
            action_end_actual_minus_command_I_m=(action_actual-command).tolist(),
            final_actual_minus_command_I_m=(actual-command).tolist(),
            sum_action_applied_target_residual_I_m=action_residual.tolist(),
            final_ideal_position_error_m=row['ideal_position_error_m'],final_applied_target_error_m=row['applied_target_error_m'],
            final_ideal_rotation_error_deg=row['ideal_rotation_error_deg'],
            final_applied_rotation_error_deg=row['applied_rotation_error_deg'],
            final_applied_target_reached=row['applied_target_error_m']<=.00025 and row['applied_rotation_error_deg']<=.1,
            ideal_tcp97_target_world_m=ideal['tcp97_target_world_m'],ideal_hand_target_quat_world_xyzw=ideal['hand_target_quat_world_xyzw'])
        runtime.save_images('final');return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        if trace is not None:trace.close()
        if runtime is not None:runtime.trace.close()
        save();print(f'[Target reference] Report: {output/"online_registered_result.json"}',flush=True)
