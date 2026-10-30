"""Short live-observation comparison of current and accumulated reference modes."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile
import numpy as np

from online_registered_probe import prepare_capture_observation,query_once
from openpi_health import check_openpi_health
from openpi_step_probe import read_server_metadata
from replanning_probe import capture_round,feedback_change
from target_reference_probe import TargetReference
from replay_probe import quaternion_distance_deg


def validate_initial(actual,expected):
    for value in (actual,expected):
        for key,size in [('tcp_world_m',3),('hand_quat_world_xyzw',4),('joint_positions_rad',7),('object_pos_world_m',3)]:
            if np.asarray(value[key]).shape!=(size,) or not np.isfinite(value[key]).all():raise RuntimeError('nonfinite/incomplete initial state')
    metrics={'tcp_error_m':math.dist(actual['tcp_world_m'],expected['tcp_world_m']),
        'hand_rotation_error_deg':quaternion_distance_deg(actual['hand_quat_world_xyzw'],expected['hand_quat_world_xyzw']),
        'max_joint_error_rad':float(np.max(np.abs(np.array(actual['joint_positions_rad'])-expected['joint_positions_rad']))),
        'object_error_m':math.dist(actual['object_pos_world_m'],expected['object_pos_world_m'])}
    if metrics['tcp_error_m']>.0005 or metrics['hand_rotation_error_deg']>.1 or metrics['max_joint_error_rad']>.001 or metrics['object_error_m']>.001:
        raise RuntimeError('online A/B initialization differs from validated source')
    return metrics


def run_online_reference(env,base,arm,settings,output,tensor):
    from openpi_grasp_runtime import GraspRuntime
    mode=settings['reference_mode'];count=settings['queries']
    if mode not in ('current','accumulated') or count not in (10,20):raise ValueError('expected current/accumulated and 10/20 queries')
    output=Path(output)
    report={'status':'preparing','mode':'online-reference-ab','reference_mode':mode,'queries_requested':count,
        'policy_queries':0,'inference_attempts':0,'action_controls':0,'hold_controls':0,'native_controls_executed':0,
        'queries':[],'requested_close_actions':0,'scope':'each condition queries its own current observations; open gripper; experimental target convention, not grasp validation',
        'timing':'synchronous physics paused during inference, velocities retained; no real-time guarantee'}
    def save():(output/'online_registered_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    runtime=None;trace=None;save()
    try:
        if json.loads((output/'joint_gains.json').read_bytes())!=settings['expected_joint_gains']:raise RuntimeError('gains differ from source')
        report['server_health']=check_openpi_health(settings['server_host'],settings['server_port']);save()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port']);report['server_metadata']=read_server_metadata(client)
        runtime=GraspRuntime(env,base,arm,settings,output,tensor,report,save);initial=runtime.state()
        report.update(initial=initial,initial_checks=validate_initial(initial,settings['expected_initial']))
        reference=TargetReference(initial,settings['calibration']['world_from_isaac_rotation'])
        trace=(output/'reference_trace.jsonl').open('w',encoding='utf-8')
        maximum_backlog=0.;maximum_rotation=0.;maximum_applied=0.;maximum_prior_gap=0.;action_end=None;target=None;ideal=None

        def sample_query(index):
            row={'query':index,'status':'capturing','executed_prefix_controls':0};report['queries'].append(row);save()
            archive=output/'queries'/f'query_{index:04d}.zip'
            with tempfile.TemporaryDirectory(prefix='online_reference_') as tmp:
                folder=Path(tmp)
                try:
                    geometry,step=capture_round(base,arm,folder,output,tensor,index)
                    obs,request,meta=prepare_capture_observation(folder,settings['calibration'],settings['prompt'],geometry)
                    row.update(input=meta,capture_physics_step=step,camera_frames={n:int(np.array(geometry['cameras'][n]['frame']).reshape(-1)[0]) for n in ('external_camera','wrist_camera')})
                    if index>1:row['feedback_change']=feedback_change(report['queries'][-2],row)
                    report['inference_attempts']+=1;row['status']='querying';save()
                    def received(response,elapsed):
                        report['policy_queries']+=1;row['inference_wall_s']=elapsed
                        chunk=response.get('actions') if isinstance(response,dict) else response
                        if hasattr(chunk,'tolist'):chunk=chunk.tolist()
                        try:(folder/'online_model_response.json').write_text(json.dumps({'actions':chunk},indent=2,allow_nan=False),encoding='utf-8')
                        except (TypeError,ValueError):(folder/'invalid_model_response.txt').write_text(repr(chunk),encoding='utf-8')
                        save()
                    request,_=query_once(client,obs,request,on_response=received)
                    row['post_query_physics_step']=int(base.sim.get_physics_step_count())
                    if row['post_query_physics_step']!=step:raise RuntimeError('physics advanced during inference')
                    # Compare against the actual geometry snapshot, not the virtual reference.
                    from kinematic_attribution import rotate
                    hand=geometry['sensor_body_world']
                    captured={'tcp_world_m':(np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.097])).tolist(),
                        'hand_quat_world_xyzw':hand['quat_xyzw'],'joint_positions_rad':request['source_arm_positions_rad'],
                        'object_pos_world_m':geometry['object_root_world']['position_m']}
                    row['post_query_state_checks']=validate_initial(runtime.state(),captured)
                    request['source_repeat']=index
                    (folder/'online_replay_request.json').write_text(json.dumps(request,indent=2,allow_nan=False),encoding='utf-8')
                    row['status']='validated';return request,row
                except Exception as error:
                    row.update(status='failed',error=str(error));raise
                finally:
                    (folder/'query_record.json').write_text(json.dumps(row,indent=2,allow_nan=False),encoding='utf-8')
                    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                        for f in sorted(folder.iterdir()):
                            if f.is_file():z.write(f,f.name)
                    row.update(archive=archive.relative_to(output).as_posix(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());save()

        def execute(selected,ideal_target,phase,source_query,source_prefix,prior_gap):
            nonlocal maximum_backlog,maximum_rotation,maximum_applied,maximum_prior_gap
            before=runtime.state();backlog=math.dist(selected['tcp97_target_world_m'],before['tcp_world_m'])
            rotation=quaternion_distance_deg(selected['hand_target_quat_world_xyzw'],before['hand_quat_world_xyzw'])
            maximum_backlog=max(maximum_backlog,backlog);maximum_rotation=max(maximum_rotation,rotation)
            maximum_prior_gap=max(maximum_prior_gap,prior_gap)
            chosen={**selected,'source_query':source_query,'source_prefix_index':source_prefix}
            row={'control':runtime.controls+1,'phase':phase,'source_query':source_query,'source_prefix_index':source_prefix,
                'applied_tcp97_target_world_m':chosen['tcp97_target_world_m'],'ideal_tcp97_target_world_m':ideal_target['tcp97_target_world_m'],
                'ideal_hand_target_quat_world_xyzw':ideal_target['hand_target_quat_world_xyzw'],
                'pre_step_applied_backlog_m':backlog,'pre_step_applied_backlog_deg':rotation}
            row['pre_update_ideal_gap_m']=prior_gap
            trace.write(json.dumps({'event':'reference',**row},allow_nan=False)+'\n');trace.flush()
            after=runtime.step(chosen,1.,'approach' if phase=='model-action' else 'hold')
            row.update(after=after,applied_error_m=math.dist(after['tcp_world_m'],chosen['tcp97_target_world_m']),
                ideal_error_m=math.dist(after['tcp_world_m'],ideal_target['tcp97_target_world_m']),
                applied_rotation_error_deg=quaternion_distance_deg(after['hand_quat_world_xyzw'],chosen['hand_target_quat_world_xyzw']),
                ideal_rotation_error_deg=quaternion_distance_deg(after['hand_quat_world_xyzw'],ideal_target['hand_target_quat_world_xyzw']))
            maximum_applied=max(maximum_applied,row['applied_error_m'])
            trace.write(json.dumps({'event':'response',**row},allow_nan=False)+'\n');trace.flush()
            if phase=='model-action':report['action_controls']+=1
            else:report['hold_controls']+=1
            save()
            if math.dist(after['tcp_world_m'],initial['tcp_world_m'])>.060 or runtime.path>.080 or quaternion_distance_deg(after['hand_quat_world_xyzw'],initial['hand_quat_world_xyzw'])>15:
                raise RuntimeError('short online comparison exceeded 60mm/15deg or 80mm path envelope')
            if math.dist(after['object_pos_world_m'],initial['object_pos_world_m'])>.002 or math.dist(after['tcp_world_m'],after['object_pos_world_m'])<.08 or min(after['finger_positions_m'])<.03:
                raise RuntimeError('online comparison must remain open-gripper and distant from stationary object')
            return after,row

        report['status']='running';save()
        for index in range(1,count+1):
            request,query_row=sample_query(index)
            for prefix,action in enumerate(request['raw_action_chunk'][:5],1):
                report['requested_close_actions']+=int(action[6]>0)
                actual=runtime.state();prior_gap=math.dist(reference.position,actual['tcp_world_m'])
                target,ideal=reference.advance(action,actual,mode)
                action_end,last_row=execute(target,ideal,'model-action',index,prefix,prior_gap)
                query_row['executed_prefix_controls']+=1;save()
            query_row['status']='completed';save()
            print(f'[Online reference] {mode} query {index}/{count}; distance={math.dist(action_end["tcp_world_m"],action_end["object_pos_world_m"])*1000:.1f}mm',flush=True)
        for _ in range(10):final,last_row=execute(target,ideal,'fixed-target-hold',count,5,math.dist(reference.position,runtime.state()['tcp_world_m']))
        initial_distance=math.dist(initial['tcp_world_m'],initial['object_pos_world_m']);final_distance=math.dist(final['tcp_world_m'],final['object_pos_world_m'])
        report.update(status='completed',final=final,action_end=action_end,first_input_sha256=report['queries'][0]['input']['input_sha256'],
            all_followup_inputs_changed=all(q['feedback_change']['input_changed'] for q in report['queries'][1:]),
            initial_object_distance_m=initial_distance,final_object_distance_m=final_distance,distance_reduction_m=initial_distance-final_distance,
            final_xy_offset_m=math.hypot(final['tcp_world_m'][0]-final['object_pos_world_m'][0],final['tcp_world_m'][1]-final['object_pos_world_m'][1]),
            command_delta_sum_I_m=reference.command_sum.tolist(),actual_delta_I_m=(np.array(final['tcp_world_m'])-initial['tcp_world_m']).tolist(),
            final_ideal_position_error_m=last_row['ideal_error_m'],final_applied_target_error_m=last_row['applied_error_m'],
            final_ideal_rotation_error_deg=last_row['ideal_rotation_error_deg'],final_applied_rotation_error_deg=last_row['applied_rotation_error_deg'],
            max_applied_backlog_m=maximum_backlog,max_applied_backlog_deg=maximum_rotation,max_applied_target_error_m=maximum_applied,
            max_pre_update_ideal_gap_m=maximum_prior_gap,
            backlog_definition='pre-step applied backlog includes the newly issued increment; pre-update ideal gap excludes it')
        runtime.save_images('final');return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        if trace is not None:trace.close()
        if runtime is not None:runtime.trace.close()
        save();print(f'[Online reference] Report: {output/"online_registered_result.json"}',flush=True)
