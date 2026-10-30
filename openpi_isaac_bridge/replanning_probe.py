"""Three observations/queries in one continuing scene, with only a final hold."""
import json
from pathlib import Path
import shutil
import socket

from online_registered_probe import prepare_capture_observation,query_once
from openpi_step_probe import read_server_metadata
from registered_replay_probe import run_registered_replay,global_envelope


def capture_round(base,arm,folder,initial,tensor,index):
    import numpy as np
    from PIL import Image
    folder=Path(folder);initial=Path(initial)
    if index==1:
        for name in ('geometry_audit.json','camera_alignment_setup.json','external_camera.png','wrist_camera.png',
                'external_camera_policy224.png','wrist_camera_policy224.png'):
            shutil.copyfile(initial/name,folder/name)
    else:
        from geometry_audit import capture_geometry
        shutil.copyfile(initial/'camera_alignment_setup.json',folder/'camera_alignment_setup.json')
        images={}
        for name in ('external_camera','wrist_camera'):
            images[name]=tensor(base.scene[name].data.output['rgb'])[0,...,:3].detach().cpu().numpy().copy()
            Image.fromarray(images[name]).save(folder/f'{name}.png')
        capture_geometry(base,base.cfg,arm,images,folder,tensor,warmup_controls=0)
    geometry=json.loads((folder/'geometry_audit.json').read_bytes())
    if geometry['scene_setup'].get('contains_nonphysical_camera_calibration_markers',True):raise RuntimeError('unexpected calibration markers')
    step=int(base.sim.get_physics_step_count())
    if geometry['physics_step_count_before']!=step or geometry['physics_step_count_after']!=step:raise RuntimeError('capture is not bound to current physics step')
    for name in ('external_camera','wrist_camera'):
        with Image.open(folder/f'{name}_policy224.png') as im:rgb=np.asarray(im).copy()
        Image.fromarray(np.ascontiguousarray(rgb[:,::-1])).save(folder/f'{name}_policy224_flip_x.png')
    return geometry,step


def feedback_change(previous,current):
    import numpy as np
    if current['capture_physics_step']-previous['capture_physics_step']!=25:raise RuntimeError('expected exactly five controls between observations')
    for name in ('external_camera','wrist_camera'):
        if current['camera_frames'][name]<=previous['camera_frames'][name]:raise RuntimeError('camera frame did not advance after motion')
    before=previous['input'];after=current['input']
    return {'input_changed':before['input_sha256']!=after['input_sha256'],
        'state_changed':before['state_sent_float32']!=after['state_sent_float32'],
        'position_change_m':float(np.linalg.norm(np.array(after['state_sent_float32'][:3])-before['state_sent_float32'][:3])),
        'image_array_changed':{key:before['frames'][key]['array_sha256']!=after['frames'][key]['array_sha256'] for key in before['frames']}}


def run_replanning(env,base,arm,settings,output,tensor):
    import numpy as np
    from kinematic_attribution import rotate
    output=Path(output)
    report={'status':'preparing','mode':'three-round-replan','policy_queries':0,'inference_attempts':0,
        'completed_action_periods':0,'completed_settle_periods':0,'rounds':[],
        'scope':'three queries in one scene, five actions per query; no reset or intermediate hold; final ten-period absolute target hold',
        'timing':'synchronous simulation paused during each inference; velocity retained, no real-time guarantee'}
    def save():(output/'online_registered_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    def totals():
        for key in ('completed_action_periods','completed_settle_periods'):
            report[key]=sum(json.loads((output/r['motion_result']).read_bytes()).get(key,0) for r in report['rounds'] if 'motion_result' in r and (output/r['motion_result']).is_file())
    save()
    try:
        with socket.create_connection((settings['server_host'],settings['server_port']),timeout=3):pass
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port'])
        report['server_metadata']=read_server_metadata(client)
        reference=None
        for index in range(1,4):
            folder=output/f'round_{index:02d}';folder.mkdir()
            row={'round':index,'status':'capturing','motion_result':f'round_{index:02d}/registered_replay_result.json'}
            report['rounds'].append(row);save()
            geometry,step=capture_round(base,arm,folder,output,tensor,index)
            obs,request,meta=prepare_capture_observation(folder,settings['calibration'],settings['prompt'],geometry)
            row.update(capture_physics_step=step,input=meta,
                camera_frames={name:int(np.asarray(geometry['cameras'][name]['frame']).reshape(-1)[0]) for name in ('external_camera','wrist_camera')})
            if reference is None:
                hand=geometry['sensor_body_world']
                reference={'tcp97_world_m':(np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.097])).tolist(),
                    'hand_quat_world_xyzw':hand['quat_xyzw'],'object_position_world_m':geometry['object_root_world']['position_m']}
                report['global_reference']=reference
            else:row['feedback_change']=feedback_change(report['rounds'][-2],row)
            hand=geometry['sensor_body_world']
            current={'tcp97_world_m':(np.array(hand['position_m'])+rotate(hand['quat_xyzw'],[0,0,.097])).tolist(),
                'hand_quat_world_xyzw':hand['quat_xyzw'],'object_position_world_m':geometry['object_root_world']['position_m']}
            row['capture_global_envelope']=global_envelope(current,reference)
            report['status']='querying';row['status']='querying';report['inference_attempts']+=1;save()
            def received(response,elapsed):
                report['policy_queries']+=1;row['inference_wall_s']=elapsed
                chunk=response.get('actions') if isinstance(response,dict) else response
                if hasattr(chunk,'tolist'):chunk=chunk.tolist()
                try:
                    (folder/'online_model_response.json').write_text(json.dumps({'actions':chunk},indent=2,allow_nan=False),encoding='utf-8')
                except (TypeError,ValueError):(folder/'invalid_model_response.txt').write_text(repr(chunk),encoding='utf-8')
                save()
            request,elapsed=query_once(client,obs,request,on_response=received)
            row['post_query_physics_step']=int(base.sim.get_physics_step_count())
            if row['post_query_physics_step']!=step:raise RuntimeError('physics advanced during query')
            request['source_repeat']=index
            (folder/'online_replay_request.json').write_text(json.dumps(request,indent=2,allow_nan=False),encoding='utf-8')
            report['status']='executing';row['status']='executing';save()
            result=run_registered_replay(env,base,arm,request,folder,tensor,settle_steps=10 if index==3 else 0,global_reference=reference)
            row.update(status='completed',chunk_end_target_reached=result['final_target_reached'],
                final_tcp97_target_error_m=result['final_tcp97_target_error_m'],final_rotation_error_deg=result['final_rotation_error_deg'])
            totals();save();print(f'[Replanning] round {index}/3 done; controls={report["completed_action_periods"]}, holds={report["completed_settle_periods"]}',flush=True)
        report.update(status='completed',final_target_reached=report['rounds'][-1]['chunk_end_target_reached'],
            all_followup_inputs_changed=all(r['feedback_change']['input_changed'] for r in report['rounds'][1:]))
        return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        if report['rounds'] and report['rounds'][-1]['status']!='completed':report['rounds'][-1]['status']='failed'
        raise
    finally:
        totals();save();print(f'[Replanning] Report: {output/"online_registered_result.json"}',flush=True)
