"""One live registered observation/query followed by the proven short replay."""
import hashlib
import json
from pathlib import Path
import socket
import time

from frozen_coordinate_probe import digest_observation
from frozen_image_probe import registered_state
from openpi_step_probe import read_server_metadata,validate_action_chunk
from registered_replay_probe import validate_replay,run_registered_replay


def prepare_observation(output,calibration,prompt):
    import numpy as np
    from PIL import Image
    output=Path(output);geometry=json.loads((output/'geometry_audit.json').read_bytes())
    clean=json.loads((output/'clean_scene_result.json').read_bytes())
    if not clean['geometry_checks_passed'] or clean['calibration_markers_present']:raise ValueError('need valid current clean capture')
    if not isinstance(prompt,str) or not prompt.strip():raise ValueError('empty prompt')
    state,_,provenance=registered_state(geometry,calibration)
    obs={'observation/state':np.asarray(state,dtype=np.float32),'prompt':prompt};frames={}
    for view,key in [('external_camera','observation/image'),('wrist_camera','observation/wrist_image')]:
        plain=output/f'{view}_policy224.png';flipped=output/f'{view}_policy224_flip_x.png'
        if hashlib.sha256(plain.read_bytes()).hexdigest()!=geometry['cameras'][view]['policy_png_sha256']:raise ValueError('current policy image SHA mismatch')
        with Image.open(plain) as im:original=np.asarray(im).copy()
        with Image.open(flipped) as im:image=np.asarray(im).copy()
        if image.shape!=(224,224,3) or image.dtype!=np.uint8 or not np.array_equal(image,original[:,::-1]):raise ValueError('expected exact horizontal flip of current 224 RGB')
        obs[key]=np.ascontiguousarray(image)
        frames[key]={'file':flipped.name,'png_sha256':hashlib.sha256(flipped.read_bytes()).hexdigest(),
            'array_sha256':hashlib.sha256(obs[key].tobytes()).hexdigest()}
    joint_map=dict(zip(geometry['joint_names'],geometry['joint_positions_rad']))
    request={'condition':'flip_x','source_repeat':1,'source_input_sha256':digest_observation(obs),
        'source_state_float32':obs['observation/state'].tolist(),
        'source_mapped_hand_quat_xyzw':provenance['mapped_hand_quat_xyzw'],
        'registration_quat_xyzw':provenance['registration_quat_xyzw'],
        'world_rotation_L_from_I':provenance['registration_rotation'],
        'world_translation_L_from_I_m':provenance['registration_translation_m'],
        'source_arm_positions_rad':[joint_map[f'panda_joint{k}'] for k in range(1,8)]}
    return obs,request,{'state_provenance':provenance,'frames':frames,'input_sha256':request['source_input_sha256'],
        'state_sent_float32':obs['observation/state'].tolist(),'prompt':prompt,
        'scope':'current capture after warmup, not a previous-run image or state'}


def query_once(client,observation,request,on_response=None):
    digest=digest_observation(observation);start=time.perf_counter();response=client.infer(observation)
    elapsed=time.perf_counter()-start
    if on_response is not None:on_response(response,elapsed)
    if digest_observation(observation)!=digest:raise RuntimeError('policy client mutated current observation')
    chunk=response['actions']
    if hasattr(chunk,'tolist'):chunk=chunk.tolist()
    # Validate the entire future chunk, including actions that will not execute.
    validate_action_chunk(chunk)
    result=validate_replay({**request,'raw_action_chunk':chunk})
    return result,elapsed


def run_online_registered(env,base,arm,settings,output,tensor):
    output=Path(output)
    report={'status':'preparing','policy_queries':0,'inference_attempts':0,
        'server_host':settings['server_host'],'server_port':settings['server_port'],
        'completed_action_periods':0,'completed_settle_periods':0,
        'scope':'one current observation and one inference; first five actions then ten fixed-target settle periods; no replanning',
        'motion_result':'registered_replay_result.json'}
    def save():(output/'online_registered_result.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        obs,request,meta=prepare_observation(output,settings['calibration'],settings['prompt'])
        report['input']=meta;save()
        with socket.create_connection((settings['server_host'],settings['server_port']),timeout=3):pass
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port'])
        report['server_metadata']=read_server_metadata(client)
        step=int(base.sim.get_physics_step_count());report['capture_physics_step']=step
        report.update(status='querying',inference_attempts=1);save()
        def received(response,elapsed):
            report.update(policy_queries=1,inference_wall_s=elapsed)
            chunk=response.get('actions') if isinstance(response,dict) else response
            if hasattr(chunk,'tolist'):chunk=chunk.tolist()
            try:
                (output/'online_model_response.json').write_text(json.dumps({'actions':chunk},indent=2,allow_nan=False),encoding='utf-8')
                report['response_file']='online_model_response.json'
            except (TypeError,ValueError):
                (output/'invalid_model_response.txt').write_text(repr(chunk),encoding='utf-8')
                report['response_file']='invalid_model_response.txt'
            save()
        request,elapsed=query_once(client,obs,request,on_response=received)
        report.update(policy_queries=1,inference_wall_s=elapsed,raw_action_chunk=request['raw_action_chunk'])
        after=int(base.sim.get_physics_step_count());report['post_query_physics_step']=after
        if step!=after:raise RuntimeError('physics advanced while querying; refusing stale observation execution')
        (output/'online_replay_request.json').write_text(json.dumps(request,indent=2,allow_nan=False),encoding='utf-8')
        report['status']='executing';save()
        result=run_registered_replay(env,base,arm,request,output,tensor)
        report.update(status='completed',final_target_reached=result['final_target_reached'],
            completed_action_periods=result['completed_action_periods'],completed_settle_periods=result['completed_settle_periods'])
        return report
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        motion=output/'registered_replay_result.json'
        if motion.is_file():
            d=json.loads(motion.read_bytes());report['motion_status']=d['status']
            report['completed_action_periods']=d['completed_action_periods'];report['completed_settle_periods']=d['completed_settle_periods']
        save();print(f'[Online registered] Report: {output/"online_registered_result.json"}',flush=True)
