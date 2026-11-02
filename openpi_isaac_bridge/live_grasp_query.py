"""Archive each fresh registered inference with variable executed-prefix timing."""
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from geometry_audit import multiply
from online_registered_probe import prepare_capture_observation,query_once
from replanning_probe import capture_round
from replay_probe import quaternion_distance_deg


def feedback_check(previous,current):
    controls=previous['executed_prefix_steps']
    if type(controls) is not int or controls<1:raise RuntimeError('no completed prefix before next observation')
    if current['capture_physics_step']-previous['capture_physics_step']!=5*controls:raise RuntimeError('capture interval differs from executed prefix')
    for name in ('external_camera','wrist_camera'):
        if current['camera_frames'][name]<=previous['camera_frames'][name]:raise RuntimeError('camera frame did not advance')
    return {'executed_controls_between_observations':controls,
        'input_changed':current['input']['input_sha256']!=previous['input']['input_sha256'],
        'state_changed':current['input']['state_sent_float32']!=previous['input']['state_sent_float32'],
        'image_array_changed':{key:current['input']['frames'][key]['array_sha256']!=previous['input']['frames'][key]['array_sha256'] for key in previous['input']['frames']}}


def make_live_query(runtime,settings,report,save,client):
    output=runtime.output;base=runtime.base;arm=runtime.arm;tensor=runtime.tensor
    def query(index):
        row={'query':index,'status':'capturing','executed_prefix_steps':0};report['queries'].append(row);save()
        archive=output/'queries'/f'query_{index:04d}.zip'
        with tempfile.TemporaryDirectory(prefix='hybrid_capture_') as tmp:
            folder=Path(tmp)
            try:
                geometry,step=capture_round(base,arm,folder,output,tensor,index)
                obs,request,meta=prepare_capture_observation(folder,settings['calibration'],settings['prompt'],geometry)
                row.update(input=meta,capture_physics_step=step,
                    camera_frames={n:int(np.array(geometry['cameras'][n]['frame']).reshape(-1)[0]) for n in ('external_camera','wrist_camera')})
                if index>1:row['feedback_change']=feedback_check(report['queries'][-2],row)
                row['status']='querying';report['inference_attempts']+=1;save()
                def received(response,elapsed):
                    report['policy_queries']+=1;row['inference_wall_s']=elapsed
                    chunk=response.get('actions') if isinstance(response,dict) else response
                    if hasattr(chunk,'tolist'):chunk=chunk.tolist()
                    try:(folder/'online_model_response.json').write_text(json.dumps({'actions':chunk},allow_nan=False),encoding='utf-8')
                    except (TypeError,ValueError):(folder/'invalid_model_response.txt').write_text(repr(chunk),encoding='utf-8')
                    save()
                request,_=query_once(client,obs,request,on_response=received)
                row['post_query_physics_step']=int(base.sim.get_physics_step_count())
                if row['post_query_physics_step']!=step:raise RuntimeError('physics advanced during inference')
                state=runtime.state();R=np.array(request['world_rotation_L_from_I']);t=np.array(request['world_translation_L_from_I_m'])
                if np.linalg.norm(R@np.array(state['tcp_world_m'])+t-request['source_state_float32'][:3])>.0005 or quaternion_distance_deg(multiply(request['registration_quat_xyzw'],state['hand_quat_world_xyzw']),request['source_mapped_hand_quat_xyzw'])>.1:
                    raise RuntimeError('live TCP differs from queried state')
                if np.max(np.abs(np.array(state['joint_positions_rad'])-request['source_arm_positions_rad']))>.001:raise RuntimeError('live joints differ from queried state')
                request['source_repeat']=index
                (folder/'online_replay_request.json').write_text(json.dumps(request,allow_nan=False),encoding='utf-8')
                row['status']='validated';return request
            except Exception as error:row.update(status='failed',error=str(error));raise
            finally:
                (folder/'query_record.json').write_text(json.dumps(row,allow_nan=False),encoding='utf-8')
                with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                    for file in sorted(folder.iterdir()):
                        if file.is_file():z.write(file,file.name)
                row.update(archive=archive.relative_to(output).as_posix(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());save()
    return query
