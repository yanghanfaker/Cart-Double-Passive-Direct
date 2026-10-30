"""Frozen original/horizontal-flip policy query comparison on clean proxy scenes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import time

from frozen_coordinate_probe import digest_observation
from frozen_observation_probe import analyze_chunk
from openpi_step_probe import libero_state_vector,read_server_metadata,validate_action_chunk

CONDITIONS=('no_flip','flip_x')
CASES=('arm-only','arm-and-target-xy')


def registered_state(audit,calibration):
    import numpy as np
    from camera_alignment_probe import quat_xyzw
    from geometry_audit import multiply,pose_matrix
    if audit.get('status')!='completed' or not audit.get('same_sensor_ik_body') or not audit['sensor_offset_closure']['within_0p1mm_0p05deg']:
        raise ValueError('completed clean geometry with valid TCP closure required')
    if audit['sensor_body_offset']['position_m']!=[0.,0.,.1034] or audit['sensor_body_offset']['quat_xyzw']!=[0.,0.,0.,1.]:
        raise ValueError('expected 103.4mm source sensor with hand-aligned orientation')
    if not calibration.get('candidate_geometry_within_thresholds'):raise ValueError('validated multi-pose registration required')
    R=np.asarray(calibration['world_from_isaac_rotation'],dtype=float);t=np.asarray(calibration['world_from_isaac_translation_m'],dtype=float)
    if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
        raise ValueError('invalid proper world registration')
    hand=audit['sensor_body_world'];H=np.array(pose_matrix(hand))
    source=np.array(audit['sensor_tcp_world']['position_m']);corrected=source+H[:3,:3]@np.array([0.,0.,-.0064])
    position=R@corrected+t
    # Compose a near-identity registration quaternion with the ORIGINAL hand
    # quaternion. Do not rederive/canonicalize the whole hand rotation matrix.
    rotation_q=quat_xyzw(R);mapped_q=multiply(rotation_q,hand['quat_xyzw'])
    state=libero_state_vector(position.tolist(),mapped_q,audit['finger_positions_isaac_m'])
    target=R@np.asarray(audit['object_root_world']['position_m'])+t
    if not np.isfinite(state).all() or not np.isfinite(target).all():raise ValueError('nonfinite mapped state or diagnostic target')
    provenance={'registration_rotation':R.tolist(),'registration_translation_m':t.tolist(),
        'source_sensor_position_world_m':source.tolist(),'tcp_delta_hand_m':[0,0,-.0064],
        'source_hand_quat_xyzw':hand['quat_xyzw'],'registration_quat_xyzw':rotation_q,
        'mapped_hand_quat_xyzw':mapped_q,'mapped_tcp_world_m':position.tolist(),
        'semantics':'LIBERO world TCP97mm xyz; hand rotation vector without w-sign canonicalization; Isaac finger q1,-q2',
        'diagnostic_target_usage':'mapped cube center only for projection analysis, never in model observation'}
    return state,target.tolist(),provenance


def prepare(source,folder):
    import numpy as np
    from PIL import Image
    source=Path(source);folder=Path(folder);root=source.parent;manifest=json.loads(source.read_bytes())
    if manifest.get('status')!='completed' or not manifest.get('all_geometry_checks_passed'):raise ValueError('completed valid clean capture required')
    calibration_payload=(root/'world_registration.json').read_bytes();calibration=json.loads(calibration_payload)
    (folder/'source_world_registration.json').write_bytes(calibration_payload)
    (folder/'source_clean_manifest.json').write_bytes(source.read_bytes())
    prepared=[]
    for name in CASES:
        matches=[c for c in manifest['cases'] if c['name']==name]
        if len(matches)!=1:raise ValueError('missing or duplicate source case '+name)
        entry=matches[0];result_path=root/entry['result'];payload=result_path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=entry['result_sha256']:raise ValueError('clean result SHA mismatch')
        result=json.loads(payload);capture=result_path.parent
        if result['status']!='completed' or not result['geometry_checks_passed'] or result['calibration_markers_present']:
            raise ValueError('invalid/marked clean capture')
        geometry_bytes=(capture/'geometry_audit.json').read_bytes();audit=json.loads(geometry_bytes)
        setup_bytes=(capture/'camera_alignment_setup.json').read_bytes();setup=json.loads(setup_bytes)
        if setup.get('scene_contains_markers') or any(c['markers'] for c in setup['cameras'].values()):raise ValueError('source has calibration markers')
        state,target,provenance=registered_state(audit,calibration)
        item={'name':name,'state_registered_float32':np.asarray(state,dtype=np.float32).tolist(),
            'state_provenance':provenance,'diagnostic_cube_world_registered_m':target,
            'source_result_sha256':entry['result_sha256'],'source_geometry_sha256':hashlib.sha256(geometry_bytes).hexdigest(),
            'source_setup_sha256':hashlib.sha256(setup_bytes).hexdigest(),'source_cameras':result['cameras'],
            'image_files':{},'image_png_sha256':{},'image_array_sha256':{},'queries':[]}
        (folder/(name+'_source_geometry.json')).write_bytes(geometry_bytes)
        (folder/(name+'_source_setup.json')).write_bytes(setup_bytes)
        (folder/(name+'_source_result.json')).write_bytes(payload)
        variants={c:{} for c in CONDITIONS}
        for view in ('external_camera','wrist_camera'):
            meta=result['cameras'][view]
            if not meta['camera_parameters_passed']:raise ValueError('camera metadata check failed')
            item['image_files'][view]={};item['image_png_sha256'][view]={};item['image_array_sha256'][view]={}
            arrays={}
            for condition,key in [('no_flip','no_flip'),('flip_x','horizontal_flip')]:
                file=meta['frames'][key]
                if Path(file).name!=file or any(c in file for c in '/\\'):raise ValueError('unsafe source image name')
                content=(capture/file).read_bytes()
                with Image.open(capture/file) as picture:image=np.asarray(picture).copy()
                if image.shape!=(224,224,3) or image.dtype!=np.uint8:raise ValueError('source must already be 224 RGB')
                if condition=='no_flip' and hashlib.sha256(content).hexdigest()!=audit['cameras'][view]['policy_png_sha256']:
                    raise ValueError('source policy PNG SHA mismatch')
                dest=f'{name}_{view}_{condition}.png';(folder/dest).write_bytes(content)
                item['image_files'][view][condition]=dest
                item['image_png_sha256'][view][condition]=hashlib.sha256(content).hexdigest()
                item['image_array_sha256'][view][condition]=hashlib.sha256(image.tobytes()).hexdigest()
                arrays[condition]=image;variants[condition][view]=image
            if not np.array_equal(arrays['no_flip'][:,::-1],arrays['flip_x']):raise ValueError('source flip is not exactly horizontal')
        prepared.append((item,variants))
    return prepared,hashlib.sha256(calibration_payload).hexdigest()


def run_queries(client,item,variants,prompt,repeats,save,on_attempt=None):
    import numpy as np
    for repeat in range(repeats):
        for condition in (CONDITIONS if repeat%2==0 else CONDITIONS[::-1]):
            frames=variants[condition]
            obs={'observation/state':np.asarray(item['state_registered_float32'],dtype=np.float32).copy(),
                'observation/image':frames['external_camera'].copy(),'observation/wrist_image':frames['wrist_camera'].copy(),'prompt':prompt}
            digest=digest_observation(obs)
            if on_attempt is not None:on_attempt({'snapshot':item['name'],'condition':condition,'repeat':repeat+1,'input_sha256':digest})
            start=time.perf_counter();response=client.infer(obs);elapsed=time.perf_counter()-start
            if digest_observation(obs)!=digest:raise RuntimeError('policy client mutated frozen observation')
            chunk=response['actions']
            if hasattr(chunk,'tolist'):chunk=chunk.tolist()
            validate_action_chunk(chunk)
            if len(chunk)<10:raise ValueError('need at least ten predicted actions for prefix comparison')
            analysis=analyze_chunk(chunk,item['state_registered_float32'],item['diagnostic_cube_world_registered_m'])
            row={'condition':condition,'repeat':repeat+1,'input_sha256':digest,'inference_wall_s':elapsed,
                'state_sent_float32':obs['observation/state'].tolist(),'actions_executed':0,
                **analysis,'first10':analyze_chunk(chunk[:10],item['state_registered_float32'],item['diagnostic_cube_world_registered_m'])['whole_chunk']}
            save(row)
            print(f'[Image A/B] {item["name"]} {condition} repeat={repeat+1}; first10 projected approach={row["first10"]["bounded_local_approach_projection_sum_mm"]:.3f} mm; executed=0',flush=True)


def summarize(queries):
    import statistics
    result={}
    for condition in CONDITIONS:
        group=[q for q in queries if q['condition']==condition];result[condition]={}
        for window in ('first5','first10','whole_chunk'):
            values=[q[window]['bounded_local_approach_projection_sum_mm'] for q in group]
            if values:result[condition][window]={'queries':len(values),'mean_mm':statistics.mean(values),
                'sample_std_mm':statistics.stdev(values) if len(values)>1 else None,'min_mm':min(values),'max_mm':max(values),
                'positive_queries':sum(v>0 for v in values),'close_requests':sum(q[window]['close_requests'] for q in group)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);repo=Path(__file__).resolve().parent.parent
    parser.add_argument('--source-manifest',type=Path)
    parser.add_argument('--output-root',type=Path,default=repo/'diagnostics/openpi_image_ab')
    parser.add_argument('--server-host',default='127.0.0.1');parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--prompt',default='pick up the cube')
    parser.add_argument('--repeats',type=int,choices=range(5,21),default=10)
    parser.add_argument('--policy-server-log',type=Path)
    args=parser.parse_args()
    sources=sorted((repo/'diagnostics/clean_scene_capture').glob('*/run.json'))
    source=args.source_manifest or next((p for p in reversed(sources) if json.loads(p.read_text()).get('all_geometry_checks_passed')),None)
    if source is None:parser.error('no valid completed clean capture')
    source=source.absolute();folder=args.output_root.absolute()/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','actions_executed':0,'policy_queries':0,'inference_attempts':0,'prompt':args.prompt,
        'source_manifest':str(source),'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'server_host':args.server_host,'server_port':args.server_port,'repeats_per_condition':args.repeats,'snapshots':[],
        'scope':'input sensitivity on two distinct cube-proxy scenes; repeated unpaired stochastic queries, no simulation or deployment',
        'condition_change':'both cameras horizontally flipped together; state and prompt unchanged within each snapshot',
        'action_frame':'LIBERO reference world; hypothetical caps/projections only, no IK or rotation lever-arm execution'}
    def save():(folder/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    try:
        prepared,sha=prepare(source,folder);report['world_registration_sha256']=sha;report['snapshots']=[item for item,_ in prepared];save()
        if args.policy_server_log:
            payload=args.policy_server_log.read_bytes()[:128*1024];(folder/'policy_server_startup.log').write_bytes(payload)
            report['server_log_prefix_sha256']=hashlib.sha256(payload).hexdigest();report['server_log_scope']='user-supplied prefix; not endpoint/checkpoint ownership proof'
        # Fail before the client's retry loop if the existing service is unavailable.
        with socket.create_connection((args.server_host,args.server_port),timeout=3):pass
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client=WebsocketClientPolicy(host=args.server_host,port=args.server_port);report['server_metadata']=read_server_metadata(client)
        report['status']='running';save()
        for item,variants in prepared:
            def append(row):
                item['queries'].append(row);report['policy_queries']+=1;report.pop('pending_query',None);item['summary']=summarize(item['queries']);save()
            def attempt(context):
                report['inference_attempts']+=1;report['pending_query']=context;save()
            run_queries(client,item,variants,args.prompt,args.repeats,append,on_attempt=attempt)
        report['status']='completed'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        save();print(f'[Image A/B] Report: {folder/"report.json"}',flush=True)


if __name__=='__main__':main()
