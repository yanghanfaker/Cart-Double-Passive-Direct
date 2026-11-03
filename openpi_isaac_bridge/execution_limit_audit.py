"""Frozen perception/direction probes and matched archived-action cap sweeps."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from grasp_probe import GraspAbort


def limit_cases():
    from execution_hybrid import hybrid_cases
    base=copy.deepcopy(hybrid_cases()[0]);base.pop('hybrid_role')
    start={**copy.deepcopy(base),'name':'limits-reference-start','hybrid_role':'reference','limit_audit_role':'reference'}
    result=[start]
    variants=[('center',[0.,0.],'both','pick up the cube'),
        ('x-minus',[-.02,0.],'both','pick up the cube'),('x-plus',[.02,0.],'both','pick up the cube'),
        ('y-minus',[0.,-.02],'both','pick up the cube'),('y-plus',[0.,.02],'both','pick up the cube'),
        ('external-blank',[0.,0.],'external-blank','pick up the cube'),
        ('wrist-blank',[0.,0.],'wrist-blank','pick up the cube'),
        ('object-prompt',[0.,0.],'both','pick up the object')]
    for name,offset,views,prompt in variants:
        result.append({**copy.deepcopy(base),'name':'limits-frozen-'+name,'limit_audit_role':'frozen',
            'object_offset_xy_m':offset,'views':views,'prompt':prompt,'query_repeats':6,
            'requires':['limits-reference-start'],'requires_success':['limits-reference-start']})
    for prefix in (1,5):
        for cap in (.0005,.001,.002):
            result.append({**copy.deepcopy(base),'name':f'limits-replay-p{prefix}-cap{cap*1000:g}',
                'limit_audit_role':'replay','source_prefix':prefix,'cap_m':cap,'replay_controls':100,
                'requires':['limits-reference-start'],'requires_success':['limits-reference-start']})
    result.append({**base,'name':'limits-reference-end','hybrid_role':'reference','limit_audit_role':'reference'})
    return result


def prepare_replay_sources(source_manifest,destination):
    from execution_hybrid import hybrid_cases
    from native_libero_rollout import save_json
    source_manifest=Path(source_manifest);destination=Path(destination);destination.mkdir()
    payload=source_manifest.read_bytes();m=json.loads(payload)
    if m['status']!='completed' or [x['case'] for x in m['cases']]!=hybrid_cases():raise ValueError('completed hybrid matrix required')
    for row in m['cases']:
        if row['case'].get('hybrid_role')=='reference' and not (row.get('summary',{}).get('grasp_result') or {}).get('success'):
            raise ValueError('both prior program references must pass')
    (destination/'source_hybrid_run.json').write_bytes(payload);sources={}
    for prefix in (1,5):
        row=next(x for x in m['cases'] if x['case']['name']==f'hybrid-p{prefix}-geometry-r1')
        path=source_manifest.parent/row['result'];raw=path.read_bytes();r=json.loads(raw)
        if row['status']!='completed' or hashlib.sha256(raw).hexdigest()!=row['summary']['result_sha256'] or r['status']!='completed':raise ValueError('hybrid source result SHA/status differs')
        if len(r['trace'])<100 or any(x['phase']!='policy-approach' for x in r['trace'][:100]):raise ValueError('need first 100 executed policy approach controls')
        # Verify every archive that supplied the selected prefix, not only its summary.
        for q in r['queries'][:math.ceil(100/prefix)]:
            archive=(path.parent/q['archive']).read_bytes()
            if hashlib.sha256(archive).hexdigest()!=q['archive_sha256']:raise ValueError('source inference archive SHA differs')
            import io,zipfile
            with zipfile.ZipFile(io.BytesIO(archive)) as z:response=json.loads(z.read('online_model_response.json'))
            index=(q['query']-1)*prefix
            if [x['raw_action'] for x in r['trace'][index:index+prefix]]!=response['actions'][:prefix]:raise ValueError('archived actions differ from executed source')
        name=f'p{prefix}_source_result.json';(destination/name).write_bytes(raw)
        request={'source_prefix':prefix,'source_result_file':name,'source_result_sha256':hashlib.sha256(raw).hexdigest(),
            'source_manifest_sha256':hashlib.sha256(payload).hexdigest(),
            'actions':[x['raw_action'] for x in r['trace'][:100]]}
        output=destination/f'p{prefix}.json';save_json(output,request);load_replay(output)
        sources[prefix]=output
    return sources


def load_replay(path):
    from openpi_step_probe import validate_action_chunk
    path=Path(path);request=json.loads(path.read_bytes());raw=(path.parent/request['source_result_file']).read_bytes()
    manifest=(path.parent/'source_hybrid_run.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=request['source_result_sha256'] or hashlib.sha256(manifest).hexdigest()!=request['source_manifest_sha256']:
        raise ValueError('replay source SHA mismatch')
    source=json.loads(raw);m=json.loads(manifest)
    case_name=f"hybrid-p{request['source_prefix']}-geometry-r1"
    row=next(x for x in m['cases'] if x['case']['name']==case_name)
    if row['summary']['result_sha256']!=request['source_result_sha256'] or source['settings']['execution_case']['name']!=case_name:
        raise ValueError('replay manifest does not identify this source')
    actions=request['actions'];validate_action_chunk(actions)
    if len(actions)!=100 or actions!=[x['raw_action'] for x in source['trace'][:100]]:raise ValueError('replay input must exactly match source first 100 executed actions')
    return request,source


def direction_metrics(actions,state,rotation):
    from openpi_step_probe import validate_action_chunk
    validate_action_chunk(actions)
    R=np.asarray(rotation);desired=np.array(state['object_pos_world_m'])-state['tcp_world_m'];desired[2]+=.005
    vectors=(R.T@(np.clip(np.array(actions)[:,:3],-1,1)*.05).T).T
    def metrics(v):
        n=float(np.linalg.norm(v));d=float(np.linalg.norm(desired));xy=float(np.linalg.norm(v[:2]));dy=float(np.linalg.norm(desired[:2]))
        return {'decoded_world_delta_m':v.tolist(),'desired_dot':float(v@desired),
            'cosine_to_grasp_point':None if n*d<1e-12 else float(v@desired/(n*d)),
            'xy_cosine_to_object':None if xy*dy<1e-12 else float(v[:2]@desired[:2]/(xy*dy))}
    return {'desired_world_delta_m':desired.tolist(),'first_action':metrics(vectors[0]),
        'mean_first5':metrics(vectors[:5].mean(axis=0)),
        'scope':'directional diagnostic at the frozen state; not a motion-planning success criterion'}


def run_frozen(r,settings,report,save):
    import tempfile,zipfile
    from PIL import Image
    from openpi_health import check_openpi_health
    from openpi_step_probe import read_server_metadata
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from replanning_probe import capture_round
    from online_registered_probe import prepare_capture_observation,query_once
    from frozen_coordinate_probe import digest_observation
    case=settings['execution_case'];report.update(policy_queries=0,inference_attempts=0,frozen_queries=[])
    report['server_health']=check_openpi_health(settings['server_host'],settings['server_port'])
    client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port']);report['server_metadata']=read_server_metadata(client)
    first=r.state();initial_step=int(r.base.sim.get_physics_step_count());archive=r.output/'frozen_query_bundle.zip'
    with tempfile.TemporaryDirectory(prefix='limits_frozen_') as temp:
        folder=Path(temp)
        try:
            geometry,capture_step=capture_round(r.base,r.arm,folder,r.output,r.tensor,1)
            if capture_step!=initial_step:raise RuntimeError('frozen capture advanced physics')
            obs,request,meta=prepare_capture_observation(folder,settings['calibration'],settings['prompt'],geometry)
            if case['views']!='both':
                key='observation/image' if case['views']=='external-blank' else 'observation/wrist_image'
                obs[key]=np.zeros_like(obs[key])
            consumed={'prompt':settings['prompt'],'state_float32':obs['observation/state'].tolist(),'views':case['views'],
                'base_capture':meta,'input_sha256':digest_observation(obs),'images':{}}
            for view,key in (('external','observation/image'),('wrist','observation/wrist_image')):
                name=f'consumed_{view}.png';Image.fromarray(obs[key]).save(folder/name)
                consumed['images'][key]={'file':name,'array_sha256':hashlib.sha256(obs[key].tobytes()).hexdigest()}
            (folder/'consumed_input.json').write_text(json.dumps(consumed,allow_nan=False),encoding='utf-8')
            request.update(source_input_sha256=consumed['input_sha256'],image_intervention=case['views'])
            report['frozen_input']=consumed;save()
            for i in range(1,case['query_repeats']+1):
                row={'sample':i,'status':'querying','input_sha256':digest_observation(obs),'physics_step':initial_step}
                report['frozen_queries'].append(row);report['inference_attempts']+=1;save()
                def received(response,elapsed):
                    report['policy_queries']+=1;row['inference_wall_s']=elapsed
                    chunk=response.get('actions') if isinstance(response,dict) else response
                    if hasattr(chunk,'tolist'):chunk=chunk.tolist()
                    try:(folder/f'response_{i:02d}.json').write_text(json.dumps({'actions':chunk},allow_nan=False),encoding='utf-8')
                    except (TypeError,ValueError):(folder/f'invalid_response_{i:02d}.txt').write_text(repr(chunk),encoding='utf-8')
                    save()
                try:
                    returned,_=query_once(client,obs,request,on_response=received)
                    if int(r.base.sim.get_physics_step_count())!=initial_step or r.state()!=first:raise RuntimeError('state or physics changed during frozen queries')
                    if row['input_sha256']!=consumed['input_sha256'] or returned['source_input_sha256']!=consumed['input_sha256']:raise RuntimeError('frozen policy input differs')
                    (folder/f'request_{i:02d}.json').write_text(json.dumps(returned,allow_nan=False),encoding='utf-8')
                    row.update(status='validated',metrics=direction_metrics(returned['raw_action_chunk'],first,settings['calibration']['world_from_isaac_rotation']))
                    save()
                except Exception as error:row.update(status='failed',error=str(error));save();raise
        finally:
            (folder/'query_rows.json').write_text(json.dumps(report['frozen_queries'],allow_nan=False),encoding='utf-8')
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for p in sorted(folder.iterdir()):
                    if p.is_file():z.write(p,p.name)
            report['frozen_archive']={'file':archive.name,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()};save()
    vectors=np.array([x['metrics']['first_action']['decoded_world_delta_m'] for x in report['frozen_queries']])
    return {'success':False,'outcome':'frozen_queries_complete','zero_action_controls':r.controls==0,
        'first_action_mean_world_delta_m':vectors.mean(axis=0).tolist(),
        'first_action_std_world_delta_m':vectors.std(axis=0).tolist(),
        'positive_world_x_fraction':float(np.mean(vectors[:,0]>0)),
        'physics_step_before':initial_step,'physics_step_after':int(r.base.sim.get_physics_step_count()),
        'scope':'fixed robot, same image/state per six queries; deliberate black-image probes are out-of-distribution interventions, not valid benchmark scores'}


def run_replay(r,settings,report,save,execute):
    from registered_replay_probe import target_from_action
    case=settings['execution_case'];request,source=load_replay(settings['limit_replay_path'])
    if request['source_prefix']!=case['source_prefix']:raise ValueError('wrong replay prefix source')
    if source['settings']['calibration']!=settings['calibration'] or source['settings']['expected_joint_gains']!=settings['expected_joint_gains']:
        raise ValueError('replay source calibration/gains differ')
    first=r.state();initial=source['initial']
    if max(abs(a-b) for a,b in zip(first['joint_positions_rad'],initial['joint_positions_rad']))>.001 or math.dist(first['tcp_world_m'],initial['tcp_world_m'])>.0005:
        raise RuntimeError('replay start differs from source')
    if math.dist(first['object_pos_world_m'],initial['object_pos_world_m'])>.001:raise RuntimeError('replay object differs from source')
    report['replay_input']={**request,'request_sha256':hashlib.sha256(Path(settings['limit_replay_path']).read_bytes()).hexdigest()};save()
    result={'success':False,'outcome':'running','executed_controls':0,'source_prefix':case['source_prefix'],
        'max_source_tcp_difference_m':0. if case['cap_m']==.0005 else None,
        'max_source_joint_difference_rad':0. if case['cap_m']==.0005 else None,
        'scope':'same first 100 archived approach actions, open fingers, no new policy inference or grasp attempt'}
    try:
        for i,action in enumerate(request['actions'],1):
            state=r.state();target=target_from_action(state['hand_position_world_m'],state['hand_quat_world_xyzw'],settings['calibration']['world_from_isaac_rotation'],
                action,translation_cap=case['cap_m'],rotation_cap=math.radians(case['rotation_cap_deg']))
            target.update(source_control=i,source_result_sha256=request['source_result_sha256'])
            after=execute(target,1.,'policy-approach',None,list(action));result['executed_controls']=i
            if case['cap_m']==.0005:
                reference=source['trace'][i-1]['after']
                result['max_source_tcp_difference_m']=max(result['max_source_tcp_difference_m'],math.dist(after['tcp_world_m'],reference['tcp_world_m']))
                result['max_source_joint_difference_rad']=max(result['max_source_joint_difference_rad'],max(abs(a-b) for a,b in zip(after['joint_positions_rad'],reference['joint_positions_rad'])))
            if after['tcp_world_m'][2]<first['object_pos_world_m'][2]-.01:raise GraspAbort('replay TCP below object floor')
            if math.dist(after['tcp_world_m'],after['object_pos_world_m'])>math.dist(first['tcp_world_m'],first['object_pos_world_m'])+.02:raise GraspAbort('replay object distance increased >2cm')
        result['outcome']='archived_prefix_complete'
        if case['cap_m']==.0005:
            result['baseline_replay_matches_source']=result['max_source_tcp_difference_m']<=.0001 and result['max_source_joint_difference_rad']<=.001
            if not result['baseline_replay_matches_source']:result['outcome']='baseline_replay_mismatch'
    except GraspAbort as error:result['outcome']=str(error)
    finally:result['final_state']=r.state();report['replay_result']=result;save()
    return result
