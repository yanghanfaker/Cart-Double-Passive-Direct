"""Isaac policy gripper/cap factorial and matched archived-action comparisons."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile
import numpy as np
from grasp_probe import GraspAbort,angle_deg
from native_libero_rollout import save_json
from openpi_step_probe import validate_action_chunk

SOURCE='semantics-live-open-0.5mm-0.1deg-r1'
START='semantics-reference-start'


def semantics_cases():
    from execution_contact import contact_cases
    base=copy.deepcopy(next(c for c in contact_cases() if c['name']=='contact-h5-kp350-kd50-r1'))
    result=[{**copy.deepcopy(base),'name':START,'hybrid_role':'reference'}]
    combinations=[(cap,rotation,grip) for cap,rotation in ((.0005,.1),(.002,.1),(.005,.1),(.005,.5)) for grip in ('open','model')]
    for repeat,order in ((1,combinations),(2,list(reversed(combinations)))):
        for cap,rotation,grip in order:
            result.append({**copy.deepcopy(base),'name':f'semantics-live-{grip}-{cap*1000:g}mm-{rotation:g}deg-r{repeat}',
                'semantics_role':'live','gripper_policy':grip,'cap_m':cap,'rotation_cap_deg':rotation,'repeat':repeat,
                'prefix_length':5,'max_policy_controls':300,'prompt':'pick up the cube',
                'requires':[START],'requires_success':[START]})
    for cap,rotation,grip in combinations:
        identity=cap==.0005 and rotation==.1 and grip=='open'
        result.append({**copy.deepcopy(base),'name':f'semantics-replay-{grip}-{cap*1000:g}mm-{rotation:g}deg',
            'semantics_role':'replay','gripper_policy':grip,'cap_m':cap,'rotation_cap_deg':rotation,
            'prefix_length':5,'max_policy_controls':100,'replay_source':SOURCE,'identity_replay':identity,
            'requires':[START,SOURCE]+([] if identity else ['semantics-replay-open-0.5mm-0.1deg']),
            'requires_success':[START]})
    result.append({**copy.deepcopy(base),'name':'semantics-reference-end','hybrid_role':'reference'})
    return result


def grip_command(action,policy,previous=1.):
    validate_action_chunk([list(action)])
    if policy=='open':return 1.
    if policy!='model':raise ValueError('unknown gripper policy')
    return (-1. if action[6]>0 else 1.) if action[6]!=0 else previous


def lift_window(states,initial):
    """Read-only outcome; no ground-truth targets or program-generated lift."""
    if len(states)<40:return {'passed':False,'samples':len(states)}
    tail=states[-40:];height=min(s['object_pos_world_m'][2]-initial['object_pos_world_m'][2] for s in tail)
    drift=max(math.dist(s['object_in_tcp_pos_m'],tail[0]['object_in_tcp_pos_m']) for s in tail)
    rotation=max(angle_deg(s['object_in_tcp_quat_xyzw'],tail[0]['object_in_tcp_quat_xyzw']) for s in tail)
    fingers=all(all(.002<v<.039 for v in s['finger_positions_m']) for s in tail)
    return {'passed':height>=.035 and drift<=.01 and rotation<=10 and fingers,'samples':40,
        'min_lift_m':height,'relative_drift_m':drift,'relative_rotation_deg':rotation,'fingers_nonempty_and_not_fully_open':fingers,
        'scope':'40 successive model-driven controls; kinematic lift/relative-pose evidence, no contact-force measurement'}


def run_policy_sequence(read_state,query,step_action,case,save=lambda x:None,archived=None):
    if case not in semantics_cases() or case.get('semantics_role') not in ('live','replay'):raise ValueError('declared policy case required')
    replay=case['semantics_role']=='replay'
    if replay!=(archived is not None):raise ValueError('replay source presence differs from case')
    if replay:validate_action_chunk(archived)
    initial=copy.deepcopy(read_state());distance=lambda s:math.dist(s['tcp_world_m'],s['object_pos_world_m'])
    result={'success':False,'outcome':'running','controls':0,'queries':0,'model_close_requests':0,'applied_close_controls':0,
        'gripper_policy':case['gripper_policy'],'min_distance_m':distance(initial),'initial_distance_m':distance(initial),
        'maximum_object_lift_m':0.,'decisions':[],
        'scope':'model pose and declared gripper policy throughout; no geometric close gate, refinement or program lift; cube proxy, not LIBERO benchmark',
        'timing_scope':'50ms controls; physics paused during inference; gripper ramp remains 0.1m/s, not native dynamics equivalence'}
    previous_distance=distance(initial);states=[];previous_grip=1.;budget=len(archived) if replay else case['max_policy_controls']
    if replay and not 0<budget<=100:raise ValueError('replay requires 1..100 actual archived controls')
    try:
        while result['controls']<budget:
            if replay:
                actions=archived[result['controls']:result['controls']+5];query_id=0
            else:
                request=query(result['queries']+1);result['queries']+=1;query_id=result['queries']
                validate_action_chunk(request['raw_action_chunk'])
                if len(request['raw_action_chunk'])<10:raise ValueError('at least ten predicted actions required')
                actions=request['raw_action_chunk'][:5]
            for prefix,raw in enumerate(actions,1):
                before=copy.deepcopy(read_state());grip=grip_command(raw,case['gripper_policy'],previous_grip);previous_grip=grip
                event={'control':result['controls']+1,'query':query_id,'prefix_index':prefix,'raw_action':list(raw),
                    'isaac_gripper':grip,'executed':False,'before':before};result['decisions'].append(event);save(result)
                after=step_action(list(raw),grip,query_id,prefix)
                event.update(executed=True,after=copy.deepcopy(after));result['controls']+=1
                result['model_close_requests']+=int(raw[6]>0);result['applied_close_controls']+=int(grip<0)
                states.append(copy.deepcopy(after));result['min_distance_m']=min(result['min_distance_m'],distance(after))
                result['maximum_object_lift_m']=max(result['maximum_object_lift_m'],after['object_pos_world_m'][2]-initial['object_pos_world_m'][2])
                result['lift_check']=lift_window(states,initial)
                if after['tcp_world_m'][2]<initial['object_pos_world_m'][2]-.01:raise GraspAbort('policy TCP below initial object floor')
                if result['maximum_object_lift_m']<.01 and distance(after)>distance(initial)+.02:raise GraspAbort('policy distance increased >2cm before lift')
                if result['lift_check']['passed']:
                    result.update(success=True,outcome='model_lift_hold_observed');return result
                if not replay and result['controls']%100==0 and result['maximum_object_lift_m']<.01:
                    progress=previous_distance-distance(after);previous_distance=distance(after)
                    if progress<.002:raise GraspAbort('policy_progress_less_than_2mm_per100_controls')
                save(result)
                if result['controls']>=budget:break
        result['outcome']='archived_prefix_complete' if replay else 'policy_control_budget_exhausted'
    except GraspAbort as error:result['outcome']=str(error)
    finally:result['final_state']=copy.deepcopy(read_state());save(result)
    return result


def prepare_semantics_replay(source,row,destination):
    source=Path(source);destination=Path(destination);destination.mkdir()
    payload=source.read_bytes();r=json.loads(payload)
    if row['status']!='completed' or row['case']['name']!=SOURCE or r['status']!='completed' or hashlib.sha256(payload).hexdigest()!=row['summary']['result_sha256']:
        raise ValueError('completed declared live source and digest required')
    if r['settings']['execution_case']!=row['case'] or not r.get('policy_queries'):raise ValueError('source settings/inference missing')
    traces=r['trace'][:100]
    if len(traces)<5:raise ValueError('need at least five executed source controls')
    (destination/'source_result.json').write_bytes(payload);save_json(destination/'source_row.json',row)
    records=[]
    for q in r['queries']:
        selected=[t for t in traces if t['target']['source_query']==q['query']]
        if not selected:continue
        archive=source.parent/q['archive'];data=archive.read_bytes()
        if hashlib.sha256(data).hexdigest()!=q['archive_sha256']:raise ValueError('source query archive SHA differs')
        name=f"query_{q['query']:04d}.zip";shutil.copyfile(archive,destination/name)
        records.append({'query':q['query'],'file':name,'sha256':q['archive_sha256']})
    request={'source_result_sha256':hashlib.sha256(payload).hexdigest(),
        'source_row_sha256':hashlib.sha256((destination/'source_row.json').read_bytes()).hexdigest(),
        'controls':len(traces),'actions':[t['raw_action'] for t in traces],'archives':records,
        'scope':'same actual first up-to-100 Isaac controls; zero new inference, no padding or additional settling'}
    path=destination/'request.json';save_json(path,request);load_semantics_replay(path);return path


def load_semantics_replay(path):
    path=Path(path);req=json.loads(path.read_bytes());raw=(path.parent/'source_result.json').read_bytes()
    row_bytes=(path.parent/'source_row.json').read_bytes();row=json.loads(row_bytes);source=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=req['source_result_sha256'] or hashlib.sha256(row_bytes).hexdigest()!=req['source_row_sha256']:
        raise ValueError('replay source digest mismatch')
    if row['status']!='completed' or row['case']['name']!=SOURCE or row['summary']['result_sha256']!=req['source_result_sha256']:
        raise ValueError('replay source row mismatch')
    if source['status']!='completed' or source['settings']['execution_case']!=row['case']:raise ValueError('source case/status differs')
    validate_action_chunk(req['actions']);traces=source['trace'][:100]
    if not 5<=req['controls']<=100 or req['controls']!=len(traces) or req['actions']!=[t['raw_action'] for t in traces]:raise ValueError('replay action prefix differs')
    query_ids=sorted({t['target']['source_query'] for t in traces})
    if [a['query'] for a in req['archives']]!=query_ids:raise ValueError('query archive coverage differs')
    for item in req['archives']:
        data=(path.parent/item['file']).read_bytes();q=next(q for q in source['queries'] if q['query']==item['query'])
        if hashlib.sha256(data).hexdigest()!=item['sha256'] or item['sha256']!=q['archive_sha256'] or q['status']!='validated':raise ValueError('query source digest/status differs')
        if q['capture_physics_step']!=q['post_query_physics_step']:raise ValueError('source physics advanced during inference')
        with zipfile.ZipFile(path.parent/item['file']) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate archive entries')
            response=json.loads(z.read('online_model_response.json'));request=json.loads(z.read('online_replay_request.json'))
        validate_action_chunk(response['actions'])
        if request['raw_action_chunk']!=response['actions']:raise ValueError('request/response action mismatch')
        for t in (t for t in traces if t['target']['source_query']==item['query']):
            if t['phase']!='policy-open' or t['native_gripper_sign']!=-1. or t['raw_action']!=response['actions'][t['target']['source_prefix_index']-1]:
                raise ValueError('source applied action differs from archived response')
    return req,source


def compare_identity(source,trace):
    if len(trace)!=min(100,len(source['trace'])):return {'passed':False,'reason':'control count differs'}
    maxima={'tcp_m':0.,'joint_rad':0.,'finger_m':0.,'object_m':0.,'hand_deg':0.,'object_deg':0.,'joint_velocity_rad_s':0.}
    for a,b in zip(source['trace'],trace):
        for which in ('before','after'):
            x=a[which];y=b[which]
            maxima['tcp_m']=max(maxima['tcp_m'],math.dist(x['tcp_world_m'],y['tcp_world_m']))
            maxima['object_m']=max(maxima['object_m'],math.dist(x['object_pos_world_m'],y['object_pos_world_m']))
            maxima['hand_deg']=max(maxima['hand_deg'],angle_deg(x['hand_quat_world_xyzw'],y['hand_quat_world_xyzw']))
            maxima['object_deg']=max(maxima['object_deg'],angle_deg(x['object_quat_world_xyzw'],y['object_quat_world_xyzw']))
            maxima['joint_velocity_rad_s']=max(maxima['joint_velocity_rad_s'],max(abs(p-q) for p,q in zip(x['joint_velocities_rad_s'],y['joint_velocities_rad_s'])))
            for field,key in (('joint_positions_rad','joint_rad'),('finger_positions_m','finger_m')):
                maxima[key]=max(maxima[key],max(abs(p-q) for p,q in zip(x[field],y[field])))
    return {'passed':maxima['tcp_m']<=.0001 and maxima['object_m']<=.0001 and maxima['finger_m']<=.0001 and maxima['joint_rad']<=.001 and maxima['joint_velocity_rad_s']<=.001 and maxima['hand_deg']<=.05 and maxima['object_deg']<=.05,
        'maximum_differences':maxima,'scope':'source and replay common prefix, before/after states; tolerances 0.1mm/1mrad/0.05deg'}


def run_semantics_runtime(r,settings,report,save,execute):
    from registered_replay_probe import target_from_action
    case=settings['execution_case'];source=None;archived=None
    report.update(queries=[],policy_queries=0,inference_attempts=0)
    if case['semantics_role']=='live':
        from openpi_health import check_openpi_health
        from openpi_step_probe import read_server_metadata
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from live_grasp_query import make_live_query
        report['server_health']=check_openpi_health(settings['server_host'],settings['server_port'])
        client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port']);report['server_metadata']=read_server_metadata(client)
        query=make_live_query(r,settings,report,save,client)
    else:
        req,source=load_semantics_replay(settings['semantics_replay_path']);archived=req['actions']
        for key in ('calibration','expected_joint_gains','expected_runtime_arm_gains'):
            if source['settings'][key]!=settings[key]:raise ValueError('replay calibration/gains differ: '+key)
        initial=r.state();expected=source['initial']
        if math.dist(initial['tcp_world_m'],expected['tcp_world_m'])>.0005 or math.dist(initial['object_pos_world_m'],expected['object_pos_world_m'])>.001 or max(abs(a-b) for a,b in zip(initial['joint_positions_rad'],expected['joint_positions_rad']))>.001:
            raise RuntimeError('replay initial configuration differs')
        report['semantics_replay_input']=req;query=lambda index:(_ for _ in ()).throw(RuntimeError('offline replay queried model'))
    def step(raw,grip,index,prefix):
        before=r.state();target=target_from_action(before['hand_position_world_m'],before['hand_quat_world_xyzw'],settings['calibration']['world_from_isaac_rotation'],raw,
            translation_cap=case['cap_m'],rotation_cap=math.radians(case['rotation_cap_deg']))
        target.update(source_query=index,source_prefix_index=prefix,raw_model_action=list(raw))
        phase='policy-open' if case['gripper_policy']=='open' else 'policy-model'
        try:return execute(target,grip,phase,None,raw)
        finally:
            if index and report['queries']:report['queries'][-1]['executed_prefix_steps']=r.controls-(index-1)*5
            save()
    def update(result):report['policy_semantics']=result;save()
    result=run_policy_sequence(r.state,query,step,case,update,archived)
    if case.get('identity_replay'):
        result['identity_check']=compare_identity(source,report['trace'])
    result['new_policy_queries']=report['policy_queries'];save();return result
