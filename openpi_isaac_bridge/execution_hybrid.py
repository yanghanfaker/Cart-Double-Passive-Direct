"""Live policy approach with explicit handoff to a validated program grasp."""
import copy
import hashlib
import json
import math
from pathlib import Path
from grasp_probe import GraspAbort
from openpi_grasp_logic import close_gate


def hybrid_cases():
    from execution_contact import contact_cases
    base=copy.deepcopy(next(c for c in contact_cases() if c['name']=='contact-h5-kp350-kd50-r1'))
    start={**base,'name':'hybrid-reference-start','hybrid_role':'reference'}
    combinations=[(n,mode) for n in (1,5) for mode in ('geometry','model-close')]
    result=[start]
    for repeat,order in ((1,combinations),(2,list(reversed(combinations)))):
        for prefix,mode in order:
            result.append({**copy.deepcopy(base),'name':f'hybrid-p{prefix}-{mode}-r{repeat}',
                'hybrid_role':'policy','prefix_length':prefix,'handoff_mode':mode,'repeat':repeat,
                'max_approach_controls':600,'prompt':'pick up the cube',
                'requires':['hybrid-reference-start'],'requires_success':['hybrid-reference-start']})
    result.append({**base,'name':'hybrid-reference-end','hybrid_role':'reference'})
    return result


def validate_contact_source(path):
    from execution_contact import contact_cases
    path=Path(path);m=json.loads(path.read_bytes())
    if m['status']!='completed' or [x['case'] for x in m['cases']]!=contact_cases():raise ValueError('complete prior contact matrix required')
    results={}
    for row in m['cases']:
        p=path.parent/row['result'];payload=p.read_bytes();r=json.loads(payload)
        digest=hashlib.sha256(payload).hexdigest()
        if row['status']!='completed' or digest!=row['summary']['result_sha256'] or r.get('status')!='completed':
            raise ValueError('prior contact result incomplete or SHA mismatch')
        g=r['grasp_result']
        if row['case']['negative_control']:
            if g['outcome']!='negative_rejected' or not g['negative_check_passed']:raise ValueError('prior negative control failed')
        elif not g['success'] or g['outcome']!='verified_lift_hold':raise ValueError('prior contact grasp did not pass')
        results[row['case']['name']]=digest
    return {'manifest':str(path.absolute()),'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'result_sha256':results}


def run_policy_approach(read_state,query,step_model,case,on_update=lambda result:None):
    if case not in hybrid_cases() or case.get('hybrid_role')!='policy':raise ValueError('declared live approach case required')
    initial=read_state();distance=lambda s:math.dist(s['tcp_world_m'],s['object_pos_world_m'])
    initial_distance=distance(initial);previous_distance=initial_distance;last=initial
    result={'outcome':'running','handoff':False,'controls':0,'queries':0,'model_close_requests':0,
        'blocked_close_requests':0,'geometry_gate_visits':0,'decisions':[],
        'initial_distance_m':initial_distance,'min_distance_m':initial_distance,
        'finger_policy':'model gripper recorded; actual fingers held open throughout approach',
        'timing_scope':'50ms simulated controls; inference pauses physics, no real-time inference claim'}
    try:
        while result['controls']<case['max_approach_controls']:
            request=query(result['queries']+1);result['queries']+=1
            from openpi_step_probe import validate_action_chunk
            actions=request['raw_action_chunk'];validate_action_chunk(actions)
            if len(actions)<10:raise ValueError('at least ten predicted actions required')
            for index,action in enumerate(actions[:case['prefix_length']],1):
                state=read_state();gate=close_gate(state);requested=action[6]>0
                result['model_close_requests']+=int(requested)
                result['geometry_gate_visits']+=int(gate['allowed'])
                if requested and not gate['allowed']:result['blocked_close_requests']+=1
                event={'query':result['queries'],'prefix_index':index,'raw_action':list(action),
                    'state_before':copy.deepcopy(state),'gate':gate,'model_close_requested':requested,'executed':False}
                result['decisions'].append(event)
                if gate['allowed'] and (case['handoff_mode']=='geometry' or requested):
                    result.update(handoff=True,outcome='handoff_ready',handoff_state=copy.deepcopy(state),
                        handoff_query=result['queries'],handoff_prefix_index=index)
                    event['decision']='program_handoff_without_executing_selected_model_action';on_update(result);return result
                event['decision']='execute_pose_with_fingers_open'
                last=step_model(list(action),result['queries'],index);event['executed']=True;result['controls']+=1
                event['state_after']=copy.deepcopy(last);result['min_distance_m']=min(result['min_distance_m'],distance(last))
                if min(last['finger_positions_m'])<.03:raise GraspAbort('policy approach fingers not open')
                if math.dist(last['object_pos_world_m'],initial['object_pos_world_m'])>.02:raise GraspAbort('object moved >2cm during approach')
                if last['tcp_world_m'][2]<initial['object_pos_world_m'][2]-.01:raise GraspAbort('policy TCP below object floor')
                if distance(last)>initial_distance+.02:raise GraspAbort('object distance increased >2cm')
                event['gate_after']=close_gate(last)
                if case['handoff_mode']=='geometry' and event['gate_after']['allowed']:
                    result.update(handoff=True,outcome='handoff_ready',handoff_state=copy.deepcopy(last),
                        handoff_query=result['queries'],handoff_prefix_index=index)
                    event['decision']='executed_pose_then_program_handoff';on_update(result);return result
                if result['controls']%100==0:
                    progress=previous_distance-distance(last);previous_distance=distance(last)
                    if progress<.002:raise GraspAbort('approach_progress_less_than_2mm_per100_controls')
                on_update(result)
                if result['controls']>=case['max_approach_controls']:break
        result['outcome']='approach_control_budget_exhausted'
    except GraspAbort as error:result['outcome']=str(error)
    finally:result['final_state']=copy.deepcopy(read_state());on_update(result)
    return result


def run_hybrid_runtime(r,settings,report,save,step_goal,execute):
    from openpi_health import check_openpi_health
    from openpi_step_probe import read_server_metadata
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from live_grasp_query import make_live_query
    from registered_replay_probe import target_from_action
    from execution_contact import run_contact_experiment
    case=settings['execution_case'];report.update(queries=[],inference_attempts=0,policy_queries=0)
    report['server_health']=check_openpi_health(settings['server_host'],settings['server_port'])
    client=WebsocketClientPolicy(host=settings['server_host'],port=settings['server_port'])
    report['server_metadata']=read_server_metadata(client);save()
    query=make_live_query(r,settings,report,save,client)
    def step_model(action,query_index,prefix_index):
        state=r.state()
        target=target_from_action(state['hand_position_world_m'],state['hand_quat_world_xyzw'],
            settings['calibration']['world_from_isaac_rotation'],action,
            translation_cap=case['cap_m'],rotation_cap=math.radians(case['rotation_cap_deg']))
        target.update(source_query=query_index,source_prefix_index=prefix_index,raw_model_action=list(action))
        after=execute(target,1.,'policy-approach',None,action)
        report['queries'][-1]['executed_prefix_steps']=prefix_index;save();return after
    announced=[None]
    def update(approach):
        report['policy_approach']=approach;save()
        mark=(approach['controls'],approach['outcome'])
        if mark!=announced[0] and (approach['controls']%50==0 or approach['outcome']!='running'):
            print(f"[Hybrid approach] controls={approach['controls']}, queries={approach['queries']}, min_distance={approach['min_distance_m']*1000:.2f}mm, outcome={approach['outcome']}",flush=True)
            announced[0]=mark
    approach=run_policy_approach(r.state,query,step_model,case,update)
    if not approach['handoff']:
        return {'success':False,'outcome':approach['outcome'],'policy_approach':approach,
            'close_attempted':False,'lift_attempted':False,'scope':'live policy approach ended before program handoff'}
    result=run_contact_experiment(r.state,step_goal,case,handoff=True)
    result['policy_approach']=approach
    result['scope']='live OpenPI approach with explicit simulator-truth gate and program refinement, closure, lift and hold; not autonomous VLA grasp'
    return result
