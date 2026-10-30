"""OpenPI approach/close requests with a deterministic lift verification stage."""
import math
from grasp_probe import GraspAbort,angle_deg


def close_gate(state):
    from kinematic_attribution import rotate
    tcp=state['tcp_world_m'];obj=state['object_pos_world_m']
    xy=math.hypot(tcp[0]-obj[0],tcp[1]-obj[1]);dz=tcp[2]-obj[2]
    z=rotate(state['hand_quat_world_xyzw'],[0,0,1])
    tilt=math.degrees(math.acos(max(-1.,min(1.,-z[2]))))
    total=math.hypot(xy,dz)
    return {'allowed':xy<=.012 and -.005<=dz<=.020 and total<=.015 and tilt<=20,
        'lateral_distance_m':xy,'tcp_above_object_m':dz,'tcp_object_distance_m':total,'hand_down_tilt_deg':tilt}


def bounded_goal_target(hand_position,hand_q,tcp_goal,q_goal):
    """Keep the same 0.5mm/0.1deg intermediate TCP target convention."""
    import numpy as np
    from geometry_audit import multiply,inverse
    from kinematic_attribution import rotate
    from registered_replay_probe import target_from_action
    current=np.array(hand_position)+rotate(hand_q,[0,0,.097])
    dq=np.array(multiply(q_goal,inverse(hand_q)))
    if dq[3]<0:dq=-dq
    n=np.linalg.norm(dq[:3]);dr=np.zeros(3) if n<1e-14 else dq[:3]*(2*math.atan2(n,dq[3])/n)
    return target_from_action(hand_position,hand_q,np.eye(3),[*((np.array(tcp_goal)-current)/.05).tolist(),*(dr/.5).tolist(),1.])


def run_grasp_experiment(read_state,query,step_model,step_goal,max_queries=200,announce=print):
    """Callbacks advance exactly one 50ms control, except query which pauses physics."""
    if type(max_queries) is not int or not 1<=max_queries<=200:raise ValueError('max_queries must be 1..200')
    initial=read_state();last=initial;controls=0;approach_controls=0;segment_distance=None
    trace=[];report={'success':False,'outcome':'running','model_queries':0,'model_close_requests':0,'blocked_close_requests':0,
        'phase_counts':{},'progress':trace,'close_events':[],'close_attempted':False,'lift_attempted':False,
        'scope':'OpenPI controls approach and requests closure; simulator-truth gate; program-generated 5cm lift and 2s hold verification, not autonomous VLA lift',
        'thresholds':{'close_xy_m':.012,'close_tcp_dz_m':[-.005,.020],'close_distance_m':.015,'close_tilt_deg':20,
            'min_held_object_lift_m':.035,'hold_controls':40,'max_object_speed_control_estimate_m_s':.5}}
    distance=lambda s:math.dist(s['tcp_world_m'],s['object_pos_world_m'])
    initial_distance=distance(initial);segment_distance=initial_distance
    report['initial_object_distance_m']=initial_distance

    def checked(state,phase):
        nonlocal last,controls
        for key,size in [('tcp_world_m',3),('hand_quat_world_xyzw',4),('object_pos_world_m',3),('object_quat_world_xyzw',4),
                ('object_in_tcp_pos_m',3),('object_in_tcp_quat_xyzw',4),('finger_positions_m',2)]:
            if len(state[key])!=size or not all(math.isfinite(v) for v in state[key]):raise GraspAbort('nonfinite/incomplete grasp state: '+key)
        controls+=1;report['phase_counts'][phase]=report['phase_counts'].get(phase,0)+1
        speed=math.dist(state['object_pos_world_m'],last['object_pos_world_m'])/.05
        state={**state,'object_lift_m':state['object_pos_world_m'][2]-initial['object_pos_world_m'][2],
            'object_speed_estimate_m_s':speed}
        last=state
        if speed>.5:raise GraspAbort('object speed estimate exceeded 0.5m/s')
        if state['object_lift_m']<-.01:raise GraspAbort('object fell below initial support')
        if phase in ('approach','close') and math.dist(state['object_pos_world_m'],initial['object_pos_world_m'])>.02:raise GraspAbort('object moved >2cm before verification lift')
        if phase=='approach' and state['tcp_world_m'][2]<initial['object_pos_world_m'][2]-.01:raise GraspAbort('TCP descended below diagnostic floor')
        return state

    def goal_step(p,q,grip,phase):return checked(step_goal(p,q,grip,phase),phase)

    def close_lift_hold():
        report['close_attempted']=True;state=read_state();p=list(state['tcp_world_m']);q=list(state['hand_quat_world_xyzw'])
        announce('[OpenPI grasp] allowed model close; hold pose for finger settling')
        closed=[goal_step(p,q,-1.,'close') for _ in range(60)]
        tail=closed[-10:];fingers=tail[-1]['finger_positions_m']
        stable=all(max(s['finger_positions_m'][i] for s in tail)-min(s['finger_positions_m'][i] for s in tail)<=.0005 for i in (0,1))
        obstruction=all(v>.002 for v in fingers) and .006<=sum(fingers)<=.07 and abs(fingers[0]-fingers[1])<=.004
        centered=distance(tail[-1])<=.015
        pose_ok=all(math.dist(s['tcp_world_m'],p)<=.002 and angle_deg(s['hand_quat_world_xyzw'],q)<=2 for s in tail)
        report['close_check']={'stable_fingers':stable,'obstruction_candidate':obstruction,'object_near_tcp':centered,
            'pose_held':pose_ok,'finger_positions_m':fingers,'passed':stable and obstruction and centered and pose_ok}
        if not report['close_check']['passed']:raise GraspAbort('close_candidate_failed')
        relative=tail[-1]['object_in_tcp_pos_m'];lift=list(tail[-1]['tcp_world_m']);lift[2]+=.05
        report['lift_attempted']=True;lift_states=[];reached=False
        announce('[OpenPI grasp] obstruction candidate; program verification lift +5cm')
        for _ in range(250):
            s=goal_step(lift,q,-1.,'lift');lift_states.append(s)
            if len(lift_states)>=5 and all(math.dist(v['tcp_world_m'],lift)<=.002 and angle_deg(v['hand_quat_world_xyzw'],q)<=2 for v in lift_states[-5:]):reached=True;break
        if not reached:raise GraspAbort('verification_lift_target_timeout')
        held=[goal_step(lift,q,-1.,'hold') for _ in range(40)]
        minimum=min(s['object_lift_m'] for s in held)
        drift=max(math.dist(s['object_in_tcp_pos_m'],held[0]['object_in_tcp_pos_m']) for s in held)
        rotation=max(angle_deg(s['object_in_tcp_quat_xyzw'],held[0]['object_in_tcp_quat_xyzw']) for s in held)
        transfer=max(math.dist(s['object_in_tcp_pos_m'],relative) for s in lift_states+held)
        passed=(minimum>=.035 and drift<=.010 and rotation<=10 and transfer<=.015
            and all(all(v>.002 for v in s['finger_positions_m']) for s in held)
            and all(math.dist(s['tcp_world_m'],lift)<=.002 and angle_deg(s['hand_quat_world_xyzw'],q)<=2 for s in held))
        report['hold_check']={'min_object_lift_m':minimum,'relative_position_drift_m':drift,'relative_rotation_drift_deg':rotation,
            'max_transfer_drift_m':transfer,'passed':passed}
        report.update(success=passed,outcome='verified_lift_hold' if passed else 'lift_hold_failed')

    try:
        if not all(abs(v-.04)<=.002 for v in initial['finger_positions_m']):raise GraspAbort('initial gripper not open')
        for query_index in range(1,max_queries+1):
            request=query(query_index);report['model_queries']+=1
            from openpi_step_probe import validate_action_chunk
            validate_action_chunk(request['raw_action_chunk'])
            if len(request['raw_action_chunk'])<10:raise ValueError('need at least ten predicted actions')
            for prefix_index,action in enumerate(request['raw_action_chunk'][:5],1):
                state=read_state();gate=close_gate(state)
                if action[6]>0:
                    report['model_close_requests']+=1
                    report['close_events'].append({'query':query_index,'prefix_index':prefix_index,'gate':gate,
                        'tcp_world_m':state['tcp_world_m'],'object_pos_world_m':state['object_pos_world_m']})
                    if gate['allowed']:
                        report['accepted_close_gate']=gate;close_lift_hold();return report
                    report['blocked_close_requests']+=1
                state=checked(step_model(action,1.),'approach');approach_controls+=1
                if min(state['finger_positions_m'])<.03:raise GraspAbort('gripper closed during approach')
                if distance(state)>initial_distance+.02:raise GraspAbort('object distance increased >2cm')
                if approach_controls%100==0:
                    progress=segment_distance-distance(state);segment_distance=distance(state)
                    trace.append({'approach_control':approach_controls,'query':query_index,'object_distance_m':segment_distance,'distance_reduction_last100_m':progress})
                    announce(f'[OpenPI grasp] controls={approach_controls}; distance={segment_distance*1000:.1f}mm; progress100={progress*1000:.1f}mm')
                    if progress<.002:raise GraspAbort('approach_progress_less_than_2mm_per100_controls')
        report['outcome']='approach_query_budget_exhausted'
    except GraspAbort as error:report['outcome']=str(error)
    finally:
        report.update(recorded_controls=controls,approach_controls=approach_controls,
            final_object_distance_m=distance(last),final_state=last)
    return report
