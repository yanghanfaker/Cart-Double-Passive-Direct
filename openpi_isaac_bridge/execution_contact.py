"""Deterministic cube contact/lift comparison, independent of policy inference."""
import copy
import math
import numpy as np
from grasp_probe import GraspAbort,angle_deg,down_quat_from_object
from openpi_grasp_logic import close_gate


def compress_trace(path):
    """Keep long contact runs below GitHub's individual-file size limit."""
    import gzip
    import hashlib
    import shutil
    destination=path.with_suffix(path.suffix+'.gz')
    with path.open('rb') as source,gzip.open(destination,'wb',compresslevel=1) as target:shutil.copyfileobj(source,target)
    with path.open('rb') as source,gzip.open(destination,'rb') as restored:
        if hashlib.file_digest(source,'sha256').digest()!=hashlib.file_digest(restored,'sha256').digest():raise RuntimeError('compressed trace integrity check failed')
    path.unlink()
    return destination


def contact_cases():
    cases=[]
    for height in (0.,.005):
        for profile in ('kp350-kd50','kp1000-kd100'):
            for repeat in (1,2):
                case={'name':f'contact-h{height*1000:g}-{profile}-r{repeat}','kind':'contact-grasp',
                    'arm_profile':'k1687p5-d50','finger_profile':profile,'gripper_mode':'substep',
                    'cap_m':.0005,'rotation_cap_deg':.1,'grasp_height_m':height,'repeat':repeat,
                    'negative_control':False,'requires':[]}
                if profile=='kp1000-kd100':case['finger_armature']=1.
                cases.append(case)
        cases.append({**cases[-4],'name':f'contact-h{height*1000:g}-open-negative','negative_control':True,'repeat':1})
    return cases


def contact_scene(config,settings):
    from clean_scene_probe import validate_clean_scene
    result=copy.deepcopy(config);cal=settings['calibration']
    R=np.array(cal['world_from_isaac_rotation']);t=np.array(cal['world_from_isaac_translation_m'])
    tcp=R.T@(np.array(settings['native_initial']['robot0_eef_pos'])-t)
    result.update(name='deterministic-contact',object_xy_world_m=[float(tcp[0]+.01),float(tcp[1])])
    validate_clean_scene(result)
    return result


def run_contact_experiment(read_state,step_goal,case,handoff=False):
    """Fixed world goals, common feedback tolerances; step_goal advances 50 ms."""
    if case not in contact_cases():
        from execution_hybrid import hybrid_cases
        if case not in hybrid_cases():raise ValueError('undeclared contact case')
    if handoff and case.get('hybrid_role')!='policy':raise ValueError('handoff requires declared live policy case')
    initial=copy.deepcopy(read_state());obj=list(initial['object_pos_world_m'])
    q,yaw=down_quat_from_object(initial['object_quat_world_xyzw'])
    grasp=[obj[0],obj[1],obj[2]+case['grasp_height_m']]
    above=[obj[0],obj[1],obj[2]+.08]
    report={'success':False,'outcome':'running','negative_control':case['negative_control'],
        'program_goals':{'initial_tcp_world_m':initial['tcp_world_m'],'above_world_m':above,
            'grasp_world_m':grasp,'hand_quat_world_xyzw':q,'object_yaw_rad':yaw},
        'phases':[],'recorded_controls':0,'close_attempted':False,'lift_attempted':False,
        'contact_measurement':'unavailable; obstruction and relative pose are geometric proxies, not measured contacts',
        'scope':'simulator-truth fixed goals, feedback-bounded approach, closure, program lift and hold; no policy inference',
        'thresholds':{'position_m':.002,'rotation_deg':2.,'lift_m':.05,'minimum_held_lift_m':.035,
            'close_controls':60,'hold_controls':40,'relative_hold_drift_m':.01,'transfer_drift_m':.015,'hold_rotation_deg':10}}
    last=initial;lift_samples=[]
    def sample(p,quat,grip,phase):
        nonlocal last
        s=step_goal(list(p),list(quat),float(grip),phase)
        for key,size in (('tcp_world_m',3),('hand_quat_world_xyzw',4),('object_pos_world_m',3),
                ('object_in_tcp_pos_m',3),('object_in_tcp_quat_xyzw',4),('finger_positions_m',2)):
            if len(s[key])!=size or not np.isfinite(s[key]).all():raise GraspAbort('invalid contact state '+key)
        report['recorded_controls']+=1
        if phase=='lift':lift_samples.append(copy.deepcopy(s))
        speed=math.dist(s['object_pos_world_m'],last['object_pos_world_m'])/.05;last=s
        if speed>.5:raise GraspAbort('object speed exceeded 0.5m/s control estimate')
        if s['object_pos_world_m'][2]<obj[2]-.01:raise GraspAbort('object fell below initial support')
        if phase in ('orient','above','descend','refine','close') and math.dist(s['object_pos_world_m'],obj)>.02:raise GraspAbort('object moved >2cm before lift')
        return s
    def move(label,p,quat,grip,limit):
        tail=[]
        row={'name':label,'controls':0,'reached':False};report['phases'].append(row)
        for _ in range(limit):
            s=sample(p,quat,grip,label);row['controls']+=1;tail=(tail+[s])[-5:]
            if len(tail)==5 and all(math.dist(v['tcp_world_m'],p)<=.002 and angle_deg(v['hand_quat_world_xyzw'],quat)<=2 for v in tail):
                row['reached']=True;return s
        raise GraspAbort(label+' target timeout')
    try:
        if min(initial['finger_positions_m'])<.035:raise GraspAbort('initial gripper not open')
        if handoff:
            if not close_gate(initial)['allowed']:raise GraspAbort('program handoff outside close gate')
            report['handoff_refinement']={'start':copy.deepcopy(initial),'scope':'program truth refines position and yaw-aligned down orientation after policy approach'}
            move('refine',grasp,q,1.,1200)
        else:
            if initial['tcp_world_m'][2]<obj[2]+.08:raise GraspAbort('initial TCP below orientation clearance')
            move('orient',initial['tcp_world_m'],q,1.,1000)
            move('above',above,q,1.,1000)
            move('descend',grasp,q,1.,2500)
        if not close_gate(last)['allowed']:raise GraspAbort('deterministic close gate rejected')
        report['close_attempted']=not case['negative_control'];report['close_gate']=close_gate(last)
        grip=1. if case['negative_control'] else -1.
        closed=[sample(grasp,q,grip,'close') for _ in range(60)];tail=closed[-10:];fingers=tail[-1]['finger_positions_m']
        stable=all(max(s['finger_positions_m'][i] for s in tail)-min(s['finger_positions_m'][i] for s in tail)<=.0005 for i in (0,1))
        obstruction=all(v>.002 for v in fingers) and .006<=sum(fingers)<=.07 and abs(fingers[0]-fingers[1])<=.004
        centered=math.dist(last['tcp_world_m'],last['object_pos_world_m'])<=.015
        pose_ok=all(math.dist(s['tcp_world_m'],grasp)<=.002 and angle_deg(s['hand_quat_world_xyzw'],q)<=2 for s in tail)
        passed=stable and obstruction and centered and pose_ok
        report['close_check']={'stable_fingers':stable,'obstruction_candidate':obstruction,'object_near_tcp':centered,
            'pose_held':pose_ok,'finger_positions_m':fingers,'passed':passed,
            'total_opening_50_100_200_400ms_mm':[sum(closed[i]['finger_positions_m'])*1000 for i in (0,1,3,7)]}
        if case['negative_control']:
            report.update(outcome='negative_rejected' if not passed else 'negative_false_positive',negative_check_passed=not passed)
            return report
        if not passed:raise GraspAbort('close_candidate_failed')
        relative=list(last['object_in_tcp_pos_m']);lift=list(grasp);lift[2]+=.05
        report['program_goals']['lift_world_m']=lift;report['lift_attempted']=True
        move('lift',lift,q,-1.,2000)
        # Include every lift state in the transfer check, not just its endpoint.
        held=[sample(lift,q,-1.,'hold') for _ in range(40)]
        minimum=min(s['object_pos_world_m'][2]-obj[2] for s in held)
        drift=max(math.dist(s['object_in_tcp_pos_m'],held[0]['object_in_tcp_pos_m']) for s in held)
        rotation=max(angle_deg(s['object_in_tcp_quat_xyzw'],held[0]['object_in_tcp_quat_xyzw']) for s in held)
        transfer=max(math.dist(s['object_in_tcp_pos_m'],relative) for s in lift_samples+held)
        passed=(minimum>=.035 and drift<=.01 and rotation<=10 and transfer<=.015
            and all(all(v>.002 for v in s['finger_positions_m']) for s in held)
            and all(math.dist(s['tcp_world_m'],lift)<=.002 and angle_deg(s['hand_quat_world_xyzw'],q)<=2 for s in held))
        report['hold_check']={'min_object_lift_m':minimum,'relative_drift_m':drift,'relative_rotation_drift_deg':rotation,
            'max_transfer_drift_m':transfer,'passed':passed}
        report.update(success=passed,outcome='verified_lift_hold' if passed else 'lift_hold_failed')
    except GraspAbort as error:report['outcome']=str(error)
    finally:report['final_state']=copy.deepcopy(last)
    return report


def run_contact_runtime(r,settings,first,report,save,recorder,override):
    from openpi_grasp_logic import bounded_goal_target
    case=settings['execution_case']
    def execute(target,grip,phase,fixed_goal,raw_action=None):
        before=r.state()
        override.begin(-grip);index=r.controls+1
        recorder.begin_control({'case':case['name'],'phase':phase,'case_control_step':index})
        runtime_phase='approach' if phase in ('orient','above','descend','refine','policy-approach') else phase
        try:after=r.step(target,grip,runtime_phase,fixed_goal)
        except Exception as error:report['execution_abort']=str(error);raise
        finally:
            if recorder.pending is None and recorder.period_substeps==5:recorder.end_control()
            if r.controls==index:
                report['trace'].append({'step':index,'phase':phase,'before':before,'after':r.state(),'raw_action':raw_action,
                    'target':target,'fixed_goal':fixed_goal,'native_gripper_sign':-grip,
                    'gripper_apply_calls':override.calls,'ramp_target_m':override.value})
            save()
        if override.calls!=5:raise RuntimeError('expected five gripper substeps')
        return after
    def step_goal(p,q,grip,phase):
        before=r.state();target=bounded_goal_target(before['hand_position_world_m'],before['hand_quat_world_xyzw'],p,q)
        return execute(target,grip,phase,{'tcp_world_m':p,'hand_quat_world_xyzw':q,'program_phase':phase})
    if case.get('hybrid_role')=='policy':
        from execution_hybrid import run_hybrid_runtime
        result=run_hybrid_runtime(r,settings,report,save,step_goal,execute)
    else:result=run_contact_experiment(r.state,step_goal,case)
    rows=recorder.rows
    result['physics_checks']={'samples':len(rows),
        'max_object_speed_10ms_estimate_m_s':max((x['object_speed_10ms_estimate_m_s'] for x in rows),default=0.),
        'min_object_height_from_initial_m':min((x['object_pos_world_m'][2]-first['object_pos_world_m'][2] for x in rows),default=0.),
        'finger_force_scope':'computed/applied drive buffers, not object contact forces'}
    if result['physics_checks']['max_object_speed_10ms_estimate_m_s']>.5 or result['physics_checks']['min_object_height_from_initial_m']<-.01:
        result.update(success=False,outcome='10ms_object_motion_check_failed')
    held=[x['after'] for x in report['trace'] if x['phase']=='hold']
    if held:
        result['physics_checks']['minimum_hold_lift_from_scene_initial_m']=min(x['object_pos_world_m'][2]-first['object_pos_world_m'][2] for x in held)
        if result['physics_checks']['minimum_hold_lift_from_scene_initial_m']<.035:
            result.update(success=False,outcome='scene_initial_lift_check_failed')
    report['grasp_result']=result;save()
    if report.get('execution_abort'):raise RuntimeError(report['execution_abort'])
    if case['negative_control'] and not result.get('negative_check_passed',False):raise RuntimeError('open negative control did not reject grasp candidate')
    report.update(status='completed',case=case,final=r.state(),physics_controls=recorder.control_count,
        physics_substeps=recorder.substep_count,scope=result['scope'])
    r.save_images('contact_final');save()
    return report
